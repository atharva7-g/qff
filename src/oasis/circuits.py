"""Quantum circuits (Qiskit) for the campus walk.

Registers (in this qubit order — it keeps MPS bond dimensions small)
--------------------------------------------------------------------
route : L qubits, the password bits b_0..b_{L-1}
tag   : L qubits, one per step; records which of the (at most two) spots with
        the same road bit led to the new spot, so the step is reversible
spot  : ceil(log2 #spots) qubits, index of the current curve in the survey
flag  : 1 qubit, "are we at ANC?"

Why a tag and not the full history (Bennett)?  For a fixed road bit b the map
s -> succ_b(s) is two-to-one at worst (A = 0 sends both roads to A = 6), so one
extra bit per step suffices instead of a whole spot register per step.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Instruction
from qiskit.circuit.library import MCXGate, UnitaryGate

from .field import Fp2
from .survey import Survey

STEP_NAMES = ("step", "step_dg")


def step_permutation(S: Survey) -> np.ndarray:
    """perm[x] = y on 2^(n+2) basis states, with x = s + 2^n b + 2^(n+1) tag.

    Valid inputs (s < #spots, tag = 0) go to (succ_b(s), b, c) where c is the
    rank of s among the preimages of succ_b(s) under road b.  The remaining
    inputs are paired with the remaining outputs in increasing order so the
    whole map is a permutation (those states never occur in the algorithm).
    """
    n = S.spot_bits
    size = 1 << (n + 2)
    perm = -np.ones(size, dtype=np.int64)
    used = np.zeros(size, dtype=bool)
    for b in (0, 1):
        pre: dict[int, list[int]] = {}
        for s in range(S.n_spots):
            pre.setdefault(S.succ[s][b], []).append(s)
        for t, ss in pre.items():
            if len(ss) > 2:
                raise ValueError("a road bit has >2 preimages; one tag qubit is not enough")
            for c, s in enumerate(sorted(ss)):
                y = t | (b << n) | (c << (n + 1))
                perm[s | (b << n)] = y
                used[y] = True
    free_out = iter(np.flatnonzero(~used))
    for x in range(size):
        if perm[x] < 0:
            perm[x] = next(free_out)
    assert sorted(perm.tolist()) == list(range(size))
    return perm


class StepGate(Gate):
    """The one-step circuit: |s, b, 0> -> |succ_b(s), b, c>.  One use = one query.

    It is its own named gate (definition: the exact permutation unitary) so that
    queries can be counted by name anywhere in a circuit, and so that
    ``inverse()`` gives a gate called ``step_dg``.  Qiskit 2.x has no
    classical-function -> reversible-circuit synthesiser, and decomposing a generic
    10-qubit unitary into CNOTs would give ~10^6 gates, so we keep the exact
    unitary; Aer simulates ``unitary`` natively.
    """

    def __init__(self, perm: np.ndarray):
        nq = int(len(perm)).bit_length() - 1
        super().__init__("step", nq, [])
        self._perm = perm

    def _define(self):
        size = len(self._perm)
        U = np.zeros((size, size), dtype=complex)
        U[self._perm, np.arange(size)] = 1.0
        qc = QuantumCircuit(self.num_qubits, name="step")
        qc.append(UnitaryGate(U, check_input=False), range(self.num_qubits))
        self.definition = qc

    def validate_parameter(self, parameter):  # no parameters
        return parameter


def anc_spots(S: Survey, j_star: Fp2) -> list[int]:
    """Spots whose j-invariant is j*.  Survey table look-up on the small campus;
    on the real campus this becomes j-invariant arithmetic (Part 3)."""
    return [s for s in range(S.n_spots) if S.jtab[s] == j_star]


class WalkCircuits:
    """Builds every circuit needed for one (A0, L, j*) on a surveyed campus."""

    def __init__(self, S: Survey, A0: Fp2, L: int, j_star: Fp2):
        if A0 not in S.index:
            raise ValueError("A0 is not a spot of the surveyed campus")
        self.S, self.L, self.A0, self.j_star = S, L, A0, j_star
        self.N = 1 << L
        n = S.spot_bits
        self.route = QuantumRegister(L, "route")
        self.tag = QuantumRegister(L, "tag")
        self.spot = QuantumRegister(n, "spot")
        self.flag = QuantumRegister(1, "flag")
        self.regs = (self.route, self.tag, self.spot, self.flag)
        self.step = StepGate(step_permutation(S))
        self.step_dg = self.step.inverse()
        self.targets = anc_spots(S, j_star)

    # -- helpers ---------------------------------------------------------
    def new(self, name: str = "") -> QuantumCircuit:
        return QuantumCircuit(*self.regs, name=name)

    @property
    def num_qubits(self) -> int:
        return self.new().num_qubits

    def qubit_index(self, reg: QuantumRegister, i: int = 0) -> int:
        return self.new().find_bit(reg[i]).index

    @property
    def route_qubits(self) -> list[int]:
        return [self.qubit_index(self.route, t) for t in range(self.L)]

    @property
    def flag_qubit(self) -> int:
        return self.qubit_index(self.flag)

    # -- building blocks -------------------------------------------------
    def load_start(self) -> QuantumCircuit:
        """|0> -> |index(A0)> on the spot register.  The only place A0 enters."""
        qc = self.new("load_A0")
        s0 = self.S.index[self.A0]
        for k in range(len(self.spot)):
            if (s0 >> k) & 1:
                qc.x(self.spot[k])
        return qc

    def uniform_routes(self) -> QuantumCircuit:
        qc = self.new("H_route")
        qc.h(self.route)
        return qc

    def walk(self) -> QuantumCircuit:
        """L steps; step t is controlled (inside the permutation) by route[t]."""
        qc = self.new("walk")
        for t in range(self.L):
            qc.append(self.step, [*self.spot, self.route[t], self.tag[t]])
        return qc

    def unwalk(self) -> QuantumCircuit:
        qc = self.new("unwalk")
        for t in reversed(range(self.L)):
            qc.append(self.step_dg, [*self.spot, self.route[t], self.tag[t]])
        return qc

    def check(self) -> QuantumCircuit:
        """flag ^= [j(spot) == j*], reading only the spot register.

        One MCX per matching spot, with the spot index as control state.  If j*
        matches no spot the circuit is empty and nothing is ever marked.
        """
        qc = self.new("at_ANC?")
        n = len(self.spot)
        for s in self.targets:
            qc.append(MCXGate(n, ctrl_state=s), [*self.spot, self.flag[0]])
        return qc

    # -- oracles ---------------------------------------------------------
    def bit_oracle(self) -> QuantumCircuit:
        """|r,0,s0,f> -> |r,0,s0,f ^ [H(r) == j*]>          (2L queries)."""
        qc = self.new("O_bit")
        qc.compose(self.walk(), inplace=True)
        qc.compose(self.check(), inplace=True)
        qc.compose(self.unwalk(), inplace=True)
        return qc

    def phase_oracle(self) -> QuantumCircuit:
        """|r,0,s0,0> -> (-1)^[H(r) == j*] |r,0,s0,0>       (2L queries)."""
        qc = self.new("O_phase")
        qc.compose(self.walk(), inplace=True)
        qc.compose(self.check(), inplace=True)
        qc.z(self.flag[0])
        qc.compose(self.check(), inplace=True)
        qc.compose(self.unwalk(), inplace=True)
        return qc


def count_queries(qc: QuantumCircuit | Instruction) -> int:
    """Uses of the one-step circuit (``step`` or ``step_dg``), found by walking
    the circuit and all nested definitions (e.g. ``Q.power(k)`` repeats)."""
    memo: dict[int, int] = {}

    def rec(circ: QuantumCircuit) -> int:
        total = 0
        for inst in circ.data:
            op = inst.operation
            if op.name in STEP_NAMES:
                total += 1
                continue
            key = id(op)
            if key not in memo:
                d = getattr(op, "definition", None)
                memo[key] = rec(d) if d is not None else 0
            total += memo[key]
        return total

    if isinstance(qc, Instruction):
        return rec(qc.definition) if qc.definition is not None else 0
    return rec(qc)
