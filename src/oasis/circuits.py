"""Quantum circuits (Qiskit) for the campus walk.

Registers
---------
route : L qubits, the password bits b_0..b_{L-1}
tag   : L qubits, one per step; records which of the (at most two) spots with
        the same road bit led to the new spot, so the step is reversible
spot  : ceil(log2 #spots) qubits, index of the current curve in the survey
flag  : 1 qubit, "are we at ANC?"

Why a tag and not the full history (Bennett)?  For a fixed road bit b the map
s -> succ_b(s) is two-to-one at worst (e.g. A = 0 sends both roads to A = 6), so
exactly one extra bit per step is needed, not a whole spot register.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCXGate, UnitaryGate

from .field import Fp2
from .survey import Survey


def step_permutation(S: Survey) -> np.ndarray:
    """perm[x] = y on 2^(n+2) basis states; x = s + 2^n * b + 2^(n+1) * tag.

    Valid inputs (s < #spots, tag = 0) go to (succ_b(s), b, c) with c the rank of s
    among the preimages of succ_b(s) under road b.  The remaining inputs are
    paired with the remaining outputs in increasing order so the whole thing is
    a permutation (those states never occur).
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
                x = s | (b << n)
                y = t | (b << n) | (c << (n + 1))
                perm[x] = y
                used[y] = True
    free_out = iter(np.flatnonzero(~used))
    for x in range(size):
        if perm[x] < 0:
            perm[x] = next(free_out)
    assert sorted(perm.tolist()) == list(range(size))
    return perm


def perm_to_gate(perm: np.ndarray, label: str) -> UnitaryGate:
    size = len(perm)
    U = np.zeros((size, size), dtype=complex)
    U[perm, np.arange(size)] = 1.0
    return UnitaryGate(U, label=label, check_input=False)


def step_gate(S: Survey) -> UnitaryGate:
    """One query: |s, b, 0> -> |succ_b(s), b, c>.  Qubit order: spot..., b, tag."""
    return perm_to_gate(step_permutation(S), "step")


def anc_spots(S: Survey, j_star: Fp2) -> list[int]:
    """Spots whose j-invariant is j*.  This is the survey's table look-up; on the
    real campus it becomes j-invariant arithmetic on the spot register (Part 3)."""
    return [s for s in range(S.n_spots) if S.jtab[s] == j_star]


class WalkCircuits:
    """Builds the oracle and state-preparation circuits for one (A0, L, j*)."""

    def __init__(self, S: Survey, A0: Fp2, L: int, j_star: Fp2):
        if A0 not in S.index:
            raise ValueError("A0 is not a spot of the surveyed campus")
        self.S, self.L, self.A0, self.j_star = S, L, A0, j_star
        n = S.spot_bits
        # Qubit order (matters for MPS): route_t next to tag_t, spot + flag last.
        self.route = QuantumRegister(L, "route")
        self.tag = QuantumRegister(L, "tag")
        self.spot = QuantumRegister(n, "spot")
        self.flag = QuantumRegister(1, "flag")
        self.regs = (self.route, self.tag, self.spot, self.flag)
        self.step = step_gate(S)
        self.step_inv = self.step.inverse()
        self.step_inv.label = "step†"
        self.targets = anc_spots(S, j_star)
        self.queries_per_oracle = 2 * L

    def new(self, name: str = "") -> QuantumCircuit:
        return QuantumCircuit(*self.regs, name=name)

    def load_start(self) -> QuantumCircuit:
        """|0> -> |index(A0)> on the spot register (A0 enters only here)."""
        qc = self.new("load_A0")
        s0 = self.S.index[self.A0]
        for k in range(len(self.spot)):
            if (s0 >> k) & 1:
                qc.x(self.spot[k])
        return qc

    def walk(self) -> QuantumCircuit:
        """L controlled steps: step t reads road bit route[t]."""
        qc = self.new("walk")
        for t in range(self.L):
            qc.append(self.step, [*self.spot, self.route[t], self.tag[t]])
        return qc

    def check(self) -> QuantumCircuit:
        """flag ^= [j(spot) == j*], looking only at the spot register."""
        qc = self.new("at_ANC?")
        n = len(self.spot)
        for s in self.targets:
            qc.append(MCXGate(n, ctrl_state=s), [*self.spot, self.flag[0]])
        return qc

    def compute_flag(self) -> QuantumCircuit:
        """walk, then check: leaves flag = [H(route) == j*] (spot/tag dirty)."""
        qc = self.new("compute")
        qc.compose(self.walk(), inplace=True)
        qc.compose(self.check(), inplace=True)
        return qc

    def uncompute_walk(self) -> QuantumCircuit:
        qc = self.new("unwalk")
        for t in reversed(range(self.L)):
            qc.append(self.step_inv, [*self.spot, self.route[t], self.tag[t]])
        return qc

    def bit_oracle(self) -> QuantumCircuit:
        """|r>|0>|0>|f> -> |r>|0>|0>|f ^ [H(r) == j*]>   (2L queries)."""
        qc = self.new("O_bit")
        qc.compose(self.compute_flag(), inplace=True)
        qc.compose(self.uncompute_walk(), inplace=True)
        return qc

    def phase_oracle(self) -> QuantumCircuit:
        """|r> -> (-1)^[H(r) == j*] |r>, ancillas clean   (2L queries).

        Uses the phase-kickback-free form: compute flag, Z, uncompute flag.
        """
        qc = self.new("O_phase")
        qc.compose(self.walk(), inplace=True)
        qc.compose(self.check(), inplace=True)
        qc.z(self.flag[0])
        qc.compose(self.check(), inplace=True)
        qc.compose(self.uncompute_walk(), inplace=True)
        return qc

    def uniform_routes(self) -> QuantumCircuit:
        qc = self.new("H_route")
        qc.h(self.route)
        return qc


def count_queries(qc: QuantumCircuit) -> int:
    """Number of one-step-circuit uses (step or step†) after flattening."""
    ops = qc.decompose(reps=10).count_ops() if False else _count(qc)
    return ops


def _count(qc: QuantumCircuit) -> int:
    total = 0
    for inst in qc.data:
        op = inst.operation
        if getattr(op, "label", None) in ("step", "step†"):
            total += 1
        elif op.definition is not None and op.name not in ("unitary",):
            sub = _count(op.definition)
            if hasattr(op, "power") is False:
                pass
            total += sub
    return total
