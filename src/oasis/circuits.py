"""Quantum circuits (Qiskit) for the campus walk.

Registers
---------
route : L qubits, the password bits.  Step t *overwrites* route[t] with a
        "which predecessor" bit and the uncompute restores it, so the walk
        needs no per-step history.
merge : V qubits (V = 2 for p = 1019, L = 12), a tiny shift register that logs
        arrivals at the one spot where roads merge (see ``StepDesign``).
spot  : 8 qubits, the current spot's code.

Total L + V + 8 = 22 qubits for Ankan's instance (64 MB statevector).  There
is no flag qubit: "are we at ANC?" is a phase on the spot register
(``anc_phase``), and amplitude estimation measures the spot register itself.

Why in place: a step (s, b) -> t loses at most the bit b (each spot has two
incoming roads), so writing "which of the two" back into the road qubit keeps
the step a permutation.  The only exception on this campus is A = 0
(j = 1728): both its roads lead to A = 6, so A = 6 has three incoming roads.
Those arrivals are routed through a spare spot code and logged in ``merge``.
"""

from __future__ import annotations

from collections import deque
from functools import cached_property

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Instruction
from qiskit.circuit.library import DiagonalGate, UCGate

from .field import Fp2
from .reversible import apply_gates, control_table, decompose
from .survey import Survey

STEP_NAMES = ("step", "step_dg")
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_I = np.eye(2, dtype=complex)


class StepDesign:
    """Classical design of the in-place step for a surveyed campus and length L.

    * ``code[s]``: spot index -> spot-register value.
    * ``word_perm``: permutation of (code + 2^n * b) -> (code' + 2^n * c), the
      "main" part of a step; synthesised into single-target gates.
    * merge handling (only if some spot has 3 incoming roads): that spot T gets
      code 2^n - 2, the spare code 2^n - 1 is an alias "at T, arrived from the
      spot with two equal roads".  After the main permutation, every arrival at
      T shifts the merge register, the alias case writes a 1 into the fresh
      slot and the alias is folded back to T's code (a single bit flip).
    """

    def __init__(self, S: Survey, L: int):
        self.S, self.L = S, L
        pre: dict[int, list[tuple[int, int]]] = {}
        for s in range(S.n_spots):
            for b in (0, 1):
                pre.setdefault(S.succ[s][b], []).append((s, b))
        over = {t: v for t, v in pre.items() if len(v) > 2}
        self.merge_target: int | None = None
        self.merge_source: int | None = None
        if over:
            if len(over) != 1:
                raise NotImplementedError("more than one merge spot")
            (T, v), = over.items()
            srcs = [s for s, _ in v if S.succ[s][0] == S.succ[s][1] == T]
            if len(v) != 3 or len(set(srcs)) != 1:
                raise NotImplementedError("unsupported merge pattern")
            self.merge_target, self.merge_source = T, srcs[0]
        need = S.n_spots + (1 if over else 0)
        self.n = max(1, (need - 1).bit_length())  # spot-register width
        top = (1 << self.n) - 1
        self.alias = top if over else None
        others = [s for s in range(S.n_spots) if s != self.merge_target]
        self.code = {s: i for i, s in enumerate(others)}
        if over:
            self.code[self.merge_target] = top - 1
        # main permutation on n+1 bits
        n = self.n
        mapping: dict[int, int] = {}
        for t, v in pre.items():
            for s, b in v:
                if over and s == self.merge_source:
                    mapping[self.code[s] | (b << n)] = self.alias | (b << n)
            rest = [(s, b) for s, b in v if not (over and s == self.merge_source)]
            assert len(rest) <= 2
            for c, (s, b) in enumerate(sorted(rest, key=lambda sb: (sb[1], sb[0]))):
                mapping[self.code[s] | (b << n)] = self.code[t] | (c << n)
        size = 1 << (n + 1)
        used = set(mapping.values())
        free = iter(y for y in range(size) if y not in used)
        self.word_perm = [mapping[x] if x in mapping else next(free) for x in range(size)]
        assert sorted(self.word_perm) == list(range(size))
        # merge register size: arrivals at T are >= (shortest T->T cycle) apart
        if over:
            c = self._shortest_cycle(self.merge_target)
            self.V = 1 + (L - 1) // c if c else 1
        else:
            self.V = 0

    def _shortest_cycle(self, T: int) -> int | None:
        d: dict[int, int] = {}
        q: deque[int] = deque()
        for y in self.S.succ[T]:
            if y not in d:
                d[y] = 1
                q.append(y)
        while q:
            x = q.popleft()
            for y in self.S.succ[x]:
                if y not in d:
                    d[y] = d[x] + 1
                    q.append(y)
        return d.get(T)

    @cached_property
    def gates(self):
        return decompose(self.word_perm, self.n + 1)

    def classical_step(self, code: int, b: int, merge: int) -> tuple[int, int, int]:
        """What the step circuit does to a basis state (for tests/emulation)."""
        n = self.n
        w = apply_gates(self.gates, code | (b << n))
        code, b = w & ((1 << n) - 1), w >> n
        if self.V:
            hi = ((1 << n) - 1) & ~1
            if code & hi == hi:  # arrived at T (code or alias): shift merge reg
                assert merge >> (self.V - 1) == 0, "merge register overflow"
                merge = (merge << 1) & ((1 << self.V) - 1)
                if code == self.alias:
                    merge |= 1
                    code ^= 1
        return code, b, merge


class StepGate(Gate):
    """One query: the in-place step on (spot..., road bit, merge...)."""

    def __init__(self, design: StepDesign):
        super().__init__("step", design.n + 1 + design.V, [])
        self.design = design

    def _ops(self):
        """(kind, args) list of the gates, in application order."""
        D = self.design
        n, V = D.n, D.V
        word = list(range(n + 1))          # spot bits 0..n-1, road bit n
        merge = list(range(n + 1, n + 1 + V))
        ops = []
        for j, f in D.gates:
            table = control_table(j, f, n + 1)
            if not any(table):
                continue
            ctrls = [k for k in word if k != j]
            if all(table):
                ops.append(("x", [j]))
            else:
                ops.append(("uc", (table, [j, *ctrls])))
        if V:
            hi = list(range(1, n))          # spot bits that are 1 for both T and alias
            for i in range(V - 1, 0, -1):   # shift: merge[i] <- merge[i-1], merge[0] <- 0
                a, b = merge[i], merge[i - 1]
                ops += [("mcx", (hi + [a], b)), ("mcx", (hi + [b], a)), ("mcx", (hi + [a], b))]
            ops.append(("mcx", (list(range(n)), merge[0])))   # alias -> mark fresh slot
            ops.append(("mcx", (hi + [merge[0]], 0)))         # alias -> T's code
        return ops

    @staticmethod
    def _emit(qc: QuantumCircuit, ops):
        for kind, a in ops:
            if kind == "x":
                qc.x(a[0])
            elif kind == "uc":
                table, qubits = a
                qc.append(UCGate([_X if t else _I for t in table]), qubits)
            else:
                qc.mcx(*a)

    def _define(self):
        qc = QuantumCircuit(self.num_qubits, name="step")
        self._emit(qc, self._ops())
        self.definition = qc

    def inverse(self, annotated: bool = False):
        """Every gate in the step is an involution (I/X multiplexers, X, MCX), so
        the inverse is the same list reversed.  (Qiskit's generic
        ``UCGate.inverse()`` is *not* a multiplexer any more and would be
        transpiled into thousands of gates.)"""
        return StepDgGate(self)


class StepDgGate(Gate):
    """Inverse of one step; also one query."""

    def __init__(self, step: StepGate):
        super().__init__("step_dg", step.num_qubits, [])
        self._step = step

    def _define(self):
        qc = QuantumCircuit(self.num_qubits, name="step_dg")
        StepGate._emit(qc, self._step._ops()[::-1])
        self.definition = qc

    def inverse(self, annotated: bool = False):
        return self._step


def mcx_on_value(qc: QuantumCircuit, qubits, value: int, target):
    """target ^= [qubits == value] using a native (closed-control) MCX.

    Open controls (``ctrl_state``) are not native in Aer and get decomposed into
    long gate chains, so we conjugate the 0-controls with X instead."""
    zeros = [q for k, q in enumerate(qubits) if not (value >> k) & 1]
    if zeros:
        qc.x(zeros)
    qc.mcx(list(qubits), target)
    if zeros:
        qc.x(zeros)


class WalkCircuits:
    """Every circuit needed for one (A0, L, j*) on a surveyed campus."""

    def __init__(self, S: Survey, A0: Fp2, L: int, j_star: Fp2 | None):
        """j_star = None marks the Clock Tower (every spot with j in F_p)
        instead of one j-invariant (Part 2)."""
        if A0 not in S.index:
            raise ValueError("A0 is not a spot of the surveyed campus")
        self.S, self.L, self.A0, self.j_star = S, L, A0, j_star
        self.N = 1 << L
        self.design = D = StepDesign(S, L)
        self.route = QuantumRegister(L, "route")
        self.spot = QuantumRegister(D.n, "spot")
        self.merge = QuantumRegister(D.V, "merge") if D.V else None
        self.regs = tuple(r for r in (self.route, self.merge, self.spot) if r is not None)
        self.step = StepGate(D)
        self.step_dg = self.step.inverse()
        marked = (lambda j: j[1] == 0) if j_star is None else (lambda j: j == j_star)
        self.targets = [D.code[s] for s in range(S.n_spots) if marked(S.jtab[s])]

    # -- helpers ---------------------------------------------------------
    def new(self, name: str = "") -> QuantumCircuit:
        return QuantumCircuit(*self.regs, name=name)

    @property
    def num_qubits(self) -> int:
        return sum(len(r) for r in self.regs)

    def _index(self, bit) -> int:
        return self.new().find_bit(bit).index

    @property
    def route_qubits(self) -> list[int]:
        return [self._index(q) for q in self.route]

    @property
    def spot_qubits(self) -> list[int]:
        return [self._index(q) for q in self.spot]

    def is_marked_code(self, code: int) -> bool:
        return code in self.targets

    def _step_qubits(self, t: int):
        return [*self.spot, self.route[t], *(self.merge or [])]

    # -- building blocks -------------------------------------------------
    def load_start(self) -> QuantumCircuit:
        """|0> -> |code(A0)> on the spot register.  The only place A0 enters."""
        qc = self.new("load_A0")
        c0 = self.design.code[self.S.index[self.A0]]
        for k in range(len(self.spot)):
            if (c0 >> k) & 1:
                qc.x(self.spot[k])
        return qc

    def uniform_routes(self) -> QuantumCircuit:
        qc = self.new("H_route")
        qc.h(self.route)
        return qc

    def walk(self) -> QuantumCircuit:
        qc = self.new("walk")
        for t in range(self.L):
            qc.append(self.step, self._step_qubits(t))
        return qc

    def unwalk(self) -> QuantumCircuit:
        qc = self.new("unwalk")
        for t in reversed(range(self.L)):
            qc.append(self.step_dg, self._step_qubits(t))
        return qc

    def anc_phase(self) -> QuantumCircuit:
        """|spot> -> (-1)^[spot holds a curve with j = j*] |spot>.

        Reads only the spot register (one ``DiagonalGate``, native in Aer).  If no
        spot has j = j*, it is the identity, so nothing can ever be marked."""
        qc = self.new("at_ANC?")
        if self.targets:
            diag = [-1.0 if c in self.targets else 1.0 for c in range(1 << len(self.spot))]
            qc.append(DiagonalGate(diag), list(self.spot))
        return qc

    # -- composite circuits ------------------------------------------------
    def phase_oracle(self) -> QuantumCircuit:
        """|r, 0, s0> -> (-1)^[H(r) = j*] |r, 0, s0>      (2L queries)."""
        qc = self.new("O_phase")
        qc.compose(self.walk(), inplace=True)
        qc.compose(self.anc_phase(), inplace=True)
        qc.compose(self.unwalk(), inplace=True)
        return qc

    def amplitude_prep(self) -> QuantumCircuit:
        """A|0> = sum_r |garbage(r)>|merge(r)>|spot(r)> / sqrt(N)   (L queries).

        The good states are those whose spot register is ANC.  The walk is left
        uncomputed on purpose: amplitude estimation only looks at the spot
        register, and Q = A S0 A^dagger S_ANC then costs 2L queries per call."""
        qc = self.new("A")
        qc.compose(self.uniform_routes(), inplace=True)
        qc.compose(self.load_start(), inplace=True)
        qc.compose(self.walk(), inplace=True)
        return qc


def count_queries(qc: QuantumCircuit | Instruction) -> int:
    """Uses of the one-step circuit (``step``/``step_dg``) in a circuit, looking
    through nested definitions such as ``Q.power(k)``."""
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
