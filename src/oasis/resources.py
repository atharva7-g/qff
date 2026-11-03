"""Resources after transpiling to a stated basis (what the rules ask for).

Basis: {cx, rz, sx, x}, the IBM hardware basis, at optimization level 1.
The simulator itself runs the multiplexers natively; these numbers are what
the same circuits cost as plain CX + single-qubit gates.

Every step acts on the whole spot register, so steps cannot overlap in time
and the depth of a walk is the sum of the step depths.  We therefore
transpile each building block once (step, step_dg, phase, diffusion) and add
them up, instead of transpiling a circuit with millions of gates.
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import grover_operator

from .circuits import WalkCircuits

BASIS = ["cx", "rz", "sx", "x"]


@dataclass(frozen=True)
class Cost:
    cx: int
    depth: int
    size: int

    def __add__(self, o: "Cost") -> "Cost":
        return Cost(self.cx + o.cx, self.depth + o.depth, self.size + o.size)

    def __mul__(self, k: int) -> "Cost":
        return Cost(self.cx * k, self.depth * k, self.size * k)

    __rmul__ = __mul__

    def __str__(self) -> str:
        return f"cx {self.cx:,}, depth {self.depth:,}, gates {self.size:,}"


ZERO = Cost(0, 0, 0)


def cost(qc: QuantumCircuit) -> Cost:
    tq = transpile(qc, basis_gates=BASIS, optimization_level=1)
    return Cost(tq.count_ops().get("cx", 0), tq.depth(), tq.size())


def _gate(g) -> QuantumCircuit:
    qc = QuantumCircuit(g.num_qubits)
    qc.append(g, range(g.num_qubits))
    return qc


class WalkCosts:
    """Transpiled cost of each building block of ``WalkCircuits``."""

    def __init__(self, W: WalkCircuits):
        self.W = W
        self.step = cost(_gate(W.step))
        self.step_dg = cost(_gate(W.step_dg))
        self.phase = cost(W.anc_phase())
        self.diffusion = cost(grover_operator(W.new(), state_preparation=W.uniform_routes(),
                                              reflection_qubits=W.route_qubits))
        self.prep = cost(W.uniform_routes().compose(W.load_start()))

    @property
    def oracle(self) -> Cost:
        L = self.W.L
        return L * self.step + self.phase + L * self.step_dg

    @property
    def grover_iteration(self) -> Cost:
        return self.oracle + self.diffusion

    def search(self, r: int) -> Cost:
        """load A0, H on routes, then r Grover iterations (measurement not counted)."""
        return self.prep + r * self.grover_iteration

    def report(self, say=print):
        W = self.W
        say(f"resources after transpiling to {{{', '.join(BASIS)}}} (optimization level 1):")
        say(f"  qubits: {W.num_qubits}")
        rows = [("one step (1 query)", self.step), ("phase on spot register", self.phase),
                ("diffusion", self.diffusion),
                (f"oracle ({2 * W.L} queries)", self.oracle),
                (f"Grover iteration ({2 * W.L} queries)", self.grover_iteration)]
        for name, c in rows:
            say(f"  {name:<32} cx {c.cx:>10,}  depth {c.depth:>10,}  gates {c.size:>10,}")
