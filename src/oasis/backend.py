"""Simulator back end plus query accounting.

``CountingSampler`` wraps Aer's ``SamplerV2``: before running a circuit it counts
the one-step-circuit uses in the *logical* circuit (before transpiling, so the
``step`` gates are still visible) and multiplies by the shots actually taken.
That makes every query reported by the program a measured number, not a formula.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qiskit import QuantumCircuit, transpile
from qiskit.primitives import BaseSamplerV2
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler

from .circuits import count_queries

# Above this many qubits a dense statevector does not fit in ~16 GB.
STATEVECTOR_MAX_QUBITS = 28


def choose_method(num_qubits: int, method: str = "auto") -> str:
    if method != "auto":
        return method
    return "statevector" if num_qubits <= STATEVECTOR_MAX_QUBITS else "matrix_product_state"


@dataclass
class QueryLog:
    queries: int = 0
    shots: int = 0
    circuits: int = 0
    by_tag: dict[str, int] = field(default_factory=dict)

    def add(self, tag: str, q: int):
        self.queries += q
        self.by_tag[tag] = self.by_tag.get(tag, 0) + q


class CountingSampler(BaseSamplerV2):
    """A SamplerV2 that transpiles for Aer and logs step-queries x shots."""

    def __init__(self, method: str, shots: int = 100, seed: int | None = None, tag: str = ""):
        self.method = method
        self.backend = AerSimulator(method=method)
        self._inner = AerSampler(
            default_shots=shots,
            seed=seed,
            options={"backend_options": {"method": method}},
        )
        self.default_shots = shots
        self.log = QueryLog()
        self.tag = tag

    def run(self, pubs, *, shots: int | None = None, charge: bool = True):
        """charge=False: the caller charges per shot it actually consumes."""
        prepared = []
        for pub in pubs:
            circ, *rest = pub if isinstance(pub, tuple) else (pub,)
            n_shots = shots or (rest[1] if len(rest) > 1 and rest[1] else None) or self.default_shots
            if charge:
                self.log.add(self.tag, count_queries(circ) * n_shots)
                self.log.shots += n_shots
            self.log.circuits += 1
            tc = transpile(circ, self.backend, optimization_level=0)
            prepared.append((tc, *rest))
        return self._inner.run(prepared, shots=shots)


def run_shots(sampler: CountingSampler, circ: QuantumCircuit, shots: int) -> list[str]:
    """Simulate ``shots`` independent runs and return the outcomes in order.
    Nothing is charged here; the caller charges each run it actually uses."""
    res = sampler.run([(circ,)], shots=shots, charge=False).result()[0]
    reg = next(iter(res.data.keys()))
    return getattr(res.data, reg).get_bitstrings()
