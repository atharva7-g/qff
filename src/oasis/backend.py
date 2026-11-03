"""Simulator back end plus query accounting.

``CountingSampler`` wraps Aer's ``SamplerV2``: before running a circuit it counts
the one-step-circuit uses in the *logical* circuit (before transpiling, so the
``step`` gates are still visible) and multiplies by the shots actually taken.
That makes every query reported by the program a measured number, not a formula.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.primitives import BaseSamplerV2
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler

from .circuits import count_queries

# Above this many qubits a dense statevector does not fit in ~16 GB.
STATEVECTOR_MAX_QUBITS = 29


# Single precision is ~15% faster and its ~1e-7 rounding is irrelevant here;
# gate fusion does not help circuits made of 9-qubit multiplexers.
AER_OPTIONS = {"precision": "single", "fusion_enable": False}


def make_backend(method: str) -> AerSimulator:
    opts = AER_OPTIONS if method == "statevector" else {}
    return AerSimulator(method=method, **opts)


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
        self.backend = make_backend(method)
        opts = {"method": method, **(AER_OPTIONS if method == "statevector" else {})}
        self._inner = AerSampler(
            default_shots=shots,
            seed=seed,
            options={"backend_options": opts},
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



class _Job:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class SnapshotSampler(BaseSamplerV2):
    """Sampler for amplitude-estimation circuits of the form  A Q^m, measure(spot).

    Every such circuit is a *prefix* of the same sequence A, Q, Q, Q, ...  So
    instead of simulating each circuit from scratch, we simulate the sequence
    once with Aer, saving the exact outcome distribution of the spot register
    after each needed power (``save_probabilities``), and continue later from a
    saved statevector (``set_statevector``) if a longer schedule is requested.
    Shots are then drawn from those exact distributions, which is the same
    distribution as running each circuit on its own.  Queries are charged as if
    every circuit were run separately on hardware: (L + 2L m) x shots.

    With ``reuse=True`` measurements are kept (a repeated power returns the
    same counts at no cost); with ``reuse=False`` (what IAE needs) every request
    draws fresh shots and is charged.
    """

    def __init__(self, W, problem, method: str = "statevector", shots: int = 16,
                 seed: int | None = None, log: QueryLog | None = None, tag: str = "count",
                 reuse: bool = False):
        self.W, self.problem = W, problem
        self.backend = make_backend(method)
        self.default_shots = shots
        self.rng = np.random.default_rng(seed)
        self.log = log or QueryLog()
        self.tag = tag
        self.probs: dict[int, np.ndarray] = {}
        self.counts: dict[int, dict[str, int]] = {}
        self.reuse = reuse
        self._M = -1            # highest power simulated so far
        self._sv = None         # statevector after A Q^_M
        self.simulated_grover_calls = 0

    def power_of(self, circ: QuantumCircuit) -> int:
        L = self.W.L
        q = count_queries(circ)
        m2 = q // L - 1
        if q % L or m2 % 2:
            raise ValueError("not an A Q^m circuit")
        return m2 // 2

    def _simulate(self, needed: set[int]):
        """Extend the simulated sequence up to max(needed), saving the spot
        register's distribution after *every* power (256 numbers each), so any
        power <= the highest one simulated is available later."""
        if not needed or max(needed) <= self._M:
            return
        top = max(needed)
        W = self.W
        qc = W.new("prefix")
        if self._sv is None:
            qc.compose(self.problem.state_preparation, inplace=True)
            start = 0
            qc.save_probabilities(W.spot_qubits, label="m0")
        else:
            # single-precision rounding drifts the norm by ~1e-6 over many gates;
            # renormalise (in double) so Aer accepts the saved state.
            sv = np.asarray(self._sv, dtype=np.complex128)
            qc.set_statevector(sv / np.linalg.norm(sv))
            start = self._M
        for m in range(start + 1, top + 1):
            qc.compose(self.problem.grover_operator, inplace=True)
            qc.save_probabilities(W.spot_qubits, label=f"m{m}")
        qc.save_statevector(label="sv")
        tq = transpile(qc, self.backend, optimization_level=0)
        data = self.backend.run(tq).result().data(0)
        for m in range(start, top + 1):
            if f"m{m}" in data:
                self.probs[m] = np.asarray(data[f"m{m}"], dtype=float)
        self.simulated_grover_calls += top - max(self._M, 0)
        self._M = top
        self._sv = data["sv"]

    def run(self, pubs, *, shots: int | None = None):
        from qiskit.primitives import PrimitiveResult, SamplerPubResult
        from qiskit.primitives.containers import BitArray, DataBin

        circs = [p[0] if isinstance(p, tuple) else p for p in pubs]
        powers = [self.power_of(c) for c in circs]
        self._simulate({m for m in powers if m not in self.probs})
        n_shots = shots or self.default_shots
        nbits = len(self.W.spot)
        out = []
        for circ, m in zip(circs, powers):
            if not self.reuse or m not in self.counts:   # new experiment: sample and pay
                p = np.clip(self.probs[m], 0, None)
                hits = self.rng.multinomial(n_shots, p / p.sum())
                self.counts[m] = {format(c, f"0{nbits}b"): int(h) for c, h in enumerate(hits) if h}
                self.log.add(self.tag, count_queries(circ) * n_shots)
                self.log.shots += n_shots
                self.log.circuits += 1
            creg = circ.cregs[0].name
            data = DataBin(**{creg: BitArray.from_counts(self.counts[m], num_bits=nbits)})
            out.append(SamplerPubResult(data, metadata={"shots": n_shots}))
        return _Job(PrimitiveResult(out))
