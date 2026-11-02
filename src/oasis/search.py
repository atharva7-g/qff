"""Grover search for all k routes, each confirmed by one classical walk.

With k known, ``r = round(pi / (4 theta) - 1/2)`` Grover iterations
(sin^2 theta = k / N) put almost all weight on the k marked routes, uniformly.
Each measurement is one random marked route, so collecting all k takes about
k * H_k runs (coupon collector).  Every measured route is walked classically
exactly once to confirm it; that is the only classical walking allowed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qiskit import ClassicalRegister
from qiskit.circuit.library import grover_operator

from .backend import CountingSampler, run_shots
from .circuits import WalkCircuits, count_queries
from .walk import Campus


def grover_iterations(k: int, N: int) -> int:
    theta = math.asin(math.sqrt(k / N))
    return max(0, round(math.pi / (4 * theta) - 0.5))


@dataclass
class SearchResult:
    routes: list[str]
    runs: int
    rejected: int
    iterations: int
    queries: int


def search_circuit(W: WalkCircuits, r: int):
    G = grover_operator(
        W.phase_oracle(),
        state_preparation=W.uniform_routes(),
        reflection_qubits=W.route_qubits,
    )
    qc = W.new("search")
    qc.compose(W.load_start(), inplace=True)
    qc.compose(W.uniform_routes(), inplace=True)
    for _ in range(r):
        qc.compose(G, inplace=True)
    c = ClassicalRegister(W.L, "m")
    qc.add_register(c)
    qc.measure(W.route, c)
    return qc


def bits_to_route(bitstring: str) -> str:
    """Qiskit prints m[L-1]..m[0]; route bit t is route[t] = m[t]."""
    return bitstring[::-1]


def find_all(
    W: WalkCircuits,
    k: int,
    sampler: CountingSampler,
    campus: Campus,
    batch: int = 32,
    max_runs: int | None = None,
) -> SearchResult:
    if k <= 0:
        return SearchResult([], 0, 0, 0, 0)
    r = grover_iterations(k, W.N)
    qc = search_circuit(W, r)
    per_shot = count_queries(qc)
    max_runs = max_runs or int(10 * k * (math.log(k) + 2)) + 10
    found: list[str] = []
    runs = rejected = 0
    while len(found) < k and runs < max_runs:
        # Simulate a batch of runs, but only charge the runs we actually use.
        for bs in run_shots(sampler, qc, batch):
            runs += 1
            sampler.log.add(sampler.tag, per_shot)
            sampler.log.shots += 1
            route = bits_to_route(bs)
            if campus.H(W.A0, route) == W.j_star:   # the one allowed classical check
                if route not in found:
                    found.append(route)
            else:
                rejected += 1
            if len(found) >= k or runs >= max_runs:
                break
    return SearchResult(sorted(found), runs, rejected, r, per_shot * runs)
