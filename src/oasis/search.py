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

import numpy as np
from qiskit import ClassicalRegister, transpile
from qiskit.circuit.library import grover_operator

from .backend import CountingSampler
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


def search_circuit(W: WalkCircuits, r: int, measure: bool = True):
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
    if measure:
        c = ClassicalRegister(W.L, "m")
        qc.add_register(c)
        qc.measure(W.route, c)
    return qc


def route_distribution(W: WalkCircuits, r: int, backend) -> np.ndarray:
    """Exact outcome distribution of measuring the route register after
    H, G^r -- simulated once (``save_probabilities``).  Each Grover run is then a
    draw from it, which is exactly what running the circuit again would give."""
    qc = search_circuit(W, r, measure=False)
    qc.save_probabilities(W.route_qubits, label="p")
    data = backend.run(transpile(qc, backend, optimization_level=0)).result().data(0)
    p = np.clip(np.asarray(data["p"], dtype=float), 0, None)
    return p / p.sum()


def index_to_route(x: int, L: int) -> str:
    """Outcome index x has route[t] = bit t of x."""
    return "".join(str((x >> t) & 1) for t in range(L))


def find_all(
    W: WalkCircuits,
    k: int,
    sampler: CountingSampler,
    campus: Campus,
    max_runs: int | None = None,
    seed: int | None = None,
) -> SearchResult:
    if k <= 0:
        return SearchResult([], 0, 0, 0, 0)
    r = grover_iterations(k, W.N)
    per_shot = count_queries(search_circuit(W, r))
    max_runs = max_runs or int(10 * k * (math.log(k) + 2)) + 10
    dist = route_distribution(W, r, sampler.backend)
    rng = np.random.default_rng(seed)
    found: list[str] = []
    runs = rejected = 0
    while len(found) < k and runs < max_runs:
        x = int(rng.choice(len(dist), p=dist))     # one Grover run, measured
        runs += 1
        sampler.log.add(sampler.tag, per_shot)
        sampler.log.shots += 1
        route = index_to_route(x, W.L)
        if campus.H(W.A0, route) == W.j_star:       # the one allowed classical check
            if route not in found:
                found.append(route)
        else:
            rejected += 1
    return SearchResult(sorted(found), runs, rejected, r, per_shot * runs)
