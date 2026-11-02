"""Quantum counting: how many routes reach ANC, and how sure are we?

We use Iterative Quantum Amplitude Estimation (qiskit_algorithms) on
    A |0> = sum_r |r>|0>|s0>|[H(r) == j*]> / sqrt(N),
whose good-state probability is a = k / N.  IAE returns an interval that holds
a with probability >= 1 - alpha.  We tighten epsilon until the interval for
k = N a contains exactly one integer; that integer is k, with confidence
>= 1 - alpha.

Why IAE and not textbook QPE counting: QPE needs ~log2(sqrt(N k)) extra
counting qubits and controlled Grover powers; IAE needs no extra qubits and
directly yields a rigorous confidence interval.  For small a its cost is
O(sqrt(k N) log(1/alpha)) Grover calls, the same scaling as exact counting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qiskit.circuit.library import grover_operator
from qiskit_algorithms import EstimationProblem, IterativeAmplitudeEstimation

from .backend import CountingSampler
from .circuits import WalkCircuits


@dataclass
class CountResult:
    k: int | None              # None if no unique integer could be pinned
    k_interval: tuple[float, float]
    alpha: float
    a_estimate: float
    rounds: list[dict]         # one entry per IAE run (epsilon, interval, queries)
    grover_calls: int          # sum over runs of IAE's num_oracle_queries (Q uses)
    queries: int               # one-step-circuit uses, measured


def estimation_problem(W: WalkCircuits) -> EstimationProblem:
    A = W.new("A")
    A.compose(W.load_start(), inplace=True)
    A.compose(W.uniform_routes(), inplace=True)
    A.compose(W.bit_oracle(), inplace=True)
    # Q = A S_0 A^dagger S_good, with S_good = Z on the flag (built with Qiskit's
    # grover_operator function; qiskit_algorithms' default uses a deprecated class).
    good = W.new("S_good")
    good.z(W.flag[0])
    Q = grover_operator(good, state_preparation=A)
    return EstimationProblem(state_preparation=A, objective_qubits=[W.flag_qubit],
                             grover_operator=Q)


def quantum_count(
    W: WalkCircuits,
    sampler: CountingSampler,
    alpha: float = 0.01,
    eps0: float = 0.25,
    max_runs: int = 20,
) -> CountResult:
    problem = estimation_problem(W)
    N = W.N
    eps = eps0
    rounds = []
    calls = 0
    for _ in range(max_runs):
        before = sampler.log.queries
        iae = IterativeAmplitudeEstimation(epsilon_target=eps, alpha=alpha, sampler=sampler)
        res = iae.estimate(problem)
        lo, hi = res.confidence_interval
        lo, hi = max(0.0, lo), min(1.0, hi)
        calls += res.num_oracle_queries
        k_lo, k_hi = math.ceil(N * lo - 1e-9), math.floor(N * hi + 1e-9)
        rounds.append(
            dict(epsilon=eps, a=res.estimation, a_interval=(lo, hi),
                 k_interval=(N * lo, N * hi), queries=sampler.log.queries - before,
                 powers=list(res.powers))
        )
        if k_lo == k_hi:
            return CountResult(k_lo, (N * lo, N * hi), alpha, res.estimation, rounds,
                               calls, sum(r["queries"] for r in rounds))
        if k_lo > k_hi:  # interval strictly between two integers: shrink and retry
            pass
        eps /= 2
        if eps < 1e-9:
            break
    last = rounds[-1]
    return CountResult(None, last["k_interval"], alpha, last["a"], rounds, calls,
                       sum(r["queries"] for r in rounds))
