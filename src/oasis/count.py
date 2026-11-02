"""Quantum counting: how many routes reach ANC, and how sure are we?

A|0> = sum_r |garbage(r)>|merge(r)>|spot(r)> / sqrt(N): Hadamards on the route
register, then the walk.  The probability that the spot register shows ANC is
a = k/N = sin^2(theta).  Q = A S0 A^dagger S_ANC rotates by 2 theta.

We use ``qiskit_algorithms.IterativeAmplitudeEstimation`` (Grinko et al.
2021).  IAE measures A Q^m A|0> for powers m it picks so that theta stays
inside one half-period, which rules out aliasing, and it returns a
Clopper-Pearson interval that contains a with probability >= 1 - alpha (a
finite-sample guarantee, not an asymptotic one).

Stopping rule: we want the interval for k = N a to contain exactly one
integer.  Near theta, dk/dtheta = N sin(2 theta), so the theta-precision
needed is ~1 / (N sin 2 theta): IAE is run with epsilon chosen from the current
interval, and re-run with a tighter epsilon until k is pinned.  Each run uses
level alpha / R (R = maximum number of runs), so the interval we stop on holds
with probability >= 1 - alpha.  Cost: O(sqrt(k N) log(1/alpha)) Grover calls.

(We tried ``MaximumLikelihoodAmplitudeEstimation`` first: with a fixed
exponential schedule and few shots its likelihood keeps a second lobe at
k +- 1 -- e.g. k = 5 vs 6 at N = 256 -- so it needs many more queries.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from qiskit.circuit.library import grover_operator
from qiskit_algorithms import EstimationProblem, IterativeAmplitudeEstimation

from .backend import SnapshotSampler
from .circuits import WalkCircuits


@dataclass
class CountResult:
    k: int | None
    k_interval: tuple[float, float]
    alpha: float
    a_estimate: float
    rounds: list[dict] = field(default_factory=list)
    queries: int = 0
    simulated_grover_calls: int = 0


def estimation_problem(W: WalkCircuits) -> EstimationProblem:
    A = W.amplitude_prep()
    # S_good = the ANC phase on the spot register; Q = A S0 A^dagger S_good
    # built with Qiskit's grover_operator function.
    Q = grover_operator(W.anc_phase(), state_preparation=A)

    def good(bitstring: str) -> bool:      # measured spot register -> its code
        return W.is_marked_code(int(bitstring, 2))

    return EstimationProblem(state_preparation=A, objective_qubits=W.spot_qubits,
                             grover_operator=Q, is_good_state=good)


def _epsilon_for(N: int, a_hi: float) -> float:
    """IAE epsilon (half-width in a, ~ half-width in theta) that makes the k
    interval narrower than 1 near the current upper estimate."""
    th = min(math.asin(math.sqrt(min(max(a_hi, 0.0), 1.0))), math.pi / 4)
    slope = max(math.sin(2 * th), 1.0 / math.sqrt(N))  # da/dtheta, floored for a ~ 0
    return min(0.5, 0.4 / (N * slope))


def quantum_count(W: WalkCircuits, alpha: float = 0.01, shots: int = 32,
                  max_runs: int = 6, method: str = "statevector",
                  seed: int | None = None, log=None) -> CountResult:
    N = W.N
    problem = estimation_problem(W)
    sampler = SnapshotSampler(W, problem, method=method, shots=shots, seed=seed, log=log)
    alpha_run = alpha / max_runs
    res = CountResult(None, (0.0, float(N)), alpha, float("nan"))
    eps = 0.05                                     # first, coarse run
    for _ in range(max_runs):
        before = sampler.log.queries
        iae = IterativeAmplitudeEstimation(epsilon_target=eps, alpha=alpha_run, sampler=sampler)
        est = iae.estimate(problem)
        lo, hi = (max(0.0, est.confidence_interval[0]), min(1.0, est.confidence_interval[1]))
        k_lo, k_hi = N * lo, N * hi
        res.rounds.append(dict(epsilon=eps, a=float(est.estimation), k_interval=(k_lo, k_hi),
                               powers=list(est.powers), queries=sampler.log.queries - before))
        res.k_interval, res.a_estimate = (k_lo, k_hi), float(est.estimation)
        if math.ceil(k_lo - 1e-9) == math.floor(k_hi + 1e-9):
            res.k = math.floor(k_hi + 1e-9)
            break
        eps = min(eps / 2, _epsilon_for(N, hi))
    res.queries = sum(r["queries"] for r in res.rounds)
    res.simulated_grover_calls = sampler.simulated_grover_calls
    return res
