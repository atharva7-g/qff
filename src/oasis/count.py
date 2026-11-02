"""Quantum counting: how many routes reach ANC, and how sure are we?

A|0> = sum_r |garbage(r)>|spot(r)> / sqrt(N) (H on the route register, then the
walk).  The probability that the spot register shows ANC is a = k/N.  We use
``qiskit_algorithms.MaximumLikelihoodAmplitudeEstimation`` (MLAE, Suzuki et
al. 2020): run A Q^m for m in an exponential schedule 0, 1, 2, 4, ..., 2^J,
take the maximum-likelihood angle and a likelihood-ratio confidence interval
(the library's likelihood-ratio interval, vectorised).  If the interval
for k = N a contains exactly one integer, that is k at confidence 1 - alpha;
otherwise J grows by one.

Why MLAE: no extra qubits (unlike QPE counting), O(sqrt(kN)) queries like exact
counting, and its circuits are prefixes of one sequence, which
``SnapshotSampler`` simulates once.  The likelihood-ratio interval is the
standard (asymptotic, Wilks) interval; we report it as such.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

from qiskit.circuit.library import grover_operator
from qiskit_algorithms import EstimationProblem, MaximumLikelihoodAmplitudeEstimation

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
    # S_good: phase -1 on good states = the ANC phase on the spot register.
    # Q = A S0 A^dagger S_good via Qiskit's grover_operator function.
    Q = grover_operator(W.anc_phase(), state_preparation=A)
    spot = W.spot_qubits

    def good(bitstring: str) -> bool:      # measured spot register -> code
        return W.is_marked_code(int(bitstring, 2))

    return EstimationProblem(state_preparation=A, objective_qubits=spot,
                             grover_operator=Q, is_good_state=good)


def _grid(K: int) -> np.ndarray:
    # the log-likelihood oscillates with period ~ pi / (2(2K+1)); 64 points per period
    eps = 1e-12
    return np.linspace(eps, np.pi / 2 - eps, 64 * (2 * K + 1) + 1001)


def _loglik(thetas: np.ndarray, schedule, good, shots) -> np.ndarray:
    ll = np.zeros_like(thetas)
    for m, h in zip(schedule, good):
        ang = (2 * m + 1) * thetas
        with np.errstate(divide="ignore"):
            ll += h * np.log(np.sin(ang) ** 2) + (shots - h) * np.log(np.cos(ang) ** 2)
    return ll


def make_minimizer(schedule):
    """Global grid search + local refinement (MLAE's default brute force uses
    ~3000*2^J points in a Python loop, which is far too slow for J ~ 8)."""
    K = max(schedule)

    def minimizer(objective_fn, bounds):
        lo, hi = bounds[0]
        xs = np.linspace(lo, hi, 16 * (2 * K + 1) + 1001)
        vals = np.array([objective_fn(x) for x in xs])
        i = int(np.argmin(vals))
        a, b = xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]
        r = minimize_scalar(objective_fn, bounds=(a, b), method="bounded")
        return r.x if r.fun <= vals[i] else xs[i]

    return minimizer


def likelihood_ratio_interval(est, alpha: float) -> tuple[float, float]:
    """Same interval as ``MaximumLikelihoodAmplitudeEstimation.compute_confidence_interval
    (kind="likelihood_ratio")`` (outer interval of {theta : 2(ll_max - ll) <= chi2_1(1-alpha)}),
    vectorised with numpy so it stays fast for long schedules."""
    th = _grid(max(est.evaluation_schedule))
    ll = _loglik(th, est.evaluation_schedule, est.good_counts, est.shots)
    ll_mle = _loglik(np.array([est.theta]), est.evaluation_schedule, est.good_counts, est.shots)[0]
    ok = th[ll >= max(ll_mle, ll.max()) - chi2.ppf(1 - alpha, df=1) / 2]
    lo, hi = (ok.min(), ok.max()) if len(ok) else (0.0, np.pi / 2)
    return float(np.sin(lo) ** 2), float(np.sin(hi) ** 2)


def quantum_count(W: WalkCircuits, alpha: float = 0.01, shots: int = 16, J0: int = 2,
                  J_max: int | None = None, method: str = "statevector",
                  seed: int | None = None, log=None) -> CountResult:
    N = W.N
    problem = estimation_problem(W)
    sampler = SnapshotSampler(W, problem, method=method, shots=shots, seed=seed, log=log)
    J_max = J_max if J_max is not None else W.L  # 2^L Grover calls always suffices
    res = CountResult(None, (0.0, float(N)), alpha, float("nan"))
    for J in range(J0, J_max + 1):
        schedule = [0] + [2 ** j for j in range(J + 1)]
        before = sampler.log.queries
        mlae = MaximumLikelihoodAmplitudeEstimation(schedule, sampler=sampler,
                                                    minimizer=make_minimizer(schedule))
        est = mlae.estimate(problem)
        lo, hi = likelihood_ratio_interval(est, alpha)
        k_lo, k_hi = N * lo, N * hi
        res.rounds.append(dict(schedule=schedule, a=float(est.estimation),
                               k_interval=(k_lo, k_hi), queries=sampler.log.queries - before))
        res.k_interval, res.a_estimate = (k_lo, k_hi), float(est.estimation)
        if math.ceil(k_lo - 1e-9) == math.floor(k_hi + 1e-9):
            res.k = math.floor(k_hi + 1e-9)
            break
    res.queries = sum(r["queries"] for r in res.rounds)
    res.simulated_grover_calls = sampler.simulated_grover_calls
    return res
