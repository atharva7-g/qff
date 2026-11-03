"""Part 2 — the phone on the Clock Tower.

Top of the Clock Tower: T = {spots with j in F_p}.

Data (``fraction_scan``): for primes p = 3 (mod 4) spaced evenly in log p,
  * the exact fraction of spots on top, from the survey (p only), and
  * the fraction of random routes of length L = 2 ceil(log2 p) that end on top,
    from classical random walks (``fastwalk``; allowed: it measures the campus).
A least-squares fit of log(fraction) against log(p) gives the exponent; theory
says -1/2 (|T| is a class number, ~ sqrt(p); the campus has ~ p/4 spots).
The points scatter around the line because class numbers do (L(1, chi)).

Demo (``grover_curve``): Grover with the tower phase on a campus with at most
1 route in 32 on top.  One Aer run applies the iterations one after another
and saves, after every iteration r,
  * the route register's exact distribution (to draw measured runs from), and
  * the spot register's distribution *inside* the next oracle, after its walk
    (the phase does not change probabilities), whose weight on T is the exact
    success probability of measuring after r iterations.
So the simulation costs no extra steps.  Theory: sin^2((2r + 1) theta) with
sin^2 theta = a, the r = 0 success.  Classical with the same queries: one
Grover iteration is 2L steps and one classical route is L steps, so r
iterations buy 2r + 1 random routes: 1 - (1 - a)^(2r + 1).

Search (``bbht``): a is not known in advance on the real campus, so the actual
algorithm is BBHT (Boyer, Brassard, Hoyer, Tapp 1998): m = 1; pick j uniform in
[0, m), run j iterations, measure, check the route with one classical walk;
on failure m <- min(6/5 m, sqrt(N)).  Expected O(sqrt(1/a)) iterations.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import grover_operator

from . import fastwalk
from .circuits import WalkCircuits
from .search import index_to_route
from .survey import survey
from .walk import Campus


# -- data ----------------------------------------------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for q in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % q == 0:
            return n == q
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):   # exact below 3.3e24
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def primes_3mod4(lo_bits: float, hi_bits: float, per_octave: int) -> list[int]:
    """The first prime p = 3 (mod 4) at or above 2^x, for x evenly spaced."""
    out = []
    steps = int(round((hi_bits - lo_bits) * per_octave))
    for i in range(steps + 1):
        n = math.ceil(2 ** (lo_bits + i / per_octave))
        n += (3 - n) % 4
        while not is_prime(n):
            n += 4
        if not out or n != out[-1]:
            out.append(n)
    return out


@dataclass
class FractionPoint:
    p: int
    L: int
    spots: int | None          # exact, from the survey (None if not surveyed)
    top_spots: int | None
    walks: int
    top_walks: int

    @property
    def spot_fraction(self) -> float | None:
        return None if self.spots is None else self.top_spots / self.spots

    @property
    def route_fraction(self) -> float:
        return self.top_walks / self.walks


def _measure(args) -> FractionPoint:
    p, walks, exact, seed = args
    L = 2 * math.ceil(math.log2(p))
    spots = top = None
    if exact:
        S = survey(p)
        spots, top = S.n_spots, sum(1 for j in S.jtab if j[1] == 0)
    hits = 0
    for k in range(0, walks, 100_000):                  # chunks bound the memory
        n = min(100_000, walks - k)
        hits += int(fastwalk.random_walks(p, L, n, seed=seed + k).sum())
    return FractionPoint(p, L, spots, top, walks, hits)


def fraction_scan(primes: list[int], walks: int, exact_max: int, seed: int = 0,
                  workers: int | None = None) -> list[FractionPoint]:
    jobs = [(p, walks, p <= exact_max, seed + i) for i, p in enumerate(primes)]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_measure, jobs))


@dataclass
class Fit:
    slope: float
    stderr: float
    intercept: float
    n: int


def fit_exponent(ps, fracs) -> Fit:
    """Least squares log f = c + beta log p (points with f = 0 are dropped)."""
    x = np.log(np.asarray(ps, float))
    y = np.asarray(fracs, float)
    keep = y > 0
    x, y = x[keep], np.log(y[keep])
    (b, c), cov = np.polyfit(x, y, 1, cov=True)
    return Fit(float(b), float(math.sqrt(cov[0, 0])), float(c), int(keep.sum()))


# -- demo: Grover on the tower --------------------------------------------

@dataclass
class GroverCurve:
    route_dists: list[np.ndarray]     # route distribution after r iterations
    success: list[float]              # exact P[measured route ends on top]
    a: float                          # success with r = 0: fraction of routes on top

    @property
    def theta(self) -> float:
        return math.asin(math.sqrt(self.a))

    def theory(self, r: int) -> float:
        return math.sin((2 * r + 1) * self.theta) ** 2

    def classical(self, r: int) -> float:
        return 1 - (1 - self.a) ** (2 * r + 1)


def diffusion(W: WalkCircuits):
    """A S0 A^dagger on the route register (Qiskit's grover_operator, empty oracle)."""
    return grover_operator(W.new(), state_preparation=W.uniform_routes(),
                           reflection_qubits=W.route_qubits)


def grover_curve(W: WalkCircuits, R: int, backend) -> GroverCurve:
    qc = W.new("tower_grover")
    qc.compose(W.load_start(), inplace=True)
    qc.compose(W.uniform_routes(), inplace=True)
    D = diffusion(W)
    for r in range(R + 1):
        qc.save_probabilities(W.route_qubits, label=f"route{r}")
        qc.compose(W.walk(), inplace=True)
        qc.save_probabilities(W.spot_qubits, label=f"spot{r}")
        if r < R:
            qc.compose(W.anc_phase(), inplace=True)
            qc.compose(W.unwalk(), inplace=True)
            qc.compose(D, inplace=True)
    data = backend.run(transpile(qc, backend, optimization_level=0)).result().data(0)
    dists, success = [], []
    for r in range(R + 1):
        p = np.clip(np.asarray(data[f"route{r}"], dtype=float), 0, None)
        dists.append(p / p.sum())
        s = np.asarray(data[f"spot{r}"], dtype=float)
        success.append(float(sum(s[c] for c in W.targets)))
    return GroverCurve(dists, success, success[0])


def check(campus: Campus, W: WalkCircuits, route: str) -> bool:
    """The one allowed classical walk of a measured route."""
    return campus.j(campus.walk(W.A0, route))[1] == 0


def measured_success(curve: GroverCurve, W: WalkCircuits, campus: Campus,
                     shots: int, rng) -> list[float]:
    """Run each depth `shots` times: draw the measured route, check it classically."""
    out = []
    for dist in curve.route_dists:
        xs = rng.choice(len(dist), size=shots, p=dist)
        out.append(float(np.mean([check(campus, W, index_to_route(int(x), W.L)) for x in xs])))
    return out


@dataclass
class BBHTStats:
    trials: int
    failures: int                     # hit the round limit without a route
    mean_iterations: float
    mean_queries: float               # quantum queries (one step = one query)
    mean_checks: float                # classical checks (L steps each)
    cap: int


def bbht(curve: GroverCurve, W: WalkCircuits, campus: Campus, trials: int, rng,
         max_rounds: int = 200) -> BBHTStats:
    """BBHT with lambda = 6/5.  Depths are drawn from the simulated distributions,
    so m is capped at the deepest simulated depth + 1 (and at sqrt(N))."""
    cap = min(int(math.isqrt(W.N)), len(curve.route_dists))
    memo: dict[int, bool] = {}
    its = checks = fails = 0
    for _ in range(trials):
        m = 1.0
        for _ in range(max_rounds):
            j = int(rng.integers(0, max(1, int(m))))
            x = int(rng.choice(W.N, p=curve.route_dists[j]))
            its += j
            checks += 1
            if x not in memo:
                memo[x] = check(campus, W, index_to_route(x, W.L))
            if memo[x]:
                break
            m = min(6 / 5 * m, cap)
        else:
            fails += 1
    return BBHTStats(trials, fails, its / trials, 2 * W.L * its / trials, checks / trials, cap)


# -- plots -----------------------------------------------------------------

INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_fraction(points: list[FractionPoint], fit_spots: Fit | None, fit_routes: Fit, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    _style(ax)
    ex = [q for q in points if q.spots is not None]
    if ex:
        ax.plot([q.p for q in ex], [q.spot_fraction for q in ex], "o", ms=5, color=BLUE,
                label="spots on top (exact, survey)")
    rt = [q for q in points if q.top_walks > 0]
    ax.plot([q.p for q in rt], [q.route_fraction for q in rt], "s", ms=4, color=ORANGE,
            mfc="none", mew=1.5, label="random routes ending on top (sampled)")
    ps = np.array([points[0].p, points[-1].p], float)
    for f, c, name in ((fit_spots, BLUE, "spots"), (fit_routes, ORANGE, "routes")):
        if f is not None:
            ax.plot(ps, np.exp(f.intercept) * ps ** f.slope, color=c, lw=1.5, alpha=0.7,
                    label=f"fit ({name}): slope {f.slope:.3f} ± {2 * f.stderr:.3f}")
    # slope fixed at -1/2, intercept by least squares over the route fractions
    c = np.mean([math.log(q.route_fraction) + 0.5 * math.log(q.p) for q in rt])
    ax.plot(ps, math.exp(c) * ps ** -0.5, ls="--", color=MUTED, lw=1,
            label="slope −1/2 (theory)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("p", color=INK)
    ax.set_ylabel("fraction on top of the Clock Tower", color=INK)
    ax.set_title("How often routes end on the Clock Tower", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_grover(curve: GroverCurve, measured: list[float], shots: int, p: int, L: int, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = len(curve.success) - 1
    rs = np.arange(R + 1)
    fine = np.linspace(0, R, 400)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    _style(ax)
    ax.plot(fine, np.sin((2 * fine + 1) * curve.theta) ** 2, ls="--", color=MUTED, lw=1.2,
            label=r"theory $\sin^2((2r+1)\theta)$")
    ax.plot(rs, curve.success, "-o", color=BLUE, lw=2, ms=5,
            label="Grover, Aer simulation (exact)")
    ax.plot(rs, measured, "x", color=AQUA, ms=8, mew=2,
            label=f"Grover, {shots} measured runs per depth, checked classically")
    ax.plot(fine, 1 - (1 - curve.a) ** (2 * fine + 1), color=ORANGE, lw=2,
            label="classical random routes, same number of queries")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(f"Grover iterations r   (one iteration = {2 * L} queries)", color=INK)
    ax.set_ylabel("P[route ends on top]", color=INK)
    ax.set_title(f"Clock Tower search, p = {p}, L = {L}, a = {curve.a:.4f} ≈ 1/{1 / curve.a:.1f}",
                 color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
