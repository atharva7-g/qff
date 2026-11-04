"""Isogenies on Oasis.

    uv run oasis                    # Part 1, Ankan's instance (p=1019, L=12)
    uv run oasis --L 4 --jstar ...  # Part 1, smaller instances for testing
    uv run oasis --part 2           # Part 2: tower data + Grover demo (p=5923)
    uv run oasis --part 3           # Part 3: arithmetic step, all 510 checks, cost at 2^256
"""

from __future__ import annotations

import argparse
import math
import time

from pathlib import Path

import numpy as np

from . import tower
from .backend import CountingSampler, QueryLog, choose_method, make_backend
from .circuits import WalkCircuits
from .count import quantum_count
from .resources import WalkCosts
from .search import find_all
from .survey import survey
from .walk import Campus

DEFAULTS = {
    1: dict(p=1019, A0="977 + 297i", L=12, jstar="797 + 921i"),
    # smallest prime with <= 1/32 of its spots on top; from A0, 103/4096 routes are
    2: dict(p=5923, A0="66 + 1530i", L=12, jstar=None),
    3: dict(p=1019, A0=None, L=None, jstar=None),
}


def optimal_quantum_find_all(N: int, k: int) -> float:
    """Oracle calls for finding all k marked items one by one with Grover,
    removing nothing: sum_{i=1..k} (pi/4) sqrt(N/i) ~ (pi/2) sqrt(N k)."""
    return sum(math.pi / 4 * math.sqrt(N / i) for i in range(1, k + 1))


def run(p: int, A0: str, L: int, jstar: str, alpha: float = 0.01, method: str = "auto",
        shots: int = 32, seed: int | None = None, quiet: bool = False) -> dict:
    say = (lambda *a, **k: None) if quiet else print
    campus = Campus(p)
    F = campus.F
    a0, js = F.parse(A0), F.parse(jstar)

    t0 = time.time()
    S = survey(p)  # independent of A0, L, j*
    say(f"survey: {S.n_spots} spots, {S.moves} roads walked (<= 2 per spot)")

    W = WalkCircuits(S, a0, L, js)
    m = choose_method(W.num_qubits, method)
    D = W.design
    say(f"circuit: {W.num_qubits} qubits (route {L}, merge {D.V}, spot {D.n}); "
        f"step = {sum(1 for _ in D.gates)} single-target gates; simulator: Aer {m}")
    say(f"ANC phase marks spot code(s) {W.targets or 'none'} (j = {F.fmt(js)})")
    if not quiet:
        costs = WalkCosts(W)
        costs.report(say)

    N = W.N
    log = QueryLog()
    def show(r):
        say(f"  IAE run eps={r['epsilon']:.2e}: max power {max(r['powers'])}, "
            f"k in [{r['k_interval'][0]:.3f}, {r['k_interval'][1]:.3f}]  queries={r['queries']}"
            f"  [{time.time() - t0:.0f}s]", flush=True)

    cnt = quantum_count(W, alpha=alpha, shots=shots, method=m, seed=seed, log=log, progress=show)
    out = dict(N=N, L=L, count=cnt, routes=[], search=None)
    if cnt.k is None:
        say("could not pin k to one integer within the run budget; "
            f"k in [{cnt.k_interval[0]:.2f}, {cnt.k_interval[1]:.2f}]")
        return out
    k = cnt.k
    say(f"quantum count: k = {k}  (confidence >= {1 - alpha:.4f})")

    if k == 0:
        say("No 12-step route from the Maths department reaches ANC." if L == 12
            else f"No {L}-step route reaches the target.")
        sr = None
    else:
        ss = CountingSampler(m, shots=32, seed=seed, tag="search")
        sr = find_all(W, k, ss, campus, seed=seed)
        out["routes"] = sr.routes
        say(f"Grover search: {sr.iterations} iterations/run, {sr.runs} runs, "
            f"{sr.rejected} rejected by the classical check")
        if len(sr.routes) < k:
            say(f"WARNING: only {len(sr.routes)} of {k} routes found within the run budget")
        say("routes (each confirmed by one classical walk):")
        for r in sr.routes:
            say(f"  {r}")
    out["search"] = sr

    q_count = cnt.queries
    q_search = sr.queries if sr else 0
    opt_calls = optimal_quantum_find_all(N, k)
    exact_count_calls = math.sqrt((k + 1) * (N - k + 1))
    say("\nqueries (one query = one use of the one-step circuit):")
    say(f"  counting  : {q_count}")
    say(f"  search    : {q_search}")
    say(f"  total     : {q_count + q_search}")
    say(f"reference (N = 2^{L} = {N}, k = {k}):")
    say(f"  classical, certain   : Theta(N) route checks = N*L = {N * L} steps")
    say(f"  quantum find-all     : Theta(sqrt(N k)) oracle calls ~ {opt_calls:.1f} "
        f"-> {2 * L * opt_calls:.0f} steps")
    say(f"  quantum exact count  : Theta(sqrt((k+1)(N-k+1))) ~ {exact_count_calls:.1f} "
        f"oracle calls -> {2 * L * exact_count_calls:.0f} steps")
    say(f"confidence the list is complete: >= {1 - alpha:.4f} "
        f"(k pinned at level 1-alpha, and {len(out['routes'])} distinct verified routes)")
    if not quiet:
        say(f"search circuit ({sr.iterations if sr else 0} iterations): "
            f"{costs.search(sr.iterations if sr else 0)}")
    say(f"time: {time.time() - t0:.1f}s")
    return out


def run_part2(p: int, A0: str, L: int, iterations: int = 16, shots: int = 200,
              trials: int = 2000, scan_bits: tuple[float, float] = (10, 26),
              per_octave: int = 4, exact_bits: float = 20, walks: int = 400_000,
              method: str = "auto", seed: int = 0, out_dir: str = "results",
              quiet: bool = False) -> dict:
    say = (lambda *a, **k: None) if quiet else print
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(seed)

    # 1. data: how often routes end on top, as p grows
    primes = tower.primes_3mod4(*scan_bits, per_octave) if scan_bits else []
    if primes:
        say(f"tower data: {len(primes)} primes p = 3 (mod 4), 2^{scan_bits[0]:g}..2^{scan_bits[1]:g}; "
            f"exact survey up to 2^{exact_bits:g}, {walks} random routes of L = 2 ceil(log2 p) each")
        pts = tower.fraction_scan(primes, walks, 2 ** exact_bits, seed=seed)
        say(f"  {'p':>10} {'L':>3} {'spots':>8} {'on top':>7} {'spot frac':>10} "
            f"{'route frac':>11} {'frac*sqrt(p)':>13}")
        for q in pts:
            sf = f"{q.spot_fraction:.5f}" if q.spots else "-"
            say(f"  {q.p:>10} {q.L:>3} {q.spots or '-':>8} {q.top_spots or '-':>7} {sf:>10} "
                f"{q.route_fraction:>11.6f} {q.route_fraction * math.sqrt(q.p):>13.3f}")
        ex = [q for q in pts if q.spots]
        fs = tower.fit_exponent([q.p for q in ex], [q.spot_fraction for q in ex]) if len(ex) > 2 else None
        fr = tower.fit_exponent([q.p for q in pts], [q.route_fraction for q in pts])
        if fs:
            say(f"  fit, exact spots : fraction ~ p^({fs.slope:.3f} +- {2 * fs.stderr:.3f})  [{fs.n} primes, 2 s.e.]")
        say(f"  fit, random walks: fraction ~ p^({fr.slope:.3f} +- {2 * fr.stderr:.3f})  [{fr.n} primes, 2 s.e.]")
        tower.plot_fraction(pts, fs, fr, out / "part2_fraction.png")
        say(f"  plot: {out / 'part2_fraction.png'}  [{time.time() - t0:.0f}s]")

    # 2. demo: Grover on a campus with <= 1 route in 32 on top
    campus = Campus(p)
    F = campus.F
    S = survey(p)
    W = WalkCircuits(S, F.parse(A0), L, None)
    D = W.design
    on_top = len(W.targets)
    say(f"\ndemo campus p = {p}: {S.n_spots} spots, {on_top} on top (1/{S.n_spots / max(on_top, 1):.1f}); "
        f"A0 = {A0}, L = {L}")
    say(f"circuit: {W.num_qubits} qubits (route {L}, merge {D.V}, spot {D.n}); "
        f"step = {len(D.gates)} single-target gates; tower phase marks {on_top} spot codes")
    costs = WalkCosts(W)
    costs.report(say)
    m = choose_method(W.num_qubits, method)
    curve = tower.grover_curve(W, iterations, make_backend(m))
    say(f"  Aer {m}: {iterations} iterations simulated  [{time.time() - t0:.0f}s]")
    res = dict(curve=curve)
    if curve.a == 0:
        say("No route from A0 reaches the Clock Tower.")
        return res
    measured = tower.measured_success(curve, W, campus, shots, rng)
    r_best = max(0, round(math.pi / (4 * curve.theta) - 0.5))
    say(f"fraction of routes on top a = {curve.a:.5f} = 1/{1 / curve.a:.1f} (simulated, r = 0); "
        f"theta = {curve.theta:.4f}; best depth ~ {r_best}")
    say(f"  {'r':>3} {'queries':>8} {'Grover (sim)':>13} {'theory':>8} {'measured':>9} {'classical':>10}")
    for r in range(iterations + 1):
        say(f"  {r:>3} {2 * L * r:>8} {curve.success[r]:>13.4f} {curve.theory(r):>8.4f} "
            f"{measured[r]:>9.3f} {curve.classical(r):>10.4f}")
    tower.plot_grover(curve, measured, shots, p, L, out / "part2_grover.png")
    say(f"  plot: {out / 'part2_grover.png'}")
    say(f"  search circuit at r = {r_best}: {costs.search(r_best)}")

    # 3. the search itself: BBHT (a unknown in advance)
    st = tower.bbht(curve, W, campus, trials, rng)
    say(f"BBHT ({st.trials} trials, m capped at {st.cap}): mean {st.mean_iterations:.2f} iterations "
        f"= {st.mean_queries:.0f} queries + {st.mean_checks:.2f} classical checks "
        f"({st.mean_checks * L:.0f} steps); {st.failures} failures")
    say(f"classical random routes: mean 1/a = {1 / curve.a:.1f} routes = {L / curve.a:.0f} steps")
    say(f"time: {time.time() - t0:.1f}s")
    res.update(measured=measured, bbht=st)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--part", type=int, choices=(1, 2, 3), default=1)
    ap.add_argument("--p", type=int)
    ap.add_argument("--A0")
    ap.add_argument("--L", type=int)
    ap.add_argument("--jstar", help="Part 1 target j (Part 2 targets every j in F_p)")
    ap.add_argument("--alpha", type=float, default=0.01, help="failure probability of the count")
    ap.add_argument("--method", default="auto",
                    help="Aer method: auto | statevector | matrix_product_state")
    ap.add_argument("--shots", type=int, default=32, help="shots per IAE round")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=16, help="Part 2: deepest Grover depth")
    ap.add_argument("--walks", type=int, default=400_000, help="Part 2: random routes per prime")
    ap.add_argument("--scan-bits", type=float, nargs=2, default=(10, 26),
                    help="Part 2: primes from 2^a to 2^b (0 0 skips the scan)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=32, help="Part 3: processes for the checks")
    ap.add_argument("--no-check", action="store_true", help="Part 3: costs only, skip the 510 checks")
    ap.add_argument("--pairs", type=int, default=64, help="Part 3: superposition pairs in the phase test")
    a = ap.parse_args(argv)
    for k, v in DEFAULTS[a.part].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    if a.part == 1:
        run(a.p, a.A0, a.L, a.jstar, a.alpha, a.method, a.shots, a.seed)
    elif a.part == 3:
        from .part3 import run_part3
        run_part3(a.p, check=not a.no_check, workers=a.workers, pairs=a.pairs)
    else:
        scan = tuple(a.scan_bits) if a.scan_bits[1] > a.scan_bits[0] else ()
        run_part2(a.p, a.A0, a.L, a.iterations, scan_bits=scan, walks=a.walks,
                  method=a.method, seed=a.seed or 0, out_dir=a.out)


if __name__ == "__main__":
    main()
