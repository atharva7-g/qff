"""Part 1 — the password.

    uv run oasis                    # Ankan's instance (p=1019, L=12)
    uv run oasis --L 4 --jstar ...  # smaller instances for testing
"""

from __future__ import annotations

import argparse
import math
import time

from .backend import CountingSampler, QueryLog, choose_method
from .circuits import WalkCircuits
from .count import quantum_count
from .search import find_all
from .survey import survey
from .walk import Campus

DEFAULTS = dict(p=1019, A0="977 + 297i", L=12, jstar="797 + 921i")


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
        sr = find_all(W, k, ss, campus)
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
    say(f"time: {time.time() - t0:.1f}s")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    ap.add_argument("--alpha", type=float, default=0.01, help="failure probability of the count")
    ap.add_argument("--method", default="auto",
                    help="Aer method: auto | statevector | matrix_product_state")
    ap.add_argument("--shots", type=int, default=32, help="shots per IAE round")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    run(a.p, a.A0, a.L, a.jstar, a.alpha, a.method, a.shots, a.seed)


if __name__ == "__main__":
    main()
