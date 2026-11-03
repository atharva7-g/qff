"""Part 3 — check the arithmetic step and tower test, and cost them at p ~ 2^256.

Check: every spot and both bits, on Aer's matrix_product_state method.  The
circuits use only X, CX, CCX and CSWAP, which map basis states to basis
states, so the MPS keeps bond dimension 1 and 850 qubits are cheap.  Each
case loads |A, b> with RX(pi) gates (one parameter per input qubit; the
phase -i is global), runs the flat circuit, measures every qubit, and checks

    out = Campus.step(A, b),   A and b unchanged,   every other qubit 0.

Why basis states are enough: the circuit is a permutation matrix, so by
linearity sum_x c_x |x>|0> -> sum_x c_x |x>|f(x)>|0> holds for every
superposition of the tested inputs as soon as it holds for each |x>.
The cases run in parallel processes (Aer runs parameter binds one by one).
"""

from __future__ import annotations

import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator

from . import tower as tower_mod
from .arith_cost import cnots, count, depth_ub, flatten, toffolis
from .arith_step import StepCircuit, TowerCircuit
from .survey import survey
from .walk import Campus


# -- simulation ------------------------------------------------------------

def _build(kind: str, p: int):
    return StepCircuit(p) if kind == "step" else TowerCircuit(p)


def _inputs(C, kind):
    return [*C.A[0], *C.A[1]] + ([C.b] if kind == "step" else [])


def _outputs(C, kind):
    return [*C.out[0], *C.out[1]] if kind == "step" else [C.out]


def _encode(kind, n, case):
    A = case[0]
    bits = [(A[0] >> k) & 1 for k in range(n)] + [(A[1] >> k) & 1 for k in range(n)]
    return bits + ([case[1]] if kind == "step" else [])


def _worker(job):
    kind, p, cases = job
    C = _build(kind, p)
    ins, outs = _inputs(C, kind), _outputs(C, kind)
    th = [Parameter(f"t{k}") for k in range(len(ins))]
    qc = QuantumCircuit(*C.qc.qregs, ClassicalRegister(C.qc.num_qubits, "m"))
    for q, t in zip(ins, th):
        qc.rx(t, q)
    qc.compose(flatten(C.qc), inplace=True)
    qc.measure(qc.qubits, qc.clbits)
    binds = {t: [math.pi * _encode(kind, C.n, c)[k] for c in cases] for k, t in enumerate(th)}
    sim = AerSimulator(method="matrix_product_state", max_parallel_threads=1)
    res = sim.run(qc, shots=1, parameter_binds=[binds]).result()
    idx = {q: i for i, q in enumerate(qc.qubits)}
    rest = [q for q in qc.qubits if q not in set(ins) | set(outs)]
    out = []
    for i, case in enumerate(cases):
        s = next(iter(res.get_counts(i)))[::-1]
        val = lambda qs: sum(int(s[idx[q]]) << k for k, q in enumerate(qs))
        n = C.n
        got = (val(outs[:n]), val(outs[n:])) if kind == "step" else val(outs)
        same_in = [int(s[idx[q]]) for q in ins] == _encode(kind, n, case)
        clean = all(s[idx[q]] == "0" for q in rest)
        out.append((case, got, same_in, clean))
    return out


def simulate(kind: str, p: int, cases: list, workers: int = 32) -> list:
    chunks = [cases[i::workers] for i in range(workers) if cases[i::workers]]
    ctx = multiprocessing.get_context("spawn")     # Aer is multi-threaded: no fork
    with ProcessPoolExecutor(len(chunks), mp_context=ctx) as ex:
        return [r for part in ex.map(_worker, [(kind, p, c) for c in chunks]) for r in part]


@dataclass
class CheckResult:
    cases: int
    wrong: list
    inputs_changed: int
    dirty: int

    @property
    def ok(self) -> bool:
        return not self.wrong and not self.inputs_changed and not self.dirty


def check_step(p: int, workers: int = 32, limit: int | None = None) -> CheckResult:
    S, C = survey(p), Campus(p)
    cases = [(A, b) for A in S.spots for b in (0, 1)][:limit]
    res = simulate("step", p, cases, workers)
    wrong = [(c, g) for c, g, _, _ in res if g != C.step(*c)]
    return CheckResult(len(res), wrong, sum(not r[2] for r in res), sum(not r[3] for r in res))


def check_tower(p: int, workers: int = 32) -> CheckResult:
    S, C = survey(p), Campus(p)
    cases = [(A,) for A in S.spots]
    res = simulate("tower", p, cases, workers)
    wrong = [(c, g) for c, g, _, _ in res if g != int(C.j(c[0])[1] == 0)]
    return CheckResult(len(res), wrong, sum(not r[2] for r in res), sum(not r[3] for r in res))


# -- cost --------------------------------------------------------------------

@dataclass
class Cost:
    p: int
    n: int
    qubits: int
    toffoli: int
    cx: int
    x: int
    depth_ub: int
    depth: int | None = None

    def t_count(self, per_toffoli: int = 7) -> int:
        return per_toffoli * self.toffoli


def cost_of(kind: str, p: int, depth: bool = False) -> Cost:
    C = _build(kind, p)
    c = count(C.qc)
    d = flatten(C.qc).depth() if depth else None
    return Cost(p, C.n, C.qc.num_qubits, toffolis(c), cnots(c), c["x"], depth_ub(C.qc), d)


def prime_3mod4(bits: int) -> int:
    """The smallest prime p = 3 (mod 4) above 2^(bits - 1/2), so p has n = bits
    bits.  Why there: the cost depends on the bits of (p-3)/4 and (p-1)/2;
    near 2^(n - 1/2) (the digits of sqrt 2) about half of them are 1, as for a
    random p, while 2^(n-1) + small would make the exponentiations too cheap."""
    n = math.isqrt(1 << (2 * bits - 1)) + 1
    n += (3 - n) % 4
    while not tower_mod.is_prime(n):
        n += 4
    return n


def fit_power(ns, ys) -> tuple[float, float]:
    """y ~ c n^k by least squares on logs; returns (c, k)."""
    import numpy as np
    k, logc = np.polyfit(np.log(ns), np.log(ys), 1)
    return float(math.exp(logc)), float(k)


def search_cost(step: Cost, tower: Cost, frac_sqrt_p: float, L: int | None = None) -> dict:
    """Part 2 at this p with BBHT-free optimal depth: iterations = (pi/4) / sqrt(a),
    a = frac_sqrt_p / sqrt(p).  One iteration = 2L steps + 1 tower test (phase
    version: compute, Z, uncompute = the same cost) + diffusion on L qubits
    (an L-controlled Z, 2L Toffolis).  Out-of-place steps keep A_0..A_L."""
    p, n = step.p, step.n
    L = L or 2 * n
    a = frac_sqrt_p / math.sqrt(p)
    iters = math.pi / 4 / math.sqrt(a)
    per_iter = 2 * L * step.toffoli + tower.toffoli + 2 * L
    work = max(step.qubits - 4 * n - 1, tower.qubits - 2 * n - 1)
    qubits = L + (L + 1) * 2 * n + work
    return dict(L=L, a=a, iterations=iters, toffoli_per_iteration=per_iter,
                toffoli=iters * per_iter, qubits=qubits)


def run_part3(p: int = 1019, check: bool = True, workers: int = 32,
              scan_bits=(10, 12, 16, 24, 32, 48, 64, 96, 128, 192), big_bits: int = 256, frac_sqrt_p: float = 7.0, say=print) -> dict:
    t0 = time.time()
    out = {}
    st, tw = cost_of("step", p, depth=True), cost_of("tower", p, depth=True)
    say(f"arithmetic step for p = {p} (n = {st.n} bits); basis {{x, cx, ccx}}, CSWAP = 1 Toffoli + 2 CX")
    for name, c in (("step", st), ("tower test", tw)):
        say(f"  {name:<10} qubits {c.qubits:>6,}  Toffoli {c.toffoli:>9,}  T {c.t_count():>10,} "
            f"(7/Toffoli)  CX {c.cx:>10,}  X {c.x:>7,}  depth {c.depth:>10,} "
            f"(block bound {c.depth_ub:,})")
    out.update(step=st, tower=tw)

    if check:
        say(f"checking every spot and both bits on Aer matrix_product_state ({workers} processes) ...")
        r = check_step(p, workers)
        say(f"  step : {r.cases} cases, {len(r.wrong)} wrong, {r.inputs_changed} changed inputs, "
            f"{r.dirty} with a work qubit not 0  -> {'PASS' if r.ok else 'FAIL'}  [{time.time() - t0:.0f}s]")
        rt = check_tower(p, workers)
        say(f"  tower: {rt.cases} spots, {len(rt.wrong)} wrong, {rt.inputs_changed} changed inputs, "
            f"{rt.dirty} with a work qubit not 0  -> {'PASS' if rt.ok else 'FAIL'}  [{time.time() - t0:.0f}s]")
        out.update(check_step=r, check_tower=rt)

    say("\ncost against the size of p (exact counts from the block hierarchy; p ~ 2^(n - 1/2)):")
    say(f"  {'n':>4} {'qubits':>9} {'Toffoli':>15} {'T (7/Tof)':>16} {'depth bound':>14}  | "
        f"{'tower qubits':>12} {'tower Toffoli':>14}")
    rows = []
    for bits in [*scan_bits, big_bits]:
        q = prime_3mod4(bits)
        s, t = cost_of("step", q), cost_of("tower", q)
        rows.append((s, t))
        say(f"  {bits:>4} {s.qubits:>9,} {s.toffoli:>15,} {s.t_count():>16,} {s.depth_ub:>14,}  | "
            f"{t.qubits:>12,} {t.toffoli:>14,}")
    ns = [s.n for s, _ in rows]
    for name, ys in (("Toffoli", [s.toffoli for s, _ in rows]), ("qubits", [s.qubits for s, _ in rows]),
                     ("depth", [s.depth_ub for s, _ in rows])):
        c_, k_ = fit_power(ns[-3:], ys[-3:])
        say(f"  {name:<8} ~ {c_:.2f} n^{k_:.2f}   (fit on the 3 largest n)")
    big_s, big_t = rows[-1]
    say(f"  p used at n = {big_bits}: {big_s.p}")
    out.update(table=rows)

    sc = search_cost(big_s, big_t, frac_sqrt_p)
    say(f"\nPart 2 at p ~ 2^{big_bits}: a ~ {frac_sqrt_p:g} / sqrt(p) = 2^{math.log2(sc['a']):.1f}, "
        f"L = {sc['L']}")
    say(f"  Grover iterations (pi/4) / sqrt(a) ~ 2^{math.log2(sc['iterations']):.1f}")
    say(f"  per iteration: 2L steps + 1 tower test + diffusion = {sc['toffoli_per_iteration']:.3e} Toffoli")
    say(f"  total ~ 2^{math.log2(sc['toffoli']):.1f} Toffoli = {sc['toffoli']:.2e}  "
        f"(T ~ {7 * sc['toffoli']:.2e}); qubits ~ {sc['qubits']:,.0f}")
    say(f"time: {time.time() - t0:.0f}s")
    out["search"] = sc
    return out
