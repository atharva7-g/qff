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

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator

from . import tower as tower_mod
from .arith_cost import cnots, count, depth_ub, flatten, toffolis
from .arith_step import BASELINE, BEST, StepCircuit, TowerCircuit, Variant
from .survey import survey
from .walk import Campus


# -- simulation ------------------------------------------------------------

def _build(kind: str, p: int, v: Variant = BEST):
    return StepCircuit(p, v) if kind == "step" else TowerCircuit(p, v)


def _inputs(C, kind):
    return [*C.A[0], *C.A[1]] + ([C.b] if kind == "step" else [])


def _outputs(C, kind):
    return [*C.out[0], *C.out[1]] if kind == "step" else [C.out]


def _encode(kind, n, case):
    A = case[0]
    bits = [(A[0] >> k) & 1 for k in range(n)] + [(A[1] >> k) & 1 for k in range(n)]
    return bits + ([case[1]] if kind == "step" else [])


def _worker(job):
    kind, p, cases, v = job
    C = _build(kind, p, v)
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


def simulate(kind: str, p: int, cases: list, workers: int = 32, v: Variant = BEST) -> list:
    chunks = [cases[i::workers] for i in range(workers) if cases[i::workers]]
    ctx = multiprocessing.get_context("spawn")     # Aer is multi-threaded: no fork
    with ProcessPoolExecutor(len(chunks), mp_context=ctx) as ex:
        return [r for part in ex.map(_worker, [(kind, p, c, v) for c in chunks]) for r in part]


def _cry(qc, theta, c, t):
    """Controlled RY(theta) from RY and CX (Aer's MPS method has no CRY)."""
    qc.ry(theta / 2, t)
    qc.cx(c, t)
    qc.ry(-theta / 2, t)
    qc.cx(c, t)


def _pair_worker(job):
    p, pairs, v_new, v_ref, fault = job
    New, Ref = StepCircuit(p, v_new), StepCircuit(p, v_ref)
    ins, outs = _inputs(New, "step"), _outputs(New, "step")
    io_ref = _inputs(Ref, "step") + _outputs(Ref, "step")
    rest_ref = [q for q in Ref.qc.qubits if q not in set(io_ref)]
    extra = QuantumRegister(len(rest_ref), "ref_work")
    c = QuantumRegister(1, "c")
    qc = QuantumCircuit(c, *New.qc.qregs, extra)
    qc.add_register(ClassicalRegister(qc.num_qubits, "m"))
    phi = Parameter("phi")
    tx = [Parameter(f"x{k}") for k in range(len(ins))]
    td = [Parameter(f"d{k}") for k in range(len(ins))]
    prep = QuantumCircuit(*qc.qregs)
    prep.ry(1.1, c[0])                                  # cos(.55)|0> + sin(.55)|1>: unequal
    prep.p(phi, c[0])                                   # weights (equal Schmidt values made
                                                        # Aer's MPS lose the phase for some phi)
    for q, a, d in zip(ins, tx, td):
        prep.rx(a, q)                                   # |x>
        _cry(prep, d, c[0], q)                          # c = 1: flip the bits where y != x
    qc.compose(prep, inplace=True)
    qc.compose(flatten(New.qc), qubits=New.qc.qubits, inplace=True)
    if fault:
        qc.z(New.b)                                     # negative control: phase (-1)^b
    where = {q: i for i, q in enumerate(io_ref)}
    mapping = [(ins + outs)[where[q]] if q in where else extra[rest_ref.index(q)] for q in Ref.qc.qubits]
    qc.compose(flatten(Ref.qc).reverse_ops(), qubits=mapping, inplace=True)   # X, CX, CCX, CSWAP: self-inverse
    qc.compose(prep.inverse(), inplace=True)
    qc.measure(qc.qubits, qc.clbits)
    binds = {phi: [ph for _, _, ph in pairs]}
    for k in range(len(ins)):
        bx = [_encode("step", New.n, x)[k] for x, _, _ in pairs]
        by = [_encode("step", New.n, y)[k] for _, y, _ in pairs]
        binds[tx[k]] = [math.pi * u for u in bx]
        binds[td[k]] = [math.pi * (u ^ w) for u, w in zip(bx, by)]
    sim = AerSimulator(method="matrix_product_state", max_parallel_threads=1)
    res = sim.run(qc, shots=8, parameter_binds=[binds]).result()
    return [(x, y, set(res.get_counts(i)) == {"0" * qc.num_qubits}) for i, (x, y, _) in enumerate(pairs)]


def pair_test(p: int, k: int | None = None, workers: int = 32, v_new: Variant = BEST,
              v_ref: Variant = BASELINE, seed: int = 7, fault: bool = False) -> tuple[int, list]:
    """Phase check against the reference circuit (already checked on basis states):

        (cos w |0>|x> + e^{i phi} sin w |1>|y>)  -> New -> Ref^-1 -> undo the preparation

    gives |0...0> with probability 1 - sin^2(2w) sin^2((theta(y) - theta(x))/2),
    where New|x> = e^{i theta(x)} Ref|x> and w = 0.55.  fault=True adds a Z on
    b after New (a wrong phase (-1)^b), so pairs with different b must fail.  The pairs form a path through k + 1 of
    the inputs (all inputs if k is None), so passing every pair gives
    theta(x) = theta for all of them: New = Ref, also on every superposition.
    Returns (pairs tested, failed pairs)."""
    import random
    rnd = random.Random(seed)
    cases = [(A, b) for A in survey(p).spots for b in (0, 1)]
    rnd.shuffle(cases)
    path = cases if k is None else cases[:k + 1]
    pairs = [(x, y, rnd.uniform(0, 2 * math.pi)) for x, y in zip(path, path[1:])]
    chunks = [pairs[i::workers] for i in range(workers) if pairs[i::workers]]
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(len(chunks), mp_context=ctx) as ex:
        res = [r for part in ex.map(_pair_worker, [(p, ch, v_new, v_ref, fault) for ch in chunks]) for r in part]
    return len(res), [(x, y) for x, y, ok in res if not ok]


def random_spots(p: int, k: int, seed: int = 3) -> list:
    """k distinct spots from a random walk from A = 0 (E_0 is supersingular for p = 3 mod 4)."""
    import random
    rnd, C, A, out = random.Random(seed), Campus(p), (0, 0), {}
    for _ in range(64):                                  # leave the start
        A = C.step(A, rnd.randrange(2))
    while len(out) < k:
        A = C.step(A, rnd.randrange(2))
        out[A] = None
    return list(out)


@dataclass
class CheckResult:
    cases: int
    wrong: list
    inputs_changed: int
    dirty: int

    @property
    def ok(self) -> bool:
        return not self.wrong and not self.inputs_changed and not self.dirty


def check_step(p: int, workers: int = 32, limit: int | None = None, v: Variant = BEST,
               spots: list | None = None) -> CheckResult:
    C = Campus(p)
    spots = survey(p).spots if spots is None else spots
    cases = [(A, b) for A in spots for b in (0, 1)][:limit]
    res = simulate("step", p, cases, workers, v)
    wrong = [(c, g) for c, g, _, _ in res if g != C.step(*c)]
    return CheckResult(len(res), wrong, sum(not r[2] for r in res), sum(not r[3] for r in res))


def check_tower(p: int, workers: int = 32, v: Variant = BEST) -> CheckResult:
    S, C = survey(p), Campus(p)
    cases = [(A,) for A in S.spots]
    res = simulate("tower", p, cases, workers, v)
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


def cost_of(kind: str, p: int, depth: bool = False, v: Variant = BEST) -> Cost:
    C = _build(kind, p, v)
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
