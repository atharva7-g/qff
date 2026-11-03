"""Part 2 tests.  Brute-force walks over all routes appear here only as a grader."""

import itertools
import math

import numpy as np
import pytest
from qiskit_aer import AerSimulator

from oasis import fastwalk, tower
from oasis.circuits import WalkCircuits
from oasis.survey import survey
from oasis.walk import Campus

P = 1019
C = Campus(P)
S = survey(P)
A0 = C.F.parse("977 + 297i")


@pytest.mark.parametrize("p", [1019, 5923, 1000003])
def test_fastwalk_matches_campus(p):
    campus = Campus(p)
    rng = np.random.default_rng(p)
    n, L = 200, 16
    bits = rng.integers(0, 2, (L, n))
    Au, Av = np.zeros(n, np.int64), np.zeros(n, np.int64)
    for t in range(L):
        Au, Av = fastwalk.step(Au, Av, bits[t], p)
    top = fastwalk.on_top(Au, Av, p)
    for i in range(n):
        end = campus.walk((0, 0), "".join(str(b) for b in bits[:, i]))
        assert end == (int(Au[i]), int(Av[i]))
        assert (campus.j(end)[1] == 0) == bool(top[i])


def test_primes_3mod4():
    ps = tower.primes_3mod4(10, 14, 2)
    assert ps[0] == 1031 and len(ps) == 9
    assert all(p % 4 == 3 and tower.is_prime(p) for p in ps)
    assert not tower.is_prime(1019 * 1031)


def test_fit_recovers_exponent():
    ps = np.array([2.0 ** k for k in range(10, 20)])
    f = tower.fit_exponent(ps, 3 * ps ** -0.5)
    assert abs(f.slope + 0.5) < 1e-9


def test_tower_targets_are_spots_with_j_in_Fp():
    W = WalkCircuits(S, A0, 4, None)
    inv = {c: s for s, c in W.design.code.items()}
    assert sorted(S.jtab[inv[c]][1] for c in W.targets) == [0] * len(W.targets)
    assert len(W.targets) == sum(1 for j in S.jtab if j[1] == 0)


def test_grover_curve_matches_theory():
    L, R = 5, 3
    W = WalkCircuits(S, A0, L, None)
    routes = ["".join(b) for b in itertools.product("01", repeat=L)]
    a = sum(C.j(C.walk(A0, r))[1] == 0 for r in routes) / len(routes)   # grader only
    curve = tower.grover_curve(W, R, AerSimulator(method="statevector"))
    assert math.isclose(curve.a, a, abs_tol=1e-6)
    for r in range(R + 1):
        assert math.isclose(curve.success[r], curve.theory(r), abs_tol=1e-5)
        assert math.isclose(curve.route_dists[r].sum(), 1.0, abs_tol=1e-9)
    rng = np.random.default_rng(0)
    st = tower.bbht(curve, W, C, 50, rng)
    assert st.failures == 0 and st.mean_checks >= 1
