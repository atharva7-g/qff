"""Quantum-layer tests on small walks (L = 3/4) so a statevector fits.

The brute-force walk over all routes below is a *test-only* grader; the
algorithm itself never does it.
"""

import itertools

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from oasis.circuits import WalkCircuits, count_queries, step_permutation
from oasis.main import run
from oasis.survey import survey
from oasis.walk import Campus

P = 1019
C = Campus(P)
F = C.F
S = survey(P)
A0 = F.parse("977 + 297i")


def all_routes(L):
    return ["".join(b) for b in itertools.product("01", repeat=L)]


def test_step_permutation_matches_survey():
    perm = step_permutation(S)
    n = S.spot_bits
    for s in range(S.n_spots):
        for b in (0, 1):
            y = perm[s | (b << n)]
            assert y & ((1 << n) - 1) == S.succ[s][b]
            assert (y >> n) & 1 == b


def basis_index(W, route, spot, tag=0, flag=0):
    L, n = W.L, W.S.spot_bits
    r = sum(int(c) << t for t, c in enumerate(route))
    return r | (tag << L) | (spot << (2 * L)) | (flag << (2 * L + n))


@pytest.mark.parametrize("route", ["0110", "1111", "0000", "1010"])
def test_bit_oracle_on_basis_routes_is_clean(route):
    L = 4
    js = C.H(A0, route)
    W = WalkCircuits(S, A0, L, js)
    s0 = S.index[A0]
    sv = Statevector.from_int(basis_index(W, route, s0), 2 ** W.num_qubits)
    out = sv.evolve(W.bit_oracle())
    want = basis_index(W, route, s0, flag=1)  # route reaches its own j
    assert np.isclose(abs(out.data[want]), 1.0)


def test_query_count_of_oracle():
    W = WalkCircuits(S, A0, 4, F.parse("1"))
    assert count_queries(W.phase_oracle()) == 8
    assert count_queries(W.bit_oracle()) == 8


@pytest.mark.parametrize("route", ["0110", "1001"])
def test_full_pipeline_small(route):
    L = 4
    js = C.H(A0, route)
    truth = sorted(r for r in all_routes(L) if C.H(A0, r) == js)  # grader only
    out = run(P, "977 + 297i", L, F.fmt(js), alpha=0.01, method="statevector",
              shots=100, seed=7, quiet=True)
    assert out["count"].k == len(truth)
    assert out["routes"] == truth


def test_no_route_does_not_crash():
    # a j that no spot has (not supersingular): the check circuit is empty
    bad = next(F.fmt((u, 1)) for u in range(P) if (u, 1) not in set(S.jtab))
    out = run(P, "977 + 297i", 3, bad, alpha=0.01, method="statevector",
              shots=100, seed=1, quiet=True)
    assert out["count"].k == 0 and out["routes"] == []


def test_supersingular_j_but_unreachable_gives_zero():
    L = 3
    reach = {C.H(A0, r) for r in all_routes(L)}  # grader only
    js = next(j for j in S.jtab if j not in reach)
    out = run(P, "977 + 297i", L, F.fmt(js), alpha=0.01, method="statevector",
              shots=100, seed=3, quiet=True)
    assert out["count"].k == 0 and out["routes"] == []
