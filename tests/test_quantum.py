"""Quantum-layer tests.  Brute-force walks over all routes appear here only as
a grader; the algorithm itself never does them."""

import itertools

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from oasis.circuits import StepDesign, StepGate, WalkCircuits, count_queries
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


def test_design_walk_matches_campus_all_routes_L12():
    D = StepDesign(S, 12)
    assert (D.n, D.V) == (8, 2)
    inv = {c: s for s, c in D.code.items()}
    for route in all_routes(12):
        code, merge = D.code[S.index[A0]], 0
        for ch in route:
            code, _, merge = D.classical_step(code, int(ch), merge)
        assert S.spots[inv[code]] == C.walk(A0, route)


SIM = AerSimulator(method="statevector")


def final_states(circuits):
    """Run circuits on Aer (native multiplexers) and return final statevectors.
    (Qiskit's Operator/Statevector would expand every UCGate generically.)"""
    for qc in circuits:
        qc.save_statevector()
    res = SIM.run(transpile(circuits, SIM, optimization_level=0)).result()
    return [np.asarray(res.get_statevector(i)) for i in range(len(circuits))]


def from_basis(nq, x, body):
    qc = QuantumCircuit(nq)
    for k in range(nq):
        if (x >> k) & 1:
            qc.x(k)
    qc.compose(body, inplace=True)
    return qc


def test_step_and_inverse_gates_are_the_designed_permutation():
    D = StepDesign(S, 12)
    g = StepGate(D)
    nq, n, V = g.num_qubits, D.n, D.V
    body, back = QuantumCircuit(nq), QuantumCircuit(nq)
    body.append(g, range(nq))
    back.append(g, range(nq))
    back.append(g.inverse(), range(nq))
    xs, ys = [], []
    for s in range(S.n_spots):
        for b in (0, 1):
            code = D.code[s]
            x = code | (b << n)
            c2, b2, m2 = D.classical_step(code, b, 0)
            xs.append(x)
            ys.append(c2 | (b2 << n) | (m2 << (n + 1)))
    outs = final_states([from_basis(nq, x, body) for x in xs])
    for y, sv in zip(ys, outs):
        assert np.isclose(abs(sv[y]), 1.0)
    outs = final_states([from_basis(nq, x, back) for x in xs[::37]])
    for x, sv in zip(xs[::37], outs):
        assert np.isclose(abs(sv[x]), 1.0)


def basis_index(W, route, code, merge=0):
    L, V = W.L, W.design.V
    r = sum(int(c) << t for t, c in enumerate(route))
    return r | (merge << L) | (code << (L + V))


@pytest.mark.parametrize("route", ["011", "100", "111"])
def test_phase_oracle_on_basis_routes(route):
    js = C.H(A0, route)
    W = WalkCircuits(S, A0, 3, js)
    c0 = W.design.code[S.index[A0]]
    xs = [basis_index(W, r, c0) for r in all_routes(3)]
    outs = final_states([from_basis(W.num_qubits, x, W.phase_oracle()) for x in xs])
    for r, x, sv in zip(all_routes(3), xs, outs):
        sign = -1 if C.H(A0, r) == js else 1
        assert np.isclose(sv[x], sign)  # phase only; route, merge, spot back as they were


def test_query_counts():
    W = WalkCircuits(S, A0, 4, F.parse("1"))
    assert count_queries(W.phase_oracle()) == 8
    assert count_queries(W.amplitude_prep()) == 4


@pytest.mark.parametrize("L,route", [(4, "0110"), (5, "10011")])
def test_full_pipeline_small(L, route):
    js = C.H(A0, route)
    truth = sorted(r for r in all_routes(L) if C.H(A0, r) == js)  # grader only
    out = run(P, "977 + 297i", L, F.fmt(js), alpha=0.01, method="statevector", seed=7, quiet=True)
    assert out["count"].k == len(truth)
    assert out["routes"] == truth


def test_no_route_when_j_not_on_campus():
    bad = next(F.fmt((u, 1)) for u in range(P) if (u, 1) not in set(S.jtab))
    out = run(P, "977 + 297i", 3, bad, method="statevector", seed=1, quiet=True)
    assert out["count"].k == 0 and out["routes"] == []


def test_no_route_when_j_unreachable():
    L = 3
    reach = {C.H(A0, r) for r in all_routes(L)}  # grader only
    js = next(j for j in S.jtab if j not in reach)
    out = run(P, "977 + 297i", L, F.fmt(js), method="statevector", seed=3, quiet=True)
    assert out["count"].k == 0 and out["routes"] == []
