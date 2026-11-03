"""Part 3 tests: the arithmetic blocks, the step and the tower test on Aer MPS."""

import math
import random
from collections import Counter

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator

from oasis.arith import ModP
from oasis.arith_cost import count, depth_ub, flatten
from oasis.arith_step import StepCircuit
from oasis.field import Field
from oasis.part3 import check_step, check_tower, prime_3mod4
from oasis.survey import survey
from oasis.walk import Campus

SIM = AerSimulator(method="matrix_product_state")


def run_block(gate, inputs, nin):
    """Load each input on the first nin qubits, run the block, read all qubits."""
    nq = gate.num_qubits
    th = [Parameter(f"t{k}") for k in range(nin)]
    qc = QuantumCircuit(nq, nq)
    for k in range(nin):
        qc.rx(th[k], k)
    qc.append(gate, range(nq))
    qc.measure(range(nq), range(nq))
    binds = {th[k]: [math.pi * ((x >> k) & 1) for x in inputs] for k in range(nin)}
    res = SIM.run(flatten(qc), shots=1, parameter_binds=[binds]).result()
    return [int(next(iter(res.get_counts(i))), 2) for i in range(len(inputs))]


P = 1019
M = ModP(P)
N = M.n
MASK = (1 << N) - 1
RND = random.Random(5)


def test_modadd():
    xs = [RND.randrange(P) | (RND.randrange(P) << N) for _ in range(30)] + [0, (P - 1) | ((P - 1) << N)]
    for x, o in zip(xs, run_block(M.modadd, xs, 2 * N)):
        a, z = x & MASK, x >> N
        assert o == a | (((a + z) % P) << N)          # a kept, z = a + z mod p, work = 0


def test_double():
    xs = [RND.randrange(P) << 1 for _ in range(30)] + [0, (P - 1) << 1]
    for x, o in zip(xs, run_block(M.double, xs, N + 1)):
        assert o == (2 * (x >> 1)) % P


def test_mac():
    xs = [RND.randrange(P) | (RND.randrange(P) << N) | (RND.randrange(P) << 2 * N) for _ in range(12)]
    for x, o in zip(xs, run_block(M.mac, xs, 3 * N)):
        a, b, z = x & MASK, (x >> N) & MASK, x >> 2 * N
        assert o == a | (b << N) | (((z + a * b) % P) << 2 * N)


def test_algorithm9_root_on_every_spot():
    F, S = Field(P), survey(P)

    def pw(a, e):
        r = (1, 0)
        for bit in bin(e)[2:]:
            r = F.mul(r, r)
            if bit == "1":
                r = F.mul(r, a)
        return r

    branch = Counter()
    for A in S.spots:
        d = F.sub(F.mul(A, A), (4, 0))
        a1 = pw(d, (P - 3) // 4)
        x0 = F.mul(a1, d)
        alpha = F.mul(a1, x0)
        m1 = alpha == (P - 1, 0)
        s = F.add(F.mul(pw(F.add((1, 0), alpha), (P - 1) // 2), x0), F.mul((0, 1), x0) if m1 else (0, 0))
        assert F.mul(s, s) == d
        branch[m1] += 1
    assert branch[True] and branch[False]          # both branches occur


def test_counts_agree_with_flat_circuit():
    S = StepCircuit(P)
    c = count(S.qc)
    flat = flatten(S.qc)
    assert dict(c) == dict(flat.count_ops())
    assert flat.depth() <= depth_ub(S.qc)


@pytest.mark.parametrize("p", [19, 43])
def test_step_every_spot_small_campus(p):
    r = check_step(p, workers=8)
    assert r.cases == 2 * survey(p).n_spots and r.ok


def test_tower_every_spot_small_campus():
    r = check_tower(43, workers=8)
    assert r.cases == survey(43).n_spots and r.ok


def test_step_p1019_special_spots():
    # A = 0 (both roads to 6) and its neighbour A = 6: the merge spot of Part 1
    r = check_step(P, workers=2, limit=4)
    assert r.cases == 4 and r.ok


def test_prime_choice():
    for bits in (10, 16, 64, 256):
        q = prime_3mod4(bits)
        assert q.bit_length() == bits and q % 4 == 3
