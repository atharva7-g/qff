"""Exact synthesis of a permutation of {0,1}^n into 2n-1 single-target gates.

A *single-target gate* flips one bit j when a Boolean function f of the other
n-1 bits is 1.  Every permutation of {0,1}^n is a product of at most 2n-1 of
them (De Vos & Van Rentergem, "Young subgroups for reversible computers",
2008).  The proof is constructive and is what ``decompose`` implements:

for bit j = 0, 1, ..., n-2:
    pair every input x with x ^ 2^j and every output y with y ^ 2^j.  The map
    "input pair -> output pair" is a 2-regular bipartite multigraph, so its
    edges split into two perfect matchings (walk the even cycles, alternating
    colours).  Flipping bit j on the input side (gate F2_j) and on the output
    side (gate F1_j) so that every colour-0 edge carries bit j = 0 leaves a
    permutation that preserves bit j.  Recurse on the next bit.
the last bit is then fixed by a single gate.

Why we want this: in Qiskit a single-target gate is one ``UCGate`` (a
multiplexer of I/X), which Aer applies natively in one pass over the
statevector.  The whole step costs 17 passes for n = 9, versus a dense
512x512 ``UnitaryGate`` (measured ~200x slower) or ~2000 MCX gates.
"""

from __future__ import annotations

import numpy as np

SingleTarget = tuple[int, np.ndarray]  # (target bit j, f[x] for x with bit j = 0)


def decompose(perm, n: int) -> list[SingleTarget]:
    """Gates in application order; f is indexed by the full word with bit j cleared."""
    p = np.asarray(perm, dtype=np.int64).copy()
    size = 1 << n
    assert sorted(p.tolist()) == list(range(size)), "not a permutation"
    first: list[SingleTarget] = []
    last: list[SingleTarget] = []
    for j in range(n - 1):
        bit = 1 << j
        by_right: dict[int, list[int]] = {}
        for x in range(size):
            by_right.setdefault(int(p[x]) & ~bit, []).append(x)
        color = -np.ones(size, dtype=np.int64)
        for start in range(size):
            if color[start] >= 0:
                continue
            x, c = start, 0
            while color[x] < 0:
                color[x] = c
                y = x ^ bit                       # the other edge at x's left node
                color[y] = 1 - c
                a, b = by_right[int(p[y]) & ~bit]  # the other edge at y's right node
                x = b if a == y else a
                c = 1 - color[y]
        f2 = np.zeros(size, dtype=bool)
        f1 = np.zeros(size, dtype=bool)
        for x in range(size):
            if color[x] == 0:
                if x & bit:
                    f2[x & ~bit] = True
                if p[x] & bit:
                    f1[int(p[x]) & ~bit] = True
        F2 = np.array([x ^ bit if f2[x & ~bit] else x for x in range(size)])
        F1 = np.array([x ^ bit if f1[x & ~bit] else x for x in range(size)])
        p = F1[p[F2]]  # remaining permutation, now preserving bit j
        first.append((j, f2))
        last.append((j, f1))
    j = n - 1
    bit = 1 << j
    fm = np.zeros(size, dtype=bool)
    for x in range(size):
        if p[x] != x:
            assert p[x] == x ^ bit
            fm[x & ~bit] = True
    return first + [(j, fm)] + last[::-1]


def apply_gates(gates: list[SingleTarget], x: int) -> int:
    for j, f in gates:
        if f[x & ~(1 << j)]:
            x ^= 1 << j
    return x


def control_table(j: int, f: np.ndarray, n: int) -> list[bool]:
    """f re-indexed by the control value i (other bits in increasing order,
    least significant first), as UCGate expects."""
    others = [k for k in range(n) if k != j]
    table = []
    for i in range(1 << (n - 1)):
        x = 0
        for pos, k in enumerate(others):
            if (i >> pos) & 1:
                x |= 1 << k
        table.append(bool(f[x]))
    return table
