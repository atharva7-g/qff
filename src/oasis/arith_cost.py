"""Gate counts and a flat circuit for the arithmetic step (Part 3).

Basis: {x, cx, ccx}.  A CSWAP is counted as 1 Toffoli + 2 CX (its usual
decomposition).  T count: 7 per Toffoli (Amy et al. 2013), or 4 per Toffoli
whose uncompute uses measurement and a classically controlled CZ (Gidney 2018).

``count`` walks the block hierarchy (mac -> modadd -> add -> MAJ/UMA ->
gates) and memoises each block, so a step at p ~ 2^256 is counted exactly
without ever building its millions of gates.  ``flatten`` builds the flat
{x, cx, ccx, cswap} circuit that Aer runs, again one block at a time.
"""

from __future__ import annotations

from collections import Counter

from qiskit import QuantumCircuit

LEAVES = {"x", "cx", "ccx", "cswap", "rx", "measure", "barrier"}


def _key(op):
    return (op.name, op.num_qubits, op.num_clbits)


def count(qc: QuantumCircuit) -> Counter:
    memo: dict = {}

    def rec(op) -> Counter:
        k = _key(op)
        if k not in memo:
            c = Counter()
            for inst in op.definition.data:
                o = inst.operation
                c.update({o.name: 1} if o.name in LEAVES else rec(o))
            memo[k] = c
        return memo[k]

    total = Counter()
    for inst in qc.data:
        o = inst.operation
        total.update({o.name: 1} if o.name in LEAVES else rec(o))
    return total


def toffolis(c: Counter) -> int:
    return c["ccx"] + c["cswap"]


def cnots(c: Counter) -> int:
    return c["cx"] + 2 * c["cswap"]


def flatten(qc: QuantumCircuit) -> QuantumCircuit:
    """The same circuit with every block expanded to leaf gates."""
    memo: dict = {}

    def rec(op):
        """Leaf ops of a block as (op, local qubit indices)."""
        k = _key(op)
        if k not in memo:
            d = op.definition
            pos = {q: i for i, q in enumerate(d.qubits)}
            out = []
            for inst in d.data:
                o, qs = inst.operation, [pos[q] for q in inst.qubits]
                if o.name in LEAVES:
                    out.append((o, qs))
                else:
                    out.extend((lo, [qs[j] for j in lq]) for lo, lq in rec(o))
            memo[k] = out
        return memo[k]

    flat = qc.copy_empty_like()
    for inst in qc.data:
        o = inst.operation
        if o.name in LEAVES:
            flat._append(inst)
            continue
        qs = inst.qubits
        for lo, lq in rec(o):
            flat._append(lo, [qs[j] for j in lq], [])
    return flat


def depth_ub(qc: QuantumCircuit) -> int:
    """Depth, block by block: a qubit clock per qubit, and every block counted
    as one unit of its own depth that starts when all its qubits are free.
    An upper bound (a block cannot overlap with its neighbours here); exact
    for leaf gates."""
    memo: dict = {}

    def run(circ) -> int:
        pos = {q: i for i, q in enumerate(circ.qubits)}
        clock = [0] * circ.num_qubits
        for inst in circ.data:
            o = inst.operation
            qs = [pos[q] for q in inst.qubits]
            d = 1 if o.name in LEAVES else block(o)
            t = max(clock[i] for i in qs) + d
            for i in qs:
                clock[i] = t
        return max(clock, default=0)

    def block(op) -> int:
        k = _key(op)
        if k not in memo:
            memo[k] = run(op.definition)
        return memo[k]

    return run(qc)
