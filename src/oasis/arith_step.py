"""Part 3 — one step of the walk from arithmetic in F = F_p(i) only.

    |A>|b>|0>|0...0>  ->  |A>|b>|A'>|0...0>,     A' = step(A, b)

No table: the circuit computes Ankan's rule (``Campus.step``).
Compute -> copy -> uncompute (Bennett): the forward part keeps every
intermediate value, A' is copied out with CX gates, and the forward part is
run backwards, which returns every work qubit to 0.

Forward part:
  d  = A^2 - 4
  s  = sqrt(d), Adj & Rodriguez-Henriquez 2014, Algorithm 9 (p = 3 mod 4):
         a1 = d^((p-3)/4),  x0 = a1 d,  alpha = a1 x0  (= d^((p-1)/2))
         s  = (1 + alpha)^((p-1)/2) x0  +  [alpha = -1] i x0
       (if alpha = -1 then 1 + alpha = 0, so the first term is 0: the "if"
       becomes one flag qubit and two controlled additions, no branch)
  rho+- = (-A +- s) / 2              (halving = inverse of the doubling block)
  sort: swap = [key(rho-) < key(rho+)], CSWAP           -> r0, r1
  road: CSWAP on b                                       -> r_b
  A'' = 2 - 4 r_b^2
  sign: neg = [key(-A'') < key(A'')], select -A'' or A''
key(u + v i) = u + p v, so key order = compare v, then u.

The tower test (``tower_circuit``) is also arithmetic: j = num / den with
num = 256 (A^2 - 3)^3, den = A^2 - 4, and j in F_p iff
Im(num conj(den)) = num_im den_re - num_re den_im = 0.  No inversion.
"""

from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister

from .arith import ModP, load_const


class Prog:
    """Builds a reversible program on one QuantumCircuit.

    Values in F_p are lists of n qubits; values in F_{p^2} are (re, im) pairs.
    ``alloc`` takes zero qubits from a pool of freed qubits first."""

    def __init__(self, p: int, name: str = "prog"):
        self.M = M = ModP(p)
        self.p, self.n = p, M.n
        self.qc = QuantumCircuit(name=name)
        work = QuantumRegister(M.work_size, "work")
        self.qc.add_register(work)
        n = self.n
        self.e = work[0]                       # spare qubit for in-place doubling
        self.t = list(work[1:n + 1])
        self.z0, self.hi, self.flag, self.help = work[n + 1:n + 5]
        self.s = list(work[n + 5:])
        self.pool: list = []
        self._regs = 0

    # -- qubits ------------------------------------------------------------
    def new_register(self, k: int, name: str):
        r = QuantumRegister(k, name)
        self.qc.add_register(r)
        return list(r)

    def alloc(self, k: int | None = None) -> list:
        k = self.n if k is None else k
        take = self.pool[:k]
        del self.pool[:k]
        if len(take) < k:
            self._regs += 1
            take += self.new_register(k - len(take), f"v{self._regs}")
        return take

    def free(self, qs):
        self.pool.extend(qs)

    SELF_INVERSE = {"x", "cx", "ccx", "cswap"}

    def bennett(self, name: str, copy_out) -> QuantumCircuit:
        """forward; copy_out(qc); forward run backwards.  Built with _append
        (compose would deep-copy every block)."""
        fwd = self.qc
        qc = fwd.copy_empty_like(name=name)
        for inst in fwd.data:
            qc._append(inst)
        copy_out(qc)
        for inst in reversed(fwd.data):
            op = inst.operation
            if op.name not in self.SELF_INVERSE:
                op = self.M.inv(op)
            qc._append(op, inst.qubits, ())
        return qc

    def alloc2(self):
        return (self.alloc(), self.alloc())

    # -- F_p ---------------------------------------------------------------
    def _ma(self):
        return [self.z0, self.hi, self.flag, self.help, *self.s]

    def modadd(self, a, z, sub=False):
        """z <- z +- a mod p."""
        g = self.M.inv(self.M.modadd) if sub else self.M.modadd
        self.qc.append(g, [*a, *z, *self._ma()])

    def mac(self, x, y, z, sub=False):
        """z <- z +- x y mod p (x, y, z distinct)."""
        g = self.M.inv(self.M.mac) if sub else self.M.mac
        self.qc.append(g, [*x, *y, *z, self.e, *self.t, *self._ma()])

    def double(self, x) -> list:
        """x <- 2x mod p in place; returns the relabelled register."""
        v = [self.e, *x]
        self.qc.append(self.M.double, [*v, self.flag, self.help, *self.s])
        self.e = v[self.n]
        return v[:self.n]

    def halve(self, x) -> list:
        v = [*x, self.e]
        self.qc.append(self.M.inv(self.M.double), [*v, self.flag, self.help, *self.s])
        self.e = v[0]
        return v[1:]

    def addconst(self, c: int, z):
        c %= self.p
        if c:
            K = self.alloc()
            load_const(self.qc, c, K)
            self.modadd(K, z)
            load_const(self.qc, c, K)
            self.free(K)

    def copy(self, x) -> list:
        y = self.alloc()
        for a, b in zip(x, y):
            self.qc.cx(a, b)
        return y

    def neg(self, x) -> list:
        z = self.alloc()
        self.modadd(x, z, sub=True)
        return z

    def cmodadd(self, ctrl, a, z, sub=False):
        """z <- z +- ctrl * a mod p."""
        for q, t in zip(a, self.t):
            self.qc.ccx(ctrl, q, t)
        self.modadd(self.t, z, sub)
        for q, t in zip(a, self.t):
            self.qc.ccx(ctrl, q, t)

    def mcx(self, ctrls, out):
        """out ^= AND(ctrls): a Toffoli ladder with k - 2 clean ancillas."""
        k = len(ctrls)
        if k == 1:
            self.qc.cx(ctrls[0], out)
            return
        if k == 2:
            self.qc.ccx(ctrls[0], ctrls[1], out)
            return
        anc = self.alloc(k - 2)
        ladder = [(ctrls[0], ctrls[1], anc[0])]
        ladder += [(ctrls[i + 1], anc[i - 1], anc[i]) for i in range(1, k - 2)]
        for g in ladder:
            self.qc.ccx(*g)
        self.qc.ccx(ctrls[-1], anc[-1], out)
        for g in reversed(ladder):
            self.qc.ccx(*g)
        self.free(anc)

    def eq_const(self, qs, c: int, out):
        """out ^= [qs == c]."""
        zeros = [q for k, q in enumerate(qs) if not (c >> k) & 1]
        for q in zeros:
            self.qc.x(q)
        self.mcx(list(qs), out)
        for q in zeros:
            self.qc.x(q)

    def lt(self, a, b, out):
        """out ^= [a < b]: the sign bit of a - b on n + 1 bits."""
        A, B = [*a, self.hi], [*b, self.z0]
        self.qc.append(self.M.inv(self.M.add), [*B, *A, self.help])
        self.qc.cx(self.hi, out)
        self.qc.append(self.M.add, [*B, *A, self.help])

    def eq(self, a, b, out):
        for x, y in zip(a, b):
            self.qc.cx(x, y)
        self.eq_const(b, 0, out)
        for x, y in zip(a, b):
            self.qc.cx(x, y)

    # -- F_{p^2} -----------------------------------------------------------
    def key_lt(self, P, Q, out):
        """out ^= [key(P) < key(Q)]: [Pv < Qv] or ([Pv = Qv] and [Pu < Qu])."""
        self.lt(P[1], Q[1], out)
        e, l = self.alloc(2)
        self.eq(P[1], Q[1], e)
        self.lt(P[0], Q[0], l)
        self.qc.ccx(e, l, out)
        self.lt(P[0], Q[0], l)
        self.eq(P[1], Q[1], e)
        self.free([e, l])

    def mul2(self, a, b):
        """(a_r + a_i i)(b_r + b_i i), schoolbook: 4 MACs into a new value."""
        c = self.alloc2()
        self.mac(a[0], b[0], c[0])
        self.mac(a[1], b[1], c[0], sub=True)
        self.mac(a[0], b[1], c[1])
        self.mac(a[1], b[0], c[1])
        return c

    def sqr2(self, a):
        """(a_r + a_i i)^2 = (a_r + a_i)(a_r - a_i) + 2 a_r a_i i: 2 MACs."""
        sp, sm = self.copy(a[0]), self.copy(a[0])
        self.modadd(a[1], sp)
        self.modadd(a[1], sm, sub=True)
        c = self.alloc2()
        self.mac(sp, sm, c[0])
        self.mac(a[0], a[1], c[1])
        c = (c[0], self.double(c[1]))
        self.modadd(a[1], sm)
        self.modadd(a[1], sp, sub=True)
        for x in (sp, sm):
            for q, r in zip(a[0], x):
                self.qc.cx(q, r)
            self.free(x)
        return c

    def pow2(self, a, e: int):
        """a^e, left-to-right square and multiply; intermediates are kept."""
        acc = a
        for bit in bin(e)[3:]:
            acc = self.sqr2(acc)
            if bit == "1":
                acc = self.mul2(acc, a)
        return acc

    def copy2(self, a):
        return (self.copy(a[0]), self.copy(a[1]))

    def neg2(self, a):
        return (self.neg(a[0]), self.neg(a[1]))


class StepCircuit:
    """The arithmetic step for one p.  Registers: A (2n), b (1), out (2n)."""

    def __init__(self, p: int):
        P = Prog(p, "fwd")
        n = P.n
        self.p, self.n = p, n
        self.A = (P.new_register(n, "A_re"), P.new_register(n, "A_im"))
        self.b = P.new_register(1, "b")[0]
        self.out = (P.new_register(n, "out_re"), P.new_register(n, "out_im"))
        A, b = self.A, self.b

        d = P.sqr2(A)
        P.addconst(-4, d[0])
        # square root, Algorithm 9
        a1 = P.pow2(d, (p - 3) // 4)
        x0 = P.mul2(a1, d)
        alpha = P.mul2(a1, x0)
        minus1 = P.alloc(1)[0]
        P.eq_const([*alpha[0], *alpha[1]], p - 1, minus1)       # alpha == -1 (+ 0 i)
        c = P.copy2(alpha)
        P.addconst(1, c[0])
        s = P.mul2(P.pow2(c, (p - 1) // 2), x0)
        P.cmodadd(minus1, x0[1], s[0], sub=True)                # s += [alpha=-1] i x0
        P.cmodadd(minus1, x0[0], s[1])
        # the two roots of x^2 + A x + 1
        rp, rm = P.neg2(A), P.neg2(A)
        for k in (0, 1):
            P.modadd(s[k], rp[k])
            P.modadd(s[k], rm[k], sub=True)
        rp = (P.halve(rp[0]), P.halve(rp[1]))
        rm = (P.halve(rm[0]), P.halve(rm[1]))
        # sort by key, then pick road b
        swap = P.alloc(1)[0]
        P.key_lt(rm, rp, swap)
        for ctrl in (swap, b):
            for k in (0, 1):
                for x, y in zip(rp[k], rm[k]):
                    P.qc.cswap(ctrl, x, y)
        # A'' = 2 - 4 r^2
        q = P.sqr2(rp)
        q = tuple(P.double(P.double(x)) for x in q)
        z = P.neg2(q)
        P.addconst(2, z[0])
        # sign: keep the smaller key of +-A''
        nz = P.neg2(z)
        flip = P.alloc(1)[0]
        P.key_lt(nz, z, flip)
        sel = P.alloc2()
        for k in (0, 1):
            for x, y, w in zip(z[k], nz[k], sel[k]):
                P.qc.cx(x, w)
                P.qc.ccx(flip, x, w)
                P.qc.ccx(flip, y, w)
        self.prog = P

        def copy_out(qc):
            for k in (0, 1):
                for x, y in zip(sel[k], self.out[k]):
                    qc.cx(x, y)
        self.qc = P.bennett("arith_step", copy_out)


class TowerCircuit:
    """|A>|0> -> |A>|[j(E_A) in F_p]>, all work qubits back to 0."""

    def __init__(self, p: int):
        P = Prog(p, "tower_fwd")
        n = P.n
        self.p, self.n = p, n
        self.A = (P.new_register(n, "A_re"), P.new_register(n, "A_im"))
        self.out = P.new_register(1, "on_top")[0]
        a2 = P.sqr2(self.A)
        t = P.copy2(a2)
        P.addconst(-3, t[0])
        num = P.mul2(P.sqr2(t), t)
        for _ in range(8):                                      # * 256
            num = tuple(P.double(x) for x in num)
        den = P.copy2(a2)
        P.addconst(-4, den[0])
        im = P.alloc()
        P.mac(num[1], den[0], im)
        P.mac(num[0], den[1], im, sub=True)
        top = P.alloc(1)[0]
        P.eq_const(im, 0, top)
        self.prog = P
        self.qc = P.bennett("tower", lambda qc: qc.cx(top, self.out))
