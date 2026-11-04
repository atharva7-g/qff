"""Reversible arithmetic mod p (Part 3).  Gates: X, CX, CCX (and CSWAP) only.

Every value x in F_p sits in n = bitlength(p) qubits, little-endian, x < p.
The base adder is Qiskit's CDKM ripple-carry adder (``adder_ripple_c04``,
kind "fixed", on n + 1 bits): b <- a + b mod 2^(n+1), 2(n+1) Toffolis.

Blocks (each one a Gate with a definition, built once per p, so the cost of
a large circuit is counted block by block -- see ``count``):

lt       out ^= [a < b] on n bits: the carry of b + (NOT a), which is 1 iff
           b - a - 1 >= 0.  Only the MAJ ladder of CDKM, a CX of the carry,
           and the ladder backwards: 2n Toffolis (a subtract and add back
           costs 4(n+1)).  Qiskit's comparators compare with a constant only.
modadd   z <- z + a mod p                 (Beauregard's pattern with registers)
           1. z <- z + a                      (n+1 bits, no overflow: a + z < 2p)
           2. z <- z - p                      (top bit = [a + z < p] = sign)
           3. flag <- top bit;  if flag: z <- z + p
           4. flag is cleared from [z >= a]: with the comparator (cmp=True),
              or as z - a, read the sign, add a back (cmp=False)
double   x <- 2x mod p, in place.  Shift = relabel the qubits ([e] + x, e = 0),
           subtract p, add p back if negative.  The flag is then the inverse
           of the low bit (2x is even, 2x - p is odd), so one CX clears it.
mac      z <- z + x y mod p  (shift and add): for each bit y_i, add
           t = y_i AND x into z (n Toffolis in, n out), then x <- 2x.
           The doublings are undone at the end, so x is unchanged.
"""

from __future__ import annotations

from functools import cached_property

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.synthesis import adder_ripple_c04


def _gate(qc: QuantumCircuit, name: str) -> Gate:
    g = qc.to_gate()
    g.name = name
    return g


def load_const(qc: QuantumCircuit, c: int, reg, ctrl=None):
    """reg ^= c (X gates), or reg ^= ctrl * c (CX gates)."""
    for k, q in enumerate(reg):
        if (c >> k) & 1:
            if ctrl is None:
                qc.x(q)
            else:
                qc.cx(ctrl, q)


class ModP:
    """Gate factory for arithmetic mod one prime p."""

    def __init__(self, p: int, cmp: bool = True):
        self.p = p
        self.n = p.bit_length()
        self.cmp = cmp
        self._inv: dict[int, Gate] = {}

    def inv(self, g: Gate) -> Gate:
        """The inverse block, built once (Qiskit builds a new one per call)."""
        if id(g) not in self._inv:
            h = g.inverse()
            self._inv[id(g)], self._inv[id(h)] = h, g
        return self._inv[id(g)]

    # -- layouts ------------------------------------------------------------
    # modadd : a(n) z(n) | z0 hi flag help | scratch(n+1)
    # double : v(n+1)    | flag help       | scratch(n+1)
    # mac    : x(n) y(n) z(n) e | t(n) | z0 hi flag help | scratch(n+1)

    @property
    def work_size(self) -> int:
        """Qubits of the shared work area used by every block (all end at 0)."""
        return 1 + self.n + 4 + self.n + 1      # e, t, z0 hi flag help, scratch

    @cached_property
    def add(self) -> Gate:
        """[a(n+1), b(n+1), help]: b <- a + b mod 2^(n+1)."""
        return _gate(adder_ripple_c04(self.n + 1, kind="fixed"), "add")

    def _addc(self, c: int, controlled: bool) -> Gate:
        """[(ctrl,) b(n+1), help, scratch(n+1)]: b <- b + (ctrl *) c mod 2^(n+1)."""
        m = self.n + 1
        k = 1 if controlled else 0
        qc = QuantumCircuit(k + m + 1 + m)
        ctrl = 0 if controlled else None
        b = list(range(k, k + m))
        help_ = k + m
        s = list(range(k + m + 1, k + 2 * m + 1))
        load_const(qc, c, s, ctrl)
        qc.append(self.add, s + b + [help_])
        load_const(qc, c, s, ctrl)
        return _gate(qc, f"{'c' if controlled else ''}addc{c}")

    @cached_property
    def sub_p(self) -> Gate:
        return self._addc((1 << (self.n + 1)) - self.p, False)

    @cached_property
    def cadd_p(self) -> Gate:
        return self._addc(self.p, True)

    @cached_property
    def lt(self) -> Gate:
        """[a(n), b(n), help, out]: out ^= [a < b]; a, b, help unchanged."""
        n = self.n
        a, b = QuantumRegister(n, "a"), QuantumRegister(n, "b")
        anc = QuantumRegister(2, "anc")             # help (carry in) out
        qc = QuantumCircuit(a, b, anc)
        c, out = anc
        maj = []                                    # MAJ(c, b_k, a_k): a_k <- carry k+1
        for k in range(n):
            maj.append((c if k == 0 else a[k - 1], b[k], a[k]))
        qc.x(a)                                         # NOT a
        for x, y, z in maj:
            qc.cx(z, y)
            qc.cx(z, x)
            qc.ccx(x, y, z)
        qc.cx(a[n - 1], out)                            # carry of b + NOT a
        for x, y, z in reversed(maj):
            qc.ccx(x, y, z)
            qc.cx(z, x)
            qc.cx(z, y)
        qc.x(a)
        return _gate(qc, "lt")

    @cached_property
    def modadd(self) -> Gate:
        n = self.n
        a, z = QuantumRegister(n, "a"), QuantumRegister(n, "z")
        anc = QuantumRegister(4, "anc")             # z0 hi flag help
        s = QuantumRegister(n + 1, "scratch")
        qc = QuantumCircuit(a, z, anc, s)
        z0, hi, flag, help_ = anc
        A, Z = [*a, z0], [*z, hi]
        qc.append(self.add, A + Z + [help_])            # z = a + z
        qc.append(self.sub_p, Z + [help_, *s])          # z = a + z - p
        qc.cx(hi, flag)                                 # flag = [a + z < p]
        qc.append(self.cadd_p, [flag, *Z, help_, *s])   # z = (a + z) mod p
        if self.cmp:
            qc.append(self.lt, [*z, *a, help_, flag])   # flag ^= [z < a]
            qc.x(flag)                                  # flag ^= 1 -> 0
        else:
            qc.append(self.inv(self.add), A + Z + [help_])  # z - a: top = [z < a]
            qc.x(hi)
            qc.cx(hi, flag)                                 # flag ^= [z >= a] -> 0
            qc.x(hi)
            qc.append(self.add, A + Z + [help_])
        return _gate(qc, "modadd" if self.cmp else "modadd0")

    @cached_property
    def double(self) -> Gate:
        """[v(n+1), flag, help, scratch(n+1)] with v = [e] + x, e = 0:
        afterwards v[0:n] = 2x mod p and v[n] = 0."""
        n = self.n
        v = QuantumRegister(n + 1, "v")
        anc = QuantumRegister(2, "anc")             # flag help
        s = QuantumRegister(n + 1, "scratch")
        qc = QuantumCircuit(v, anc, s)
        flag, help_ = anc
        qc.append(self.sub_p, [*v, help_, *s])          # 2x - p
        qc.cx(v[n], flag)                               # flag = [2x < p]
        qc.append(self.cadd_p, [flag, *v, help_, *s])   # 2x mod p
        qc.x(v[0])
        qc.cx(v[0], flag)                               # flag ^= [result even] -> 0
        qc.x(v[0])
        return _gate(qc, "double")

    @cached_property
    def mac(self) -> Gate:
        """[x(n), y(n), z(n), e, t(n), z0, hi, flag, help, scratch(n+1)]:
        z <- z + x y mod p; x, y unchanged; e, t and the rest stay 0."""
        n = self.n
        x, y, z = (QuantumRegister(n, r) for r in "xyz")
        e = QuantumRegister(1, "e")
        t = QuantumRegister(n, "t")
        anc = QuantumRegister(4, "anc")
        s = QuantumRegister(n + 1, "scratch")
        qc = QuantumCircuit(x, y, z, e, t, anc, s)
        z0, hi, flag, help_ = anc
        xs, spare = list(x), e[0]
        history = []
        for i in range(n):
            for k in range(n):
                qc.ccx(y[i], xs[k], t[k])
            qc.append(self.modadd, [*t, *z, z0, hi, flag, help_, *s])
            for k in range(n):
                qc.ccx(y[i], xs[k], t[k])
            if i < n - 1:
                v = [spare, *xs]
                qc.append(self.double, [*v, flag, help_, *s])
                history.append(v)
                xs, spare = v[:n], v[n]
        for v in reversed(history):
            qc.append(self.inv(self.double), [*v, flag, help_, *s])
        return _gate(qc, "mac")
