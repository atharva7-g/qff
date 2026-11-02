"""Arithmetic in F_{p^2} = F_p(i), i^2 = -1, p = 3 (mod 4).

Elements are tuples (u, v) meaning u + v*i.  Everything is O(log p) so the
same code would work for p ~ 2^256 (Part 3 turns it into circuits).
"""

from __future__ import annotations

Fp2 = tuple[int, int]


class Field:
    def __init__(self, p: int):
        if p % 4 != 3:
            raise ValueError("need p = 3 (mod 4) so that i^2 = -1 gives a field")
        self.p = p

    # -- basic ops -----------------------------------------------------
    def add(self, a: Fp2, b: Fp2) -> Fp2:
        p = self.p
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def sub(self, a: Fp2, b: Fp2) -> Fp2:
        p = self.p
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)

    def neg(self, a: Fp2) -> Fp2:
        p = self.p
        return ((-a[0]) % p, (-a[1]) % p)

    def mul(self, a: Fp2, b: Fp2) -> Fp2:
        p = self.p
        return ((a[0] * b[0] - a[1] * b[1]) % p, (a[0] * b[1] + a[1] * b[0]) % p)

    def scal(self, c: int, a: Fp2) -> Fp2:
        p = self.p
        return ((c * a[0]) % p, (c * a[1]) % p)

    def inv(self, a: Fp2) -> Fp2:
        # 1/(u+vi) = (u-vi)/(u^2+v^2); the norm lives in F_p.
        p = self.p
        n = (a[0] * a[0] + a[1] * a[1]) % p
        if n == 0:
            raise ZeroDivisionError("inverse of 0 in F_p^2")
        ni = pow(n, p - 2, p)
        return ((a[0] * ni) % p, (-a[1] * ni) % p)

    def key(self, a: Fp2) -> int:
        """Ordering used by Ankan: key(u + v i) = u + p v."""
        return a[0] + self.p * a[1]

    # -- square roots ----------------------------------------------------
    def _is_sq_fp(self, x: int) -> bool:
        p = self.p
        x %= p
        return x == 0 or pow(x, (p - 1) // 2, p) == 1

    def _sqrt_fp(self, x: int) -> int:
        # p = 3 (mod 4): sqrt(x) = x^((p+1)/4) whenever x is a square.
        return pow(x % self.p, (self.p + 1) // 4, self.p)

    def sqrt(self, a: Fp2) -> Fp2:
        """A square root of a in F_{p^2}.

        a is a square in F_{p^2} iff its norm u^2 + v^2 is a square in F_p
        (every element of F_p itself is a square in F_{p^2}).  On the campus
        A^2 - 4 always is one, because all 2-torsion of a supersingular E_A
        over F_{p^2} is rational; otherwise we raise ValueError.

        Write a = u + v i and look for (x + y i)^2 = a, i.e.
            x^2 - y^2 = u,   2xy = v.
        The norm n = u^2 + v^2 = (x^2 + y^2)^2 is a square in F_p, with
        s = sqrt(n) = +-(x^2 + y^2).  Hence x^2 = (u + s)/2 for the right sign
        of s; we try both signs, take x = sqrt_Fp(x^2), y = v / (2x), and keep
        the sign whose (x + y i)^2 really equals a.
        If x = 0 then v = 0 and a = -y^2, so y = sqrt_Fp(-u).
        """
        p = self.p
        u, v = a[0] % p, a[1] % p
        if v == 0:
            if self._is_sq_fp(u):
                r = (self._sqrt_fp(u), 0)
            else:  # -u is a square, a = (y i)^2
                r = (0, self._sqrt_fp(-u))
        else:
            n = (u * u + v * v) % p
            if not self._is_sq_fp(n):
                raise ValueError(f"{self.fmt((u, v))} is not a square in F_p^2")
            s = self._sqrt_fp(n)
            half = pow(2, p - 2, p)
            for cand in ((u + s) * half % p, (u - s) * half % p):
                if cand != 0 and self._is_sq_fp(cand):
                    x = self._sqrt_fp(cand)
                    y = v * pow(2 * x, p - 2, p) % p
                    # both (u+s)/2 and (u-s)/2 can be squares in F_p; only the
                    # one with s = x^2 + y^2 also satisfies x^2 - y^2 = u.
                    if self.mul((x, y), (x, y)) == (u, v):
                        r = (x, y)
                        break
            else:  # pragma: no cover - impossible for a field
                raise ArithmeticError("no square root found")
        assert self.mul(r, r) == (u, v), "sqrt self-check failed"
        return r

    def parse(self, s: str) -> Fp2:
        """Parse strings like '977 + 297i', '138', '5i'."""
        s = s.replace(" ", "")
        u = v = 0
        for term in s.replace("-", "+-").split("+"):
            if not term:
                continue
            if term.endswith("i"):
                c = term[:-1]
                v += int(c) if c not in ("", "-") else (1 if c == "" else -1)
            else:
                u += int(term)
        return (u % self.p, v % self.p)

    def fmt(self, a: Fp2) -> str:
        u, v = a
        if v == 0:
            return str(u)
        if u == 0:
            return f"{v}i"
        return f"{u} + {v}i"
