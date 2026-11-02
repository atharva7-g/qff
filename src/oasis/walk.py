"""The classical walk on Ankan's campus (Montgomery curves E_A: y^2 = x^3 + A x^2 + x).

Used for (a) the one-off survey, and (b) the single classical check of a route
that was *measured* from the quantum computer.  Never used to plan routes.
"""

from __future__ import annotations

from .field import Field, Fp2


class Campus:
    def __init__(self, p: int):
        self.F = Field(p)
        self.p = p

    def normalize(self, A: Fp2) -> Fp2:
        """E_A and E_{-A} are the two forms with (0,0) as the back road: keep the smaller key."""
        F = self.F
        return min(A, F.neg(A), key=F.key)

    def roots(self, A: Fp2) -> tuple[Fp2, Fp2]:
        """x-coordinates r0, r1 of the 2-torsion points other than (0,0), sorted by key.

        They are the roots of x^2 + A x + 1, i.e. (-A +- sqrt(A^2 - 4)) / 2.
        """
        F = self.F
        d = F.sub(F.mul(A, A), (4, 0))
        if d == (0, 0):
            raise ValueError("A^2 = 4: singular curve")
        s = F.sqrt(d)
        half = pow(2, self.p - 2, self.p)
        r1 = F.scal(half, F.add(F.neg(A), s))
        r2 = F.scal(half, F.sub(F.neg(A), s))
        return tuple(sorted((r1, r2), key=F.key))  # type: ignore[return-value]

    def step(self, A: Fp2, b: int) -> Fp2:
        """Take road b: the 2-isogeny with kernel <(r_b, 0)>.

        For a kernel point (r, 0) with r != 0 the codomain is the Montgomery curve
        A' = 2(1 - 2 r^2), and (0,0) maps to (0,0), which generates the dual's kernel
        (the road back).  Then normalise A' vs -A'.
        """
        F = self.F
        r = self.roots(A)[b]
        Ap = F.scal(2, F.sub((1, 0), F.scal(2, F.mul(r, r))))
        return self.normalize(Ap)

    def j(self, A: Fp2) -> Fp2:
        """j(E_A) = 256 (A^2 - 3)^3 / (A^2 - 4)."""
        F = self.F
        A2 = F.mul(A, A)
        t = F.sub(A2, (3, 0))
        return F.mul(F.scal(256, F.mul(t, F.mul(t, t))), F.inv(F.sub(A2, (4, 0))))

    def walk(self, A0: Fp2, route: str) -> Fp2:
        A = A0
        for c in route:
            A = self.step(A, int(c))
        return A

    def H(self, A0: Fp2, route: str) -> Fp2:
        return self.j(self.walk(A0, route))
