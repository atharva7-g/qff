"""Many classical random walks at once (numpy), for the Part 2 data only.

Ankan allows classical random routes for *measuring* the campus.  This is the
same rule as ``Campus.step``, vectorised over n walks with int64 arrays, so it
needs p < 2^31 (every product of two residues then fits in 63 bits).

Square root: any root of d = A^2 - 4 works, because the two roads
(-A +- s)/2 are sorted by key afterwards, so the choice of s cancels.
"""

from __future__ import annotations

import numpy as np

MAX_P = 1 << 31


def _pow(x: np.ndarray, e: int, p: int) -> np.ndarray:
    r = np.ones_like(x)
    x = x % p
    while e:
        if e & 1:
            r = r * x % p
        x = x * x % p
        e >>= 1
    return r


def _mul(au, av, bu, bv, p):
    return (au * bu % p - av * bv % p) % p, (au * bv % p + av * bu % p) % p


def _sqrt(u, v, p):
    """A square root of u + v i in F_{p^2} (input must be a square)."""
    e = (p + 1) // 4
    half = (p + 1) // 2
    out_x, out_y = np.zeros_like(u), np.zeros_like(u)
    done = np.zeros(u.shape, dtype=bool)
    # v = 0: u or -u is a square in F_p
    real = v == 0
    xr = _pow(u, e, p)
    ok = real & (xr * xr % p == u)
    out_x[ok], done[ok] = xr[ok], True
    yr = _pow(-u % p, e, p)
    ok = real & ~done
    out_y[ok], done[ok] = yr[ok], True
    # v != 0: x^2 = (u +- s) / 2 with s = sqrt(u^2 + v^2); keep the sign that works
    s = _pow((u * u % p + v * v % p) % p, e, p)
    for sign in (1, -1):
        c = (u + sign * s) % p * half % p
        x = _pow(c, e, p)
        y = v * _pow(2 * x % p, p - 2, p) % p
        su, sv = _mul(x, y, x, y, p)
        ok = ~done & (x != 0) & (su == u) & (sv == v)
        out_x[ok], out_y[ok], done[ok] = x[ok], y[ok], True
    if not done.all():
        raise ValueError("A^2 - 4 is not a square: start spot is not on the campus")
    return out_x, out_y


def step(Au: np.ndarray, Av: np.ndarray, b: np.ndarray, p: int):
    """One road for every walk: the same rule as ``Campus.step``."""
    half = (p + 1) // 2
    du, dv = _mul(Au, Av, Au, Av, p)
    du = (du - 4) % p
    su, sv = _sqrt(du, dv, p)
    r1u, r1v = (-Au + su) % p * half % p, (-Av + sv) % p * half % p
    r2u, r2v = (-Au - su) % p * half % p, (-Av - sv) % p * half % p
    swap = (r2u + p * r2v < r1u + p * r1v)          # r0 = smaller key
    r0u, r0v = np.where(swap, r2u, r1u), np.where(swap, r2v, r1v)
    s1u, s1v = np.where(swap, r1u, r2u), np.where(swap, r1v, r2v)
    ru, rv = np.where(b == 1, s1u, r0u), np.where(b == 1, s1v, r0v)
    qu, qv = _mul(ru, rv, ru, rv, p)
    Bu, Bv = (2 - 4 * qu) % p, (-4 * qv) % p        # A' = 2(1 - 2 r^2)
    Nu, Nv = -Bu % p, -Bv % p
    neg = Nu + p * Nv < Bu + p * Bv
    return np.where(neg, Nu, Bu), np.where(neg, Nv, Bv)


def on_top(Au: np.ndarray, Av: np.ndarray, p: int) -> np.ndarray:
    """j(E_A) in F_p, tested without a division: j = num/den is in F_p iff
    Im(num * conj(den)) = 0, with num = 256 (A^2 - 3)^3 and den = A^2 - 4."""
    a2u, a2v = _mul(Au, Av, Au, Av, p)
    tu = (a2u - 3) % p
    t2u, t2v = _mul(tu, a2v, tu, a2v, p)
    nu, nv = _mul(t2u, t2v, tu, a2v, p)
    du, dv = (a2u - 4) % p, a2v
    _, im = _mul(nu, nv, du, -dv % p, p)
    return im == 0


def random_walks(p: int, L: int, n: int, start: tuple[int, int] = (0, 0),
                 seed: int | None = None) -> np.ndarray:
    """End spots of n uniformly random L-step routes; returns the on-top mask."""
    if p >= MAX_P:
        raise ValueError("fastwalk needs p < 2^31")
    rng = np.random.default_rng(seed)
    Au = np.full(n, start[0], dtype=np.int64)
    Av = np.full(n, start[1], dtype=np.int64)
    for _ in range(L):
        Au, Av = step(Au, Av, rng.integers(0, 2, n), p)
    return on_top(Au, Av, p)
