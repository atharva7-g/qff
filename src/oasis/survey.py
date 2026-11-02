"""The survey: walk every road once, before knowing where we are going.

Starts from E_0 : y^2 = x^3 + x, which is supersingular because p = 3 (mod 4),
and does a BFS over forward moves.  Every spot is expanded exactly once, so at
most two moves are made per spot.  It takes **only p** — no A0, no L, no j*.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .field import Fp2
from .walk import Campus


@dataclass(frozen=True)
class Survey:
    p: int
    spots: tuple[Fp2, ...]          # spot index -> A
    index: dict[Fp2, int]           # A -> spot index
    succ: tuple[tuple[int, int], ...]  # succ[s][b] = index of step(spots[s], b)
    jtab: tuple[Fp2, ...]           # jtab[s] = j(E_{spots[s]})
    moves: int                      # number of roads walked (<= 2 * len(spots))

    @property
    def n_spots(self) -> int:
        return len(self.spots)

    @property
    def spot_bits(self) -> int:
        return max(1, (self.n_spots - 1).bit_length())


def survey(p: int) -> Survey:
    campus = Campus(p)
    start = campus.normalize((0, 0))
    spots: list[Fp2] = [start]
    index: dict[Fp2, int] = {start: 0}
    raw: dict[int, tuple[Fp2, Fp2]] = {}
    moves = 0
    q = deque([start])
    while q:
        A = q.popleft()
        nxt = []
        for b in (0, 1):
            B = campus.step(A, b)
            moves += 1
            nxt.append(B)
            if B not in index:
                index[B] = len(spots)
                spots.append(B)
                q.append(B)
        raw[index[A]] = (nxt[0], nxt[1])
    succ = tuple((index[raw[s][0]], index[raw[s][1]]) for s in range(len(spots)))
    jtab = tuple(campus.j(A) for A in spots)
    return Survey(p, tuple(spots), index, succ, jtab, moves)
