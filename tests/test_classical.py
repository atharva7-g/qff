import inspect

import pytest

from oasis.field import Field
from oasis.survey import survey
from oasis.walk import Campus

P = 1019
C = Campus(P)
F = C.F
A0 = F.parse("977 + 297i")

CHECKS = [
    ("0", "747 + 288i", "98 + 85i"),
    ("1", "79 + 218i", "138"),
    ("01", None, "967 + 249i"),
    ("000000000000", None, "348 + 660i"),
    ("111111111111", None, "34"),
    ("011010010110", None, "98 + 934i"),
    ("101100111000", None, "119"),
]


@pytest.mark.parametrize("route,A,j", CHECKS)
def test_ankan_checks(route, A, j):
    end = C.walk(A0, route)
    if A is not None:
        assert end == F.parse(A)
    assert C.j(end) == F.parse(j)


def test_sqrt_all_squares_small_field():
    G = Field(43)
    squares = {G.mul((x, y), (x, y)) for x in range(43) for y in range(43)}
    for u in range(43):
        for v in range(43):
            if (u, v) in squares:
                r = G.sqrt((u, v))
                assert G.mul(r, r) == (u, v)
            else:
                with pytest.raises(ValueError):
                    G.sqrt((u, v))


def test_survey_is_independent_of_A0_L_jstar():
    assert list(inspect.signature(survey).parameters) == ["p"]
    S = survey(P)
    assert S.moves <= 2 * S.n_spots
    assert A0 in S.index
    assert S.n_spots == 255 and S.spot_bits == 8
    for s, A in enumerate(S.spots):
        for b in (0, 1):
            assert S.spots[S.succ[s][b]] == C.step(A, b)
