# Isogenies on Oasis

**Team:** The Three Qiskiteers

**Contributors:** Uday Lathar ([@uday232523](https://github.com/uday232523)), Nihar Maheshwari ([@Cognoscenti007](https://github.com/Cognoscenti007)), Atharva Gupta ([@atharva7-g](https://github.com/atharva7-g))

**Report:** [`report/report.pdf`](report/report.pdf) · **Slides:** [`slides/slides.pdf`](slides/slides.pdf) · **Repo:** <https://github.com/atharva7-g/qff>

Qiskit Fall Fest 2026 × BITS Pilani, Track 1. The full problem statement is in
[`PROBLEM.md`](PROBLEM.md).

## Problem setup

Every spot on campus is a curve

$$
E_A:\quad y^2=x^3+Ax^2+x,\qquad A\in\mathbb F=\mathbb F_p(i),\quad p\equiv3\pmod4,\quad i^2=-1.
$$

* **Roads.** Each spot has three roads, which are 2-isogenies. A route never
  goes back along the road it arrived by, so there are two roads forward, 0 and 1.
  Their kernels are the 2-torsion points $(r_0,0)$ and $(r_1,0)$ other than
  $(0,0)$. The roots $r_{0,1}=\tfrac{-A\pm\sqrt{A^2-4}}{2}$ are sorted by
  $\mathrm{key}(u+vi)=u+pv$.
* **One step.** Bit $b$ takes the road through $r_b$ and lands on $E_{A'}$ or
  $E_{-A'}$ with $A'=2-4r_b^2$. Of the two, the one with the smaller key is kept.
* **Passwords.** A password is a route $x=b_1\cdots b_L$. Its value is the
  $j$-invariant of the spot where it ends:
  $H(x)=j(E_{A_L})=\dfrac{256(A_L^2-3)^3}{A_L^2-4}$.

| Parameter | Value |
|---|---|
| $p$ | 1019 |
| Maths department (start) | $A_0 = 977 + 297i$ |
| Route length | $L = 12$ |
| ANC (target) | $j^* = 797 + 921i$ |
| Clock Tower | every spot with $j \in \mathbb F_p$ |

The tasks:

1. **The password.** Find every route from $A_0$ to ANC with a quantum
   procedure, give the confidence that the list is complete, and count the queries.
2. **The phone on the Clock Tower.** Give the quantum cost, as a power of $p$,
   of reaching the Clock Tower, with an upper bound, a matching lower bound, and data.
3. **The Oasis map.** Build one step from arithmetic in $\mathbb F$ alone, check
   it at $p=1019$, and give its cost at $p\approx2^{256}$.
4. **The Djinns.** Shuffle the spot names at random and say what still works.

The rules (one walk of every road before the search, no classical route
planning, the ANC test inside the circuit) are in [`PROBLEM.md`](PROBLEM.md).

## Reproduction

Needs Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/atharva7-g/qff
cd qff
uv sync
uv run pytest
```

Every result comes from one command, `oasis`. It takes `--p`, `--A0`, `--L` and
`--jstar`. The defaults are the instance above.

```bash
# Part 1: all routes to ANC, the count and the queries
uv run oasis --part 1 --p 1019 --A0 "977 + 297i" --L 12 --jstar "797 + 921i" | tee results/part1_L12.txt

# Part 2: Clock-Tower data over primes and the Grover demonstration (p = 5923)
uv run oasis --part 2 | tee results/part2.txt

# Part 3: the arithmetic step, all 510 checks, and the cost at p ≈ 2^256
uv run oasis --part 3 | tee results/part3.txt
```

The full Part 1 run took 65 minutes on 4 CPU cores. A small instance finishes
in seconds:

```bash
uv run oasis --L 6 --jstar "937 + 321i"
```

Part 2 writes its plots to `results/`. Part 4 is a theoretical answer, written
up in the report and the slides. Saved outputs of all runs are in
[`results/`](results/).

The report and the slides build with `pdflatex`:

```bash
cd report && pdflatex report.tex && pdflatex report.tex
cd ../slides && pdflatex slides.tex && pdflatex slides.tex
```
