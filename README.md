# Isogenies on Oasis — Part 1: the password

Find **every** L-step route from the Maths department (A0) to ANC (j*) on Ankan's
2-isogeny campus with a quantum procedure, say how confident we are that the list
is complete, and count the queries (uses of the one-step circuit).

```bash
uv sync
uv run pytest                      # classical checks + gate-level + end-to-end (small L)
uv run oasis                       # Ankan's instance: p = 1019, L = 12, 22 qubits
uv run oasis --L 6 --jstar "937 + 321i"   # small instance, seconds
```

The output of the full run is in [`results/part1_L12.txt`](results/part1_L12.txt).

## Pipeline

| Step | Where | What, and why |
|---|---|---|
| Field | `src/oasis/field.py` | F_{p²} arithmetic. √ via the norm: with p ≡ 3 (mod 4) a square root in F_p is one exponentiation. O(log p), so it would work at p ≈ 2^256. |
| Walk | `src/oasis/walk.py` | Roads r0 < r1 (roots of x²+Ax+1, ordered by key), A' = 2(1−2r_b²), keep the smaller-key form of ±A', j = 256(A²−3)³/(A²−4). Reproduces all 7 of Ankan's checks. Used only by the survey and to check a *measured* route once. |
| Survey | `src/oasis/survey.py` | BFS from A = 0 (E_0 is supersingular since p ≡ 3 mod 4). Takes only `p`: no A0, L or j*. 255 spots, 510 roads, ≤ 2 per spot. |
| Step design | `src/oasis/circuits.py` `StepDesign` | Turns the survey into a reversible, *in-place* step (below). |
| Synthesis | `src/oasis/reversible.py` | The 9-bit step permutation → 17 single-target gates (2n−1 theorem), each one Qiskit `UCGate` (an I/X multiplexer, native in Aer). |
| At ANC? | `WalkCircuits.anc_phase` | One `DiagonalGate` on the spot register: −1 on the codes of curves with j = j*. Never evaluates H. If no spot has j*, it is the identity, so k = 0. |
| Count | `src/oasis/count.py` | `qiskit_algorithms.IterativeAmplitudeEstimation` on a = k/N, re-run with a tighter ε until the interval for k holds exactly one integer. |
| Search | `src/oasis/search.py` | Grover (`qiskit.circuit.library.grover_operator`, reflection on the route register only), round(π/4θ − ½) iterations, measure, check each route with one classical walk, repeat until k distinct routes. |
| Queries | `src/oasis/backend.py` | Every query reported is counted from the circuits (`step` / `step_dg` gates × shots actually used), not computed from a formula. |

## The one-step circuit (22 qubits in total)

| Register | Qubits | Role |
|---|---|---|
| `route` | L = 12 | the password bits |
| `merge` | V = 2 | logs arrivals at the one spot where roads merge |
| `spot` | 8 | code of the current curve (255 spots + 1 spare code) |

**In place, no history.** Every spot has exactly two incoming roads, so a step
(s, b) → t loses only one bit: which of the two roads it came in on. Step t
writes that bit back into `route[t]`. The step is then a permutation of
(spot, road bit), and the uncompute restores the route bits. Keeping the
full history (Bennett) would cost 8 qubits per step, and the first version,
with one tag qubit per step, needed 33 qubits.

**The one exception.** A = 0 (j = 1728, which has the extra automorphism i)
sends *both* roads to A = 6, so A = 6 has three incoming roads and A = 0 has
only one. Arrivals at 6 coming from 0 are written to the spare code 255
(A = 6 itself is coded 254). The step then:

1. shifts the `merge` register on every arrival at 6;
2. writes a 1 into the freed slot if the arrival came via 255;
3. folds 255 back to 254 with a single bit flip.

Arrivals at 6 are at least 8 steps apart (the shortest cycle through 6, a
survey fact), so V = 1 + ⌊(L−1)/8⌋ = 2 qubits suffice.

**Why single-target gates.** We measured these costs per gate in Aer on 23 qubits:

| Gate | Time |
|---|---|
| dense 10-qubit `UnitaryGate` | 17.7 s |
| dense 9-qubit `UnitaryGate` | 6.2 s |
| 8-control `UCGate` | 28 ms |
| native `mcx` | 2 ms |

An MCX with *open* controls (`ctrl_state`) is not native and gets decomposed,
so every multi-controlled gate here uses closed controls. Any permutation of
n bits is a product of at most 2n−1 single-target gates (De Vos & Van
Rentergem 2008). We implement the constructive proof (edge-colouring a
2-regular bipartite multigraph per bit), giving 17 multiplexers per step.
`step_dg` is the same list reversed: every gate is its own inverse.
(Qiskit's generic `UCGate.inverse()` is not a multiplexer, and transpiles
into about 4,000 gates.)

## Confidence

IAE returns a Clopper–Pearson interval that contains a = k/N with
probability ≥ 1 − α. That is a finite-sample guarantee, not an asymptotic
one.
- **Stopping rule:** we stop when the interval for k = N·a contains exactly
  one integer.
- **Repeated looks:** each run uses level α/R (union bound over the at most R
  runs), so the interval we stop on is valid with probability ≥ 1 − α.
- **Completeness:** the search then holds k distinct routes, each confirmed by
  one classical walk. So "the list is complete" holds with probability
  ≥ 1 − α (default α = 0.01).

We tried `MaximumLikelihoodAmplitudeEstimation` first. With a fixed
exponential schedule and few shots, its likelihood kept a second lobe at
k ± 1 (k = 5 vs 6 at N = 256). IAE chooses its powers so that θ never
leaves one half-period, which rules that aliasing out.

## Queries (N = 2^L, k routes, one query = one step)

* **Classical, with certainty:** every route must be checked, Θ(N) routes,
  i.e. N·L steps.
* **Best possible quantum, finding all k:** Θ(√(Nk)) oracle calls
  (Σ_{i≤k} (π/4)√(N/i)), each 2L steps.
* **Exact quantum counting:** Θ(√((k+1)(N−k+1))) oracle calls.
* **What we use:** IAE costs O(√(kN)·log(1/α)) Grover calls, plus Grover
  search with about k·H_k runs of ≈ (π/4)√(N/k) calls.

On this small campus (N = 4096) the constant factors of the counting step
(shots per round, log(1/α), the union bound) make the measured total larger
than walking all 4096 routes classically. The asymptotic gap, √(kN) against N,
is what matters on the real campus, where N = 2^L and walking every route is
out of the question. The program prints the measured numbers next to these
references.

## Rules followed

* The survey depends on p only. The step circuit is built from it.
* There is no path counting, backward search, meet-in-the-middle or route
  listing in the algorithm. The only classical walk is the single check of a
  measured route. (The tests brute-force small L, and L = 12 for the
  classical step design only, purely to grade the answer.)
* The ANC test is inside the circuit, on the spot register. It is a table on
  this campus and becomes arithmetic in Part 3.
* k = 0 is reported ("no route reaches ANC") instead of crashing; both ways it
  can happen are tested.

## How it's simulated

22 qubits fit a dense Aer statevector (64 MB). Amplitude-estimation circuits
A Q^m are prefixes of one sequence, so `SnapshotSampler` simulates the
sequence once, saving the spot register's exact distribution after every
power (`save_probabilities`). It resumes from a saved statevector
(`set_statevector`) when IAE asks for a deeper power, and draws shots from
those exact distributions. Shots drawn this way have the same distribution as
running each circuit separately, and queries are charged as if each circuit
were run on hardware.
