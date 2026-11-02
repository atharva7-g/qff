# Isogenies on Oasis — Part 1: the password

Find **every** L-step route from the Maths department (A0) to ANC (j*) on Ankan's
2-isogeny campus, with a quantum procedure, and say how sure we are and how many
queries it took.

```bash
uv sync
uv run pytest            # classical checks + quantum pipeline on small L
uv run oasis --L 4 --jstar "<j>"   # small instance (17 qubits, statevector)
uv run oasis             # Ankan's instance: p=1019, L=12 (33 qubits, see below)
```

## Pipeline

| Step | File | What / why |
|---|---|---|
| Field | `src/oasis/field.py` | F_{p²} arithmetic, `key`, square roots via the norm (p ≡ 3 mod 4 ⇒ √ in F_p is one exponentiation). O(log p), so it scales to p ≈ 2^256. |
| Walk | `src/oasis/walk.py` | Roads r0 < r1 (roots of x²+Ax+1 by key), A' = 2(1−2r_b²), keep min-key of ±A', j = 256(A²−3)³/(A²−4). Used only by the survey and to check a *measured* route once. |
| Survey | `src/oasis/survey.py` | BFS from A = 0 (E_0 supersingular since p ≡ 3 mod 4). Takes only `p`: no A0, L, j*. 255 spots, ≤ 2 roads walked per spot. |
| One-step circuit | `src/oasis/circuits.py` `StepGate` | \|s, b, 0⟩ → \|succ_b(s), b, c⟩ on spot(8) + road bit + tag. One use = one query. |
| At ANC? | `circuits.py` `check` | `MCXGate(8, ctrl_state=s)` onto the flag for each spot s with j(s) = j*. Reads the spot register, never H. Empty if no spot has j* ⇒ k = 0. |
| Count | `src/oasis/count.py` | `qiskit_algorithms.IterativeAmplitudeEstimation` on a = k/N; shrink ε until the (1−α) interval for k contains one integer. |
| Search | `src/oasis/search.py` | Grover (`qiskit.circuit.library.grover_operator`, reflection on the route qubits only), round(π/4θ − ½) iterations, measure, verify each route by one classical walk, repeat until k distinct routes. |
| Queries | `src/oasis/backend.py` | `CountingSampler` wraps Aer's `SamplerV2` and counts `step`/`step_dg` gates in each logical circuit × shots actually used. |

### Why a tag qubit per step
A step must be reversible.  For a fixed road bit, s ↦ succ_b(s) is at most
two-to-one (A = 0, j = 1728, sends both roads to A = 6), so one bit per step
("which of the two predecessors") is all the garbage needed — instead of
Bennett's full history (8 qubits per step).  The oracle is
walk → check → (Z) → check → unwalk, i.e. **2L queries**, and returns every
ancilla to |0⟩ (tested).

### Confidence
IAE's interval holds a = k/N with probability ≥ 1 − α (Clopper–Pearson).  When
it pins a single integer k and the search has k distinct classically-verified
routes, the list is complete with probability ≥ 1 − α.

### Query comparison (N = 2^L, k routes)
* classical, with certainty: Θ(N) route checks (N·L steps)
* best quantum, finding all k: Θ(√(Nk)) oracle calls (Σ_{i≤k} (π/4)√(N/i))
* exact quantum counting: Θ(√((k+1)(N−k+1))) oracle calls

Each oracle call is 2L one-step queries; the program prints measured totals.

## Rules followed
* Survey depends on p only; the step circuit is built from it.
* No path counting, backward search, meet-in-the-middle or route listing in the
  algorithm; the only classical walk is the single check of a measured route.
  (`tests/` brute-forces small L purely to grade the answer.)
* The ANC test is inside the circuit on the spot register (table now, arithmetic
  in Part 3).
* k = 0 is reported ("no route reaches ANC") instead of crashing.

## Simulation limits
Qubits = 2L + 9.  L ≤ 9 fits a dense statevector (`--method statevector`); L = 12
needs 33 qubits, which is beyond a 16 GB dense statevector, so `auto` picks Aer's
`matrix_product_state` method.  Aer's pre-run memory estimate for this circuit
is ~1 TB, so whether L = 12 actually runs on a given machine is an open
question; this is a simulation limit, not a property of the algorithm.
