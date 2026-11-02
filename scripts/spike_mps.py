"""Feasibility: one Grover iteration of the full circuit on Aer MPS."""
import time

from qiskit import transpile
from qiskit.circuit.library import grover_operator
from qiskit_aer import AerSimulator

from oasis.circuits import WalkCircuits
from oasis.survey import survey
from oasis.walk import Campus

S = survey(1019)
F = Campus(1019).F
W = WalkCircuits(S, F.parse("977+297i"), 12, F.parse("797+921i"))
print("qubits:", W.new().num_qubits, "targets:", W.targets)
prep = W.load_start().compose(W.uniform_routes())
G = grover_operator(W.phase_oracle(), reflection_qubits=list(range(12)))
qc = prep.copy()
for _ in range(int(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 1):
    qc.compose(G, inplace=True)
qc.measure_all()
sim = AerSimulator(method="matrix_product_state")
t = time.time()
tq = transpile(qc, sim, optimization_level=0)
print("transpile", time.time() - t, tq.count_ops())
t = time.time()
res = sim.run(tq, shots=200).result()
print("run", time.time() - t)
c = res.get_counts()
print(sorted(c.items(), key=lambda kv: -kv[1])[:5])
