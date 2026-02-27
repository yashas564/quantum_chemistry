import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------
# Molecular Hamiltonian (H2 model)
# ------------------------------------
H = (
    -1.0523732 * qml.PauliZ(0)
    -1.0523732 * qml.PauliZ(1)
    +0.0112801 * qml.PauliZ(0) @ qml.PauliZ(1)
    +0.3979374 * qml.PauliX(0) @ qml.PauliX(1)
)

# ------------------------------------
# Classical Solver (Exact Diagonalization)
# ------------------------------------
H_matrix = qml.matrix(H)
exact_energies, _ = np.linalg.eigh(H_matrix)
exact_energies = np.sort(exact_energies)

exact_ground = exact_energies[0]
exact_excited = exact_energies[1]

print("Exact Ground State Energy:", exact_ground)
print("Exact First Excited State Energy:", exact_excited)

# ------------------------------------
# Quantum Solver (VQE for Ground State)
# ------------------------------------
dev = qml.device("default.qubit", wires=2)

energy_history = []

@qml.qnode(dev)
def vqe_circuit(theta):
    qml.BasisState(np.array([1, 1]), wires=[0, 1])
    qml.RY(theta[0], wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(H)

theta = np.array([0.0], requires_grad=True)
opt = qml.GradientDescentOptimizer(0.4)

# VQE optimization loop
for _ in range(40):
    theta, energy = opt.step_and_cost(vqe_circuit, theta)
    energy_history.append(energy)

vqe_ground = vqe_circuit(theta)

print("VQE Ground State Energy:", vqe_ground)

# ------------------------------------
# Error Analysis
# ------------------------------------
ground_error = abs(vqe_ground - exact_ground)

print("Ground State Error:", ground_error)

# ------------------------------------
# Plots
# ------------------------------------

# Convergence plot
plt.figure()
plt.plot(energy_history)
plt.xlabel("Iteration")
plt.ylabel("Energy (Hartree)")
plt.title("VQE Convergence Behavior")
plt.grid()
plt.show()

# Accuracy comparison plot
plt.figure()
plt.bar(
    ["Exact (Classical)", "VQE (Quantum)"],
    [exact_ground, vqe_ground]
)
plt.ylabel("Energy (Hartree)")
plt.title("Classical vs Quantum Energy Comparison")
plt.show()
