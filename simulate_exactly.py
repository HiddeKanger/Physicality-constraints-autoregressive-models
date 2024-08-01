import cirq
import numpy as np
from utils.utils import read_config
from utils.ncon import ncon
from utils.POVM import POVM
from scipy.linalg import expm, logm


def rho_to_P(rho, povm, N_qubits):
    rho = rho.reshape((2,)*2*N_qubits)
    M = povm.M

    tensors = (rho, ) + (M,) * N_qubits

    indices = (np.arange(2*N_qubits) + 1,) + \
        tuple([[-i, i+N_qubits, i] for i in range(1, N_qubits + 1)])
    return ncon(tensors, indices)


def discretize_matrix(U, steps):
    H = logm(U)
    dt = 1 / steps
    return expm(H * dt)


class ExactSimulator:
    def __init__(self, config, povm):
        self.config = config
        self.povm = povm

        self.circuit, self.qubits = self.prepare_initial_state(
            config["START_STATE"], config["N_QUBITS"])

    def prepare_initial_state(self, start_state, n_qubits):
        # Prepare the initial state as a Cirq circuit
        circuit = cirq.Circuit()
        qubits = cirq.LineQubit.range(n_qubits)

        if start_state == '+':
            for qubit in qubits:
                circuit.append(cirq.H(qubit))
        else:
            for qubit in qubits:
                circuit.append(cirq.I(qubit))
        return circuit, qubits

    def apply_gate(self, circuit, qubits, gate_matrix, sites):
        gate = cirq.MatrixGate(gate_matrix)
        circuit.append(gate(qubits[sites[0]], qubits[sites[1]]))
        return circuit

    def append_and_simulate(self, gate_matrix, sites):
        # Apply the specified gates
        self.circuit = self.apply_gate(
            self.circuit, self.qubits, gate_matrix, sites)
        # Example of how to simulate the circuit and get the final state vector
        simulator = cirq.DensityMatrixSimulator()
        result = simulator.simulate(self.circuit)
        density_matrix = result.final_density_matrix

        P = rho_to_P(rho=density_matrix, povm=self.povm,
                     N_qubits=self.config["N_QUBITS"])
        assert np.isclose(P.sum(), 1), "error simulating exact evolution"

        print(self.circuit)

        return P.reshape((2**(2*self.config["N_QUBITS"]), )).real


if __name__ == "__main__":
    config = read_config("config_adiabatic.ini")
    povm = POVM(POVM=config["POVM"], N_Qubits=config["N_QUBITS"])

    exactSim = ExactSimulator(config=config, povm=povm)

    P = exactSim.append_and_simulate(povm.cz.reshape(4, 4), sites=[0, 1])
    print(P, P.shape)
