import numpy as np
from utils.ncon import ncon
# import tensorly as tl
# from tensorly.decomposition import matrix_product_state
from copy import deepcopy


class POVM():
    def __init__(self, POVM='4Pauli', N_Qubits=2):

        self.N = N_Qubits
        # Hamiltonian for calculation of energy (TFIM in 1d)
        # self.Jz = Jz
        # self.hx = hx

        # POVMs and other operators
        # Pauli matrices,gates,simple states
        self.I = np.array([[1, 0], [0, 1]])
        self.X = np.array([[0, 1], [1, 0]])
        self.Z = np.array([[1, 0], [0, -1]])
        self.Y = np.array([[0, -1j], [1j, 0]])

        # Pauli matrices
        self.s1 = self.X
        self.s3 = self.Z
        self.s2 = self.Y

        # gates for random circuit
        self.sx = np.array(
            [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]])
        self.sy = np.array(
            [[0.5 + 0.5j, -0.5 - 0.5j], [0.5 + 0.5j, 0.5 + 0.5j]])
        self.sw = (self.sx + self.sy)/np.sqrt(3)
        self.c = np.array([[1, 0, 0, 0], [0, 0, -1.j, 0],
                          [0, -1.j, 0, 0], [0, 0, 0, np.exp(-1.j * np.pi / 6)]])

        self.h = 1.0/np.sqrt(2.0)*np.array([[1, 1], [1, -1]])
        self.Sp = np.array([[1.0, 0.0], [0.0, -1j]])
        self.oxo = np.array([[1.0, 0.0], [0.0, 0.0]])
        self.IxI = np.array([[0.0, 0.0], [0.0, 1.0]])
        self.Phase = np.array([[1.0, 0.0], [0.0, 1j]])  # =S = (Sp)^{\dag}
        self.T = np.array([[1.0, 0], [0, np.exp(-1j*np.pi/4.0)]])
        self.U1 = np.array(
            [[np.exp(-1j*np.pi/3.0),  0], [0, np.exp(1j*np.pi/3.0)]])

        # two-qubit gates
        self.cx = ncon((self.oxo, self.I), ([-1, -3], [-2, -4])) + \
            ncon((self.IxI, self.X), ([-1, -3], [-2, -4]))
        self.cy = ncon((self.oxo, self.I), ([-1, -3], [-2, -4])) + \
            ncon((self.IxI, self.Y), ([-1, -3], [-2, -4]))
        self.cz = ncon((self.oxo, self.I), ([-1, -3], [-2, -4])) + \
            ncon((self.IxI, self.Z), ([-1, -3], [-2, -4]))
        self.cnot = ncon((self.oxo, self.I), ([-1, -3], [-2, -4])) + \
            ncon((self.IxI, self.X), ([-1, -3], [-2, -4]))
        self.cu1 = ncon((self.oxo, self.I), ([-1, -3], [-2, -4])) + \
            ncon((self.IxI, self.U1), ([-1, -3], [-2, -4]))

        self.swap = np.array([[1, 0, 0, 0],
                              [0, 0, 1, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1]])

        self.single_qubit = [self.h, self.Phase, self.T, self.U1]

        # here we make 2 qubit gates out of single qubits by using kron identity
        self.single_qubit2 = [np.kron(x, self.I).reshape(
            2, 2, 2, 2) for x in self.single_qubit]

        self.two_qubit = [self.cnot, self.cz, self.cy,
                          self.cu1, np.matmul(self.h, self.cx)]

        if POVM.upper() == '4PAULI':
            self.K = 4

            self.M = np.zeros((self.K, 2, 2), dtype=complex)

            self.M[0, :, :] = 1.0/3.0*np.array([[1, 0], [0, 0]])
            self.M[1, :, :] = 1.0/6.0*np.array([[1, 1], [1, 1]])
            self.M[2, :, :] = 1.0/6.0*np.array([[1, -1j], [1j, 1]])
            self.M[3, :, :] = 1.0/3.0*(np.array([[0, 0], [0, 1]]) +
                                       0.5*np.array([[1, -1], [-1, 1]])
                                       + 0.5*np.array([[1, 1j], [-1j, 1]]))

        if POVM.upper() == 'TETRA':
            self.K = 4

            self.M = np.zeros((self.K, 2, 2), dtype=complex)

            self.v1 = np.array([0, 0, 1.0])
            self.M[0, :, :] = 1.0/4.0 * \
                (self.I + self.v1[0]*self.s1 +
                 self.v1[1]*self.s2+self.v1[2]*self.s3)

            self.v2 = np.array([2.0*np.sqrt(2.0)/3.0, 0.0, -1.0/3.0])
            self.M[1, :, :] = 1.0/4.0 * \
                (self.I + self.v2[0]*self.s1 +
                 self.v2[1]*self.s2+self.v2[2]*self.s3)

            self.v3 = np.array([-np.sqrt(2.0)/3.0, np.sqrt(2.0/3.0), -1.0/3.0])
            self.M[2, :, :] = 1.0/4.0 * \
                (self.I + self.v3[0]*self.s1 +
                 self.v3[1]*self.s2+self.v3[2]*self.s3)

            self.v4 = np.array(
                [-np.sqrt(2.0)/3.0, -np.sqrt(2.0/3.0), -1.0/3.0])
            self.M[3, :, :] = 1.0/4.0 * \
                (self.I + self.v4[0]*self.s1 +
                 self.v4[1]*self.s2+self.v4[2]*self.s3)

        elif POVM.upper() == 'TRINE':
            self.K = 3
            self.M = np.zeros((self.K, 2, 2), dtype=complex)
            phi0 = 0.0
            for k in range(self.K):
                phi = phi0 + (k)*2*np.pi/3.0
                self.M[k, :, :] = 0.5 * \
                    (self.I + np.cos(phi)*self.Z + np.sin(phi)*self.X)*2/3.0

        # % T matrix and its inverse
        self.t = ncon((self.M, self.M), ([-1, 1, 2], [-2, 2, 1]))
        self.it = np.linalg.inv(self.t)
        # Tensor for expectation value
        self.Trsx = np.zeros((self.N, self.K), dtype=complex)
        self.Trsy = np.zeros((self.N, self.K), dtype=complex)
        self.Trsz = np.zeros((self.N, self.K), dtype=complex)
        self.Trrho = np.zeros((self.N, self.K), dtype=complex)
        self.Trrho2 = np.zeros((self.N, self.K, self.K), dtype=complex)
        self.T2 = np.zeros((self.N, self.K, self.K), dtype=complex)

        # probability gate set single qubit
        self.p_single_qubit = []
        for i in range(len(self.single_qubit)):
            mat = ncon((self.M, self.single_qubit[i], self.M, self.it, np.transpose(
                np.conj(self.single_qubit[i]))), ([-1, 4, 1], [1, 2], [3, 2, 5], [3, -2], [5, 4]))
            self.p_single_qubit.append(mat)

        # probability gate set two qubit: these are the O matrices, see notes.
        self.p_two_qubit = []
        for i in range(len(self.two_qubit)):
            mat = ncon((self.M, self.M, self.two_qubit[i], self.M, self.M, self.it, self.it, np.conj(
                self.two_qubit[i])), ([-1, 9, 1], [-2, 10, 2], [1, 2, 3, 4], [5, 3, 7], [6, 4, 8], [5, -3], [6, -4], [9, 10, 7, 8]))
            self.p_two_qubit.append(mat)
            # print(np.real(np.sum(np.reshape(self.p_two_qubit[i],(16,16)),1)),np.real(np.sum(np.reshape(self.p_two_qubit[i],(16,16)),0)))

        # these "two qubit gates" consist of single qubit gates but with the identity acting on the second qubit
        # this way the code for applying qubits is identical (and is only done few times)
        self.p_single_qubit2 = []
        for i in range(len(self.single_qubit2)):
            mat = ncon((self.M, self.M, self.single_qubit2[i], self.M, self.M, self.it, self.it, np.conj(
                self.single_qubit2[i])), ([-1, 9, 1], [-2, 10, 2], [1, 2, 3, 4], [5, 3, 7], [6, 4, 8], [5, -3], [6, -4], [9, 10, 7, 8]))
            self.p_single_qubit2.append(mat)

        self.single_qubit_dict = {  # for the single qubit gates we use the 2 qubit version
            "H": self.p_single_qubit2[0],
            "Phase": self.p_single_qubit2[1],
            "T": self.p_single_qubit2[2],
            "U1": self.p_single_qubit2[3]
        }

        self.two_qubit_dict = {
            "CNOT": self.p_two_qubit[0],
            "CZ": self.p_two_qubit[1],
            "CY": self.p_two_qubit[2],
            "CX": self.p_two_qubit[3],
            "GHZ": self.p_two_qubit[4],  # GHZ gate, Hadammard and CX in 1 gate
        }

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def getinitialbias(self, initial_state):
        # which initial product state?
        if initial_state == '0':
            s = np.array([1, 0])
        elif initial_state == '1':
            s = np.array([0, 1])
        elif initial_state == '+':
            s = (1.0/np.sqrt(2.0))*np.array([1, 1])
        elif initial_state == '-':
            s = (1.0/np.sqrt(2.0))*np.array([1, -1])
        elif initial_state == 'r':
            s = (1.0/np.sqrt(2.0))*np.array([1, 1j])
        elif initial_state == 'l':
            s = (1.0/np.sqrt(2.0))*np.array([1, -1j])

        self.P = np.real(ncon((self.M, s, np.conj(s)), ([-1, 1, 2], [1], [2])))

        self.bias = np.log(self.P)

        if np.sum(np.abs(self.softmax(self.bias)-self.P)) > 0.00000000001:
            print("initial bias not found")
        else:
            return self.bias


if __name__ == "__main__":
    povm = POVM(POVM="4Pauli")
    print("M shape:", povm.M.shape)

    print("M0:")
    print(povm.M[0], "\n")
    print("M1:")
    print(povm.M[1], "\n")
    print("M2:")
    print(povm.M[2], "\n")
    print("M3:")
    print(povm.M[3], "\n")

    print(f"CZ: ({povm.cz.shape})")
    print(povm.cz)
