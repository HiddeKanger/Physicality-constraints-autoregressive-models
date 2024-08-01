import numpy as np
from copy import deepcopy

if __name__ == "__main__":
    from ncon import ncon
else:
    from utils.ncon import ncon


class MPS():
    def __init__(self, POVM='Trine', N_Qubits=4, state='GHZ'):

        self.N = N_Qubits
        # Hamiltonian for calculation of energy (TFIM in 1d)
        # self.Jz = Jz
        # self.hx = hx

        # POVMs and other operators
        # Pauli matrices

        self.I = np.array([[1, 0], [0, 1]])
        self.X = np.array([[0, 1], [1, 0]])
        self.s1 = self.X
        self.Z = np.array([[1, 0], [0, -1]])
        self.s3 = self.Z
        self.Y = np.array([[0, -1j], [1j, 0]])
        self.s2 = self.Y
        self.H = 1.0/np.sqrt(2.0)*np.array([[1, 1], [1, -1]])
        self.Sp = np.array([[1.0, 0.0], [0.0, -1j]])
        self.oxo = np.array([[1.0, 0.0], [0.0, 0.0]])
        self.IxI = np.array([[0.0, 0.0], [0.0, 1.0]])
        self.Phase = np.array([[1.0, 0.0], [0.0, 1j]])  # =S = (Sp)^{\dag}
        self.T = np.array([[1.0, 0], [0, np.exp(-1j*np.pi/4.0)]])
        self.U1 = np.array(
            [[np.exp(-1j*np.pi/3.0),  0], [0, np.exp(1j*np.pi/3.0)]])

        # two-qubit gates
        self.cy = ncon((self.oxo, self.I), ([-1, -3], [-2, -4])) + \
            ncon((self.IxI, self.Y), ([-1, -3], [-2, -4]))
        self.cz = ncon((self.oxo, self.I), ([-1, -3], [-2, -4])) + \
            ncon((self.IxI, self.Z), ([-1, -3], [-2, -4]))

        if POVM.upper() == '4PAULI':
            print("MPS: 4PAULI")
            self.K = 4

            self.M = np.zeros((self.K, 2, 2), dtype=complex)

            self.M[0, :, :] = 1.0/3.0*np.array([[1, 0], [0, 0]])
            self.M[1, :, :] = 1.0/6.0*np.array([[1, 1], [1, 1]])
            self.M[2, :, :] = 1.0/6.0*np.array([[1, -1j], [1j, 1]])
            self.M[3, :, :] = 1.0/3.0*(np.array([[0, 0], [0, 1]]) +
                                       0.5*np.array([[1, -1], [-1, 1]])
                                       + 0.5*np.array([[1, 1j], [-1j, 1]]))

        if POVM.upper() == 'TETRA':
            print("MPS: TETRA")

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
            print("MPS: TRINE")

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
        # self.Trsx  = np.zeros((self.N,self.K),dtype=complex);
        # self.Trsy  = np.zeros((self.N,self.K),dtype=complex);
        # self.Trsz  = np.zeros((self.N,self.K),dtype=complex);
        # self.Trrho = np.zeros((self.N,self.K),dtype=complex);
        # self.Trrho2 = np.zeros((self.N,self.K,self.K),dtype=complex);
        # self.T2 = np.zeros((self.N,self.K,self.K),dtype=complex);
        self.Trsx = np.zeros(self.K, dtype=complex)
        self.Trsy = np.zeros(self.K, dtype=complex)
        self.Trsz = np.zeros(self.K, dtype=complex)
        self.Trrho = np.zeros((self.N, self.K), dtype=complex)
        self.Trrho2 = np.zeros((self.N, self.K, self.K), dtype=complex)
        self.T2 = np.zeros((self.N, self.K, self.K), dtype=complex)

        self.Trsx = ncon((self.M, self.it, self.X),
                         ([3, 2, 1], [3, -1], [2, 1]))
        self.Trsy = ncon((self.M, self.it, self.Y),
                         ([3, 2, 1], [3, -1], [2, 1]))
        self.Trsz = ncon((self.M, self.it, self.Z),
                         ([3, 2, 1], [3, -1], [2, 1]))
        self.stab_ops = [self.Trsx, self.Trsz]

        if state.upper() == "GHZ":
            print("MPS: GHZ STATE")

            # Copy tensors used to construct GHZ as an MPS. The procedure below should work for any other MPS
            cc = np.zeros((2, 2))  # corner
            cc[0, 0] = 2**(-1.0/(2*self.N))
            cc[1, 1] = 2**(-1.0/(2*self.N))

            cb = np.zeros((2, 2, 2))  # bulk
            cb[0, 0, 0] = 2**(-1.0/(2*self.N))
            cb[1, 1, 1] = 2**(-1.0/(2*self.N))

            self.MPS = []
            self.MPS.append(cc)
            for i in range(self.N-2):
                self.MPS.append(cb)
            self.MPS.append(cc)

            # c = np.zeros((2, 2))

            # c[0, 0] = 1/np.sqrt(2)
            # c[0, 1] = 0
            # c[1, 0] = 0
            # c[1, 1] = 1/np.sqrt(2)

            # X, Y, Z = np.linalg.svd(c)

            # self.MPS.append(np.matmul(X, np.diag(Y)))
            # self.MPS.append(Z)

        elif state.upper() == "GRAPH":
            print("MPS: GRAPH STATE")

            state = []

            chi = 2  # this particular graph state has bond dimension of 2

            plus = (1.0/np.sqrt(2.0))*np.ones(2)

            # first qubit
            temp = ncon((plus, plus, self.cz), ([1], [2], [-1, -2, 1, 2]))
            X, Y, Z = np.linalg.svd(temp, full_matrices=0)
            # truncation
            chi2 = np.min([np.sum(Y > 10.**(-10)), chi])
            piv = np.zeros(len(Y), bool)
            piv[(np.argsort(Y)[::-1])[:chi2]] = True
            Y = Y[piv]
            invsq = np.sqrt(sum(Y**2))
            X = X[:, piv]
            Z = Z[piv, :]

            state.append(np.matmul(X, np.diag(Y)))

            for i in range(1, self.N-1):
                temp = ncon((Z, plus, self.cz), ([-1, 1], [2], [-2, -3, 1, 2]))
                s0 = temp.shape[0]
                s1 = temp.shape[1]
                s2 = temp.shape[2]

                temp = np.reshape(temp, (s0*s1, s2))
                X, Y, Z = np.linalg.svd(temp, full_matrices=0)
                # truncation
                chi2 = np.min([np.sum(Y > 10.**(-10)), chi])
                piv = np.zeros(len(Y), bool)
                piv[(np.argsort(Y)[::-1])[:chi2]] = True
                Y = Y[piv]
                invsq = np.sqrt(sum(Y**2))

                X = X[:, piv]
                Z = Z[piv, :]

                state.append(np.reshape(
                    np.matmul(X, np.diag(Y)), (s0, s1, chi2)))

            # last qubit
            # transpose is to keep the indexing order consistent with my poor choice
            state.append(np.transpose(Z))

            # testing
            # GS = ncon((state[0], state[1]),
            #           ([-1, 1], [1, -2]))
            # GS = np.reshape(GS, (2**2))
            # print(GS)
            self.MPS = state

        elif state == "plus":
            print("MPS: PLUS")

            plus = (1.0/np.sqrt(2.0))*np.ones(2)
            self.MPS = []

            self.MPS.append(np.reshape(plus, [2, 1]))

            for i in range(1, self.N-1):
                self.MPS.append(np.reshape(plus, [1, 2, 1]))

            self.MPS.append(np.reshape(plus, [2, 1]))

    def Fidelity(self, S):
        Fidelity = 0.0
        F2 = 0.0
        Ns = S.shape[0]
        for i in range(Ns):

            # contracting the entire TN for each sample S[i,:]
            # eT = ncon(( self.TB[0][:,:,S[i,0]], self.TB[1][:,:,:,:,S[i,1]]) ,( [1,2],[-1,-2,1,2 ]));
            eT = ncon((self.it[:, S[i, 0]], self.M, self.MPS[0],
                      self.MPS[0]), ([3], [3, 2, 1], [1, -1], [2, -2]))

            for j in range(1, self.N-1):
                # eT = ncon((eT,self.TB[j][:,:,:,:,S[i,j]]),([ 1,2],[ -1,-2, 1,2 ]));
                eT = ncon((eT, self.it[:, S[i, j]], self.M, self.MPS[j], self.MPS[j]), ([
                          2, 4], [1], [1, 5, 3], [2, 3, -1], [4, 5, -2]))

            # eT = ncon((eT, self.TB[self.N-1][:,:,S[i,self.N-1]]),([1,2],[1,2 ]));
            j = self.N-1
            eT = ncon((eT, self.it[:, S[i, j]], self.M, self.MPS[j], self.MPS[j]), ([
                      2, 5], [1], [1, 4, 3], [3, 2], [4, 5]))
            # print i, eT
            Fidelity = Fidelity + eT
            F2 = F2 + eT**2
            Fest = Fidelity/float(i+1)
            F2est = F2/float(i+1)
            Error = np.sqrt(np.abs(F2est-Fest**2)/float(i+1))
            # print i,np.real(Fest),Error
            # disp([i,i/Ns, real(Fest), real(Error)])
            # fflush(stdout);

        F2 = F2/float(Ns)

        Fidelity = np.abs(Fidelity/float(Ns))

        Error = np.sqrt(np.abs(F2-Fidelity**2)/float(Ns))

        return np.real(Fidelity), Error

    def cFidelity(self, S, logP):
        Fidelity = 0.0
        F2 = 0.0
        Ns = S.shape[0]
        KL = 0.0
        K2 = 0.0

        for i in range(Ns):

            P = ncon((self.MPS[0], self.MPS[0], self.M[S[i, 0], :, :]), ([
                     1, -1], [2, -2], [1, 2]))

            # contracting the entire TN for each sample S[i,:]
            for j in range(1, self.N-1):
                P = ncon((P, self.MPS[j], self.MPS[j], self.M[S[i, j], :, :]), ([
                         1, 2], [1, 3, -1], [2, 4, -2], [3, 4]))

            # full contraction  --> float as result
            P = ncon((P, self.MPS[self.N-1], self.MPS[self.N-1],
                     self.M[S[i, self.N-1], :, :]), ([1, 2], [3, 1], [4, 2], [3, 4]))

            ee = np.sqrt(P/np.exp(logP[i]))
            # print(P, np.exp(logP[i]), np.log(ee), KL)

            Fidelity = Fidelity + ee
            F2 = F2 + ee**2

            # 2 times because of square root term in front of log is implied by averaging
            KL = KL + 2*np.log(ee)
            K2 = K2 + 4*(np.log(ee))**2

        F2 = F2/float(Ns)

        Fidelity = np.abs(Fidelity/float(Ns))

        Error = np.sqrt(np.abs(F2-Fidelity**2)/float(Ns))

        K2 = K2/float(Ns)

        KL = np.abs(KL/float(Ns))

        ErrorKL = np.sqrt(np.abs(K2-KL**2)/float(Ns))

        return np.real(Fidelity)[0], Error[0], np.real(KL)[0], ErrorKL[0]

    def stabilizers(self, i, si, j, sj, k, sk):
        MPS = deepcopy(self.MPS)
        if i == 0:
            MPS[i] = ncon((MPS[i], si), ([1, -2], [-1, 1]))
        elif i == self.N-1:
            MPS[i] = ncon((MPS[i], si), ([-2, 1], [-1, 1]))
        else:
            MPS[i] = ncon((MPS[i], si), ([-1, 1, -3], [-2, 1]))

        if j == 0:
            MPS[j] = ncon((MPS[j], sj), ([1, -2], [-1, 1]))
        elif j == self.N-1:
            MPS[j] = ncon((MPS[j], sj), ([-2, 1], [-1, 1]))
        else:
            MPS[j] = ncon((MPS[j], sj), ([-1, 1, -3], [-2, 1]))

        if k == 0:
            MPS[k] = ncon((MPS[k], sk), ([1, -2], [-1, 1]))
        elif k == self.N-1:
            MPS[k] = ncon((MPS[k], sk), ([-2, 1], [-1, 1]))
        else:
            MPS[k] = ncon((MPS[k], sk), ([-1, 1, -3], [-2, 1]))

        C = ncon((MPS[0], np.conj(self.MPS[0])), ([1, -1], [1, -2]))
        for ii in range(1, self.N-1):
            C = ncon((C, MPS[ii], np.conj(self.MPS[ii])),
                     ([1, 2], [1, 3, -1], [2, 3, -2]))

        ii = self.N-1
        C = ncon((C, MPS[ii], np.conj(self.MPS[ii])), ([1, 2], [3, 1], [3, 2]))
        return C

    def stabilizers_samples(self, S):
        s = np.zeros(self.N)
        s2 = np.zeros(self.N)
        Ns = S.shape[0]
        for i in range(Ns):
            for j in range(self.N):

                if j == 0:
                    temp = self.Trsx[S[i, j]]*self.Trsz[S[i, j+1]]
                    s[0] = s[0] + temp
                    s2[0] = s2[0] + temp**2

                elif j == self.N-1:
                    temp = self.Trsz[S[i, j-1]]*self.Trsx[S[i, j]]
                    s[self.N-1] = s[self.N-1] + temp
                    s2[self.N-1] = s2[self.N-1] + temp**2

                else:
                    temp = self.Trsz[S[i, j-1]] * \
                        self.Trsx[S[i, j]]*self.Trsz[S[i, j+1]]
                    s[j] = s[j] + temp
                    s2[j] = s2[j] + temp**2

        s2 = s2/float(Ns)
        s = s/float(Ns)

        Error = np.sqrt(np.abs(s2-s**2)/float(Ns))

        return s, Error


if __name__ == "__main__":
    mps = MPS(POVM="4Pauli", N_Qubits=2, state="GHZ")
    print(mps.MPS[0], mps.MPS[1])
