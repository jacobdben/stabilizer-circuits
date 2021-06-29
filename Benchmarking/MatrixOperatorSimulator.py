import numpy as np


class MO_Circuit:
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    S = np.array([[1, 0], [0, 0 + 1j]])

    def __init__(self, filename):
        self.Nqubits = 0
        self.Gates = []

        with open(filename, 'r') as reader:
            self.Nqubits = int(reader.readline().split()[0])

            for line in reader:
                if line != '\n':
                    self.Gates.append([line.split()[0]] + [int(q) for q in line.split()[1:]])

        self.State = np.eye(2 ** self.Nqubits)
        self.SingleBitOps = [np.eye(2) for i in range(self.Nqubits)]

    def simulate(self):
        for gate in self.Gates:
            if gate[0] == 'H':
                self.SingleBitOps[gate[1]] = self.H @ self.SingleBitOps[gate[1]]
            elif gate[0] == 'S':
                self.SingleBitOps[gate[1]] = self.S @ self.SingleBitOps[gate[1]]
            elif gate[0] == 'CX':
                TempOp1 = self.SingleBitOps[-1]
                for i in range(len(self.SingleBitOps) - 2, -1, -1):
                    TempOp1 = np.kron(TempOp1, self.SingleBitOps[i])

                self.SingleBitOps = [np.eye(2) for i in range(self.Nqubits)]
                self.SingleBitOps[gate[1]] = np.array([[1, 0], [0, 0]])

                TempOp2 = self.SingleBitOps[-1]
                for i in range(len(self.SingleBitOps) - 2, -1, -1):
                    TempOp2 = np.kron(TempOp2, self.SingleBitOps[i])

                self.SingleBitOps[gate[1]] = np.array([[0, 0], [0, 1]])
                self.SingleBitOps[gate[2]] = np.array([[0, 1], [1, 0]])

                TempOp3 = self.SingleBitOps[-1]
                for i in range(len(self.SingleBitOps) - 2, -1, -1):
                    TempOp3 = np.kron(TempOp3, self.SingleBitOps[i])

                self.State = (TempOp2 + TempOp3) @ TempOp1 @ self.State

                self.SingleBitOps = [np.eye(2) for i in range(self.Nqubits)]

        TempOp = self.SingleBitOps[-1]
        for i in range(len(self.SingleBitOps) - 2, -1, -1):
            TempOp = np.kron(TempOp, self.SingleBitOps[i])
        self.SingleBitOps = [np.eye(2) for i in range(self.Nqubits)]
        self.State = TempOp @ self.State

    def measure(self):
        InitState = np.zeros(2 ** self.Nqubits)
        InitState[0] = 1

        return self.State @ InitState