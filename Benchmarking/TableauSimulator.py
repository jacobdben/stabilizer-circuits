import numpy as np

def XOR(a,b):
    return int(a != b)

class Tableau_Circuit:
    def __init__(self, filename):
        self.Nqubits = 0
        self.gates = []

        with open(filename, 'r') as reader:
            self.Nqubits = int(reader.readline().split()[0])

            for line in reader:
                if line != '\n':
                    self.gates.append([line.split()[0]] + [int(q) for q in line.split()[1:]])

        self.tableau = np.concatenate(
            (np.eye(2 * self.Nqubits, dtype='int'), np.zeros((2 * self.Nqubits, 1), dtype='int')), axis=1)
        self.tableau = np.concatenate((self.tableau, np.zeros((1, 2 * self.Nqubits + 1), dtype='int')), axis=0)

    def apply_Hadamard_gate(self, a):
        """
        Applies H-gate to qubit number a
        """

        for i in range(2 * self.Nqubits):
            # r_i = r_i xor x_ia * z_ia
            self.tableau[i, 2 * self.Nqubits] = XOR(self.tableau[i, 2 * self.Nqubits], \
                                                    self.tableau[i, a] * self.tableau[i, a + self.Nqubits])

            # swap x_ia with z_ia
            self.tableau[i, a], self.tableau[i, a + self.Nqubits] = self.tableau[i, a + self.Nqubits], self.tableau[
                i, a]

    def apply_S_gate(self, a):
        """
        Applies S-gate to qubit number a
        """

        for i in range(2 * self.Nqubits):
            # r_i = r_i xor x_ia * z_ia
            self.tableau[i, 2 * self.Nqubits] = XOR(self.tableau[i, 2 * self.Nqubits], \
                                                    self.tableau[i, a] * self.tableau[i, a + self.Nqubits])

            # z_ia = x_ia xor z_ia
            self.tableau[i, a + self.Nqubits] = XOR(self.tableau[i, a + self.Nqubits], self.tableau[i, a])

    def apply_CNOT_gate(self, a, b):
        """
        Applies CX-gate to qubit number a
        """

        for i in range(2 * self.Nqubits):
            # r_i = r_i xor x_ia * z_ib * (x_ib xor z_ia xor 1)
            self.tableau[i, 2 * self.Nqubits] = XOR(self.tableau[i, 2 * self.Nqubits], \
                                                    self.tableau[i, a] * self.tableau[i, b + self.Nqubits] \
                                                    * XOR(self.tableau[i, b],
                                                          XOR(self.tableau[i, a + self.Nqubits], 1)))

            # x_ib = x_ib xor x_ia
            self.tableau[i, b] = XOR(self.tableau[i, b], self.tableau[i, a])

            # z_ia = z_ia xor z_ib
            self.tableau[i, a + self.Nqubits] = XOR(self.tableau[i, a + self.Nqubits],
                                                    self.tableau[i, b + self.Nqubits])

    def simulate(self):
        """
        Build the circuit
        """
        for gate in self.gates:
            if gate[0] == 'H':
                self.apply_Hadamard_gate(gate[1])
            elif gate[0] == 'S':
                self.apply_S_gate(gate[1])
            elif gate[0] == 'CX':
                self.apply_CNOT_gate(gate[1], gate[2])

    def rowsum(self, h, j):
        """
        Helper function outlined by Aaronson, S. and Gottesman, D.
        Replaces generator g_h with g_h * g_j
        """

        def g(x1, z1, x2, z2):
            if x1 == 0 and z1 == 0:
                return 0
            elif x1 == 1 and z1 == 1:
                return z2 - x2
            elif x1 == 1 and z1 == 0:
                return z2 * (2 * x2 - 1)
            elif x1 == 0 and z1 == 1:
                return x2 * (1 - 2 * z2)

        m = (2 * self.tableau[h, 2 * self.Nqubits] + 2 * self.tableau[j, 2 * self.Nqubits] \
             + np.sum([g(self.tableau[j, k], self.tableau[j, k + self.Nqubits], self.tableau[h, k], \
                         self.tableau[h, k + self.Nqubits]) for k in range(self.Nqubits)])) % 4

        # r_h = m/2
        self.tableau[h, 2 * self.Nqubits] = m / 2

        for k in range(self.Nqubits):
            # x_hk = x_jk xor x_hk
            self.tableau[h, k] = XOR(self.tableau[j, k], self.tableau[h, k])

            # z_hk = z_jk xor z_hk
            self.tableau[h, k + self.Nqubits] = XOR(self.tableau[j, k + self.Nqubits],
                                                    self.tableau[h, k + self.Nqubits])

    def measure_qubit(self, a):
        """
        Measures qubit number a
        """
        deterministic = True
        q = 0

        # Check if measurement is deterministic or random
        for p in range(self.Nqubits, 2 * self.Nqubits):
            if self.tableau[p, a] == 1:
                deterministic = False
                q = p
                break

        if deterministic:
            # Case of deterministic measurement

            for j in range(self.Nqubits):
                if self.tableau[j, a] == 1:
                    self.rowsum(2 * self.Nqubits, j + self.Nqubits)

            res = self.tableau[2 * self.Nqubits, 2 * self.Nqubits]
            self.tableau[2 * self.Nqubits] = np.zeros(2 * self.Nqubits + 1, dtype='int')

            return res


        else:
            # Case of random measurement

            # The q'th generator g_q was found to anticommute with the measurment operator Z_q,
            # so we transform all other generators g_j!=g_q such that they do not commute with Z_q.
            # This we do by setting g_j=g_j g_q, ie. taking rowsum(j,q). Our new generators generate the
            # same group!
            for j in range(2 * self.Nqubits):
                if j == q or self.tableau[j, a] != 1:
                    continue
                self.rowsum(j, q)

            # Set (q-n)'th row equals the q'th row
            self.tableau[q - self.Nqubits] = self.tableau[q]

            # Set x_qj = z_qj = 0 for all j
            self.tableau[q] = np.zeros(2 * self.Nqubits + 1, dtype='int')
            # Set z_qa = 1
            self.tableau[q, a + self.Nqubits] = 1

            # Set r_q to be 0 or 1 with equal probability
            self.tableau[q, 2 * self.Nqubits] = np.random.randint(0, 2)

            # Return r_q as the measurement result
            return self.tableau[q, 2 * self.Nqubits]

    def measure_system(self):
        """
        Measures the wavefunction at the end of the circuit
        """
        wavefunction = np.zeros(self.Nqubits, dtype='int')
        for i in range(self.Nqubits - 1, -1, -1):
            wavefunction[i] = self.measure_qubit(i)

        return wavefunction