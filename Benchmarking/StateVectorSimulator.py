import numpy as np

def b_i(qubit, index):
    return (index // 2 ** qubit) % 2


def index(qubit, old_index, x):
    if b_i(qubit, old_index) == x:
        return old_index
    elif b_i(qubit, old_index) == 0 and x == 1:
        return old_index + 2 ** qubit
    elif b_i(qubit, old_index) == 1 and x == 0:
        return old_index - 2 ** qubit


def index2(target, control, old_index, x, y):
    if b_i(target, old_index) == x and b_i(control, old_index) == y:
        return old_index

    if b_i(target, old_index) == 0 and x == 1:
        old_index += 2 ** target
    elif b_i(target, old_index) == 1 and x == 0:
        old_index -= 2 ** target

    if b_i(control, old_index) == 0 and y == 1:
        old_index += 2 ** control
    elif b_i(control, old_index) == 1 and y == 0:
        old_index -= 2 ** control

    return old_index


class SV_Circuit:
    GATE_TYPES = {
        'H': np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
        'S': np.array([[1, 0], [0, 0 + 1j]], dtype=complex),
        'CX': np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    }

    def __init__(self, filename):
        self.Nqubits = 0
        self.Gates = []
        self.CurrPos = 0

        with open(filename, 'r') as reader:
            self.Nqubits = int(reader.readline().split()[0])

            for line in reader:
                if line != '\n':
                    self.Gates.append([line.split()[0]] + [int(q) for q in line.split()[1:]])

        self.State = np.zeros(2 ** self.Nqubits, dtype=complex)
        self.State[0] = 1 + 0j

    def apply_single_qubit_gate(self, gate, target):
        U = self.GATE_TYPES[gate]
        t = target
        OldState = np.copy(self.State)
        for i in range(2 ** self.Nqubits):
            self.State[i] = U[b_i(t, i), 0] * OldState[index(t, i, 0)] + U[b_i(t, i), 1] * OldState[index(t, i, 1)]

    def apply_two_qubit_gate(self, gate, control, target):
        U = self.GATE_TYPES[gate]
        t = target
        c = control
        OldState = np.copy(self.State)

        for i in range(2 ** self.Nqubits):
            b_t = b_i(t, i)
            b_c = b_i(c, i)
            self.State[i] = U[b_t + 2 * b_c, 0] * OldState[index2(t, c, i, 0, 0)] \
                            + U[b_t + 2 * b_c, 1] * OldState[index2(t, c, i, 1, 0)] \
                            + U[b_t + 2 * b_c, 2] * OldState[index2(t, c, i, 0, 1)] \
                            + U[b_t + 2 * b_c, 3] * OldState[index2(t, c, i, 1, 1)]

    def propagate(self):
        if self.CurrPos < len(self.Gates):
            gate = self.Gates[self.CurrPos]
            if len(gate) == 2:
                self.apply_single_qubit_gate(gate[0], gate[1])
            elif len(gate) == 3:
                self.apply_two_qubit_gate(gate[0], gate[1], gate[2])
            self.CurrPos += 1

    def simulate(self):
        for g in self.Gates:
            self.propagate()

    def measure(self):
        return self.State
