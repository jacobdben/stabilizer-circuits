import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

TwoBitLookup = [[[0 for j in range(24)] for i in range(24)], [[0 for j in range(24)] for i in range(24)]]

# Load lookup table for two bit graph state from file
with open('TwoBitLookup.txt', 'r') as f:
    for edge in range(2):
        for i in range(24):
            for j in range(24):
                line = f.readline()
                line = line.split()
                e = int(line[4].split(':')[0])
                c_vop = int(line[5].split(',')[0])
                t_vop = int(line[6])
                TwoBitLookup[edge][i][j] = np.array([e, c_vop, t_vop], dtype='int')

# Use: [new edge, new control VOP, new target VOP] = TwoBitLookup[edge, control VOP, target VOP]
TwoBitLookup = np.array(TwoBitLookup)

# Index of the list corresponds to the Clifford operator number (ie. 0-23)
CliffordOpStrings = ['XXXX', 'X', 'Z', 'XX', 'ZZ', 'XZ', 'ZX', 'XXX', 'ZZZ', 'XXZ', 'XZZ', 'ZXX', 'ZZX', 'XZX', 'XXXZ', 'XXZX', 'XZZZ', 'XZXX', 'ZZZX', 'ZZXX', 'ZXXX', 'XXXZX', 'XZXXX', 'XZZZX']
rhs_iZdag = np.array([8,16,0,11,2,1,21,17,4,3,5,19,14,6,7,13,10,12,23,9,22,15,18,20], dtype='int')
rhs_iX = np.array([7,0,20,1,10,22,2,3,15,18,19,6,4,5,23,9,21,13,8,12,11,14,17,16], dtype='int')
lhs_iZ = np.array([2,6,4,11,8,13,12,20,0,3,15,19,18,17,23,7,21,22,1,9,10,14,5,16], dtype='int')
lhs_iXdag = np.array([1,3,5,7,10,9,13,0,16,14,19,17,4,15,2,21,11,8,23,12,22,6,18,20], dtype='int')
measurement_ops = ['+Z','+Y','+Z','-Z','+Z','-X','+Y','-Y','+Z','-Z','-Y','-Z','+Y','-X','+X','-Y','+X','-X','+Y','-Z','-Y','+X','-X','+X']

Zgroup = np.array([0,2,4,8], dtype='int')

class Vertix:
    def __init__(self):
        self.edges = []
        self.VOP = 23


class Graph:
    def __init__(self, nqubits):
        self.qubit = [Vertix() for i in range(nqubits)]
        self.N = nqubits

    def add_edge(self, a, b):
        assert (max(a, b) < self.N)
        assert (a != b)

        if a not in self.qubit[b].edges:
            self.qubit[a].edges.append(b)
            self.qubit[b].edges.append(a)

    def delete_edge(self, a, b):
        assert (max(a, b) < self.N)
        assert (a != b)

        if a in self.qubit[b].edges:
            self.qubit[a].edges.remove(b)
            self.qubit[b].edges.remove(a)

    def toggle_edge(self, a, b):
        if a in self.qubit[b].edges:
            self.delete_edge(a, b)
        else:
            self.add_edge(a, b)

    def local_complementation(self, a):
        ngbh_a = self.qubit[a].edges

        for i in range(len(ngbh_a) - 1):
            for j in range(i + 1, len(ngbh_a)):
                self.toggle_edge(ngbh_a[i], ngbh_a[j])

def SQG_on_vop(gate_string, vop):
    """
    Takes the string rep. of a Single Qubit Gate operation and transforms the given VOP accordingly
    """

    for op in gate_string[::-1]:
        if op == 'X':
            vop = lhs_iXdag[vop]
        elif op == 'Z':
            vop = lhs_iZ[vop]
        else:
            raise Exception("Error in Gate String")

    return vop

class GS_Circuit:
    def __init__(self, filename):
        self.Nqubits = 0
        self.gates = []
        self.fname = filename

        with open(filename, 'r') as reader:
            self.Nqubits = int(reader.readline().split()[0])

            for line in reader:
                if line != '\n':
                    self.gates.append([line.split()[0]] + [int(q) for q in line.split()[1:]])

        self.GraphState = Graph(self.Nqubits)

    def reset(self):
        self.__init__(self.fname)

    def apply_Hadamard_gate(self, a):
        """
        Applies H-gate to qubit number a
        """

        assert (a < self.Nqubits), "Qubit " + str(a) + " out of range for circuit of " + str(self.Nqubits) + " qubits."

        vop = self.GraphState.qubit[a].VOP
        vop = SQG_on_vop(CliffordOpStrings[23], vop)
        self.GraphState.qubit[a].VOP = vop

    def apply_S_gate(self, a):
        """
        Applies S-gate to qubit number a
        """

        assert (a < self.Nqubits), "Qubit " + str(a) + " out of range for circuit of " + str(self.Nqubits) + " qubits."

        vop = self.GraphState.qubit[a].VOP
        vop = SQG_on_vop(CliffordOpStrings[8], vop)
        self.GraphState.qubit[a].VOP = vop

    def reduce_vertix(self, a, b):
        """
        Reduces VOP of qubit a to identity while leaving the state of the system invariant. The qubit b is the other operand
        vertix.
        """

        vop = self.GraphState.qubit[a].VOP
        vop_string = CliffordOpStrings[vop]

        # Find a non-operand neighbour of a
        c = self.GraphState.qubit[a].edges[0]

        assert (c != b or len(self.GraphState.qubit[a].edges) > 1)

        if c == b:
            c = self.GraphState.qubit[a].edges[1]

        for op in vop_string[::-1]:

            if op == 'X':
                self.GraphState.local_complementation(a)
                self.GraphState.qubit[a].VOP = rhs_iX[self.GraphState.qubit[a].VOP]
                for k in self.GraphState.qubit[a].edges:
                    self.GraphState.qubit[k].VOP = rhs_iZdag[self.GraphState.qubit[k].VOP]


            elif op == 'Z':
                self.GraphState.local_complementation(c)
                self.GraphState.qubit[c].VOP = rhs_iX[self.GraphState.qubit[c].VOP]
                for k in self.GraphState.qubit[c].edges:
                    self.GraphState.qubit[k].VOP = rhs_iZdag[self.GraphState.qubit[k].VOP]

            else:
                raise Exception("Error in Gate String")

    def CZ_on_2qbit_subgraph(self, a, b):
        """
        Performs the CZ gate on the subgraph of two vertices a and b, ignoring the rest of the graph.
        """

        vop_a = self.GraphState.qubit[a].VOP
        vop_b = self.GraphState.qubit[b].VOP

        has_edge = int(b in self.GraphState.qubit[a].edges)

        has_edge, vop_a, vop_b = TwoBitLookup[has_edge, vop_a, vop_b]
        self.GraphState.qubit[a].VOP = vop_a
        self.GraphState.qubit[b].VOP = vop_b
        if has_edge:
            self.GraphState.add_edge(a, b)
        else:
            self.GraphState.delete_edge(a, b)

    def apply_CZ_gate(self, a, b):
        """
        Applies CZ-gate to qubit number a
        """

        assert (max(a, b) < self.Nqubits)
        assert (a != b)

        # Case 1
        if (self.GraphState.qubit[a].VOP in Zgroup) and (self.GraphState.qubit[b].VOP in Zgroup):
            self.GraphState.toggle_edge(a, b)
        # Case 2
        else:
            a_edges = self.GraphState.qubit[a].edges
            b_edges = self.GraphState.qubit[b].edges
            a_has_non_op_ngbh = len(a_edges) > 1 or (len(a_edges) == 1 and a_edges[0] != b)
            b_has_non_op_ngbh = len(b_edges) > 1 or (len(b_edges) == 1 and b_edges[0] != a)

            # Subcase 2.1
            if a_has_non_op_ngbh and b_has_non_op_ngbh:

                self.reduce_vertix(a, b)

                if len(self.GraphState.qubit[b].edges) > 1 or self.GraphState.qubit[b].edges[0] != a:
                    self.reduce_vertix(b, a)

                    # Both vertices are now in Z_group, proceed as in Case 1
                    assert (self.GraphState.qubit[a].VOP in Zgroup)
                    assert (self.GraphState.qubit[b].VOP in Zgroup)
                    self.GraphState.toggle_edge(a, b)
                else:
                    self.CZ_on_2qbit_subgraph(a, b)

            # Subcase 2.2.1
            elif not a_has_non_op_ngbh and not b_has_non_op_ngbh:
                self.CZ_on_2qbit_subgraph(a, b)

            # Subcase 2.2.2a
            elif a_has_non_op_ngbh and not b_has_non_op_ngbh:
                self.reduce_vertix(a, b)
                self.CZ_on_2qbit_subgraph(a, b)

            # Subcase 2.2.2b
            elif not a_has_non_op_ngbh and b_has_non_op_ngbh:
                self.reduce_vertix(b, a)
                self.CZ_on_2qbit_subgraph(a, b)

    def apply_CNOT_gate(self, a, b):
        """
        Applies CX-gate to qubit number a using an equivialent circuit of H- and CZ-gates.
        """
        self.apply_Hadamard_gate(b)
        self.apply_CZ_gate(a, b)
        self.apply_Hadamard_gate(b)

    def simulate(self):
        """
        Build the circuit
        """
        for gate in self.gates:
            if gate[0] == 'H':
                self.apply_Hadamard_gate(gate[1])
            elif gate[0] == 'S':
                self.apply_S_gate(gate[1])
            elif gate[0] == 'CZ':
                self.apply_CZ_gate(gate[1], gate[2])
            elif gate[0] == 'CX':
                self.apply_CNOT_gate(gate[1], gate[2])

    def measure_qubit(self, a):
        """
        Measures qubit number a
        """
        vop_a = self.GraphState.qubit[a].VOP
        ngbh_a = self.GraphState.qubit[a].edges

        P_a = measurement_ops[vop_a]

        r = None
        r_comp = None

        # Case: P_a = +Z or -Z
        if P_a[1] == 'Z':
            r_comp = np.random.randint(0, 2)
            self.GraphState.qubit[a].VOP = SQG_on_vop(CliffordOpStrings[vop_a], 23)

            if r_comp:
                self.GraphState.qubit[a].VOP = SQG_on_vop(CliffordOpStrings[self.GraphState.qubit[a].VOP], 3)
                for b in ngbh_a:
                    self.GraphState.qubit[b].VOP = SQG_on_vop(CliffordOpStrings[self.GraphState.qubit[b].VOP], 4)

            for b in ngbh_a:
                self.GraphState.delete_edge(a, b)

            r = int(not r_comp)

        # Case: P_a = +Y or -Y
        elif P_a[1] == 'Y':
            r_comp = np.random.randint(0, 2)

            if r_comp:
                self.GraphState.qubit[a].VOP = rhs_iZdag[vop_a]
                for b in ngbh_a:
                    self.GraphState.qubit[b].VOP = rhs_iZdag[self.GraphState.qubit[b].VOP]
            else:
                self.GraphState.qubit[a].VOP = SQG_on_vop(CliffordOpStrings[self.GraphState.qubit[a].VOP], 2)
                for b in ngbh_a:
                    self.GraphState.qubit[b].VOP = SQG_on_vop(CliffordOpStrings[self.GraphState.qubit[b].VOP], 2)

            self.GraphState.local_complementation(a)

            r = int(not r_comp)

        # Case: P_a = +X or -X
        elif P_a[1] == 'X':

            if len(ngbh_a) == 0:
                if P_a[0] == '+':
                    r = 0
                elif P_a[0] == '-':
                    r = 1

            else:
                r_comp = np.random.randint(0, 2)
                b = ngbh_a[0]
                ngbh_b = self.GraphState.qubit[b].edges

                if r_comp:
                    cs = [i for i in ngbh_b if i not in ngbh_a and i != a]
                    self.GraphState.qubit[a].VOP = SQG_on_vop(CliffordOpStrings[self.GraphState.qubit[a].VOP], 4)
                    self.GraphState.qubit[b].VOP = SQG_on_vop(CliffordOpStrings[self.GraphState.qubit[b].VOP], 21)

                    for c in cs:
                        self.GraphState.qubit[c].VOP = SQG_on_vop(CliffordOpStrings[self.GraphState.qubit[c].VOP], 4)

                else:
                    cs = [i for i in ngbh_a if i not in ngbh_b and i != b]
                    self.GraphState.qubit[b].VOP = SQG_on_vop(CliffordOpStrings[self.GraphState.qubit[b].VOP], 22)

                    for c in cs:
                        self.GraphState.qubit[c].VOP = SQG_on_vop(CliffordOpStrings[self.GraphState.qubit[c].VOP], 4)

                r = int(not r_comp)

        return r

    def measure_system(self):
        """
        Measures the wavefunction at the end of the circuit
        """
        wavefunction = np.zeros(self.Nqubits, dtype='int')
        for i in range(self.Nqubits - 1, -1, -1):
            wavefunction[i] = self.measure_qubit(i)

        return wavefunction

    def bitstring_coeff(self, bitstring):
        """
        Takes a bitstring for a stabilizer state and returns it amplitude coefficient.


        F.ex:
        |psi> = 0.5 |0000> + 0.5 |0001> + 0.5 |0101> + 0.5 |1111>

        Then
        bitstring = '0000', '0001', '0101' or '1111' returns coeff = 0.5

        All other strings such as
        bitstring = '1000' or '1110' return coeff = 0
        """

        # Reverse the string since our algorithm's labeling is reversed
        bitstring = bitstring[::-1]

        # Start out with a coefficient of 1
        coeff = 1

        # Iterate over every bit
        for a in range(self.Nqubits):

            vop_a = self.GraphState.qubit[a].VOP
            ngbh_a = self.GraphState.qubit[a].edges
            P_a = measurement_ops[vop_a]

            # Check if a measurment outcome would be deterministic
            if len(ngbh_a) == 0 and P_a[1] == 'X':

                # Check if the bit does not match the measurement for this qubit, and if does not match
                # set the coefficient of the bitstring to 0
                if (P_a[0] == '+' and bitstring[a] == '1') or (P_a[0] == '-' and bitstring[a] == '0'):
                    coeff = 0
                    break

            # The measurement outcome would be random
            else:
                # We know that the measurement gives either 0 or 1 with 50% probability,
                # so the bitstring coefficient must be halved
                coeff *= 0.5

        return coeff

    def get_stabilizer_states(self):

        # Reset system
        self.reset()
        self.simulate()

        bitstrings = []

        # Get one of the stabilizer states
        stab_state = self.measure_system()

        # Make array into string
        stab_state = np.array2string(stab_state, separator='')[1:-1]

        # Reverse
        stab_state = stab_state[::-1]

        # Reset system
        self.reset()
        self.simulate()

        # Get amplitude of the stabilizer states
        amp = self.bitstring_coeff(stab_state)

        # Add this stabilizer state to the list of bitstrings found
        bitstrings.append(stab_state)

        # Get number of stabilizer states that form the complete wavefunction
        M = int(1 / amp)

        # The method runtime is probabilistic, so we set a max number of iterations
        maxiter = 10000

        assert (M < maxiter)

        for i in range(maxiter):

            # Reset system
            self.reset()
            self.simulate()

            # Get one of the stabilizer states
            stab_state = self.measure_system()

            # Make array into string
            stab_state = np.array2string(stab_state, separator='')[1:-1]

            # Reverse
            stab_state = stab_state[::-1]

            if stab_state not in bitstrings:
                # Add this stabilizer state to the list of bitstrings found
                bitstrings.append(stab_state)

            if len(bitstrings) == M:
                break

        return bitstrings, amp

    def visualise_graph(self):
        G = nx.Graph()
        
        E = []
        
        for i in range(self.Nqubits):
            for edge in self.GraphState.qubit[i].edges:
                E.append([i,edge])
        
        G.add_edges_from(E)
        nx.draw_networkx(G)
        plt.show()