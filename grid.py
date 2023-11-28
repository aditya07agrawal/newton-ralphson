"""
Defines classes:
- Node
- Line
- Grid
"""
# pylint: disable=invalid-name, missing-function-docstring, consider-using-f-string, attribute-defined-outside-init

from __future__ import annotations

from typing import List

from attrs import define, field

import numpy as np

from utils import CountMixin


@define
class Node(CountMixin):
    """Class to store information on a Node"""

    kind: int
    v: float
    theta: float
    PGi: float
    QGi: float
    PLi: float
    QLi: float
    Qmax: float = 500
    Qmin: float = -500

    v_org: float = field(init=False)

    def __attrs_post_init__(self):
        self._set_index()

        self.v_org = self.v

    @property
    def vm(self) -> complex:
        """Complex power at this node"""
        return self.v * np.exp(self.theta * 1j)


@define
class Line(CountMixin):
    """Class to store information on Line"""

    from_node: Node
    to_node: Node
    r: float
    x: float
    b_half: float

    z: complex = field(init=False)
    y: complex = field(init=False)
    b: complex = field(init=False)

    end_nodes: tuple[Node, Node] = field(init=False)
    end_nodes_id: tuple[int, int] = field(init=False)

    def __attrs_post_init__(self):
        self._set_index()

        self.z = self.r + self.x * 1j
        self.y = 1 / self.z
        self.b = self.b_half * 1j

        self.end_nodes = self.from_node, self.to_node
        self.end_nodes_id = self.from_node.index, self.to_node.index

    @property
    def voltage_drop(self) -> complex:
        """Voltage drop across the line"""
        return self.from_node.vm - self.to_node.vm

    @property
    def incoming_current(self) -> complex:
        """Current pulled by the line"""
        return self.voltage_drop * self.y + self.b * self.from_node.vm

    @property
    def outgoing_current(self) -> complex:
        """Current pushed by the line"""
        return self.voltage_drop * self.y - self.b * self.to_node.vm

    @property
    def incoming_power(self) -> complex:
        """Incoming power for this line"""
        return self.from_node.vm * self.incoming_current

    @property
    def outgoing_power(self) -> complex:
        """Outgoing power for this line"""
        return self.to_node.vm * self.outgoing_current

    @property
    def power_loss(self) -> complex:
        """Power loss in this line"""
        return self.incoming_power - self.outgoing_power


class Grid:
    """Class to store information on Grid"""

    def __init__(self, nodes: List[Node], lines: List[Line]):
        self.nodes = sorted(nodes, key=lambda node: node.index)
        self.lines = lines

        self.nl = len(self.lines)
        self.nb = len(self.nodes)

        self.V = np.array([node.v for node in self.nodes])
        self.angle = np.array([node.theta for node in self.nodes])

        self.Y = np.zeros((self.nb, self.nb), dtype=complex)
        self.G = np.zeros((self.nb, self.nb))
        self.B = np.zeros((self.nb, self.nb))

        self.create_matrix()

        self.Pl = np.vstack([node.PLi for node in self.nodes])
        self.Ql = np.vstack([node.QLi for node in self.nodes])
        self.Pg = np.vstack([node.PGi for node in self.nodes])
        self.Qg = np.vstack([node.QGi for node in self.nodes])

        self.Psp = self.Pg - self.Pl
        self.Qsp = self.Qg - self.Ql

        self.iter = 0
        self.dV = np.zeros(self.nb)
        self.dangle = np.zeros(self.nb)

    def get_node_by_id(self, index: int):
        for node in self.nodes:
            if node.index == index:
                return node
        raise ValueError(f"No node with number {index}.")

    def get_line_by_id(self, index: int):
        for line in self.lines:
            if line.index == index:
                return line
        raise ValueError(f"No line with number {index}.")

    def get_lines_by_node(self, node_id: int):
        return [line for line in self.lines if node_id in line.end_nodes_id]

    @property
    def pq_node_ids(self):
        return [node.index for node in self.nodes if node.kind == 3]

    @property
    def pv_nodes(self):
        return [node for node in self.nodes if node.kind == 2]

    def create_matrix(self):
        """Construct the Y (and hence G and B) matrix for the grid"""
        for line in self.lines:
            # Off-diagonal admittances
            from_node, to_node = line.end_nodes_id
            self.Y[to_node, from_node] = self.Y[from_node, to_node] = -line.y

            # Half-line susceptances
            self.Y[to_node, to_node] += line.b
            self.Y[from_node, from_node] += line.b

        # Diagonal admittances
        diag = range(self.nb)
        self.Y[diag, diag] -= self.Y.sum(axis=1)

        self.G = self.Y.real
        self.B = self.Y.imag

    @property
    def Vm(self):
        return np.array([node.vm for node in self.nodes])

    def update_V(self):
        for i in self.pq_node_ids:
            self.nodes[i].v += self.dV[i]
        self.V = np.array([node.v for node in self.nodes])

    def update_angle(self):
        for i in range(1, self.nb):
            self.nodes[i].theta += self.dangle[i]
        self.angle = np.array([node.theta for node in self.nodes])

    @property
    def d_angle(self):
        """Antisymmetric matrix representing the phase difference between buses"""
        return np.subtract.outer(self.angle, self.angle)

    @property
    def eff_G(self):
        """Effective conductance between buses"""
        return self.G * np.cos(self.d_angle) + self.B * np.sin(self.d_angle)

    @property
    def eff_B(self):
        """Effective susceptance between buses"""
        return self.G * np.sin(self.d_angle) - self.B * np.cos(self.d_angle)

    @property
    def P_calc(self):
        """Calculated active power"""
        return np.reshape(self.V * np.matmul(self.eff_G, self.V), (-1, 1))

    @property
    def Q_calc(self):
        """Calculated reactive power"""
        Q = np.reshape(self.V * np.matmul(self.eff_B, self.V), (-1, 1))

        for ind in range(1, self.nb):
            node = self.nodes[ind]
            if node.Qmax != 500:
                if not node.Qmin <= Q[ind] + self.Ql[ind] <= node.Qmax:
                    Q[ind] = sorted((Q[ind], node.Qmin, node.Qmax))[1]
                    node.kind = 3
                else:
                    self.V[ind] = node.v = node.v_org
                    node.kind = 2
                    Q = self.Q_calc
        return Q

    @property
    def calculated_power(self):
        """Calculated power vector"""
        return np.vstack((self.P_calc[1:], self.Q_calc[self.pq_node_ids]))

    @property
    def deltaP(self):
        """Mismatch vector for active power"""
        return (self.Psp - self.P_calc)[1:]

    @property
    def deltaQ(self):
        """Mismatch vector for reactive power"""
        return (self.Qsp - self.Q_calc)[self.pq_node_ids]

    @property
    def delta(self):
        """Mismatch vector"""
        return np.vstack((self.deltaP, self.deltaQ))

    @property
    def J11(self):
        # off diagonal elements
        J11 = np.outer(self.V, self.V) * self.eff_B

        # diagonal elements
        i, j = np.diag_indices_from(J11)
        J11[i, j] = -self.Q_calc.flatten() - np.square(self.V) * self.B.diagonal()

        return J11[1:, 1:]

    @property
    def J12(self):
        # off diagonal elements
        J12 = self.V.reshape(-1, 1) * self.eff_G

        # diagonal elements
        i, j = np.diag_indices_from(J12)
        J12[i, j] = J12.sum(axis=0) + self.V[i] * self.G[i, i]

        return J12[1:, self.pq_node_ids]

    @property
    def J21(self):
        # off diagonal elements
        J21 = -1 * np.outer(self.V, self.V) * self.eff_G

        # diagonal elements
        i, j = np.diag_indices_from(J21)
        J21[i, j] = self.P_calc.flatten() - np.square(self.V) * self.G.diagonal()

        return J21[self.pq_node_ids, 1:]

    @property
    def J22(self):
        # off diagonal elements
        J22 = self.V.reshape(-1, 1) * self.eff_B

        # diagonal elements
        i, j = np.diag_indices_from(J22)
        J22[i, j] = J22.sum(axis=0) - self.V[i] * self.B[i, i]

        return J22[np.ix_(self.pq_node_ids, self.pq_node_ids)]

    @property
    def J(self):
        return np.vstack(
            (np.hstack((self.J11, self.J12)), np.hstack((self.J21, self.J22)))
        )

    def nr(self, maxIter=2):
        self.iter = 0

        while self.iter < maxIter:
            self.iter += 1

            # J X = M -> X = J^-1 M
            X = np.linalg.solve(self.J, self.delta)
            dTh = X[0 : self.nb - 1]
            dV = X[self.nb - 1 :]

            self.complete_iteration(dV, dTh)

        # the iteration is over; calculate the power flow
        self.calculateLf()

    def decoupled(self, maxIter=2):
        self.iter = 0

        while self.iter < maxIter:
            self.iter += 1

            dTh = np.linalg.solve(self.J11, self.deltaP)
            dV = np.linalg.solve(self.J22, self.deltaQ)
            print("Check....")
            print(dV)

            self.complete_iteration(dV, dTh)

        # the iteration is over; calculate the power flow
        self.calculateLf()

    def fast_decoupled(self, maxIter=2):
        self.iter = 0

        invB1 = np.linalg.inv(self.B[1:, 1:])
        invB2 = np.linalg.inv(self.B[np.ix_(self.pq_node_ids, self.pq_node_ids)])

        while self.iter < maxIter:
            self.iter += 1

            dP_V = self.deltaP / self.V[1:].reshape(-1, 1)
            dQ_V = self.deltaQ / self.V[self.pq_node_ids].reshape(-1, 1)

            print(dP_V)
            print(dQ_V)

            dTh = -np.matmul(invB1, dP_V)
            dV = -np.matmul(invB2, dQ_V)

            self.complete_iteration(dV, dTh)

        # the iteration is over; calculate the power flow
        self.calculateLf()

    def calculateLf(self):
        """Calculate the load flow through the grid"""
        self.Si = np.array(np.conj(self.Vm) * np.matmul(self.Y, self.Vm))

        self.Pi = np.real(self.Si)
        self.Qi = -np.imag(self.Si)
        self.Pg = self.Pi.reshape([-1, 1]) + self.Pl.reshape([-1, 1])
        self.Qg = self.Qi.reshape([-1, 1]) + self.Ql.reshape([-1, 1])

    def print_results(self):
        print("\033[95mNewton-Raphson Results:\033[0m")
        print()
        print(
            "| Bus |   V    |  Angle  |     Injection     |    Generation     |      Load       |"
        )
        print(
            "| No  |   pu   |  Degree |   MW    |  MVar   |   MW    |  Mvar   |   MW   |  MVar  |"
        )
        for i in range(self.nb):
            print(
                "| %3g | %6.4f | %7.4f | %7.4f | %7.4f | %7.4f | %7.4f | %6.4f | %6.4f |"
                % (
                    i,
                    self.nodes[i].v,
                    self.nodes[i].theta,
                    self.Pi[i],
                    self.Qi[i],
                    self.Pg[i],
                    self.Qg[i],
                    self.Pl[i],
                    self.Ql[i],
                )
            )

        print(
            "------------------------------------------------------------------------------------"
        )
        print()
        print("Line flows and losses:")
        print()
        print(
            "| From |  To  |     P    |    Q     | From |  To  |    P     |    Q     |"
        )
        print(
            "| Bus  | Bus  |    MW    |   MVar   | Bus  | Bus  |    MW    |   MVar   |"
        )
        for line in self.lines:
            i, j = line.end_nodes_id
            print(
                "| %4g | %4g | %8.2f | %8.2f | %4g | %4g | %8.2f | %8.2f |"
                % (
                    i,
                    j,
                    np.real(line.outgoing_power),
                    np.imag(line.outgoing_power),
                    j,
                    i,
                    -np.real(line.incoming_power),
                    -np.imag(line.incoming_power),
                )
            )
        print(
            "-------------------------------------------------------------------------"
        )
        print()

    def print_iteration(self):
        print(f"\033[95mCurrent iteration {self.iter}:\033[0m")
        with np.printoptions(linewidth=200):
            print("Voltage: ", self.V, "\n")
            print("Angle: ", self.angle, "\n")
            print("Calculated Power: ", self.calculated_power.flatten(), "\n")
            print("Power error: ", self.delta.flatten(), "\n")
            print("Jacobian: ", "\n", self.J, "\n")
            print("Inverse Jacobian J11: ", "\n", np.linalg.inv(self.J11), "\n")
            print("Inverse Jacobian J22: ", "\n", np.linalg.inv(self.J22), "\n")
            print("Voltage change: ", self.dV, "\n")
            print("Angle change: ", self.dangle, "\n")
        print()

    def complete_iteration(self, dV: np.ndarray, dTh: np.ndarray):
        """Update voltages and angles and print the iteration information"""
        it = iter(dV.flatten())
        self.dV = np.zeros(self.nb)
        for i in self.pq_node_ids:
            self.dV[i] = next(it)

        it = iter(dTh.flatten())
        self.dangle = np.zeros(self.nb)
        for i in range(1, self.nb):
            self.dangle[i] = next(it)

        self.print_iteration()
        self.update_V()
        self.update_angle()
