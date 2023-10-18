"""
Defines classes:
- Node
- Line
- Grid
"""
# pylint: disable=C0103

from __future__ import annotations

from typing import List

import numpy as np

from utils import CountMixin


class Node(CountMixin):
    def __init__(
        self,
        kind: int,
        v: float,
        theta: float,
        PGi: float,
        QGi: float,
        PLi: float,
        QLi: float,
    ):
        super().__init__()

        self.v = v
        self.kind = kind
        self.theta = theta
        self.PGi = PGi
        self.QGi = QGi
        self.PLi = PLi
        self.QLi = QLi
        self.vLf = self.v
        self.thetaLf = self.theta

    @property
    def vm(self) -> complex:
        """Complex power at this node"""
        return self.v * np.exp(self.theta * 1j)

    @property
    def vmLf(self) -> complex:
        """Complex power(Lf) at this node"""
        return self.vLf * np.exp(self.thetaLf * 1j)


class Line(CountMixin):
    def __init__(self, fromNode: Node, toNode: Node, r: float, x: float):
        super().__init__()

        self.fromNode = fromNode
        self.toNode = toNode
        self.r = r
        self.x = x

        self.z = self.r + self.x * 1j
        self.y = 1 / self.z

    @property
    def endNodes(self):
        return self.fromNode, self.toNode

    @property
    def endNodes_id(self):
        return self.fromNode.index, self.toNode.index


class Grid:
    def __init__(self, nodes: List[Node], lines: List[Line]):
        self.nodes = sorted(nodes, key=lambda node: node.index)
        self.lines = lines

        self.V = np.array([node.v for node in self.nodes])
        self.angle = np.array([node.theta for node in self.nodes])

        self.Y = np.zeros((self.nb, self.nb), dtype=complex)
        self.G = np.zeros((self.nb, self.nb))
        self.B = np.zeros((self.nb, self.nb))

        self.nl = len(self.lines)

        self.create_matrix()

        self.Pl = np.vstack([node.PLi for node in self.nodes])
        self.Ql = np.vstack([node.QLi for node in self.nodes])
        self.Pg = np.vstack([node.PGi for node in self.nodes])
        self.Qg = np.vstack([node.QGi for node in self.nodes])

        self.Psp = self.Pg - self.Pl
        self.Qsp = self.Qg - self.Ql

    @property
    def nb(self):
        fromBus = [line.fromNode.index for line in self.lines]
        toBus = [line.toNode.index for line in self.lines]
        return max(max(fromBus), max(toBus)) + 1

    def get_node_by_id(self, index: int):
        for node in self.nodes:
            if node.index == index:
                return node
        raise ValueError("No node with number %d." % index)

    def get_line_by_id(self, index: int):
        for line in self.lines:
            if line.index == index:
                return line
        raise ValueError("No line with number %d." % index)

    def get_lines_by_node(self, node_id: int):
        return [line for line in self.lines if node_id in line.endNodes_id]

    @property
    def pq_node_ids(self):
        return [node.index for node in self.nodes if node.kind == 3]

    @property
    def pv_nodes(self):
        return [node for node in self.nodes if node.kind == 2]

    def create_matrix(self):
        # off diagonal elements
        for line in self.lines:
            fromNode, toNode = line.endNodes_id
            self.Y[toNode, fromNode] = self.Y[fromNode, toNode] = -line.y

        # diagonal elements
        diag = range(self.nb)
        self.Y[diag, diag] = -self.Y.sum(axis=1)

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
        return np.subtract.outer(self.angle, self.angle)

    @property
    def eff_G(self):
        return self.G * np.cos(self.d_angle) + self.B * np.sin(self.d_angle)

    @property
    def eff_B(self):
        return self.G * np.sin(self.d_angle) - self.B * np.cos(self.d_angle)

    @property
    def P_calc(self):
        return np.reshape(self.V * np.matmul(self.eff_G, self.V), (-1, 1))

    @property
    def Q_calc(self):
        return np.reshape(self.V * np.matmul(self.eff_B, self.V), (-1, 1))

    @property
    def f_calc(self):
        return np.vstack((self.P_calc[1:], self.Q_calc[self.pq_node_ids]))

    @property
    def deltaP(self):
        return (self.Psp - self.P_calc)[1:]

    @property
    def deltaQ(self):
        return (self.Qsp - self.Q_calc)[self.pq_node_ids]

    @property
    def delta(self):
        """Delta P and Q"""
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

    def nr(self, tol=1, maxIter=2, BMva=1):
        self.iter = 0

        while self.iter < maxIter:
            self.iter += 1

            # Maximum reactive power generation during second iteration at bus 3
            if self.iter == 2 and self.Q_calc[3] > 0.5:
                jacobian(
                    self.V,
                    self.G,
                    self.B,
                    self.eff_G,
                    self.eff_B,
                    self.pq_node_ids + [3],
                    self.Q_calc,
                    self.P_calc,
                )
                exit()

            # J X = M -> X = J^-1 M
            X = np.linalg.solve(self.J, self.delta)
            dTh = X[0 : self.nb - 1]
            dV = X[self.nb - 1 :]

            # update Angles and Voltages
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

            # update Angles and Voltages
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

            # update Angles and Voltages
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

        # the iteration is over; calculate the power flow
        self.calculateLf()

    def calculateLf(self, BMva=1):
        Vm = np.vstack([node.vm for node in self.nodes]).reshape((self.nb, -1))
        self.I = np.matmul(self.Y, Vm)
        Iij = np.zeros((self.nb, self.nb), dtype=complex)
        Sij = np.zeros((self.nb, self.nb), dtype=complex)

        self.Im = abs(self.I)
        self.Ia = np.angle(self.I)

        for line in self.lines:
            fromNode, toNode = line.endNodes
            i, j = fromNode.index, toNode.index

            Iij[i, j] = -(fromNode.vm - toNode.vm) * self.Y[i, j]
            Iij[j, i] = -Iij[i, j]

        for node in self.nodes:
            m = node.index  # node index
            lines = self.get_lines_by_node(node.index)
            for line in lines:
                if line.fromNode.index == m:
                    p = line.toNode.index  # index to
                    if m != p:
                        Iij[m, p] = -(line.fromNode.vm - line.toNode.vm) * self.Y[m, p]
                        Iij[p, m] = -(line.toNode.vm - line.fromNode.vm) * self.Y[p, m]
                else:
                    p = line.fromNode.index  # index from
                    if m != p:
                        Iij[m, p] = -(line.toNode.vm - line.fromNode.vm) * self.Y[p, m]
                        Iij[p, m] = -(line.fromNode.vm - line.toNode.vm) * self.Y[m, p]

        self.Iij = Iij
        self.Iijr = np.real(Iij)
        self.Iiji = np.imag(Iij)

        # line powerflows
        for m in range(self.nb):
            for n in range(self.nb):
                if n != m:
                    Sij[m, n] = self.nodes[m].vmLf * np.conj(self.Iij[m, n]) * BMva

        self.Sij = Sij
        self.Pij = np.real(Sij)
        self.Qij = np.imag(Sij)

        # line losses
        Lij = np.zeros(self.nl, dtype=complex)
        for line in self.lines:
            m = line.index - 1
            p = line.fromNode.index
            q = line.toNode.index
            Lij[m] = Sij[p, q] + Sij[q, p]

        self.Lij = Lij
        self.Lpij = np.real(Lij)
        self.Lqij = np.imag(Lij)

        # Bus power injection
        Si = np.zeros(self.nb, dtype=complex)
        for i in range(self.nb):
            for k in range(self.nb):
                Si[i] += np.conj(self.nodes[i].vm) * self.nodes[k].vm * self.Y[i, k]

        self.Si = Si
        self.Pi = np.real(Si)
        self.Qi = -np.imag(Si)
        self.Pg = self.Pi.reshape([-1, 1]) + self.Pl.reshape([-1, 1])
        self.Qg = self.Qi.reshape([-1, 1]) + self.Ql.reshape([-1, 1])

    def printResults(self):
        print("\033[95mNewton Raphson Results:\033[0m")
        print()
        print(
            "| Bus |    V     |  Angle   |      Injection      |     Generation      |       Load         |"
        )
        print(
            "| No  |    pu    |  Degree  |    MW    |   MVar   |    MW    |   Mvar   |   MW    |   MVar   |"
        )
        for i in range(self.nb):
            print(
                "| %3g | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f |%8.4f | %8.4f |"
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
            "----------------------------------------------------------------------------------------------"
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
        for i in range(self.nl):
            p = self.lines[i].fromNode.index
            q = self.lines[i].toNode.index
            print(
                "| %4g | %4g | %8.2f | %8.2f | %4g | %4g | %8.2f | %8.2f |"
                % (
                    p,
                    q,
                    self.Pij[p, q],
                    self.Qij[p, q],
                    q,
                    p,
                    self.Pij[q, p],
                    self.Qij[q, p],
                )
            )
        print(
            "-------------------------------------------------------------------------"
        )
        print()

    def print_iteration(self, max_it=2):
        # if self.iter > max_it:
        #     return
        print(f"\033[95mCurrent iteration {self.iter}:\033[0m")
        # print("|   V_0   |   V_1   |   V_2   |   V_3   |   V_4   |   V_5   |")
        # print("| %7.3f | %7.3f | %7.3f | %7.3f | %7.3f | %7.3f |" % tuple(self.V))
        # print("|  dV_0   |  dV_1   |  dV_2   |  dV_3   |  dV_4   |  dV_5   |")
        # print("| %7.3f | %7.3f | %7.3f | %7.3f | %7.3f | %7.3f |" % tuple(self.dV))
        # print("| theta_0 | theta_1 | theta_2 | theta_3 | theta_4 | theta_5 |")
        # print("| %7.3f | %7.3f | %7.3f | %7.3f | %7.3f | %7.3f |" % tuple(self.angle))
        # print("|  dth_0  |  dth_1  |  dth_2  |  dth_3  |  dth_4  |  dth_5  |")
        # print("| %7.3f | %7.3f | %7.3f | %7.3f | %7.3f | %7.3f |" % tuple(self.dangle))
        # print()
        with np.printoptions(linewidth=200):
            print("Voltage: ", end="")
            print(self.V)
            print()
            print("Angle: ", end="")
            print(self.angle)
            print()
            print("Calculated Power: ", end="")
            print(self.f_calc.flatten())
            print()
            print("Power error: ", end="")
            print(self.delta.flatten())
            print()
            print("Jacobian: ")
            print(self.J)
            print()
            print("Inverse Jacobian J11: ")
            print(np.linalg.inv(self.J11))
            print()
            print("Inverse Jacobian J22: ")
            print(np.linalg.inv(self.J22))
            print()
            print("Voltage change: ", end="")
            print(self.dV)
            print()
            print("Angle change: ", end="")
            print(self.dangle)
            print()
        print()


def jacobian(V, G, B, eff_G, eff_B, pq_node_ids, Q_calc, P_calc):
    def J11():
        # off diagonal elements
        J11 = np.outer(V, V) * eff_B

        # diagonal elements
        i, j = np.diag_indices_from(J11)
        J11[i, j] = -Q_calc.flatten() - np.square(V) * B.diagonal()

        return J11[1:, 1:]

    def J12():
        # off diagonal elements
        J12 = V.reshape(-1, 1) * eff_G

        # diagonal elements
        i, j = np.diag_indices_from(J12)
        J12[i, j] = J12.sum(axis=0) + V[i] * G[i, i]

        return J12[1:, pq_node_ids]

    def J21():
        # off diagonal elements
        J21 = -1 * np.outer(V, V) * eff_G

        # diagonal elements
        i, j = np.diag_indices_from(J21)
        J21[i, j] = P_calc.flatten() - np.square(V) * G.diagonal()

        return J21[pq_node_ids, 1:]

    def J22():
        # off diagonal elements
        J22 = V.reshape(-1, 1) * eff_B

        # diagonal elements
        i, j = np.diag_indices_from(J22)
        J22[i, j] = J22.sum(axis=0) - V[i] * B[i, i]

        return J22[np.ix_(pq_node_ids, pq_node_ids)]

    print(np.vstack((np.hstack((J11(), J12())), np.hstack((J21(), J22())))))
