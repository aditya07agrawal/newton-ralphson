"""
Module to run newton raphson on a grid
"""

from __future__ import annotations

from grid import Node, Line, Grid


def solve_grid_2():
    """Solves grid 2"""
    n1 = Node(1, 1.05, 0, 0.0, 0, 0, 0)
    n2 = Node(2, 1.05, 0, 0.5, 0, 0, 0)
    n3 = Node(2, 1.07, 0, 0.6, 0, 0, 0)
    n4 = Node(3, 1.0, 0, 0.0, 0, 0.7, 0.7)
    n5 = Node(3, 1.0, 0, 0.0, 0, 0.7, 0.7)
    n6 = Node(3, 1.0, 0, 0.0, 0, 0.7, 0.7)

    l1 = Line(n1, n2, 0.1, 0.2, 0)
    l2 = Line(n1, n4, 0.05, 0.2, 0)
    l3 = Line(n1, n5, 0.08, 0.3, 0)
    l4 = Line(n2, n3, 0.05, 0.25, 0)
    l5 = Line(n2, n4, 0.05, 0.1, 0)
    l6 = Line(n2, n5, 0.1, 0.3, 0)
    l7 = Line(n2, n6, 0.07, 0.2, 0)
    l8 = Line(n3, n5, 0.12, 0.26, 0)
    l9 = Line(n3, n6, 0.02, 0.1, 0)
    l10 = Line(n4, n5, 0.2, 0.4, 0)
    l11 = Line(n5, n6, 0.1, 0.3, 0)

    nodes = [n1, n2, n3, n4, n5, n6]
    lines = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11]

    grid = Grid(nodes=nodes, lines=lines)
    grid.nr()
    grid.print_results()


def solve_grid_1():
    """Solves grid 1"""
    n1 = Node(1, 1, 0, 0.0, 0.0, 0.0, 0.0)
    n2 = Node(2, 1, 0, 1.0, 0.0, 0.0, 0.0)
    n3 = Node(3, 1, 0, 0.0, 0.0, 4.0, 2.0)
    n4 = Node(2, 1, 0, 2.0, 0.0, 1.0, 0.5)

    l1 = Line(n1, n2, 0, 0.1, 0)
    l2 = Line(n1, n3, 0, 0.1, 0)
    l3 = Line(n1, n4, 0, 0.1, 0)
    l4 = Line(n2, n3, 0, 0.1, 0)
    l5 = Line(n3, n4, 0, 0.1, 0)

    nodes = [n1, n2, n3, n4]
    lines = [l1, l2, l3, l4, l5]

    grid = Grid(nodes, lines)
    grid.nr()
    grid.print_results()


if __name__ == "__main__":
    solve_grid_1()
