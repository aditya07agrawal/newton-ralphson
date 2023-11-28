"""
Defines functions for parsing files
"""

from __future__ import annotations

import csv

from grid import Grid, Bus, Line


def load_buses(buses_file) -> list[Bus]:
    """Read data from file to construct buses"""
    nodes = []
    try:
        # Open the CSV file in read mode
        with open(buses_file, "r", encoding="utf-8", newline="") as file:
            # Create a CSV reader object
            csv_reader = csv.DictReader(file)

            # Iterate over the rows in the CSV file
            # Each row is a dictionary where keys are column names
            nodes = [
                Bus(
                    int(row["Bus type"]),
                    float(row["Voltage (pu)"]),
                    0,
                    float(row["Pg (pu)"]),
                    0,
                    float(row["Pd (pu)"]),
                    float(row["Qd (pu)"]),
                    float(row["Qmax (pu)"]),
                    float(row["Qmin (pu)"]),
                )
                for row in csv_reader
            ]

    except FileNotFoundError:
        print(f"File not found at '{buses_file}'. Please provide a valid file path.")
    return nodes


def load_lines(lines_file, buses) -> list[Line]:
    """Read data from file to construct lines"""
    lines = []
    try:
        # Open the CSV file in read mode
        with open(lines_file, "r", encoding="utf-8", newline="") as file:
            # Create a CSV reader object
            csv_reader = csv.DictReader(file)

            # Iterate over the rows in the CSV file
            # Each row is a dictionary where keys are column names
            lines = [
                Line(
                    buses[int(row["From Bus"]) - 1],
                    buses[int(row["To Bus"]) - 1],
                    float(row["R (pu)"]),
                    float(row["X (pu)"]),
                    float(row["Half total line charging susceptance (pu)"]),
                )
                for row in csv_reader
            ]
    except FileNotFoundError:
        print(f"File not found at '{lines_file}'. Please provide a valid file path.")
    return lines


def load_grid(bus_file_path, line_file_path):
    """Load the grid"""
    buses = load_buses(bus_file_path)
    lines = load_lines(line_file_path, buses)
    return Grid(buses, lines)
