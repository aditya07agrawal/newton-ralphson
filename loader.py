"""
Defines functions for parsing files
"""

from __future__ import annotations

import csv
from grid import Node, Line


def load_buses(file_path_buses):
    """Read data from file to construct buses"""
    nodes = []
    try:
        # Open the CSV file in read mode
        with open(file_path_buses, "r", encoding="utf-8", newline="") as file:
            # Create a CSV reader object
            csv_reader = csv.DictReader(file)

            # Iterate over the rows in the CSV file
            for row in csv_reader:
                # Each row is a dictionary where keys are column names
                node = Node(
                    # row["Bus No."],
                    row["Bus type"],
                    row["Voltage (pu)"],
                    0,
                    row["Pg (pu)"],
                    0,
                    row["Pd (pu)"],
                    row["Qd (pu)"],
                    row["Qmax (pu)"],
                    row["Qmin (pu)"],
                )
                nodes.append(node)
    except FileNotFoundError:
        print(
            f"File not found at '{file_path_buses}'. Please provide a valid file path."
        )
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    return nodes


def load_lines(file_path_lines):
    """Read data from file to construct lines"""
    lines = []
    try:
        # Open the CSV file in read mode
        with open(file_path_lines, "r", encoding="utf-8", newline="") as file:
            # Create a CSV reader object
            csv_reader = csv.DictReader(file)

            # Iterate over the rows in the CSV file
            for row in csv_reader:
                # Each row is a dictionary where keys are column names
                line = Line(
                    row["From Bus"],
                    row["To Bus"],
                    row["R (pu)"],
                    row["X (pu)"],
                    row["Half total line charging susceptance (pu)"],
                )
                lines.append(line)
    except FileNotFoundError:
        print(
            f"File not found at '{file_path_lines}'. Please provide a valid file path."
        )
    return lines


# Get the CSV file location from the user
# def read_csv(file_path_lines, file_path_buses):
#     """Read information from file and construct nodes and lines"""
#     lines = []
#     nodes = []
#     try:
#         # Open the CSV file in read mode
#         with open(file_path_lines, "r", encoding="utf-8", newline="") as file:
#             # Create a CSV reader object
#             csv_reader = csv.DictReader(file)

#             # Iterate over the rows in the CSV file
#             for row in csv_reader:
#                 # Each row is a dictionary where keys are column names
#                 line = Line(
#                     row["From Bus"],
#                     row["To Bus"],
#                     row["R (pu)"],
#                     row["X (pu)"],
#                     # row["Half total line charging susceptance (pu)"],
#                 )
#                 lines.append(line)
#     except FileNotFoundError:
#         print(
#             f"File not found at '{file_path_lines}'. Please provide a valid file path."
#         )
#     except Exception as e:
#         print(f"An error occurred: {e}")

#     try:
#         # Open the CSV file in read mode
#         with open(file_path_buses, "r", encoding="utf-8", newline="") as file:
#             # Create a CSV reader object
#             csv_reader = csv.DictReader(file)

#             # Iterate over the rows in the CSV file
#             for row in csv_reader:
#                 # Each row is a dictionary where keys are column names
#                 node = Node(
#                     # row["Bus No."],
#                     row["Bus type"],
#                     row["Voltage (pu)"],
#                     0,
#                     row["Pg (pu)"],
#                     0,
#                     row["Pd (pu)"],
#                     row["Qd (pu)"],
#                     # row["Qmax (pu)"],
#                     # row["Qmin (pu)"],
#                 )
#                 nodes.append(node)
#     except FileNotFoundError:
#         print(
#             f"File not found at '{file_path_buses}'. Please provide a valid file path."
#         )
#     # except Exception as e:
#     #     print(f"An error occurred: {e}")
#     return (lines, nodes)
