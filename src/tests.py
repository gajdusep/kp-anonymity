import sys

import numpy as np

from node import Node
from group import Group
from p_anonymity import compute_pattern_similarity, p_anonymity_naive


def test_compute_pattern_similarity():
    Nodes = [
        Node(None, None, 3, "aabca"),
        Node(None, None, 3, "abbca"),
        Node(None, None, 3, "abcba"),
        Node(None, None, 5, "aacea"),
        Node(None, None, 5, "acdeb"),
        Node(None, None, 1, "aaaaa")
    ]
    for n in range(len(Nodes)):
        print(compute_pattern_similarity(Nodes[0], Nodes[n]))
    print(compute_pattern_similarity(Nodes[-1], Nodes[-1]))
    return


def test_p_anonymity_naive():
    # create and print test group
    group_table = np.array([
        [0, 1, 2, 3, 4],
        [-5, -3, -2, 0, 1],
        [-1, -3, -2, -10, -8],
        [10, 8, 9, 1, 3],
        # [1, 5, 6, 2, -4],
        # [10, 1, 3, -6, 0],
        [1, 10, 5, 6, -8]
    ])

    ids = [
        "A",
        "B",
        "C",
        "D",
        # "E",
        # "F",
        "G"
        ]

    group = Group(group_table, ids)
    for i, row in enumerate(group.group_table):
        print(ids[i] + ": " + str(row))

    # p-anonymity
    leaves = p_anonymity_naive(group, 2, 5, 0)

    # print output
    for leaf in leaves:
        print()
        print(leaf.pr + " rows:")
        print(leaf.table)

    return


if __name__ == "__main__":
    if sys.argv[1] == "p_naive":
        test_p_anonymity_naive()
    elif sys.argv[1] == "pattern_similarity":
        test_compute_pattern_similarity()
    exit()
