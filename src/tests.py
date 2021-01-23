import sys

import numpy as np

from node import Node
from group import Group
from p_anonymity import compute_pattern_similarity, p_anonymity_naive


def test_compute_pattern_similarity():
    Nodes = [
        Node(None, 3, "aabca", None),
        Node(None, 3, "abbca", None),
        Node(None, 3, "abcba", None),
        Node(None, 5, "aacea", None),
        Node(None, 5, "acdeb", None),
        Node(None, 1, "aaaaa", None)
    ]
    for n in range(len(Nodes)):
        print(compute_pattern_similarity(Nodes[0], Nodes[n]))
    print(compute_pattern_similarity(Nodes[-1], Nodes[-1]))
    return


def test_p_anonymity_naive():
    # create and print test group
    """
    group_table = np.array([
        [0, 1, 2, 3, 4],
        [-5, -3, -2, 0, 1],
        [-1, -3, -2, -10, -8],
        [10, 8, 9, 1, 3],
        [1, 5, 6, 2, -4]
    ])
    """
    group_table = np.array([
        [0, 1, 2, 3, 4],
        [-5, -3, -2, 0, 1],
        [-1, -3, -2, -10, -8],
        [10, 8, 9, 1, 3],
        [1, 5, 6, 2, -4],
        [10, 1, 3, -6, 0],
        [1, 10, 5, 6, -8]
    ])

    ids = ["A", "B", "C", "D", "E", "F", "G"]

    group = Group(group_table, ids)
    for i, row in enumerate(group.group_table):
        print(ids[i] + ": " + str(row))

    # p-anonymity
    leaves = p_anonymity_naive(group, 2, 5, 0)

    # print output
    for leave in leaves:
        print()
        print(leave.PR + " members:")
        for i in leave.members:
            print(group.get_row_id_at_index(i))

    return


if __name__ == "__main__":
    if sys.argv[1] == "p_naive":
        test_p_anonymity_naive()
    elif sys.argv[1] == "pattern_similarity":
        test_compute_pattern_similarity()
    exit()
