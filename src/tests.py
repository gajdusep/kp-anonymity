import sys

import numpy as np

from node import Node
from group import Group
from p_anonymity import compute_pattern_similarity, p_anonymity_naive


def test_compute_pattern_similarity():
    prs = [
        ("aabca", 3),
        ("abbca", 3),
        ("abcba", 3),
        ("aacea", 5),
        ("acdeb", 5),
        ("ccbac", 3),
        ("aaaaa", 1)
    ]
    for n in range(len(prs)):
        print(compute_pattern_similarity(prs[0][0], prs[n][0],prs[0][1], prs[n][1]))
    print(compute_pattern_similarity(prs[-1][0], prs[-1][0],prs[-1][1], prs[-1][1]))
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
