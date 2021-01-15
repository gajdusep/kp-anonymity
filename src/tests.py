import sys

import numpy as np

import kp_anonymity
from node import Node
from group import Group

def test_p_anonymity_naive():
    # create and print test group
    group_table = np.array([
        [0, 1, 2, 3, 4],
        [-5, -3, -2, 0, 1],
        [-1, -3, -2, -10, -8],
        [10, 8, 9, 1, 3]
    ])
    group = Group(group_table)
    for i, row in enumerate(group.group_table):
        print(str(i) + ": " + str(row))

    # p-anonymity
    leaves = kp_anonymity.p_anonimity_naive(group, 2, 5, 0)

    # print output
    for leave in leaves:
        print(leave.PR)
        for i in leave.members:
            print(i)

    return

if __name__ == "__main__":
    if sys.argv[1] == "p_naive":
        test_p_anonymity_naive()
    exit()