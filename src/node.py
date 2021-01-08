from typing import List
# import numpy as np
from saxpy.znorm import znorm
from saxpy.paa import paa
from saxpy.alphabet import cuts_for_asize
from saxpy.sax import ts_to_string
from group import Group

class Node:
    """
    Attributes:
    group:      The original group which generated the tree
    level:      SAX level of the node
    PR:         pattern representation
    PR_len:     length of PR
    members:    indexes of rows belonging to the node
    size:       number of rows belonging to the node
    """
    
    def __init__(self, group: Group, level: int, PR: str, members):
        self.group = group
        self.level = level
        self.PR = PR
        self.PR_len = len(PR)
        self.members = members
        self.size = len(members)
    
    def split(self):
        # Split node N in child nodes with level N.level+1
        child_level = self.level + 1
        child_nodes = {}
        for i in self.members:
            PR = SAX(self.group.get_row_at_index(i), child_level, self.PR_len)
            if PR in child_nodes:
                child_nodes[PR].members.append(i)
            else:
                child_nodes[PR] = Node(self.group, child_level, PR, [i])
        return child_nodes.values()

    def maximize_level(self, max_level: int):
        # Maximize the level of the node without splitting it
        while self.level < max_level:
            new_level = self.level + 1
            prev_PR = ""
            for i in self.members:
                new_PR = SAX(self.group.get_row_at_index(i), new_level, self.PR_len)
                if prev_PR != new_PR:
                    return
            self.level = new_level
            self.PR = new_PR

    def to_rows(self):
        rows = []
        for i in self.members:
            row = self.group.get_row_at_index(i).copy().append(self.PR)
            rows.append(row)
        return rows
    
    def copy(self):
        """
        Returns a copy of the node
        """
        return Node(self.group, self.level, self.PR, self.members)

def create_node_from_group(group: Group, PR_len: int) -> Node:
    level = 1
    PR = "a" * PR_len
    members = range(group.size())
    return Node(group, level, PR, members)

def merge_nodes(nodes: List[Node]) -> Node:
    # Merge nodes (should be all of the same level)
    # The level of the merged node is [level of the merging nodes]-1
    # The merged node should have the same PR as its parent
    group = nodes[0].group
    level = nodes[0].level - 1
    PR = SAX(group.get_row_at_index(0), level, nodes[0].PR_len)
    members = []
    for N in nodes:
        members.extend(N.members)
    return Node(group, level, PR, members)

def SAX(sequence, alphabet_size: int, length = 0) -> str:
    """
    Compute SAX string of a sequence of numbers with specified alphabet size.
    Length of the output string may be specified; length 0 will generate a string as long as the sequence.
    """
    if length == 0:
        return ts_to_string(znorm(sequence), cuts_for_asize(alphabet_size))
    return ts_to_string(paa(znorm(sequence), length), cuts_for_asize(alphabet_size))
