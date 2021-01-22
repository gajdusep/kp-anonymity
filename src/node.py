import numpy as np
from typing import List
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
    members:    indexes of rows belonging to the node
    """
    
    def __init__(self, group: Group, level: int, PR: str, members: List[int]):
        self.group = group
        self.level = level
        self.PR = PR
        self.members = members

    def PR_len(self):
        return len(self.PR)
        
    def size(self):
        return len(self.members)

    def split(self) -> List['Node']:
        # Split node N in child nodes with level N.level+1
        child_level = self.level + 1
        child_nodes = {}
        for i in self.members:
            PR = SAX(self.group.get_row_at_index(i), child_level, self.PR_len())
            if PR in child_nodes:
                child_nodes[PR].members.append(i)
            else:
                child_nodes[PR] = Node(self.group, child_level, PR, [i])
        return list(child_nodes.values())

    def maximize_level(self, max_level: int):
        # Maximize the level of the node without splitting it
        while self.level < max_level:
            new_level = self.level + 1
            new_PR = None
            for i in self.members:
                prev_PR = new_PR
                new_PR = SAX(self.group.get_row_at_index(i), new_level, self.PR_len())
                if prev_PR is not None and prev_PR != new_PR:
                    return
            self.level = new_level
            self.PR = new_PR
    
    def table(self) -> np.ndarray:
        """
        Returns a table containing the member rows of the node
        """
        rows = []
        for i in self.members:
            rows.append(self.group.get_row_at_index(i))
        table: np.ndarray = np.vstack(rows)
        return table
        
    def ids(self) -> List[str]:
        """
        Returns a list containing ids corresponding to the members of the node
        """
        ids = []
        for i in self.members:
            ids.append(self.group.get_row_id_at_index(i))
        return ids
    
    def to_group(self) -> Group:
        """
        Returns a group containing the member rows of the node
        """
        return Group(self.table(), self.ids(), self)
    
    def copy(self):
        """
        Returns a copy of the node
        """
        return Node(self.group, self.level, self.PR, self.members)


def create_node_from_group(group: Group, PR_len: int) -> Node:
    level = 1
    if PR_len != 0:
        PR = "a" * PR_len
    else:
        PR = "a" * len(group.get_row_at_index(0))
    members = range(group.size())
    return Node(group, level, PR, list(members))


def merge_tree_nodes(nodes: List[Node]) -> Node:
    """
    Merges nodes from a tree which should be all of the same level
    The level of the merged node is [level of the merging nodes]-1
    The merged node will have the same PR as the parent of the merging nodes
    """
    group = nodes[0].group
    level = nodes[0].level - 1
    row_index = nodes[0].members[0]
    row = group.get_row_at_index(row_index)
    PR = SAX(row, level, nodes[0].PR_len())
    members = []
    for N in nodes:
        members.extend(N.members)
    return Node(group, level, PR, members)


def SAX(sequence: np.ndarray, alphabet_size: int, length: int = 0) -> str:
    """
    Compute SAX string of a sequence of numbers with specified alphabet size.
    Length of the output string may be specified; length 0 will generate a string as long as the sequence.
    """
    if alphabet_size == 1:
        if length == 0:
            return "a" * len(sequence)
        else:
            return "a" * length
    else:    
        if length == 0 or length == len(sequence):
            return ts_to_string(znorm(sequence), cuts_for_asize(alphabet_size))
        else:
            return ts_to_string(paa(znorm(sequence), length), cuts_for_asize(alphabet_size))
