import numpy as np
from typing import Dict, List
from saxpy.znorm import znorm
from saxpy.paa import paa
from saxpy.alphabet import cuts_for_asize
from saxpy.sax import ts_to_string

from group import Group
from verbose import verbose


class Node:
    """
    Attributes:
    id:         An identifier number of the node
    table:      The rows of the node
    row_ids:    The IDS of the rows
    level:      SAX level of the node
    PR:         pattern representation
    """
    
    id = 0

    def __init__(self, table: np.ndarray, row_ids: List[str], level: int, pr: str):
        self.id = Node.id
        Node.id += 1
        self.table = table
        self.row_ids = row_ids
        self.level = level
        self.pr = pr
        verbose("Creating new node {} with:\nRow IDs: {}\nSize: {}\nLevel: {}\nPR: {}".format(self.id, row_ids, self.size(), level, pr))
        assert len(self.row_ids) == self.table.shape[0]

    def pr_len(self):
        return len(self.pr)
        
    def size(self):
        return len(self.row_ids)

    def split(self) -> List['Node']:
        verbose("Splitting node {} of level {}".format(self.id, self.level))
        # Split node N in child nodes with level N.level+1
        child_level = self.level + 1
        child_nodes: Dict[str, Node] = {}
        for i, row in enumerate(self.table):
            id = self.row_ids[i]
            pr = SAX(row, child_level, self.pr_len())
            if pr in child_nodes:
                child_nodes[pr].table = np.vstack([child_nodes[pr].table, row])
                child_nodes[pr].row_ids.append(id)
                verbose("Node {} updated, new size {}, IDs {}".format(child_nodes[pr].id, child_nodes[pr].size(), child_nodes[pr].row_ids))
            else:
                child_nodes[pr] = Node(np.array([row]), [id], child_level, pr)
        children = list(child_nodes.values())
        verbose("Created {} child nodes of level {}:".format(len(children), child_level))
        for n in children:
            verbose("Node {} of size {}".format(n.id, n.size()))
        return children

    def maximize_level(self, max_level: int):
        verbose("Maximizing the level of node {} (level {})".format(self.id, self.level))
        # Maximize the level of the node without splitting it
        while self.level < max_level:
            new_level = self.level + 1
            new_pr = None
            for i in range(self.size()):
                prev_pr = new_pr
                new_pr = SAX(self.table[i], new_level, self.pr_len())
                if prev_pr is not None and prev_pr != new_pr:
                    verbose('Node maximized to level {}, with PR "{}"'.format(self.level, self.pr))
                    return
            self.level = new_level
            self.pr = new_pr
        verbose('Node maximized to level {}, with PR "{}"'.format(self.level, self.pr))
    
    def extend_table_with_node(self, n: 'Node'):
        """
        Appends to the rows (and IDs) of this node the rows (and IDs) of the supplied node
        """
        verbose("Extending table of node {} with table of node {}".format(self.id, n.id))
        verbose("Node {} row IDs: {}".format(self.id, self.row_ids))
        verbose("Node {} row IDs: {}".format(n.id, n.row_ids))
        self.table = np.vstack([self.table, n.table])
        self.row_ids.extend(n.row_ids)
        verbose("Node {} row IDs after extension: {}".format(self.id, self.row_ids))
        return
        
    def to_group(self) -> Group:
        """
        Returns a group containing the member rows of the node
        """
        verbose("Converting node {} to group".format(self.id))
        pr_list = [self.pr for _ in range(len(self.row_ids))]
        return Group(self.table, self.row_ids, pr_list)

    def copy(self):
        """
        Returns a copy of the node
        """
        verbose("Creating copy of node {}".format(self.id))
        return Node(self.table, self.row_ids, self.level, self.pr)

    def __str__(self):
        return 'Node {}: size {}, PR "{}", IDs: {}'.format(self.id, self.size(), self.pr, self.row_ids)


def create_node_from_group(group: Group, pr_len: int) -> Node:
    verbose("Creating node from group")
    level = 1
    if pr_len != 0:
        pr = "a" * pr_len
    else:
        pr = "a" * len(group.get_row_at_index(0))
    return Node(group.group_table, group.ids, level, pr)


def merge_child_nodes(nodes: List[Node]) -> Node:
    """
    Merges child nodes with the same parent
    The level of the merged node is [level of the merging nodes]-1
    The merged node will have the same PR as the parent of the merging nodes
    """
    verbose("Merging child nodes with ids {}".format([n.id for n in nodes]))
    table: np.ndarray = np.vstack([n.table for n in nodes])
    row_ids: List[str] = []
    for n in nodes:
        row_ids.extend(n.row_ids)
    level = nodes[0].level - 1
    pr = SAX(table[0], level, nodes[0].pr_len())
    return Node(table, row_ids, level, pr)


def SAX(sequence: np.ndarray, alphabet_size: int, length: int = 0) -> str:
    """
    Computes SAX string of a sequence of numbers with specified alphabet size.
    Length of the output string may be specified; length 0 will generate a string as long as the sequence.
    """
    verbose("Calculating SAX of {}, with alphabet of size {}".format(sequence, alphabet_size))
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
