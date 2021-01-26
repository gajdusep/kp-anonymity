from typing import Dict, List

from group import Group
from node import Node, create_node_from_group, merge_child_nodes, SAX
from verbose import verbose, debug


def compute_pattern_similarity(n1: Node, n2: Node) -> float:
    """
    Calculate the similarity between two pattern representations as a value between 0 and 1.
    The difference is here defined as the average of the normalized distances (normalized over the number of levels)
    between the two characters in the same position of the two PRs.
    The similarity is 1 - difference.
    """
    debug('Computing similarity of patterns "{}" (node {}, level {}) and "{}" (node {}, level {})'.format(n1.pr, n1.id, n1.level, n2.pr, n2.level, n2.id))
    if n1.pr_len() != n2.pr_len():
        raise ValueError("Pattern similarity cannot be computed on PR of different lengths")
    
    diff = 0
    if n1.level == 1 or n2.level == 1:
        if n1.level == n2.level:
            return 1
        else:
            return 0
    elif n1.level == n2.level:
        for i in range(len(n1.pr)):
            diff += abs(ord(n1.pr[i]) - ord(n2.pr[i])) / (n1.level - 1)
    else:
        # If the two PRs are of different levels, their values are normalized separately
        for i in range(len(n1.pr)):
            diff += abs((ord(n1.pr[i]) - 97) / (n1.level - 1) - (ord(n2.pr[i]) - 97) / (n2.level - 1))
    
    diff = diff / len(n1.pr)
    sim = 1 - diff
    debug("Computed similarity: {}".format(sim))
    return sim


def create_p_anonymity_tree(group: Group, p: int, max_level: int, pr_len: int) -> Dict[str, List[Node]]:
    """
    The algorithm is implemented in a non-recursive way because keeping the entire tree structure is not needed
    as we only use the leaf nodes.
    The nodes_to_process is the list of nodes which have not already been processed.
    As nodes are labeled as good or bad leaves, they are added to the good_leaves or bad_leaves lists, respectively.
    The new_nodes_to_process flag indicates whether during the current cycle are new nodes are created by splits.
    If there are new nodes, the nodes_to_process list is updated and those new nodes are processed.
    A dictionary containing two keys is returned:
    - the first key is "good leaves" and contains the list of good leaf nodes, and
    - the second key is "bad leaves" and contains the list of bad leaf nodes.
    """
    verbose('Starting "create tree" step (p: {}, max PR level: {}, PR length: {}) on the following rows:'.format(p, max_level, pr_len))
    for i, row in enumerate(group.group_table):
        verbose("{}: {}".format(group.ids[i], row))
    # Initialize nodes list with the starting node, corresponding to the group
    nodes_to_process = [create_node_from_group(group, pr_len)]
    new_nodes_to_process: List[Node] = []
    good_leaves: List[Node] = []
    bad_leaves: List[Node] = []

    # Node splitting
    while nodes_to_process:
        debug("New nodes to process found")
        new_nodes_to_process = []
        for n in nodes_to_process:
            n_id = n.id
            n_size = n.size()
            if n_size < p:
                debug("Node {} labeled bad leaf (size: {})".format(n_id, n_size))
                bad_leaves.append(n)
            elif n.level == max_level:
                debug("Node {} labeled good leaf for reaching maximum PR level (size: {})".format(n_id, n_size))
                good_leaves.append(n)
            elif n_size < 2 * p:
                debug("Node {} labeled good leaf for size (size: {})".format(n_id, n_size))
                good_leaves.append(n)
                n.maximize_level(max_level)
            else:
                debug("Node {} big enough to be split (size: {})".format(n_id, n_size))
                child_nodes = n.split()
                # Split not possible
                if len(child_nodes) < 2 or max(child.size() for child in child_nodes) < p:
                    debug("Node {} labeled good leaf: split produced only one node or no child node had size >= p".format(n_id))
                    good_leaves.append(n)
                # Split possible
                else:
                    debug("Split was successful")
                    TG_nodes: List[Node] = []
                    TB_nodes: List[Node] = []
                    total_TB_size = 0
                    debug("Checking tentative node sizes:")
                    for child in child_nodes:
                        if child.size() < p:
                            TB_nodes.append(child)
                            total_TB_size += child.size()
                            debug("Node {} is a tentative bad node (size: {}), total TB nodes size: {}".format(child.id, child.size(), total_TB_size))
                        else:
                            TG_nodes.append(child)
                            debug("Node {} is a tentative good node (size: {})".format(child.id, child.size()))

                    new_nodes_to_process.extend(TG_nodes)

                    if total_TB_size >= p:
                        debug("Tentative bad nodes are big enough to be merged")
                        child_merge = merge_child_nodes(TB_nodes)
                        new_nodes_to_process.append(child_merge)
                    else:
                        debug("Tentative bad nodes are not big enough to be merged")
                        new_nodes_to_process.extend(TB_nodes)
        nodes_to_process = new_nodes_to_process

    verbose('The "create tree" step produced the following good leaves:\n{}\nand the following bad leaves:\n{}'.format([n.id for n in good_leaves],[n.id for n in bad_leaves]))
    return good_leaves, bad_leaves


def postprocess(good_leaves: List[Node], bad_leaves: List[Node]) -> List[Node]:
    """
    The postprocessing phase takes care of the bad leaves integrating them into the good leaves.
    The modified good leaves are then returned.
    """
    if not good_leaves:
        print("WARNING: no good leaves found, not enough rows in the anonymizing group?")
    elif not bad_leaves:
        verbose("No bad leaves found: no postprocessing needed")
    elif len(good_leaves) == 1:
        verbose("Starting postprocessing phase")
        good = good_leaves[0]
        verbose("Only 1 good leaf (node {}): adding all bad leaf rows to the only good leaf".format(good.id))
        for bad in bad_leaves:
            debug("Processing bad leaf {}".format(bad.id))
            debug("Similarity to good node: {})".format(compute_pattern_similarity(bad, good)))
            good.extend_table_with_node(bad)
    else:
        verbose("Starting postprocessing phase")
        bad_leaves.sort(key=lambda node: node.size())
        for bad in bad_leaves:
            debug("Processing bad leaf {}".format(bad.id))
            max_similarity = None
            most_similar_good = None
            for good in good_leaves:
                similarity = compute_pattern_similarity(bad, good)
                if max_similarity is None or similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_good = good
                    debug("Updating most similar node to node {} (better similarity: {})".format(most_similar_good.id, max_similarity))
                elif similarity == max_similarity and good.size() < most_similar_good.size():
                    most_similar_good = good
                    debug("Updating most similar node to node {} (smaller size: {})".format(most_similar_good.id, most_similar_good.size()))
            most_similar_good.extend_table_with_node(bad)

    verbose("The postprocessing phase produced the following good leaves:")
    for n in good_leaves:
        verbose('Node {}: size {}, level {}, PR "{}", ids {}'.format(n.id, n.size(), n.level, n.pr, n.row_ids))
    return good_leaves


def recycle_bad_leaves(good_leaves: List[Node], bad_leaves: List[Node], p: int) -> List[Node]:
    """
    "Recycle bad leaves" step of the KAPRA algorithm, which merges bad leaves creating good ones.
    This function returns a list of all good leaf nodes.
    """
    verbose('Starting "recycle bad leaves" phase')
    # If there are no bad leaves, then this step is not needed
    if not bad_leaves:
        verbose('No bad leaves found: no "recycle bad leaves" step needed')
        return good_leaves

    # Preparation: bad leaves are sorted in different lists depending on their level
    bad_leaves_by_level: Dict[int, List[Node]] = {}
    debug("Sorting bad leaves by level:")
    for bad_leaf in bad_leaves:
        if bad_leaf.level in bad_leaves_by_level:
            bad_leaves_by_level[bad_leaf.level].append(bad_leaf)
        else:
            bad_leaves_by_level[bad_leaf.level] = [bad_leaf]
    for level in bad_leaves_by_level:
        debug("Level {} bad leaves: {}".format(level, [n.id for n in bad_leaves_by_level[level]]))
    
    current_level = max(bad_leaves_by_level)

    # Adding empty lists for levels with no leaves to prevent errors
    for level in range(current_level):
        if level not in bad_leaves_by_level:
            bad_leaves_by_level[level] = []

    debug("Maximum bad leaf level: {}".format(current_level))
    bad_rows = sum(bad_leaf.size() for bad_leaf in bad_leaves)
    while bad_rows >= p:
        debug("{} bad rows to process".format(bad_rows))
        debug("Processing bad leaves of level {}:".format(current_level))
        for n in bad_leaves_by_level[current_level]:
            debug('Leaf {}: size {}, pr "{}"'.format(n.id, n.size(), n.pr))
        leaves_by_pr: Dict[str, Node] = {}
        for bad_leaf in bad_leaves_by_level[current_level]:
            if bad_leaf.pr not in leaves_by_pr:
                debug('New PR "{}" found in bad leaf {}'.format(bad_leaf.pr, bad_leaf.id))
                leaves_by_pr[bad_leaf.pr] = bad_leaf
            # If there are other bad leaves with the same PR, merge them
            else:
                merging_leaf = leaves_by_pr[bad_leaf.pr]
                verbose('Merging leaf {} to leaf {} (PR: "{}")'.format(bad_leaf.id, merging_leaf.id, merging_leaf.pr))
                merging_leaf.extend_table_with_node(bad_leaf)
        
        debug("The following leaves were produced:")
        for leaf in leaves_by_pr.values():
            debug("Leaf {}: size {}, ids {}".format(leaf.id, leaf.size(), leaf.row_ids))
        
        for leaf in leaves_by_pr.values():
            # If the leaf is not smaller than p, then it is a good leaf
            if leaf.size() >= p:
                good_leaves.append(leaf)
                bad_rows -= leaf.size()
                debug("Leaf {} is a good leaf, {} bad rows remaining".format(leaf.id, bad_rows))
            # Otherwise its level is decreased and it will be checked in the next step for a possible merge
            else:
                debug("Leaf {} is a bad leaf: its SAX level is decreased".format(leaf.id, bad_rows))
                leaf.level -= 1
                row = leaf.table[0]
                leaf.pr = SAX(row, leaf.level, leaf.pr_len())
                bad_leaves_by_level[leaf.level].append(leaf)
        current_level -= 1

    if bad_rows == 0:
        print("No rows were suppressed")
    else:
        print(str(bad_rows) + " rows were suppressed: they could not be merged")
    
    verbose('The "recycle bad leaves" phase produced the following good leaves:')
    for n in good_leaves:
        verbose('Node {}: size {}, level {}, PR "{}", ids {}'.format(n.id, n.size(), n.level, n.pr, n.row_ids))
    return good_leaves


def p_anonymity_naive(group: Group, p: int, max_level: int, PR_len: int) -> List[Node]:
    """
    The list of processed nodes is returned. Those can then be used to rebuild the table rows.
    """
    return postprocess(*create_p_anonymity_tree(group, p, max_level, PR_len))


def p_anonymity_kapra(group: Group, p: int, max_level: int, PR_len: int) -> List[Node]:
    """
    The list of processed nodes is returned. Those can then be used to rebuild the table rows.
    """
    return recycle_bad_leaves(*create_p_anonymity_tree(group, p, max_level, PR_len), p)
