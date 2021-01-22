from typing import Dict, List

from group import Group
from node import Node, create_node_from_group, merge_tree_nodes, SAX


def compute_pattern_similarity(N1: Node, N2: Node) -> float:
    """
    Calculate the similarity between two pattern representations as a value between 0 and 1.
    The difference is here defined as the average of the normalized distances (normalized over the number of levels)
    between the two characters in the same position of the two PRs.
    The similarity is 1 - difference.
    """
    diff = 0
    if N1.level == N2.level:
        for i in range(len(N1.PR)):
            diff += abs(ord(N1.PR[i]) - ord(N2.PR[i])) / N1.level
        diff = diff / len(N1.PR)
        return 1 - diff
    else:
        # if the two PRs are of different levels, their values are normalized separately
        for i in range(len(N1.PR)):
            diff += abs((ord(N1.PR[i]) - 97) / (N1.level - 1) - (ord(N2.PR[i]) - 97) / (N2.level - 1))
        diff = diff / len(N1.PR)
        return 1 - diff


def create_p_anonymity_tree(group: Group, p: int, max_level: int, PR_len: int) -> Dict[str, List[Node]]:
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
    # Initialize nodes list with the starting node, corresponding to group
    nodes_to_process = [create_node_from_group(group, PR_len)]
    new_nodes_to_process = True
    good_leaves: List[Node] = []
    bad_leaves: List[Node] = []

    # Node splitting
    while new_nodes_to_process:
        new_nodes_to_process = False
        for N in nodes_to_process:
            N_size = N.size()
            if N_size < p:
                bad_leaves.append(N)
            elif N.level == max_level:
                good_leaves.append(N)
            elif N_size < 2 * p:
                good_leaves.append(N)
                N.maximize_level(max_level)
            else:
                child_nodes = N.split()
                # Split not possible
                if len(child_nodes) < 2 or max(child.size() for child in child_nodes) < p:
                    good_leaves.append(N)
                # Split possible
                else:
                    new_nodes_to_process = True
                    TG_nodes: List[Node] = []
                    TB_nodes: List[Node] = []
                    total_TB_size = 0
                    for child in child_nodes:
                        if child.size() < p:
                            TB_nodes.append(child)
                            total_TB_size += child.size()
                        else:
                            TG_nodes.append(child)

                    nodes_to_process = TG_nodes

                    if total_TB_size >= p:
                        child_merge = merge_tree_nodes(TB_nodes)
                        nodes_to_process.append(child_merge)
                    else:
                        nodes_to_process.extend(TB_nodes)

    return {'good leaves': good_leaves, 'bad leaves': bad_leaves}


def postprocess(leaves_dict: Dict[str, List[Node]]) -> List[Node]:
    """
    The postprocessing phase takes care of the bad leaves integrating them into the good leaves.
    The modified good leaves are then returned.
    """
    good_leaves = leaves_dict["good leaves"]
    bad_leaves = leaves_dict["bad leaves"]

    bad_leaves.sort(key=lambda node: node.size())
    for bad in bad_leaves:
        max_similarity = None
        most_similar_good = None
        for good in good_leaves:
            similarity = compute_pattern_similarity(bad, good)
            if max_similarity is None or similarity > max_similarity:
                max_similarity = similarity
                most_similar_good = good
            elif similarity == max_similarity and good.size() < most_similar_good.size():
                most_similar_good = good
        most_similar_good.members.extend(bad.members)

    return good_leaves


def recycle_bad_leaves(leaves_dict: Dict[str, List[Node]], p: int) -> List[Node]:
    """
    "Recycle bad leaves" step of the KAPRA algorithm, which merges bad leaves creating good ones.
    This function returns a list of all good leaf nodes.
    """
    bad_leaves = leaves_dict["bad leaves"]
    good_leaves = leaves_dict["good leaves"]
    # If there are no bad leaves, then this phase is not needed
    if len(bad_leaves) == 0:
        return good_leaves

    # Preparation: bad leaves are sorted in different lists depending on their level
    bad_leaves_by_level: Dict[int, List[Node]] = {}
    for bad_leaf in bad_leaves:
        if bad_leaf.level in bad_leaves_by_level:
            bad_leaves_by_level[bad_leaf.level].append(bad_leaf)
        else:
            bad_leaves_by_level[bad_leaf.level] = [bad_leaf]

    current_level = max(bad_leaves_by_level)
    bad_rows = sum(bad_leaf.size() for bad_leaf in bad_leaves)
    while bad_rows >= p:
        merged_leaves: List[Node] = []
        bad_leaves_by_PR: Dict[str, Node] = {}
        for bad_leaf in bad_leaves_by_level[current_level]:
            if bad_leaf.PR not in bad_leaves_by_PR:
                bad_leaves_by_PR[bad_leaf.PR] = bad_leaf
            # If there are other bad leaves with the same PR, merge them
            else:
                merging_leaf = bad_leaves_by_PR[bad_leaf.PR]
                merging_leaf.members.extend(bad_leaf.members)
                if merging_leaf not in merged_leaves:
                    merged_leaves.append(merging_leaf)

        for merged_leaf in merged_leaves:
            # If the merged leaf is not smaller than p, then it is a good leaf
            if merged_leaf.size() >= p:
                good_leaves.append(merged_leaf)
                bad_rows -= merged_leaf.size()
            # Otherwise its level is decreased and it will be checked in the next step for a possible merge
            else:
                merged_leaf.level -= 1
                row_index = merged_leaf.members[0]
                row = merged_leaf.group.get_row_at_index(row_index)
                merged_leaf.PR = SAX(row, merged_leaf.level, merged_leaf.PR_len())
                bad_leaves_by_level[merged_leaf.level].append(merged_leaf)

        current_level -= 1

    print(str(bad_rows) + " were suppressed: they could not be merged")
    return good_leaves


def p_anonymity_naive(group: Group, p: int, max_level: int, PR_len: int) -> List[Node]:
    """
    The list of processed nodes is returned. Those can then be used to rebuild the table rows.
    """
    return postprocess(create_p_anonymity_tree(group, p, max_level, PR_len))


def p_anonymity_kapra(group: Group, p: int, max_level: int, PR_len: int) -> List[Node]:
    """
    The list of processed nodes is returned. Those can then be used to rebuild the table rows.
    """
    return recycle_bad_leaves(create_p_anonymity_tree(group, p, max_level, PR_len), p)
