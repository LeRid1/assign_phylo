"""Helper functions for HW3"""
import numpy as np
from copy import deepcopy
from matplotlib.axes import Axes


class Node:
    def __init__(
        self,
        name: str,
        left: "Node",
        left_distance: float,
        right: "Node",
        right_distance: float,
        confidence: float = None,
    ):
        """A node in a binary tree produced by neighbor joining algorithm.

        Parameters
        ----------
        name: str
            Name of the node.
        left: Node
            Left child.
        left_distance: float
            The distance to the left child.
        right: Node
            Right child.
        right_distance: float
            The distance to the right child.
        confidence: float
            The confidence level of the split determined by the bootstrap method.
            Only used if you implement Bonus Problem 1.

        Notes
        -----
        The current public API needs to remain as it is, i.e., don't change the
        names of the properties in the template, as the tests expect this kind
        of structure. However, feel free to add any methods/properties/attributes
        that you might need in your tree construction.

        """
        self.name = name
        self.left = left
        self.left_distance = left_distance
        self.right = right
        self.right_distance = right_distance
        self.confidence = confidence

        # Internal helper: track how many children have been assigned
        self._child_count = 0

    def add_child(self, child: "Node", branch_length: float):
        """
        Attach a child to this node. The first child becomes `left`,
        the second becomes `right`. A third child is not allowed.
        """
        if self._child_count == 0:
            self.left = child
            self.left_distance = branch_length
        elif self._child_count == 1:
            self.right = child
            self.right_distance = branch_length
        else:
            raise ValueError(
                f"Node '{self.name}' already has two children; "
                "neighbor-joining trees must be strictly binary."
            )

        self._child_count += 1

    def __repr__(self):
        return f"Node({self.name})"


def neighbor_joining(distances: np.ndarray, labels: list) -> Node:
    """The Neighbor-Joining algorithm.

    For the same results as in the later test dendrograms;
    add new nodes to the end of the list/matrix and
    in case of ties, use np.argmin to choose the joining pair.

    Parameters
    ----------
    distances: np.ndarray
        A 2d square, symmetric distance matrix containing distances between
        data points. The diagonal entries should always be zero; d(x, x) = 0.
    labels: list
        A list of labels corresponding to entries in the distances matrix.
        Use them to set names of nodes.

    Returns
    -------
    Node
        A root node of the neighbor joining tree.

    """
    
    # Create initial leaf nodes
    nodes = {label: Node(name=label) for label in labels}

    # Work on copies so original inputs remain unchanged
    D = distances.astype(float)
    L = labels.copy()

    while len(L) > 2:

        n = len(L)

        # ----------------------------------------------------
        # Step 1: Compute Q-matrix
        # ----------------------------------------------------
        Q = np.zeros((n, n))

        row_sums = np.sum(D, axis=1)

        for i in range(n):
            for j in range(n):
                if i != j:
                    Q[i, j] = (n - 2) * D[i, j] - row_sums[i] - row_sums[j]

        # Find pair with minimum Q value
        # Use argmin to break ties deterministically

        flat_index = np.argmin(Q + np.eye(n) * np.inf)
        i, j = divmod(flat_index, n)

        label_i = L[i]
        label_j = L[j]

        # Create new internal node U

        new_label = f"{label_i}_{label_j}"
        U = Node(name=new_label)

        # Compute branch lengths to U

        sum_i = row_sums[i]
        sum_j = row_sums[j]

        delta_iU = 0.5 * D[i, j] + (sum_i - sum_j) / (2 * (n - 2))
        delta_jU = D[i, j] - delta_iU

        U.add_child(nodes[label_i], delta_iU)
        U.add_child(nodes[label_j], delta_jU)


        # Compute distances from U to all other nodes
    
        new_row = []
        for k in range(n):
            if k != i and k != j:
                dUk = 0.5 * (D[i, k] + D[j, k] - D[i, j])
                new_row.append(dUk)

      
        # Update distance matrix and labels
 
        # Remove rows/columns i and j
        mask = np.ones(n, dtype=bool)
        mask[[i, j]] = False
        D = D[mask][:, mask]

        # Append new row/column
        new_row = np.array(new_row)
        D = np.vstack([D, new_row])
        new_col = np.append(new_row, 0.0)
        D = np.column_stack([D, new_col])

        # Update labels
        L = [L[k] for k in range(n) if k not in (i, j)]
        L.append(new_label)

        nodes[new_label] = U

    # Final iteration: only two nodes remain

    label_a, label_b = L
    a_idx, b_idx = 0, 1

    root = Node(name="root")
    dist = D[a_idx, b_idx]
    branch = dist / 2.0

    root.add_child(nodes[label_a], branch)
    root.add_child(nodes[label_b], branch)

    return root


def plot_nj_tree(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.
        But make sure your function can work without them, for testing purposes.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree(tree=tree, ax=ax)
    >>> fig.savefig("example.png")

    """

    return ax


def _find_a_parent_to_node(tree: Node, node: Node) -> tuple:
    """Utility function for reroot_tree"""
    stack = [tree]

    while len(stack) > 0:

        current_node = stack.pop()
        if node.name == current_node.left.name:
            return current_node, "left"
        elif node.name == current_node.right.name:
            return current_node, "right"

        stack += [
            n for n in [current_node.left, current_node.right] if n.left is not None
        ]

    return None



def _remove_child_from_parent(parent_node: Node, child_location: str) -> None:
    """Utility function for reroot_tree"""
    setattr(parent_node, child_location, None)
    setattr(parent_node, f"{child_location}_distance", 0.0)


def reroot_tree(original_tree: Node, outgroup_node: Node) -> Node:
    """A function to create a new root and invert a tree accordingly.

    This function reroots tree with nodes in original format. If you
    added any other relational parameters to your nodes, these parameters
    will not be inverted! You can modify this implementation or create
    additional functions to fix them.

    Parameters
    ----------
    original_tree: Node
        A root node of the original tree.
    outgroup_node: Node
        A Node to set as an outgroup (already included in a tree).
        Find it by it's name and then use it as parameter.

    Returns
    -------
    Node
        Inverted tree with a new root node.
    """
    tree = deepcopy(original_tree)

    parent, child_loc = _find_a_parent_to_node(tree, outgroup_node)
    distance = getattr(parent, f"{child_loc}_distance")
    _remove_child_from_parent(parent, child_loc)

    new_root = Node("new_root", parent, distance / 2, outgroup_node, distance / 2)
    child = parent

    while tree != child:
        parent, child_loc = _find_a_parent_to_node(tree, child)

        distance = getattr(parent, f"{child_loc}_distance")
        _remove_child_from_parent(parent, child_loc)

        empty_side = "left" if child.left is None else "right"
        setattr(child, f"{empty_side}_distance", distance)
        setattr(child, empty_side, parent)

        if tree.name == parent.name:
            break
        child = parent

    other_child_loc = "right" if child_loc == "left" else "left"
    other_child_distance = getattr(parent, f"{other_child_loc}_distance")

    setattr(child, f"{empty_side}_distance", other_child_distance + distance)
    setattr(child, empty_side, getattr(parent, other_child_loc))

    return new_root


def sort_children_by_leaves(tree: Node) -> None:
    """Sort the children of a tree by their corresponding number of leaves.

    The tree can be changed inplace.

    Parameters
    ----------
    tree: Node
        The root node of the tree.

    """
    raise NotImplementedError()


def plot_nj_tree_radial(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram
    with a radial layout.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree_radial(tree=tree, ax=ax)
    >>> fig.savefig("example_radial.png")

    """
    raise NotImplementedError()