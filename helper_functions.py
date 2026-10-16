"""Helper functions for HW3"""
import numpy as np
from copy import deepcopy
from matplotlib.axes import Axes
from Bio import AlignIO
import matplotlib
import matplotlib.pyplot as plt
import math
from Bio import Entrez
from Bio import SeqIO
import warnings
import matplotlib.lines


class Node:
    def __init__(
        self,
        name: str,
        left: "Node" = None,
        left_distance: float = None,
        right: "Node" = None,
        right_distance: float = None,
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


def hamming_distance_array(MSA) -> np.ndarray:
    """Given a fasta file (data/clustal_MSA_coronavirus.fasta), produce 
    a distance matrix (as a np.ndarray) by calculating hamming distance.
    """
    alignment = AlignIO.read(MSA, "fasta")
    arr = np.array([list(str(rec.seq).upper()) for rec in alignment])

    distances = np.array([(arr != row).sum(axis=1) for row in arr], dtype=float)
    return distances


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
    nodes = {
        label: Node(name=label, left_distance=0.0, right_distance=0.0)
        for label in labels
    }

    # Work on copies so original inputs remain unchanged
    D = distances.astype(float)
    L = labels.copy()

    while len(L) > 2:

        n = len(L)

        row_sums = np.sum(D, axis=1)

        # Compute Q-matrix
        
        Q = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    Q[i, j] = (n - 2) * D[i, j] - row_sums[i] - row_sums[j]

        # Find pair with minimum Q value
        # Use argmin to break ties deterministically

        Q = (n - 2) * D - row_sums[:, None] - row_sums[None, :]
        np.fill_diagonal(Q, np.inf)
        i, j = divmod(np.argmin(Q), n)

        # Remove rows/columns i and j
        if i > j:
            i, j = j, i

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


    
        # Compute distances from U to all other nodes (OLD indexing)
        new_row_old = []
        for k in range(n):
            if k != i and k != j:
                dUk = 0.5 * (D[i, k] + D[j, k] - D[i, j])
                new_row_old.append((k, dUk))

        mask = np.ones(n, dtype=bool)
        mask[[i, j]] = False
        D = D[mask][:, mask]

        # Build remaining_old_indices from OLD matrix
        remaining_old_indices = list(np.where(mask)[0])

        # Map old indices → new indices
        old_to_new = {old: new for new, old in enumerate(remaining_old_indices)}

        # Build new_row using NEW indexing
        new_row = np.zeros(len(D))
        for old_k, value in new_row_old:
            new_k = old_to_new[old_k]
            new_row[new_k] = value

        # Insert new row/column
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

GENERA = ["Alphacoronavirus", "Betacoronavirus",
          "Gammacoronavirus", "Deltacoronavirus"]

def get_leaf_groups(labels: list, email: str) -> dict:
    """Map each leaf label to its coronavirus genus using NCBI taxonomy.
    A helper function for plotting nj tree.

    Returns the genus name, "Outgroup" if the record was found but belongs
    to none of the four genera, or "Unknown" if no record matched.
    """
    Entrez.email = email
    accessions = [lab.split("|")[-1] for lab in labels]
    handle = Entrez.efetch(db="nucleotide", id=",".join(accessions),
                           rettype="gb", retmode="text")
    records = {}
    for rec in SeqIO.parse(handle, "genbank"):
        keys = {rec.id, rec.id.split(".")[0], *rec.annotations.get("accessions", [])}
        for key in keys:
            records[key] = rec
    handle.close()

    mapping, missing = {}, []
    for lab, acc in zip(labels, accessions):
        rec = records.get(acc) or records.get(acc.split(".")[0])
        if rec is None:
            mapping[lab] = "Unknown"
            missing.append(lab)
            continue
        taxonomy = rec.annotations.get("taxonomy", [])
        mapping[lab] = next((g for g in GENERA if g in taxonomy), "Outgroup")

    if missing:
        warnings.warn(f"No NCBI record matched: {missing}")
    return mapping

GROUP_COLORS = {
    "Alphacoronavirus": "#1f77b4",
    "Betacoronavirus": "#d62728",
    "Gammacoronavirus": "#2ca02c",
    "Deltacoronavirus": "#ff7f0e",
    "Outgroup": "#555555",   # record found, but not one of the four genera
    "Unknown": "#bbbbbb",    # NCBI lookup failed for this leaf
}
GROUP_NAMES = {"Unknown": "Unassigned"}  # legend text overrides

def _runs(indices):
    """Split sorted integers into runs of consecutive values."""
    runs = []
    for i in sorted(indices):
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs

def plot_nj_tree(tree: Node, ax: Axes = None, **kwargs) -> None:
    
    """A function for plotting neighbor joining phylogeny dendrogram.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Optional extras (all have defaults):
        show_confidence : bool   - draw support values at internal nodes
        show_lengths    : bool   - draw branch lengths (default False)
        length_decimals : int    - decimals for branch lengths (default 0)
        scale_length    : float  - length of the scale bar (default: auto)
        line_color      : str    - branch colour (default "#bbbbbb")
        fontsize        : int    - leaf label size (default 10)
        short_labels    : bool   - show "Name" instead of "Name|ACCESSION"
                                   (default True)
        groups          : list of (label, [leaf names], italic: bool)
                          - extra manual brackets on the right
        leaf_groups     : dict {leaf name: group name}
                          - colours labels/branches, adds one bracket per
                          contiguous block of each group
        group_colors    : dict {group name: colour} (default GROUP_COLORS)
        show_legend     : bool   - legend of the groups (default True)

    Example
    -------
    >>> fig, ax = plt.subplots(figsize=(10, 10))
    >>> plot_nj_tree(tree=tree, ax=ax)
    >>> fig.savefig("example.png", bbox_inches="tight")
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
    fig = ax.figure

    line_color = kwargs.get("line_color", "#bbbbbb")
    fontsize = kwargs.get("fontsize", 10)
    show_conf = kwargs.get("show_confidence", False)
    show_len = kwargs.get("show_lengths", False)
    decimals = kwargs.get("length_decimals", 0)
    groups = kwargs.get("groups", None)
    leaf_groups = kwargs.get("leaf_groups", None)
    colors = {**GROUP_COLORS, **kwargs.get("group_colors", {})}
    show_legend = kwargs.get("show_legend", True) and bool(leaf_groups)
    short = kwargs.get("short_labels", True)
    lw = 1.0

    def display_name(name):
        return name.split("|")[0].replace("_", " ") if short else name

    def children(node):
        out = []
        if node.left is not None:
            out.append((node.left, node.left_distance or 0.0))
        if node.right is not None:
            out.append((node.right, node.right_distance or 0.0))
        return out

    leaves = []

    def collect_leaves(node):
        kids = children(node)
        if not kids:
            leaves.append(node)
            return
        for child, _ in kids:
            collect_leaves(child)

    collect_leaves(tree)
    leaf_y = {id(leaf): i for i, leaf in enumerate(leaves)}

    # Group of each node: the shared group of all its leaves, else None
    node_group = {}

    def compute_group(node):
        kids = children(node)
        if not kids:
            g = leaf_groups.get(node.name) if leaf_groups else None
        else:
            gs = {compute_group(c) for c, _ in kids}
            g = gs.pop() if len(gs) == 1 else None
        node_group[id(node)] = g
        return g

    compute_group(tree)

    def color_of(node):
        g = node_group.get(id(node))
        return colors.get(g, line_color) if g else line_color

    xs, ys = {}, {}

    def compute_x(node, x=0.0):
        xs[id(node)] = x
        for child, dist in children(node):
            compute_x(child, x + dist)

    def compute_y(node):
        kids = children(node)
        if not kids:
            ys[id(node)] = leaf_y[id(node)]
        else:
            vals = [compute_y(child) for child, _ in kids]
            ys[id(node)] = sum(vals) / len(vals)
        return ys[id(node)]

    compute_x(tree)
    compute_y(tree)

    max_x = max(xs[id(leaf)] for leaf in leaves) or 1.0

    # Brackets: one per contiguous block of leaves
    brackets = []  # (label, [leaf indices], italic, colour)
    if leaf_groups:
        for g in dict.fromkeys(leaf_groups.get(l.name) for l in leaves):
            if g and g not in ("Outgroup", "Unknown"):
                idx = [leaf_y[id(l)] for l in leaves if leaf_groups.get(l.name) == g]
                brackets.append((g, idx, True, colors.get(g, "black")))
    for label, names, *rest in (groups or []):
        idx = [leaf_y[id(l)] for l in leaves if l.name in names]
        brackets.append((label, idx, rest[0] if rest else True, "black"))

    # Layout: measure text so nothing overlaps 
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        renderer = fig._get_renderer()

    def text_width(s, **kw):
        t = ax.text(0, 0, s, **kw)
        w = t.get_window_extent(renderer).width
        t.remove()
        return w

    leaf_px = max(text_width(display_name(l.name), fontsize=fontsize) for l in leaves)
    bracket_px = max(
        (text_width(b[0], fontsize=fontsize + 1, style="italic") for b in brackets),
        default=0,
    )
    axes_px = ax.get_window_extent(renderer).width

    stub = max_x * 0.03
    xmin = -stub * 2
    right_px = 6 + leaf_px + (24 + bracket_px + 10 if brackets else 6)
    span = (max_x - xmin) / (1 - min(right_px / axes_px, 0.8))
    xmax = xmin + span
    px2d = span / axes_px                      # data units per pixel
    bx = max_x + px2d * (6 + leaf_px + 16)     # bracket x position
    tick = px2d * 8

    # Draw branches
    def draw(node):
        kids = children(node)
        if not kids:
            return
        nx = xs[id(node)]
        child_ys = [ys[id(c)] for c, _ in kids]
        ax.plot([nx, nx], [min(child_ys), max(child_ys)],
                color=color_of(node), lw=lw, solid_capstyle="butt")
        for child, dist in kids:
            cy = ys[id(child)]
            ax.plot([nx, xs[id(child)]], [cy, cy], color=color_of(child),
                    lw=lw + 0.5 * bool(node_group[id(child)]),
                    solid_capstyle="butt")
            if show_len:
                ax.text((nx + xs[id(child)]) / 2, cy - 0.12,
                        f"{dist:.{decimals}f}", fontsize=7, color="#555555",
                        ha="center", va="bottom")
            draw(child)

    draw(tree)
    ax.plot([-stub, xs[id(tree)]], [ys[id(tree)]] * 2,
            color=color_of(tree), lw=lw, solid_capstyle="butt")

    # Leaf labels (offset in points, so spacing is size-independent)
    for leaf in leaves:
        ax.annotate(display_name(leaf.name), (xs[id(leaf)], ys[id(leaf)]),
                    xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=fontsize,
                    color=color_of(leaf) if node_group[id(leaf)] else "black")

    # Support values
    if show_conf:
        def label_conf(node):
            if not children(node):
                return
            conf = node.confidence
            if conf is not None:
                txt = f"{conf:g}" if isinstance(conf, (int, float)) else str(conf)
                ax.annotate(txt, (xs[id(node)], ys[id(node)]),
                            xytext=(-3, 2), textcoords="offset points",
                            fontsize=7, color="#555555", ha="right", va="bottom")
            for child, _ in children(node):
                label_conf(child)

        label_conf(tree)

    # Brackets 
    for label, idx, italic, col in brackets:
        for run in _runs(idx):
            y0, y1 = run[0] - 0.35, run[-1] + 0.35
            ax.plot([bx - tick, bx, bx, bx - tick], [y0, y0, y1, y1],
                    color=col, lw=1.6, solid_capstyle="butt")
            ax.annotate(label, (bx, (y0 + y1) / 2), xytext=(6, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=fontsize + 1, color=col,
                        style="italic" if italic else "normal")

    # Scale bar 
    scale = kwargs.get("scale_length")
    if scale is None:
        raw = max_x / 5
        mag = 10 ** math.floor(math.log10(raw))
        scale = round(raw / mag) * mag or raw
    y_bar = len(leaves) + 0.4
    ax.plot([0, scale], [y_bar, y_bar], color="black", lw=1.2)
    for xx in (0, scale):
        ax.plot([xx, xx], [y_bar - 0.15, y_bar + 0.15], color="black", lw=1.2)
    ax.text(scale / 2, y_bar + 0.3, f"{scale:g}", ha="center", va="top", fontsize=8)

    # Legend in its own band below the tree
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(len(leaves) + (3.2 if show_legend else 1.5), -1)
    if show_legend:
        present = [g for g in colors if g in set(leaf_groups.values())]
        handles = [matplotlib.lines.Line2D([0], [0], color=colors[g], lw=3,
                                           label=GROUP_NAMES.get(g, g))
                   for g in present]
        ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1, 0),
                  ncol=min(len(handles), 3), frameon=False,
                  fontsize=fontsize - 1, borderaxespad=0)

    ax.axis("off")
    return ax

def get_phylogenetic_tree_coronaviruses(fasta_file) -> Node:
    """
    Usage: 
    tree = plot_phylogenetic_tree_coronaviruses(data/clustal_MSA_coronavirus.fasta)
    """
    alignment = AlignIO.read(fasta_file, "fasta")
    labels = [rec.id for rec in alignment]   # e.g. "Breda|NC_007447.1"
    D = hamming_distance_array("data/clustal_MSA_coronavirus.fasta")
    tree = neighbor_joining(D, labels)
    return tree

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

def find_node(tree: Node, name: str) -> Node | None:
    """Return the first node whose name matches `name`, or None if not found."""
    if tree is None:
        return None
    if tree.name == name:
        return tree
    left_result = find_node(tree.left, name)
    if left_result is not None:
        return left_result
    return find_node(tree.right, name)


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
    def sort_and_count(node: Node) -> int:
        """Sort the subtree below `node` and return its number of leaves."""
        if node is None:
            return 0
        if node.left is None and node.right is None:
            return 1

        n_left = sort_and_count(node.left)
        n_right = sort_and_count(node.right)

        # Swap each child together with its branch length.
        if n_right < n_left:
            node.left, node.right = node.right, node.left
            node.left_distance, node.right_distance = (
                node.right_distance,
                node.left_distance,
            )

        return n_left + n_right

    sort_and_count(tree)


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

def fetch_spike_protein(accession_id, Entrez_email):
    from Bio import Entrez, SeqIO
    Entrez.email = Entrez_email

    # Fetch annotated GenBank record 
    handle = Entrez.efetch(
        db="nucleotide",
        id=accession_id,
        idtype="acc",
        rettype="gb",
        retmode="text"
    )
    record = SeqIO.read(handle, "genbank")
    handle.close()

    # Search for CDS with gene="S" or product containing "spike"
    for feature in record.features:
        if feature.type == "CDS":
            gene = feature.qualifiers.get("gene", [""])
            product = feature.qualifiers.get("product", [""])

            gene_name = gene[0].lower()
            product_name = product[0].lower()

            if gene_name == "s" or "spike" in product_name:
                cds_seq = feature.extract(record.seq)
                protein = cds_seq.translate(to_stop=True)
                return str(protein)

    raise ValueError("Spike protein (S gene) not found in GenBank record.")
    
def scoring_function(a, b, substitution_matrix):

    #chosen gap penalty
    if a == '-' or b == '-':
        return -11  
    
    try:
        return substitution_matrix[a, b]
    except KeyError:
        return -1000
    
def global_alignment(seq1, seq2, scoring_function, substitution_matrix):

    """Global sequence alignment using the Needleman–Wunsch algorithm.

    Indels should be denoted with the "-" character.

    Parameters
    ----------
    seq1: str
        First sequence to be aligned.
    seq2: str
        Second sequence to be aligned.
    scoring_function: Callable

    Returns
    -------
    str
        First aligned sequence.
    str
        Second aligned sequence.
    float
        Final score of the alignment.

    Examples
    --------
    >>> global_alignment("abracadabra", "dabarakadara", lambda x, y: [-1, 1][x == y])
    ('-ab-racadabra', 'dabarakada-ra', 5.0)

    Other alignments are not possible.

    """

    n, m = len(seq1), len(seq2)

    score = [[0] * (m + 1) for _ in range(n + 1)]
    score[0][0] = 0
    back  = [[None] * (m + 1) for _ in range(n + 1)]

    gap_penalty = scoring_function('-', '-', substitution_matrix)

    # initialization
    for i in range(1, n + 1):
        score[i][0] = i * gap_penalty
        back[i][0] = "up"

    for j in range(1, m + 1):
        score[0][j] = j * gap_penalty
        back[0][j] = "left"

    # recurrence
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i - 1][j - 1] + scoring_function(seq1[i-1], seq2[j-1], substitution_matrix)
            up = score[i - 1][j] + gap_penalty
            left = score[i][j - 1] + gap_penalty
            best = max(diag, up, left)
            score[i][j] = best

            if best == diag:
                back[i][j] = "diag"
            elif best == up:
                back[i][j] = "up"
            else:
                back[i][j] = "left"

    # traceback
    aligned1 = []
    aligned2 = []
    freq_counter = 0  # amino acid frequency counter

    i, j = n, m

    while i > 0 or j > 0:

        if i == 0:
            direction = "left"
        elif j == 0:
            direction = "up"
        else:
            direction = back[i][j]
        if direction == None:
            raise RuntimeError(f"Traceback stuck at i={i}, j={j}, direction={direction}")

        if direction == "diag":
            a1, a2 = seq1[i - 1], seq2[j - 1]
            aligned1.append(a1)
            aligned2.append(a2)
            if a1 == a2 and a1 != '-':  # count identical amino acids
                freq_counter += 1
            i -= 1
            j -= 1

        elif direction == "up":
            aligned1.append(seq1[i - 1])
            aligned2.append('-')
            i -= 1

        elif direction == "left":
            aligned1.append('-')
            aligned2.append(seq2[j - 1])
            j -= 1

    aligned1.reverse()
    aligned2.reverse()

    final_score = score[n][m]
    percent_identity = (freq_counter / len(aligned1)) * 100

    return "".join(aligned1), "".join(aligned2), final_score, percent_identity

import numpy as np
import matplotlib.pyplot as plt


def sliding_similarity(seq1, seq2, window=100, gap_chars="-"):
    """Fraction of matching positions in each window of a pairwise alignment.
    Positions where both sequences have a gap are skipped."""
    s1, s2 = str(seq1), str(seq2)
    assert len(s1) == len(s2), "Aligned sequences must have equal length"

    # Keep only columns where at least one sequence has a residue
    cols = [(a, b) for a, b in zip(s1, s2) if not (a in gap_chars and b in gap_chars)]
    match = np.array([1 if (a == b and a not in gap_chars) else 0 for a, b in cols],
                     dtype=float)

    if len(match) < window:
        return np.array([match.mean()]), np.array([0])
    kernel = np.ones(window) / window
    sim = np.convolve(match, kernel, mode="valid")     # one value per window start
    positions = np.arange(len(sim)) + window // 2      # window centre
    return sim, positions


def plot_recombination(aligned_human_spike_to_bat, aligned_bat,
                       aligned_human_spike_to_pangolin, aligned_pangolin,
                       HvB_genome, Bat_genome, HvP_genome, Pan_genome,
                       spike_window=100, genome_window=1000,
                       rbd_aa=(319, 541),  # SARS-CoV-2 RBD in spike aa coordinates
                       outfile="recombination.svg"):
    # Spike sliding similarity (at the protein level)
    sim_bat, pos_bat = sliding_similarity(aligned_human_spike_to_bat, aligned_bat, spike_window)
    sim_pan, pos_pan = sliding_similarity(aligned_human_spike_to_pangolin, aligned_pangolin, spike_window)

    # Convert alignment-column positions to human spike residue positions
    def to_human_coords(aligned_human, pos):
        h = np.array([c != "-" for c in str(aligned_human)])
        cum = np.cumsum(h)
        return cum[np.clip(pos, 0, len(cum) - 1)]

    x_bat = to_human_coords(aligned_human_spike_to_bat, pos_bat)
    x_pan = to_human_coords(aligned_human_spike_to_pangolin, pos_pan)

    # Whole genome sliding similarity (nucleotide level, from the Pairwise Alignment)
    g_bat, gp_bat = sliding_similarity(HvB_genome, Bat_genome, genome_window)
    g_pan, gp_pan = sliding_similarity(HvP_genome, Pan_genome, genome_window)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    # Plot 1: whole genome
    ax = axes[0]
    ax.plot(gp_bat, g_bat, color="tab:blue", label="Bat-CoV RaTG13")
    ax.plot(gp_pan, g_pan, color="tab:orange", label="Pangolin-CoV MP789")
    ax.set_title("Whole genome similarity to SARS-CoV-2 (window = %d)" % genome_window)
    ax.set_xlabel("Alignment position (bp)")
    ax.set_ylabel("Similarity")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="lower left")

    # Plot 2: spike region
    ax = axes[1]
    ax.axvspan(rbd_aa[0], rbd_aa[1], color="khaki", alpha=0.5, label="RBD")
    ax.plot(x_bat, sim_bat, color="tab:blue", label="Bat-CoV RaTG13")
    ax.plot(x_pan, sim_pan, color="tab:orange", label="Pangolin-CoV MP789")
    ax.set_title("Spike protein amino-acid similarity to SARS-CoV-2 (window = %d)" % spike_window)
    ax.set_xlabel("Spike residue position (human SARS-CoV-2)")
    ax.set_ylabel("Similarity")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig("recombination.svg")

    # Numeric similarity: inside vs outside RBD
    def mean_in_out(x, sim):
        inside = (x >= rbd_aa[0]) & (x <= rbd_aa[1])
        return sim[inside].mean(), sim[~inside].mean()

    results = {"Bat": mean_in_out(x_bat, sim_bat), "Pangolin": mean_in_out(x_pan, sim_pan)}
    for name, (i, o) in results.items():
        print(f"{name}: mean similarity inside RBD = {i:.3f}, outside RBD = {o:.3f}")
    return results


def pairwise_genome_alignment(ref, query):
    from Bio.Align import PairwiseAligner
    """Global nucleotide alignment of two raw sequences 
    using Biopython's PairwiseAligner.
    Returns two gapped strings of equal length."""
    ref, query = str(ref).upper(), str(query).upper()

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -0.5

    aln = aligner.align(ref, query)[0]

    a, b = [], []
    prev_r = prev_q = 0
    for (r0, r1), (q0, q1) in zip(*aln.aligned):
        if r0 > prev_r:                       # ref residues missing in query
            a.append(ref[prev_r:r0]); b.append("-" * (r0 - prev_r))
        if q0 > prev_q:                       # query residues missing in ref
            a.append("-" * (q0 - prev_q)); b.append(query[prev_q:q0])
        a.append(ref[r0:r1]); b.append(query[q0:q1])
        prev_r, prev_q = r1, q1
    # trailing overhangs
    if prev_r < len(ref):
        a.append(ref[prev_r:]); b.append("-" * (len(ref) - prev_r))
    if prev_q < len(query):
        a.append("-" * (len(query) - prev_q)); b.append(query[prev_q:])

    ga, gb = "".join(a), "".join(b)
    assert len(ga) == len(gb)
    return ga, gb