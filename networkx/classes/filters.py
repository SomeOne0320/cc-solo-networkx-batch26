"""Filter factories to hide or show sets of nodes and edges.

These filters return the function used when creating `SubGraph`.
"""

__all__ = [
    "no_filter",
    "hide_nodes",
    "hide_edges",
    "hide_multiedges",
    "hide_diedges",
    "hide_multidiedges",
    "show_nodes",
    "show_edges",
    "show_multiedges",
    "show_diedges",
    "show_multidiedges",
]


def no_filter(*items):
    """Returns a filter function that always evaluates to True."""
    return True


class hide_nodes:
    """Filter class to hide specific nodes."""

    def __init__(self, nodes):
        # Avoid the attribute name ``nodes``: coreviews uses that attribute as
        # an inclusion-set optimization for show-style filters.
        self.hidden_nodes = set(nodes)

    def __call__(self, node):
        return node not in self.hidden_nodes


class hide_diedges:
    """Filter class to hide specific directed edges."""

    def __init__(self, edges):
        self.edges = {(u, v) for u, v in edges}

    def __call__(self, u, v):
        return (u, v) not in self.edges


class hide_edges:
    """Filter class to hide specific undirected edges."""

    def __init__(self, edges):
        edges = list(edges)
        self.edges = set(edges) | {(v, u) for (u, v) in edges}

    def __call__(self, u, v):
        return (u, v) not in self.edges


class hide_multidiedges:
    """Filter class to hide specific multi-directed edges."""

    def __init__(self, edges):
        self.edges = {(u, v, k) for u, v, k in edges}

    def __call__(self, u, v, k):
        return (u, v, k) not in self.edges


class hide_multiedges:
    """Filter class to hide specific multi-undirected edges."""

    def __init__(self, edges):
        edges = list(edges)
        self.edges = set(edges) | {(v, u, k) for (u, v, k) in edges}

    def __call__(self, u, v, k):
        return (u, v, k) not in self.edges


# write show_nodes as a class to make SubGraph pickleable
class show_nodes:
    """Filter class to show specific nodes.

    Attach the set of nodes as an attribute to speed up this commonly used filter

    Note that another allowed attribute for filters is to store the number of nodes
    on the filter as attribute `length` (used in `__len__`). It is a user
    responsibility to ensure this attribute is accurate if present.
    """

    def __init__(self, nodes):
        self.nodes = set(nodes)

    def __call__(self, node):
        return node in self.nodes


class show_diedges:
    """Filter class to show specific directed edges."""

    def __init__(self, edges):
        self.edges = {(u, v) for u, v in edges}

    def __call__(self, u, v):
        return (u, v) in self.edges


class show_edges:
    """Filter class to show specific undirected edges."""

    def __init__(self, edges):
        edges = list(edges)
        self.edges = set(edges) | {(v, u) for (u, v) in edges}

    def __call__(self, u, v):
        return (u, v) in self.edges


class show_multidiedges:
    """Filter class to show specific multi-directed edges."""

    def __init__(self, edges):
        self.edges = {(u, v, k) for u, v, k in edges}

    def __call__(self, u, v, k):
        return (u, v, k) in self.edges


class show_multiedges:
    """Filter class to show specific multi-undirected edges."""

    def __init__(self, edges):
        edges = list(edges)
        self.edges = set(edges) | {(v, u, k) for (u, v, k) in edges}

    def __call__(self, u, v, k):
        return (u, v, k) in self.edges
