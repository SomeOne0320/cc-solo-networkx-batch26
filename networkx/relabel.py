import networkx as nx

__all__ = ["convert_node_labels_to_integers", "relabel_nodes"]


@nx._dispatchable(
    preserve_all_attrs=True, mutates_input={"not copy": 2}, returns_graph=True
)
def relabel_nodes(G, mapping, copy=True):
    """Relabel the nodes of the graph G according to a given mapping.

    The original node ordering may not be preserved if `copy` is `False` and the
    mapping includes overlap between old and new labels.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    mapping : dictionary
       A dictionary with the old labels as keys and new labels as values.
       A partial mapping is allowed. Mapping 2 nodes to a single node is allowed.
       Any non-node keys in the mapping are ignored.

    copy : bool (optional, default=True)
       If True return a copy, or if False relabel the nodes in place.

    Examples
    --------
    To create a new graph with nodes relabeled according to a given
    dictionary:

    >>> G = nx.path_graph(3)
    >>> sorted(G)
    [0, 1, 2]
    >>> mapping = {0: "a", 1: "b", 2: "c"}
    >>> H = nx.relabel_nodes(G, mapping)
    >>> sorted(H)
    ['a', 'b', 'c']

    Nodes can be relabeled with any hashable object, including numbers
    and strings:

    >>> import string
    >>> G = nx.path_graph(26)  # nodes are integers 0 through 25
    >>> sorted(G)[:3]
    [0, 1, 2]
    >>> mapping = dict(zip(G, string.ascii_lowercase))
    >>> G = nx.relabel_nodes(G, mapping)  # nodes are characters a through z
    >>> sorted(G)[:3]
    ['a', 'b', 'c']
    >>> mapping = dict(zip(G, range(1, 27)))
    >>> G = nx.relabel_nodes(G, mapping)  # nodes are integers 1 through 26
    >>> sorted(G)[:3]
    [1, 2, 3]

    To perform a partial in-place relabeling, provide a dictionary
    mapping only a subset of the nodes, and set the `copy` keyword
    argument to False:

    >>> G = nx.path_graph(3)  # nodes 0-1-2
    >>> mapping = {0: "a", 1: "b"}  # 0->'a' and 1->'b'
    >>> G = nx.relabel_nodes(G, mapping, copy=False)
    >>> sorted(G, key=str)
    [2, 'a', 'b']

    A mapping can also be given as a function:

    >>> G = nx.path_graph(3)
    >>> H = nx.relabel_nodes(G, lambda x: x**2)
    >>> list(H)
    [0, 1, 4]

    In a multigraph, relabeling two or more nodes to the same new node
    will retain all edges, but may change the edge keys in the process:

    >>> G = nx.MultiGraph()
    >>> G.add_edge(0, 1, value="a")  # returns the key for this edge
    0
    >>> G.add_edge(0, 2, value="b")
    0
    >>> G.add_edge(0, 3, value="c")
    0
    >>> mapping = {1: 4, 2: 4, 3: 4}
    >>> H = nx.relabel_nodes(G, mapping, copy=True)
    >>> print(H[0])
    {4: {0: {'value': 'a'}, 1: {'value': 'b'}, 2: {'value': 'c'}}}

    This works for in-place relabeling too:

    >>> G = nx.relabel_nodes(G, mapping, copy=False)
    >>> print(G[0])
    {4: {0: {'value': 'a'}, 1: {'value': 'b'}, 2: {'value': 'c'}}}

    Notes
    -----
    Only the nodes specified in the mapping will be relabeled.
    Any non-node keys in the mapping are ignored.

    The keyword setting copy=False modifies the graph in place.
    Relabel_nodes avoids naming collisions by building a
    directed graph from ``mapping`` which specifies the order of
    relabelings. Naming collisions, such as a->b, b->c, are ordered
    such that "b" gets renamed to "c" before "a" gets renamed "b".
    In cases of circular mappings (e.g. a->b, b->a), modifying the
    graph is not possible in-place and an exception is raised.
    In that case, use copy=True.

    If a relabel operation on a multigraph would cause two or more
    edges to have the same source, target and key, the second edge must
    be assigned a new key to retain all edges. The new key is set
    to the lowest non-negative integer not already used as a key
    for edges between these two nodes. Note that this means non-numeric
    keys may be replaced by numeric keys.

    See Also
    --------
    convert_node_labels_to_integers
    """
    # you can pass any callable e.g. f(old_label) -> new_label or
    # e.g. str(old_label) -> new_label, but we'll just make a dictionary here regardless
    m = {n: mapping(n) for n in G} if callable(mapping) else mapping

    if copy:
        return _relabel_copy(G, m)
    else:
        return _relabel_inplace(G, m)


def _canonical_edge_pair(u, v, directed):
    """Return a hashable identifier for an endpoint pair.

    For undirected graphs ``(u, v)`` and ``(v, u)`` are the same pair.
    """
    return (u, v) if directed else frozenset((u, v))


def _resolve_merge_keys(edges, directed, occupied=None):
    """Resolve multigraph edge keys after endpoints have been merged.

    `edges` is an iterable of ``(u, v, key, data)`` tuples whose endpoints
    are already the final ones. `occupied`, if provided, maps a canonical
    endpoint pair (see :func:`_canonical_edge_pair`) to an iterable of keys
    already present in the destination graph. Those keys have priority and
    are never reassigned.

    Every edge whose original key does not collide keeps that key. All
    keys that can be kept are reserved first, so an edge that needs a new
    key can never take a key another edge could have kept. Colliding edges
    then receive, independently for each endpoint pair, the lowest free
    non-negative integer as their new key.

    The returned list contains ``(u, v, key, data)`` tuples in the same
    order as the input.
    """
    used = {}
    if occupied:
        for pair, keys in occupied.items():
            used[pair] = set(keys)

    edge_list = list(edges)
    pairs = [_canonical_edge_pair(u, v, directed) for u, v, _, _ in edge_list]
    keep_key = [False] * len(edge_list)

    # Reserve incumbent and original keys that can stay. Processing order
    # breaks ties when several remapped edges share the same original key.
    for i, ((_, _, key, _), pair) in enumerate(zip(edge_list, pairs)):
        keys = used.setdefault(pair, set())
        if key not in keys:
            keys.add(key)
            keep_key[i] = True

    resolved = []
    for i, ((u, v, key, data), pair) in enumerate(zip(edge_list, pairs)):
        if keep_key[i]:
            resolved.append((u, v, key, data))
            continue
        keys = used[pair]
        new_key = 0
        while new_key in keys:
            new_key += 1
        keys.add(new_key)
        resolved.append((u, v, new_key, data))
    return resolved


def _relabel_multigraph_edges(G, mapping):
    """Build the globally key-resolved edge list for a multigraph relabel."""
    directed = G.is_directed()
    result = [None] * G.number_of_edges()
    occupied = {}
    remapped_edges = []
    remapped_positions = []

    for i, (u, v, key, data) in enumerate(G.edges(keys=True, data=True)):
        new_u = mapping.get(u, u)
        new_v = mapping.get(v, v)
        edge = (new_u, new_v, key, data.copy())
        if new_u == u and new_v == v:
            # Endpoints unchanged: this edge stays in place and its key has
            # priority over edges realigned onto the same endpoint pair.
            pair = _canonical_edge_pair(new_u, new_v, directed)
            occupied.setdefault(pair, set()).add(key)
            result[i] = edge
        else:
            remapped_edges.append(edge)
            remapped_positions.append(i)

    resolved = _resolve_merge_keys(remapped_edges, directed, occupied)
    for position, edge in zip(remapped_positions, resolved):
        result[position] = edge
    return result


def _relabel_inplace(G, mapping):
    if len(mapping.keys() & mapping.values()) > 0:
        # labels sets overlap
        # can we topological sort and still do the relabeling?
        D = nx.DiGraph(list(mapping.items()))
        D.remove_edges_from(nx.selfloop_edges(D))
        try:
            nodes = reversed(list(nx.topological_sort(D)))
        except nx.NetworkXUnfeasible as err:
            raise nx.NetworkXUnfeasible(
                "The node label sets are overlapping and no ordering can "
                "resolve the mapping. Use copy=True."
            ) from err
    else:
        # non-overlapping label sets, sort them in the order of G nodes
        nodes = [n for n in G if n in mapping]

    multigraph = G.is_multigraph()
    directed = G.is_directed()

    # Resolve all multigraph keys up front, before the graph is mutated, so
    # that key allocation is global per endpoint pair and matches copy=True.
    final_multiedges = _relabel_multigraph_edges(G, mapping) if multigraph else None

    for old in nodes:
        # Test that old is in both mapping and G, otherwise ignore.
        try:
            new = mapping[old]
            G.add_node(new, **G.nodes[old])
        except KeyError:
            continue
        if new == old:
            continue
        if multigraph:
            # Edges are restored in one pass after the node loop below.
            # Removing the old node also drops its (possibly duplicated, in
            # directed graphs) incident edges exactly once.
            G.remove_node(old)
            continue
        new_edges = [
            (new, new if old == target else target, data)
            for (_, target, data) in G.edges(old, data=True)
        ]
        if directed:
            new_edges += [
                (new if old == source else source, new, data)
                for (source, _, data) in G.in_edges(old, data=True)
            ]
        G.remove_node(old)
        G.add_edges_from(new_edges)

    if multigraph:
        # Edges between nodes that were never removed are still present;
        # clear_edges() leaves the relabeled node order intact. Re-adding
        # the full resolved edge set makes this equivalent to copy=True.
        G.clear_edges()
        G.add_edges_from(final_multiedges)
    return G


def _relabel_copy(G, mapping):
    H = G.__class__()
    H.add_nodes_from(mapping.get(n, n) for n in G)
    H._node.update((mapping.get(n, n), d.copy()) for n, d in G.nodes.items())
    if G.is_multigraph():
        H.add_edges_from(_relabel_multigraph_edges(G, mapping))
    else:
        H.add_edges_from(
            (mapping.get(n1, n1), mapping.get(n2, n2), d.copy())
            for (n1, n2, d) in G.edges(data=True)
        )
    H.graph.update(G.graph)
    return H


@nx._dispatchable(preserve_all_attrs=True, returns_graph=True)
def convert_node_labels_to_integers(
    G, first_label=0, ordering="default", label_attribute=None
):
    """Returns a copy of the graph G with the nodes relabeled using
    consecutive integers.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    first_label : int, optional (default=0)
       An integer specifying the starting offset in numbering nodes.
       The new integer labels are numbered first_label, ..., n-1+first_label.

    ordering : string
       "default" : inherit node ordering from G.nodes()
       "sorted"  : inherit node ordering from sorted(G.nodes())
       "increasing degree" : nodes are sorted by increasing degree
       "decreasing degree" : nodes are sorted by decreasing degree

    label_attribute : string, optional (default=None)
       Name of node attribute to store old label.  If None no attribute
       is created.

    Notes
    -----
    Node and edge attribute data are copied to the new (relabeled) graph.

    There is no guarantee that the relabeling of nodes to integers will
    give the same two integers for two (even identical graphs).
    Use the `ordering` argument to try to preserve the order.

    See Also
    --------
    relabel_nodes
    """
    N = G.number_of_nodes() + first_label
    if ordering == "default":
        mapping = dict(zip(G.nodes(), range(first_label, N)))
    elif ordering == "sorted":
        nlist = sorted(G.nodes())
        mapping = dict(zip(nlist, range(first_label, N)))
    elif ordering == "increasing degree":
        dv_pairs = [(d, n) for (n, d) in G.degree()]
        dv_pairs.sort()  # in-place sort from lowest to highest degree
        mapping = dict(zip([n for d, n in dv_pairs], range(first_label, N)))
    elif ordering == "decreasing degree":
        dv_pairs = [(d, n) for (n, d) in G.degree()]
        dv_pairs.sort()  # in-place sort from lowest to highest degree
        dv_pairs.reverse()
        mapping = dict(zip([n for d, n in dv_pairs], range(first_label, N)))
    else:
        raise nx.NetworkXError(f"Unknown node ordering: {ordering}")
    H = relabel_nodes(G, mapping)
    # create node attribute with the old label
    if label_attribute is not None:
        nx.set_node_attributes(H, {v: k for k, v in mapping.items()}, label_attribute)
    return H
