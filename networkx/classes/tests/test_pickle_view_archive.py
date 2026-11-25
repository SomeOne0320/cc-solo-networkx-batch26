import pickle

import pytest

import networkx as nx


class KeyLineFilter:
    """Pickleable caller filter selecting parallel edges by their key."""

    def __init__(self, line):
        self.line = line

    def __call__(self, u, v, key):
        return key == self.line


def make_road_graph():
    G = nx.MultiDiGraph()
    G.add_edge("C", "A", key="L1", delay_min=0)
    G.add_edge("A", "C", key="L1", delay_min=0)
    G.add_edge("C", "A", key="B1")
    G.add_edge("C", "R", key="L2")
    G.add_edge("R", "C", key="L2", closed=True)
    G.add_edge("A", "D", key="OLD")
    G.add_edge("D", "C", key="N9")
    return G


@pytest.mark.parametrize(
    ("factory", "items", "query", "expected"),
    [
        (nx.filters.hide_nodes, [1], (1,), False),
        (nx.filters.show_nodes, [1], (1,), True),
        (nx.filters.hide_diedges, [(1, 2)], (1, 2), False),
        (nx.filters.show_diedges, [(1, 2)], (2, 1), False),
        (nx.filters.hide_edges, [(1, 2)], (2, 1), False),
        (nx.filters.show_edges, [(1, 2)], (2, 1), True),
        (nx.filters.hide_multidiedges, [(1, 2, "k")], (1, 2, "k"), False),
        (nx.filters.show_multidiedges, [(1, 2, "k")], (2, 1, "k"), False),
        (nx.filters.hide_multiedges, [(1, 2, "k")], (2, 1, "k"), False),
        (nx.filters.show_multiedges, [(1, 2, "k")], (2, 1, "k"), True),
    ],
)
def test_builtin_filters_survive_pickle(factory, items, query, expected):
    filter_obj = factory(items)
    restored = pickle.loads(pickle.dumps(filter_obj, protocol=-1))
    assert restored(*query) is expected


def test_pickled_views_share_restored_graph_and_state(tmp_path):
    G = make_road_graph()
    closed = nx.restricted_view(G, [], [("R", "C", "L2")])
    line = nx.edge_subgraph(G, [("C", "A", "L1"), ("A", "C", "L1")])
    route = nx.subgraph_view(G, filter_edge=KeyLineFilter("L1"))
    layered = nx.reverse_view(nx.subgraph_view(closed, filter_edge=KeyLineFilter("L2")))
    inbound = nx.reverse_view(G)
    closed_inbound = nx.reverse_view(closed)
    line_inbound = nx.reverse_view(line)

    # Exercise reporting paths before writing the archive.
    for view in (closed, line, route, layered, inbound, closed_inbound):
        list(view.nodes)
        list(view.edges(keys=True))
        list(view.in_edges(keys=True))

    archive = {
        "base": G,
        "closed": closed,
        "line": line,
        "route": route,
        "layered": layered,
        "inbound": inbound,
        "closed_inbound": closed_inbound,
        "line_inbound": line_inbound,
    }
    path = tmp_path / "road-views.pkl"
    with path.open("wb") as fh:
        pickle.dump(archive, fh)
    with path.open("rb") as fh:
        restored = pickle.load(fh)

    H = restored["base"]
    closed = restored["closed"]
    line = restored["line"]
    route = restored["route"]
    layered = restored["layered"]
    inbound = restored["inbound"]
    closed_inbound = restored["closed_inbound"]
    line_inbound = restored["line_inbound"]

    # All views in the archive observe one restored base graph.
    assert closed._graph is H
    assert line._graph is H
    assert route._graph is H
    assert inbound._graph is H
    assert closed_inbound._graph is closed
    assert line_inbound._graph is line
    assert layered._graph._graph is closed
    assert layered._graph._graph._graph is H

    # Attribute writes through either object are visible to all views.
    H.nodes["C"]["staffed"] = True
    line.nodes["A"]["terminal"] = "T2"
    line["C"]["A"]["L1"]["delay_min"] = 7
    H.remove_edge("A", "D", key="OLD")
    H.add_edge("A", "H", key="L1")

    assert closed.nodes["C"]["staffed"] is True
    assert H.nodes["A"]["terminal"] == "T2"
    assert closed.nodes["A"]["terminal"] == "T2"
    assert H["C"]["A"]["L1"]["delay_min"] == 7
    assert closed["C"]["A"]["L1"]["delay_min"] == 7

    base_edges = set(H.edges(keys=True))
    closed_edges = set(closed.edges(keys=True))
    line_edges = set(line.edges(keys=True))
    route_edges = set(route.edges(keys=True))

    assert ("A", "D", "OLD") not in base_edges
    assert ("A", "H", "L1") in route_edges
    assert closed_edges == base_edges - {("R", "C", "L2")}
    assert line_edges == {("C", "A", "L1"), ("A", "C", "L1")}
    assert route_edges == {
        ("C", "A", "L1"),
        ("A", "C", "L1"),
        ("A", "H", "L1"),
    }

    # Directions and parallel keys are preserved after reversal.
    assert set(inbound.edges(keys=True)) == {(v, u, k) for u, v, k in base_edges}
    assert set(closed_inbound.edges(keys=True)) == {
        (v, u, k) for u, v, k in closed_edges
    }
    assert set(line_inbound.edges(keys=True)) == {(v, u, k) for u, v, k in line_edges}

    # closed hides R->C L2; the next L2 layer keeps C->R L2; reversing it
    # yields R->C L2 without reintroducing the hidden C->R L2 edge.
    assert set(layered.edges(keys=True)) == {("R", "C", "L2")}

    # Restored views remain structurally read-only.
    for view in (closed, line, route, layered, inbound, closed_inbound):
        with pytest.raises(nx.NetworkXError):
            view.add_node("X")
        with pytest.raises(nx.NetworkXError):
            view.add_edge("X", "Y")
        with pytest.raises(nx.NetworkXError):
            view.remove_node("C")
