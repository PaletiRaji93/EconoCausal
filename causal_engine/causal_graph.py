
"""Document the assumed causal relationships in EconoCausal."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "reports" / "causal_graph.png"


def build_causal_graph() -> nx.DiGraph:
    """Build the assumed causal DAG for the binary campaign treatment."""
    graph = nx.DiGraph()

    edges = [
        ("Age", "Discount assignment"),
        ("Age", "Conversion"),
        ("Past purchases", "Discount assignment"),
        ("Past purchases", "Conversion"),
        ("Previous spend", "Conversion"),
        ("Loyalty score", "Discount assignment"),
        ("Loyalty score", "Conversion"),
        ("Discount assignment", "Conversion"),
        ("Discount assignment", "Discount amount"),
        ("Discount amount", "Conversion"),
        ("Conversion", "Order value"),
    ]

    graph.add_edges_from(edges)
    return graph


def save_causal_graph(graph: nx.DiGraph) -> None:
    """Save the graph as a PNG for project documentation."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    positions = {
        "Age": (0, 2),
        "Past purchases": (0, 1),
        "Previous spend": (0, 0),
        "Loyalty score": (0, -1),
        "Discount assignment": (2, 1),
        "Discount amount": (4, 1),
        "Conversion": (4, -1),
        "Order value": (6, -1),
    }

    plt.figure(figsize=(13, 7))

    nx.draw_networkx(
        graph,
        pos=positions,
        node_color="lightblue",
        node_size=2600,
        font_size=8,
        arrows=True,
        arrowsize=18,
        edge_color="gray",
    )

    plt.title("EconoCausal: Assumed Retail Campaign Causal DAG")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Causal graph saved to: {OUTPUT_PATH}")


def main() -> None:
    graph = build_causal_graph()
    save_causal_graph(graph)
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")


if __name__ == "__main__":
    main()
