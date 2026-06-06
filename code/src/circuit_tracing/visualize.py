"""
Interactive HTML dashboard for circuit tracing results.

Generates a self-contained HTML file with plotly charts:
  1. Ablation sweep curves  (faithfulness / completeness vs k)
  2. Layer attribution heatmap
  3. Circuit overlap matrix
  4. Top neurons table
  5. Per-example model behaviour  (full vs circuit vs ablated)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


# ──────────────────────────────────────────────────────────────────────
# 1. Ablation sweep curves
# ──────────────────────────────────────────────────────────────────────

def plot_ablation_sweep(
    sweep_data: Dict[str, List[dict]],
    title: str = "Ablation Sweep",
) -> go.Figure:
    """
    sweep_data: {property_tag: [{k, faithfulness, completeness, ...}, ...]}
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Faithfulness (higher=better)", "Completeness (lower=better)"),
        shared_xaxes=True,
    )

    colors = px.colors.qualitative.Set2
    for i, (tag, rows) in enumerate(sweep_data.items()):
        ks = [r["k"] for r in rows]
        faith = [r["faithfulness"] for r in rows]
        compl = [r["completeness"] for r in rows]
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=ks, y=faith, mode="lines+markers", name=tag,
            line=dict(color=color), legendgroup=tag,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=ks, y=compl, mode="lines+markers", name=tag,
            line=dict(color=color, dash="dash"), legendgroup=tag,
            showlegend=False,
        ), row=1, col=2)

    # reference lines
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray",
                  annotation_text="perfect", row=1, col=1)
    fig.add_hline(y=0.0, line_dash="dot", line_color="gray",
                  annotation_text="perfect", row=1, col=2)

    fig.update_xaxes(title_text="Circuit size (k)", type="log")
    fig.update_layout(title=title, height=450, template="plotly_white")
    return fig


# ──────────────────────────────────────────────────────────────────────
# 2. Layer attribution heatmap
# ──────────────────────────────────────────────────────────────────────

def plot_layer_attributions(
    layer_attrs: Dict[str, Dict[int, float]],
    title: str = "Per-layer attribution (total abs score)",
) -> go.Figure:
    """
    layer_attrs: {property_tag: {layer_idx: total_attribution}}
    """
    tags = list(layer_attrs.keys())
    all_layers = sorted(set(l for d in layer_attrs.values() for l in d))

    z = []
    for tag in tags:
        row = [layer_attrs[tag].get(l, 0.0) for l in all_layers]
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"L{l}" for l in all_layers],
        y=tags,
        colorscale="YlOrRd",
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        hovertemplate="Property: %{y}<br>Layer: %{x}<br>Attribution: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=title, height=max(300, 60 * len(tags)),
        template="plotly_white",
        xaxis_title="Layer", yaxis_title="Property",
    )
    return fig


# ──────────────────────────────────────────────────────────────────────
# 3. Circuit overlap matrix
# ──────────────────────────────────────────────────────────────────────

def plot_overlap_matrix(
    circuits: Dict[str, set],
    title: str = "Circuit overlap (Jaccard similarity)",
) -> go.Figure:
    """circuits: {property_tag: set of (layer, neuron) tuples}"""
    tags = list(circuits.keys())
    n = len(tags)
    z = [[0.0] * n for _ in range(n)]
    text = [[""] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            shared = circuits[tags[i]] & circuits[tags[j]]
            union  = circuits[tags[i]] | circuits[tags[j]]
            jacc = len(shared) / len(union) if union else 0.0
            z[i][j] = jacc
            text[i][j] = f"{len(shared)}"

    # short labels
    short = [t.split("/")[-1] if "/" in t else t for t in tags]

    fig = go.Figure(go.Heatmap(
        z=z, x=short, y=short,
        colorscale="Blues", zmin=0, zmax=1,
        text=text, texttemplate="%{text}",
        hovertemplate="%{y} ∩ %{x}<br>Jaccard: %{z:.3f}<br>Shared: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title=title, height=max(400, 60 * n), width=max(500, 60 * n),
        template="plotly_white",
    )
    return fig


# ──────────────────────────────────────────────────────────────────────
# 4. Top neurons table
# ──────────────────────────────────────────────────────────────────────

def plot_top_neurons(
    attributions: Dict[int, "torch.Tensor"],
    top_n: int = 30,
    title: str = "Top neurons by attribution",
) -> go.Figure:
    """attributions: {layer: [d_ffn] tensor}"""
    entries = []
    for layer, scores in attributions.items():
        for neuron_idx in range(scores.shape[0]):
            val = scores[neuron_idx].item()
            if abs(val) > 0:
                entries.append((abs(val), val, layer, neuron_idx))

    entries.sort(reverse=True)
    top = entries[:top_n]

    fig = go.Figure(go.Bar(
        x=[f"L{l}:N{n}" for _, _, l, n in top],
        y=[v for _, v, _, _ in top],
        marker_color=["#e74c3c" if v > 0 else "#3498db" for _, v, _, _ in top],
        hovertemplate="Layer %{customdata[0]}, Neuron %{customdata[1]}<br>"
                      "Score: %{y:.4f}<extra></extra>",
        customdata=[(l, n) for _, _, l, n in top],
    ))
    fig.update_layout(
        title=f"{title} (top {top_n})",
        xaxis_title="Layer:Neuron", yaxis_title="Attribution score",
        height=400, template="plotly_white",
    )
    return fig


# ──────────────────────────────────────────────────────────────────────
# 5. Per-example behaviour
# ──────────────────────────────────────────────────────────────────────

def plot_example_behaviour(
    example_data: List[dict],
    title: str = "Per-example metric values",
) -> go.Figure:
    """
    example_data: [{"input": ..., "m_full": ..., "m_circuit": ..., "m_complement": ...}, ...]
    """
    labels = [d.get("input", f"ex {i}")[:25] for i, d in enumerate(example_data)]
    m_full = [d["m_full"] for d in example_data]
    m_circ = [d["m_circuit"] for d in example_data]
    m_comp = [d["m_complement"] for d in example_data]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Full model", x=labels, y=m_full,
                         marker_color="#2ecc71"))
    fig.add_trace(go.Bar(name="Circuit only", x=labels, y=m_circ,
                         marker_color="#3498db"))
    fig.add_trace(go.Bar(name="Circuit ablated", x=labels, y=m_comp,
                         marker_color="#e74c3c"))

    fig.update_layout(
        barmode="group", title=title,
        yaxis_title="Logit difference",
        height=400, template="plotly_white",
    )
    return fig


# ──────────────────────────────────────────────────────────────────────
# 6. Layer distribution per property
# ──────────────────────────────────────────────────────────────────────

def plot_layer_distributions(
    circuits: Dict[str, set],
    n_layers: int = 28,
    title: str = "Circuit neurons per layer",
) -> go.Figure:
    """circuits: {property_tag: set of (layer, neuron) tuples}"""
    fig = go.Figure()
    colors = px.colors.qualitative.Set2

    for i, (tag, nodes) in enumerate(circuits.items()):
        counts = [0] * n_layers
        for l, _ in nodes:
            if l < n_layers:
                counts[l] += 1
        fig.add_trace(go.Bar(
            name=tag, x=list(range(n_layers)), y=counts,
            marker_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        barmode="group", title=title,
        xaxis_title="Layer", yaxis_title="# neurons in circuit",
        height=400, template="plotly_white",
    )
    return fig


# ──────────────────────────────────────────────────────────────────────
# 7. Edge layer-pair heatmap
# ──────────────────────────────────────────────────────────────────────

def plot_edge_layer_heatmap(
    edge_scores: Dict,
    title: str = "Edge weight by layer pair",
) -> go.Figure:
    """
    Heatmap of total absolute edge weight between each source/target layer pair.

    edge_scores: { ((src_l, src_n), (tgt_l, tgt_n)): score }
    """
    # accumulate per layer pair
    pair_sums: Dict[tuple, float] = {}
    all_layers = set()
    for (src, tgt), score in edge_scores.items():
        sl, tl = src[0], tgt[0]
        all_layers.add(sl)
        all_layers.add(tl)
        pair_sums[(sl, tl)] = pair_sums.get((sl, tl), 0.0) + abs(score)

    layers = sorted(all_layers)
    n = len(layers)
    l2i = {l: i for i, l in enumerate(layers)}

    z = [[0.0] * n for _ in range(n)]
    for (sl, tl), val in pair_sums.items():
        z[l2i[tl]][l2i[sl]] = val

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"L{l}" for l in layers],
        y=[f"L{l}" for l in layers],
        colorscale="Viridis",
        hovertemplate="Source: %{x}<br>Target: %{y}<br>Total weight: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Source layer",
        yaxis_title="Target layer",
        height=max(400, 30 * n),
        width=max(500, 30 * n),
        template="plotly_white",
    )
    return fig


# ──────────────────────────────────────────────────────────────────────
# 8. Edge Sankey diagram
# ──────────────────────────────────────────────────────────────────────

def plot_edge_sankey(
    edge_scores: Dict,
    top_n: int = 80,
    title: str = "Circuit edge flow (Sankey)",
) -> go.Figure:
    """
    Sankey diagram of the top edges in the circuit.

    Nodes are neurons (L{layer}:N{idx}), grouped by layer.
    Links are edges with width proportional to absolute score.
    """
    # take top edges
    sorted_edges = sorted(edge_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    top_edges = sorted_edges[:top_n]

    if not top_edges:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white")
        return fig

    # collect unique nodes
    node_set = set()
    for (src, tgt), _ in top_edges:
        node_set.add(src)
        node_set.add(tgt)
    nodes = sorted(node_set)
    node_idx = {n: i for i, n in enumerate(nodes)}

    labels = [f"L{l}:N{n}" for l, n in nodes]

    sources = [node_idx[e[0]] for e, _ in top_edges]
    targets = [node_idx[e[1]] for e, _ in top_edges]
    values = [abs(v) for _, v in top_edges]
    colors = [
        "rgba(231,76,60,0.5)" if v > 0 else "rgba(52,152,219,0.5)"
        for _, v in top_edges
    ]

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=20,
            label=labels,
            color="rgba(100,100,100,0.8)",
        ),
        link=dict(
            source=sources, target=targets,
            value=values, color=colors,
        ),
    ))
    fig.update_layout(
        title=title,
        height=max(500, 20 * len(nodes)),
        template="plotly_white",
    )
    return fig


# ──────────────────────────────────────────────────────────────────────
# Assemble full dashboard
# ──────────────────────────────────────────────────────────────────────

def build_dashboard(
    sweep_data: Dict[str, List[dict]],
    layer_attrs: Dict[str, Dict[int, float]],
    circuits: Dict[str, set],
    example_data: Optional[Dict[str, List[dict]]] = None,
    top_neurons_per_prop: Optional[Dict[str, Dict[int, "torch.Tensor"]]] = None,
    edge_data: Optional[Dict[str, Dict]] = None,
    n_layers: int = 28,
    title: str = "Circuit Tracing Dashboard",
) -> str:
    """
    Build a self-contained HTML dashboard.

    Returns the HTML string.
    """
    figs = []

    # 1. ablation sweep
    if sweep_data:
        figs.append(("Ablation Sweep", plot_ablation_sweep(sweep_data)))

    # 2. layer attributions
    if layer_attrs:
        figs.append(("Layer Attributions", plot_layer_attributions(layer_attrs)))

    # 3. layer distribution
    if circuits:
        figs.append(("Layer Distribution", plot_layer_distributions(circuits, n_layers)))

    # 4. overlap matrix
    if circuits and len(circuits) > 1:
        figs.append(("Circuit Overlap", plot_overlap_matrix(circuits)))

    # 5. top neurons per property
    if top_neurons_per_prop:
        for tag, attrs in top_neurons_per_prop.items():
            figs.append((f"Top Neurons: {tag}", plot_top_neurons(attrs, title=f"Top neurons — {tag}")))

    # 6. edge visualizations
    if edge_data:
        for tag, scores in edge_data.items():
            if scores:
                figs.append((
                    f"Edge Flow: {tag}",
                    plot_edge_sankey(scores, title=f"Edge flow (Sankey) — {tag}"),
                ))
                figs.append((
                    f"Edge Layers: {tag}",
                    plot_edge_layer_heatmap(scores, title=f"Edge layer heatmap — {tag}"),
                ))

    # 7. per-example behaviour
    if example_data:
        for tag, data in example_data.items():
            if data:
                figs.append((f"Examples: {tag}", plot_example_behaviour(data, title=f"Per-example — {tag}")))

    # ── assemble HTML ──
    chart_divs = []
    for i, (section_title, fig) in enumerate(figs):
        div_id = f"chart_{i}"
        chart_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)
        chart_divs.append(f"""
        <div class="chart-section">
            <h2>{section_title}</h2>
            {chart_html}
        </div>
        """)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #fafafa;
            color: #333;
        }}
        h1 {{
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
        }}
        .chart-section {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chart-section h2 {{
            margin-top: 0;
            color: #555;
            font-size: 1.1em;
        }}
        .info {{
            background: #e8f4fd;
            border-left: 4px solid #3498db;
            padding: 12px 16px;
            margin: 16px 0;
            border-radius: 0 4px 4px 0;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="info">
        <strong>How to read this dashboard:</strong><br>
        <b>Ablation Sweep</b> — faithfulness should approach 1 and completeness should approach 0
        as circuit size k increases. The curve's knee shows the minimal circuit size.<br>
        <b>Layer Attributions</b> — brighter = more important layer for that property.<br>
        <b>Circuit Overlap</b> — high Jaccard = two properties share neurons (common computation).<br>
        <b>Top Neurons</b> — red = positive attribution, blue = negative.
        Click/hover for details.<br>
        <b>Edge Flow</b> — Sankey diagram of neuron-to-neuron edges; width = attribution flow magnitude.<br>
        <b>Edge Layers</b> — which layer pairs have the strongest edges.<br>
        <b>Per-example</b> — green (full model) should match blue (circuit only);
        red (circuit ablated) should drop to zero.
    </div>
    {"".join(chart_divs)}
</body>
</html>"""
    return html


def save_dashboard(html: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    print(f"Dashboard saved to {path}")
