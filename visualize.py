"""Shared code for flattening document embeddings to 2D and plotting them by domain and topic.

The notebooks import from here so every model is projected and plotted the same way.
"""

import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version
from IPython.display import HTML
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

SEED = 42
SVD_COMPONENTS = 50

# Colours and marker shapes are assigned in this fixed order to the sorted domains
# (or topics). Five or more colours can't all be told apart in a scatter plot (e.g.
# with colour vision deficiency), so plot_by_domain also gives each domain a shape.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DOMAIN_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "x"]
LINE_COLOR = "#d4d3cf"
MUTED_COLOR = "#6b6a64"
TEXT_COLOR = "#0b0b0b"

PLOT_SIZE = 700
PANEL_WIDTH = 400


def project_2d(embeddings: np.ndarray, doc_ids: pd.Index, seed: int = SEED) -> pd.DataFrame:
    """Project document embeddings to 2D.

    Rows are L2-normalised so document length doesn't dominate, reduced to
    SVD_COMPONENTS dimensions with truncated SVD, and then projected with t-SNE.
    ``doc_ids`` are "<domain>/<filename>" strings, one per row.
    """
    embeddings_normed = normalize(embeddings)
    embeddings_svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=seed).fit_transform(
        embeddings_normed
    )
    coords = TSNE(n_components=2, init="pca", random_state=seed).fit_transform(embeddings_svd)

    projection = pd.DataFrame(coords, index=doc_ids, columns=["x", "y"])
    projection["domain"] = projection.index.str.split("/").str[0]
    return projection


def plot_by_domain(projection: pd.DataFrame, texts: pd.Series, title: str) -> HTML:
    """Interactive scatter plot with one colour and marker shape per domain.

    Hovering a point shows its domain, and its text is shown in a panel to the right
    of the plot, cut off where the panel ends. Clicking a domain in the legend hides
    or shows it, and double-clicking shows that domain alone. ``texts`` is indexed
    like ``projection``.
    """
    domains = sorted(projection["domain"].unique())
    doc_texts = texts[projection.index]

    fig = go.Figure()
    for domain, color, symbol in zip(domains, COLORS, DOMAIN_SYMBOLS):
        in_domain = projection["domain"] == domain
        fig.add_trace(
            go.Scatter(
                x=projection.loc[in_domain, "x"],
                y=projection.loc[in_domain, "y"],
                text=projection.loc[in_domain, "domain"],
                customdata=doc_texts[in_domain],
                mode="markers",
                marker={
                    "size": 9,
                    "color": color,
                    "symbol": symbol,
                    "opacity": 0.85,
                    "line": {"width": 1, "color": "white"},
                },
                hovertemplate="%{text}<extra></extra>",
                name=domain,
            )
        )
    fig.update_layout(
        title={"text": title, "font": {"color": TEXT_COLOR}},
        width=PLOT_SIZE,
        height=PLOT_SIZE,
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": 1.02,
            "yanchor": "bottom",
            "font": {"color": TEXT_COLOR},
        },
        margin={"t": 110},
    )
    return _with_text_panel(fig, scrollable=False)


def plot_topics_by_domain(
    projection: pd.DataFrame, topics: pd.Series, texts: pd.Series, title: str
) -> HTML:
    """Interactive scatter plot coloured by topic and shaped by domain.

    ``topics`` holds the topic label of each document and is indexed like
    ``projection`` and ``texts``. The legend lists the topics and the domains.
    Clicking one hides or shows its documents, and double-clicking shows only that
    topic or domain. A document is shown when both its topic and its domain are.
    Hovering a point shows its topic and domain, and its full text is shown in a
    scrollable panel to the right of the plot.
    """
    topic_names = sorted(topics.unique())
    domains = sorted(projection["domain"].unique())
    topic_colors = dict(zip(topic_names, COLORS))
    domain_symbols = dict(zip(domains, DOMAIN_SYMBOLS))
    doc_topics = topics[projection.index]
    labels = doc_topics + " · " + projection["domain"]
    doc_texts = texts[projection.index]

    # One trace per topic and domain, so the legend can filter on both. The traces
    # are left out of the legend, which instead holds one empty key trace per
    # topic and per domain; _with_text_panel wires their clicks to the traces
    # through ``meta``.
    fig = go.Figure()
    for topic in topic_names:
        for domain in domains:
            selected = (doc_topics == topic) & (projection["domain"] == domain)
            if not selected.any():
                continue
            fig.add_trace(
                go.Scatter(
                    x=projection.loc[selected, "x"],
                    y=projection.loc[selected, "y"],
                    text=labels[selected],
                    customdata=doc_texts[selected],
                    mode="markers",
                    marker={
                        "size": 9,
                        "color": topic_colors[topic],
                        "symbol": domain_symbols[domain],
                        "opacity": 0.85,
                        "line": {"width": 1, "color": "white"},
                    },
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                    meta={"topic": topic, "domain": domain},
                )
            )
    for topic in topic_names:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 9, "color": topic_colors[topic], "symbol": "square"},
                name=topic,
                legendgroup="topics",
                legendgrouptitle_text="Topic (colour)",
                meta={"filter": "topic", "value": topic},
            )
        )
    for domain in domains:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 9, "color": MUTED_COLOR, "symbol": domain_symbols[domain]},
                name=domain,
                legendgroup="domains",
                legendgrouptitle_text="Domain (shape)",
                meta={"filter": "domain", "value": domain},
            )
        )
    fig.update_layout(
        title={"text": title, "font": {"color": TEXT_COLOR}},
        width=PLOT_SIZE,
        height=PLOT_SIZE + 150,
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": -0.02,
            "yanchor": "top",
            "font": {"color": TEXT_COLOR},
        },
        margin={"t": 60, "b": 190},
    )
    return _with_text_panel(fig)


def _with_text_panel(fig: go.Figure, scrollable: bool = True) -> HTML:
    """Style the plot and put it next to a panel showing the hovered document's text.

    Every point needs ``text`` (the panel heading) and ``customdata`` (the document
    text). A ``scrollable`` panel can be scrolled through the whole text, and
    clicking a point pins its text in the panel so the mouse can move there without
    passing over other points; clicking it again unpins it. Otherwise the text is
    cut off where the panel ends.

    Legend key traces with ``meta={"filter": category, "value": value}`` hide and
    show the traces whose ``meta[category]`` is that value, instead of themselves.
    """
    fig.update_layout(
        plot_bgcolor="white",
        hoverlabel={
            "bgcolor": "rgba(255, 255, 255, 0.8)",
            "font": {"size": 11, "color": TEXT_COLOR},
        },
    )
    axis = {
        "showticklabels": False,
        "showgrid": False,
        "zeroline": False,
        "linecolor": LINE_COLOR,
        "mirror": True,
    }
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis, scaleanchor="x")

    # Notebook frontends like VS Code run output scripts without waiting for external
    # scripts to load, so plotly.js is loaded here and the plot drawn once it's ready.
    # On hover, the text panel is filled with the point's label and text, unless a
    # point has been pinned by clicking it.
    plot_id = f"plot-{uuid.uuid4().hex}"
    plotly_url = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"
    fig_json = fig.to_json().replace("</", "<\\/")
    script = f"""
    (() => {{
        const fig = {fig_json};
        const render = () => Plotly.newPlot("{plot_id}", fig.data, fig.layout).then((plot) => {{
            const scrollable = {str(scrollable).lower()};
            let pinned = null;
            const show = (point, isPinned) => {{
                const panel = document.getElementById("{plot_id}-text");
                panel.querySelector(".label").textContent = point.text;
                panel.querySelector(".status").textContent = !scrollable ? ""
                    : isPinned ? "Pinned. Click the point again to unpin."
                    : "Click the point to pin its text.";
                panel.querySelector(".doc-text").textContent = point.customdata;
                panel.scrollTop = 0;
            }};
            const key = (point) => `${{point.curveNumber}}:${{point.pointNumber}}`;
            plot.on("plotly_hover", (event) => {{
                if (pinned === null) show(event.points[0], false);
            }});
            plot.on("plotly_click", (event) => {{
                if (!scrollable) return;
                const point = event.points[0];
                pinned = pinned === key(point) ? null : key(point);
                show(point, pinned !== null);
            }});

            // Legend key traces filter the data traces by category, and a data
            // trace is shown when none of its categories is hidden. A hidden key
            // is faded in the legend.
            const isKey = (trace) => trace.meta && trace.meta.filter !== undefined;
            const keyIndices = plot.data.flatMap((trace, i) => (isKey(trace) ? [i] : []));
            const hidden = {{}};
            const update = () => {{
                const isHidden = (category, value) => (hidden[category] || new Set()).has(value);
                const visible = plot.data.map((trace) => isKey(trace)
                    || !Object.entries(trace.meta || {{}}).some(([category, value]) => isHidden(category, value)));
                Plotly.restyle(plot, {{visible: visible}});
                Plotly.restyle(plot, {{
                    opacity: keyIndices.map((i) => isHidden(plot.data[i].meta.filter, plot.data[i].meta.value) ? 0.3 : 1),
                }}, keyIndices);
            }};
            plot.on("plotly_legendclick", (event) => {{
                const meta = plot.data[event.curveNumber].meta;
                if (!isKey(plot.data[event.curveNumber])) return true;
                const values = hidden[meta.filter] = hidden[meta.filter] || new Set();
                values.has(meta.value) ? values.delete(meta.value) : values.add(meta.value);
                update();
                return false;
            }});
            plot.on("plotly_legenddoubleclick", (event) => {{
                const meta = plot.data[event.curveNumber].meta;
                if (!isKey(plot.data[event.curveNumber])) return true;
                const others = keyIndices
                    .map((i) => plot.data[i].meta)
                    .filter((other) => other.filter === meta.filter && other.value !== meta.value)
                    .map((other) => other.value);
                const current = hidden[meta.filter] || new Set();
                const isolated = current.size === others.length && others.every((value) => current.has(value));
                hidden[meta.filter] = isolated ? new Set() : new Set(others);
                update();
                return false;
            }});
        }});
        if (window.Plotly) {{
            render();
            return;
        }}
        let loader = document.querySelector('script[src="{plotly_url}"]');
        if (!loader) {{
            loader = document.createElement("script");
            loader.src = "{plotly_url}";
            document.head.appendChild(loader);
        }}
        loader.addEventListener("load", render);
    }})();
    """
    return HTML(
        f"""
        <div style="display: flex; gap: 16px; align-items: flex-start;">
          <div id="{plot_id}" style="width: {fig.layout.width}px; height: {fig.layout.height}px;"></div>
          <div id="{plot_id}-text" style="width: {PANEL_WIDTH}px; height: {PLOT_SIZE - 120}px;
               margin-top: 100px; padding: 12px; overflow-y: {"auto" if scrollable else "hidden"}; box-sizing: border-box;
               border: 1px solid {LINE_COLOR}; background: white; color: {TEXT_COLOR};
               font-family: sans-serif; font-size: 12px;">
            <div class="label" style="font-weight: bold;"></div>
            <div class="status" style="color: {MUTED_COLOR}; font-size: 11px; margin-bottom: 8px;"></div>
            <div class="doc-text" style="white-space: pre-wrap;">Hover a point to see its text.</div>
          </div>
        </div>
        <script>{script}</script>
        """
    )
