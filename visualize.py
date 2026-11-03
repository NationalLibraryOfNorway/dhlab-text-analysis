"""Shared code for flattening document embeddings to 2D and plotting them by domain.

The embedding notebooks import from here so every model is projected and plotted
the same way.
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

# One colour and one marker shape per domain, assigned in this fixed order to the
# domains sorted by name. Five colours can't all be told apart in a scatter plot
# (e.g. with colour vision deficiency), so the shape is a second cue.
DOMAIN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
DOMAIN_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "x"]
LINE_COLOR = "#d4d3cf"
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

    Hovering a point shows its domain, and its full text is shown in a scrollable
    panel to the right of the plot. Clicking a point pins its text in the panel so
    the mouse can move there without passing over other points; clicking it again
    unpins it. Clicking a domain in the legend hides or shows it, and
    double-clicking shows that domain alone. ``texts`` is indexed like ``projection``.
    """
    domains = sorted(projection["domain"].unique())
    doc_texts = texts[projection.index]

    fig = go.Figure()
    for domain, color, symbol in zip(domains, DOMAIN_COLORS, DOMAIN_SYMBOLS):
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
        plot_bgcolor="white",
        hoverlabel={
            "bgcolor": "rgba(255, 255, 255, 0.8)",
            "font": {"size": 11, "color": TEXT_COLOR},
        },
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
    # On hover, the text panel is filled with the point's domain and text, unless a
    # point has been pinned by clicking it.
    plot_id = f"plot-{uuid.uuid4().hex}"
    plotly_url = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"
    fig_json = fig.to_json().replace("</", "<\\/")
    script = f"""
    (() => {{
        const fig = {fig_json};
        const render = () => Plotly.newPlot("{plot_id}", fig.data, fig.layout).then((plot) => {{
            let pinned = null;
            const show = (point, isPinned) => {{
                const panel = document.getElementById("{plot_id}-text");
                panel.querySelector(".domain").textContent = point.text;
                panel.querySelector(".status").textContent = isPinned
                    ? "Pinned. Click the point again to unpin."
                    : "Click the point to pin its text.";
                panel.querySelector(".doc-text").textContent = point.customdata;
                panel.scrollTop = 0;
            }};
            const key = (point) => `${{point.curveNumber}}:${{point.pointNumber}}`;
            plot.on("plotly_hover", (event) => {{
                if (pinned === null) show(event.points[0], false);
            }});
            plot.on("plotly_click", (event) => {{
                const point = event.points[0];
                pinned = pinned === key(point) ? null : key(point);
                show(point, pinned !== null);
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
          <div id="{plot_id}" style="width: {PLOT_SIZE}px; height: {PLOT_SIZE}px;"></div>
          <div id="{plot_id}-text" style="width: {PANEL_WIDTH}px; height: {PLOT_SIZE - 120}px;
               margin-top: 100px; padding: 12px; overflow-y: auto; box-sizing: border-box;
               border: 1px solid {LINE_COLOR}; background: white; color: {TEXT_COLOR};
               font-family: sans-serif; font-size: 12px;">
            <div class="domain" style="font-weight: bold;"></div>
            <div class="status" style="color: #6b6a64; font-size: 11px; margin-bottom: 8px;"></div>
            <div class="doc-text" style="white-space: pre-wrap;">Hover a point to see its text.</div>
          </div>
        </div>
        <script>{script}</script>
        """
    )
