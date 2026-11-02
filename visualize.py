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

HIGHLIGHT_COLOR = "#2a78d6"
BACKGROUND_COLOR = "#d4d3cf"
TEXT_COLOR = "#0b0b0b"

SNIPPET_CHARS = 1000
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


def snippet(text: str) -> str:
    """The first SNIPPET_CHARS characters of the text."""
    return text[:SNIPPET_CHARS] + ("…" if len(text) > SNIPPET_CHARS else "")


def plot_by_domain(projection: pd.DataFrame, texts: pd.Series, title: str) -> HTML:
    """Interactive scatter plot highlighting one domain against the rest in grey.

    The highlighted domain is picked from a dropdown. Hovering a point shows its
    domain, and the start of its text is shown in a panel to the right of the plot.
    ``texts`` is indexed like ``projection``. The grey layer holds every document,
    so only the highlighted points change when another domain is picked.
    """
    domains = sorted(projection["domain"].unique())
    snippets = texts[projection.index].map(snippet)

    def highlighted(domain: str) -> dict:
        in_domain = projection["domain"] == domain
        return {
            "x": [projection.loc[in_domain, "x"].tolist()],
            "y": [projection.loc[in_domain, "y"].tolist()],
            "text": [projection.loc[in_domain, "domain"].tolist()],
            "customdata": [snippets[in_domain].tolist()],
        }

    initial = highlighted(domains[0])
    fig = go.Figure(
        [
            go.Scatter(
                x=projection["x"],
                y=projection["y"],
                text=projection["domain"],
                customdata=snippets,
                mode="markers",
                marker={"size": 6, "color": BACKGROUND_COLOR},
                hovertemplate="%{text}<extra></extra>",
                name="All documents",
            ),
            go.Scatter(
                x=initial["x"][0],
                y=initial["y"][0],
                text=initial["text"][0],
                customdata=initial["customdata"][0],
                mode="markers",
                marker={
                    "size": 10,
                    "color": HIGHLIGHT_COLOR,
                    "line": {"width": 1, "color": "white"},
                },
                hovertemplate="%{text}<extra></extra>",
                name="Selected domain",
            ),
        ]
    )
    fig.update_layout(
        title={"text": title, "font": {"color": TEXT_COLOR}},
        width=PLOT_SIZE,
        height=PLOT_SIZE,
        plot_bgcolor="white",
        showlegend=False,
        hoverlabel={
            "bgcolor": "rgba(255, 255, 255, 0.8)",
            "font": {"size": 11, "color": TEXT_COLOR},
        },
        updatemenus=[
            {
                "buttons": [
                    {"label": domain, "method": "restyle", "args": [highlighted(domain), [1]]}
                    for domain in domains
                ],
                "x": 0,
                "xanchor": "left",
                "y": 1.02,
                "yanchor": "bottom",
            }
        ],
        margin={"t": 110},
    )
    axis = {
        "showticklabels": False,
        "showgrid": False,
        "zeroline": False,
        "linecolor": BACKGROUND_COLOR,
        "mirror": True,
    }
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis, scaleanchor="x")

    # Notebook frontends like VS Code run output scripts without waiting for external
    # scripts to load, so plotly.js is loaded here and the plot drawn once it's ready.
    # On hover, the text panel is filled with the point's domain and snippet.
    plot_id = f"plot-{uuid.uuid4().hex}"
    plotly_url = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"
    fig_json = fig.to_json().replace("</", "<\\/")
    script = f"""
    (() => {{
        const fig = {fig_json};
        const render = () => Plotly.newPlot("{plot_id}", fig.data, fig.layout).then((plot) => {{
            plot.on("plotly_hover", (event) => {{
                const panel = document.getElementById("{plot_id}-text");
                const point = event.points[0];
                panel.querySelector(".domain").textContent = point.text;
                panel.querySelector(".snippet").textContent = point.customdata;
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
               border: 1px solid {BACKGROUND_COLOR}; background: white; color: {TEXT_COLOR};
               font-family: sans-serif; font-size: 12px;">
            <div class="domain" style="font-weight: bold; margin-bottom: 8px;"></div>
            <div class="snippet" style="white-space: pre-wrap;">Hover a point to see its text.</div>
          </div>
        </div>
        <script>{script}</script>
        """
    )
