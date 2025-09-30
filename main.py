# Histogram Explorer — Streamlit + Plotly (Multi‑Histogram + Interval Metrics)
# ---------------------------------------------------------------------------
# Features
# - Multiple histogram sessions, each inside its own expander
# - Per‑session: label, file upload, numeric column, binning controls
# - Output: Count or Normalized (0–1)
# - Type: Standard / Cumulative / Inverse cumulative
# - Optional X‑range and forced bin START/END for all modes
# - Per‑session CSV export of the binned table (with metadata)
# - NEW: Interval analytics
#   • X‑interval metrics: sum of Count/Probability (Standard), ΔProbability (Cumulative)
#   • Y‑limit metrics: intercept bins (Standard); below/above‑limit split (Cumulative)

import io
import re
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------
# Helpers
# ---------------------------

def load_table(uploaded_file: Optional[io.BytesIO]) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            content = uploaded_file.read()
            try:
                df = pd.read_csv(io.BytesIO(content), sep=";")
                if df.shape[1] == 1:
                    df = pd.read_csv(io.BytesIO(content), sep=",")
            except Exception:
                df = pd.read_csv(io.BytesIO(content))
        elif name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        elif name.endswith((".json", ".ndjson")):
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload CSV, Excel, or JSON.")
            return None
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None
    df.columns = [str(c).strip() for c in df.columns]
    return df


def numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def compute_default_range(series: pd.Series) -> Tuple[float, float]:
    col_data = pd.to_numeric(series, errors="coerce").dropna()
    if col_data.empty:
        return 0.0, 1.0
    return float(np.nanmin(col_data)), float(np.nanmax(col_data))


def build_histogram(
    df: pd.DataFrame,
    col: str,
    bin_mode: str,
    bin_width: Optional[float],
    n_bins: Optional[int],
    hist_output: str,
    hist_type: str,
    xrange: Optional[Tuple[float, float]],
    start_override: Optional[float],
    end_override: Optional[float],
) -> go.Figure:
    histnorm = None
    if hist_output == "Normalized (0–1)":
        histnorm = "probability"

    base = px.histogram(
        df,
        x=col,
        nbins=n_bins if (bin_mode == "Number of bins" and n_bins) else None,
        histnorm=histnorm,
        opacity=0.85,
    )

    x = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
    has_data = x.size > 0
    xmin = np.min(x) if has_data else 0.0
    xmax = np.max(x) if has_data else 1.0

    if xrange is not None:
        xmin = max(xmin, xrange[0])
        xmax = min(xmax, xrange[1])

    xbins = {}
    if start_override is not None:
        xbins["start"] = float(start_override)
    if end_override is not None:
        xbins["end"] = float(end_override)

    if bin_mode == "Bin width" and (bin_width is not None) and (bin_width > 0):
        if "start" not in xbins:
            start = np.floor(xmin / bin_width) * bin_width
            xbins["start"] = start
        if "end" not in xbins:
            end = np.ceil(xmax / bin_width) * bin_width
            xbins["end"] = end
        xbins["size"] = bin_width

    if xbins:
        base.update_traces(xbins=xbins)

    if hist_type == "Cumulative":
        base.update_traces(cumulative_enabled=True, cumulative_direction="increasing")
    elif hist_type == "Inverse cumulative":
        base.update_traces(cumulative_enabled=True, cumulative_direction="decreasing")

    if xrange is not None:
        base.update_xaxes(range=list(xrange))

    base.update_layout(
        margin=dict(l=40, r=30, t=60, b=40),
        bargap=0.02,
        title=f"Histogram — {col}",
        xaxis_title=col,
        yaxis_title=("Probability" if hist_output == "Normalized (0–1)" else "Count"),
        hovermode="x unified",
    )

    return go.Figure(base)


def _edges_from_binwidth(start: float, end: float, size: float) -> np.ndarray:
    if size <= 0:
        return np.array([start, end])
    n = int(np.ceil((end - start) / size))
    n = max(n, 1)
    edges = start + np.arange(n + 1) * size
    if not np.isclose(edges[-1], end):
        edges = np.append(edges, end)
    return edges


def compute_histogram_table(
    df: pd.DataFrame,
    col: str,
    label: str,
    dataset_name: str,
    bin_mode: str,
    bin_width: Optional[float],
    n_bins: Optional[int],
    hist_output: str,
    hist_type: str,
    xrange: Optional[Tuple[float, float]],
    start_override: Optional[float],
    end_override: Optional[float],
) -> pd.DataFrame:
    """Return a per-bin table matching the app settings.

    Columns: bin_left, bin_right, bin_center, count, probability, y
    + metadata columns for reproducibility.
    """
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return pd.DataFrame(columns=[
            "bin_left","bin_right","bin_center","count","probability","y",
            "session_label","dataset","column","bin_mode","hist_output","hist_type",
            "bin_width","n_bins","start_override","end_override","xrange_min","xrange_max"
        ])

    if xrange is not None:
        s = s[(s >= xrange[0]) & (s <= xrange[1])]
        if s.empty:
            return pd.DataFrame(columns=[
                "bin_left","bin_right","bin_center","count","probability","y",
                "session_label","dataset","column","bin_mode","hist_output","hist_type",
                "bin_width","n_bins","start_override","end_override","xrange_min","xrange_max"
            ])

    if bin_mode == "Bin width" and (bin_width is not None) and (bin_width > 0):
        data_min, data_max = float(s.min()), float(s.max())
        start = float(start_override) if start_override is not None else np.floor(data_min / bin_width) * bin_width
        end = float(end_override) if end_override is not None else np.ceil(data_max / bin_width) * bin_width
        if start >= end:
            end = start + bin_width
        edges = _edges_from_binwidth(start, end, bin_width)
        counts, edges = np.histogram(s, bins=edges)
    elif bin_mode == "Number of bins" and (n_bins is not None):
        data_min, data_max = float(s.min()), float(s.max())
        start = float(start_override) if start_override is not None else data_min
        end = float(end_override) if end_override is not None else data_max
        if start == end:
            end = start + 1e-9
        counts, edges = np.histogram(s, bins=int(n_bins), range=(start, end))
    else:  # Auto
        rng = None
        if xrange is not None:
            rng = (float(xrange[0]), float(xrange[1]))
        counts, edges = np.histogram(s, bins="auto", range=rng)

    total = counts.sum()
    probability = (counts / total) if total > 0 else np.zeros_like(counts, dtype=float)

    # y as plotted (depends on hist_output + hist_type)
    y = probability if hist_output == "Normalized (0–1)" else counts.astype(float)
    if hist_type == "Cumulative":
        y = np.cumsum(y)
    elif hist_type == "Inverse cumulative":
        y = np.cumsum(y[::-1])[::-1]

    centers = (edges[:-1] + edges[1:]) / 2.0

    meta = {
        "session_label": label,
        "dataset": dataset_name,
        "column": col,
        "bin_mode": bin_mode,
        "hist_output": hist_output,
        "hist_type": hist_type,
        "bin_width": bin_width,
        "n_bins": n_bins,
        "start_override": start_override,
        "end_override": end_override,
        "xrange_min": (None if xrange is None else float(xrange[0])),
        "xrange_max": (None if xrange is None else float(xrange[1])),
    }

    table = pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "bin_center": centers,
        "count": counts,
        "probability": probability,
        "y": y,
    })
    for k, v in meta.items():
        table[k] = v
    return table


def _interval_overlap(mask_series: pd.Series, left: float, right: float) -> pd.Series:
    """Return boolean mask for bins overlapping [left, right]."""
    return (mask_series["bin_right"] > left) & (mask_series["bin_left"] < right)


# ---------------------------
# App UI
# ---------------------------

st.set_page_config(page_title="Histogram Explorer — Multiple", layout="wide")
st.title("Histogram Explorer — Multiple")

st.sidebar.markdown("### Global Settings")
num_sessions = st.sidebar.number_input(
    "Number of histograms", min_value=1, max_value=12, value=2, step=1, key="num_sessions"
)
show_global_help = st.sidebar.checkbox("Show quick how‑to", value=True)

if show_global_help:
    st.sidebar.info(
        "Each histogram has its own expander below. Upload a file, choose the column and options."
        "Use the 'X‑interval metrics' and 'Y‑limit / intercept' sections for calculations."
    )

# Seed per‑session label defaults
for i in range(int(num_sessions)):
    st.session_state.setdefault(f"label_{i}", f"Histogram {i+1}")

# Render sessions
for i in range(int(num_sessions)):
    session_default_label = st.session_state.get(f"label_{i}", f"Histogram {i+1}")
    with st.expander(f"Session {i+1}: {session_default_label}", expanded=(i == 0)):
        # Top row: label + toggles
        ctop1, ctop2 = st.columns([1.6, 1])
        with ctop1:
            label = st.text_input("Label", value=session_default_label, key=f"label_input_{i}")
            st.session_state[f"label_{i}"] = label
        with ctop2:
            show_table = st.checkbox("Show data table", value=False, key=f"table_{i}")
            show_stats = st.checkbox("Show quick stats", value=True, key=f"stats_{i}")

        # File upload inside the session
        uploaded = st.file_uploader(
            "Upload CSV, Excel, or JSON",
            type=["csv", "xls", "xlsx", "json", "ndjson"],
            key=f"uploader_{i}",
            help="CSV delimiter is auto‑detected (semicolon/comma).",
        )

        # Load data or build a per‑session demo
        if uploaded is not None:
            df = load_table(uploaded)
            dataset_name = uploaded.name
        else:
            st.info("No file uploaded for this session. Using a small demo dataset.")
            rng = np.random.default_rng(100 + i)
            df = pd.DataFrame({
                "normal": rng.normal(loc=50, scale=10, size=2000),
                "uniform": rng.uniform(low=10, high=90, size=2000),
            })
            dataset_name = "demo"

        if df is None or df.empty:
            st.warning("Empty dataset.")
            continue

        num_cols = numeric_columns(df)
        if not num_cols:
            st.error("No numeric columns found in this dataset.")
            continue

        # Controls row
        st.markdown("**Controls**")
        col1, col2, col3, col4 = st.columns([1.6, 1.4, 1.2, 1.2])
        with col1:
            chosen = st.selectbox("Numeric column", options=num_cols, index=0, key=f"col_{i}")
        with col2:
            bin_mode = st.selectbox("Binning mode", ["Auto", "Bin width", "Number of bins"], index=0, key=f"binmode_{i}")
        with col3:
            hist_output = st.selectbox("Output", ["Count", "Normalized (0–1)"], index=0, key=f"out_{i}")
        with col4:
            hist_type = st.selectbox("Type", ["Standard", "Cumulative", "Inverse cumulative"], index=0, key=f"type_{i}")

        # Dynamic bin inputs
        bin_width = None
        n_bins = None
        if bin_mode == "Bin width":
            bin_width = st.number_input("Bin width", min_value=1e-12, value=1.0, step=1.0, format="%f", key=f"binw_{i}")
        elif bin_mode == "Number of bins":
            n_bins = st.number_input("Number of bins", min_value=1, value=30, step=1, key=f"nbins_{i}")

        # Optional X range
        with st.expander("Optional: Limit X range"):
            default_min, default_max = compute_default_range(df[chosen])
            use_range = st.checkbox("Enable X range filter", value=False, key=f"use_range_{i}")
            xrange = None
            if use_range:
                sub1, sub2 = st.columns(2)
                xmin = sub1.number_input("X min", value=default_min, key=f"xmin_{i}")
                xmax = sub2.number_input("X max", value=default_max, key=f"xmax_{i}")
                if xmin < xmax:
                    xrange = (xmin, xmax)
                else:
                    st.warning("X min must be less than X max.")

        # Force start/end
        with st.expander("Optional: Force bin start/end (applies to all modes)"):
            force_edges = st.checkbox("Enable bin edge overrides", value=False, key=f"force_edges_{i}")
            start_override = None
            end_override = None
            if force_edges:
                sm1, sm2 = st.columns(2)
                dmin, dmax = compute_default_range(df[chosen])
                if xrange is not None:
                    dmin = max(dmin, xrange[0])
                    dmax = min(dmax, xrange[1])
                start_val = sm1.number_input("Force START", value=float(dmin), key=f"start_{i}")
                end_val = sm2.number_input("Force END", value=float(dmax), key=f"end_{i}")
                if start_val >= end_val:
                    st.warning("START must be strictly less than END. Overrides will be ignored.")
                else:
                    start_override = float(start_val)
                    end_override = float(end_val)

        # Chart
        st.subheader("Chart")
        fig = build_histogram(
            df=df,
            col=chosen,
            bin_mode=bin_mode,
            bin_width=bin_width,
            n_bins=n_bins,
            hist_output=hist_output,
            hist_type=hist_type,
            xrange=xrange,
            start_override=start_override,
            end_override=end_override,
        )

        # Export bins (CSV) & analytics base table
        export_table = compute_histogram_table(
            df=df,
            col=chosen,
            label=label,
            dataset_name=dataset_name,
            bin_mode=bin_mode,
            bin_width=bin_width,
            n_bins=n_bins,
            hist_output=hist_output,
            hist_type=hist_type,
            xrange=xrange,
            start_override=start_override,
            end_override=end_override,
        )

        # --- X‑interval metrics ---        with st.expander("X-interval metrics"):
        dmin = float(export_table["bin_left"].min()) if not export_table.empty else 0.0
        dmax = float(export_table["bin_right"].max()) if not export_table.empty else 1.0
        xi1, xi2 = st.columns(2)
        x0 = xi1.number_input("Start (x0)", value=dmin, key=f"x0_{i}")
        x1 = xi2.number_input("End (x1)", value=dmax, key=f"x1_{i}")
        show_x_guides = st.checkbox("Show x-lines (guides)", value=False, key=f"show_x_guides_{i}")
        if x0 >= x1:
            st.warning("x0 must be < x1")
        else:
            # Optional vertical guide lines at x0/x1 (red, thicker)
            if show_x_guides:
                try:
                    fig.add_vline(x=x0, line_color="red", line_width=3)
                    fig.add_vline(x=x1, line_color="red", line_width=3)
                except Exception:
                    pass
            if not export_table.empty:
                mask = (export_table["bin_right"] > x0) & (export_table["bin_left"] < x1)
                sel = export_table[mask]
                if not sel.empty:
                    if hist_type == "Standard":
                        if hist_output == "Count":
                            val = float(sel["count"].sum())
                            st.write(f"Sum of count in [x0,x1] = **{val:.0f}**")
                        else:
                            val = float(sel["probability"].sum())
                            st.write(f"Sum of probability in [x0,x1] = **{val:.4f}**")
                    else:  # cumulative metrics
                        cumprob = export_table["probability"].to_numpy().cumsum()
                        p0 = float(cumprob[(export_table["bin_right"] <= x0)].max()) if (export_table["bin_right"] <= x0).any() else 0.0
                        p1 = float(cumprob[(export_table["bin_right"] <= x1)].max()) if (export_table["bin_right"] <= x1).any() else 0.0
                        st.write(f"Δ probability = |p(x1)−p(x0)| = **{abs(p1 - p0):.4f}**")
                else:
                    st.info("No bins intersect this interval.")

        # --- Y-limit / intercept ---
        with st.expander("Y-limit / intercept"):
            if hist_type == "Standard":
                ymax = float(export_table["y"].max()) if not export_table.empty else 1.0
                y_limit = st.number_input("Y limit", value=max(0.0, round(ymax * 0.2, 6)), min_value=0.0, key=f"yl_{i}")
                show_intercepts = st.checkbox("Show intercept guides & values", value=False, key=f"show_intercepts_{i}")
                if show_intercepts:
                    try:
                        fig.add_hline(y=y_limit, line_color="purple", line_width=3)
                    except Exception:
                        pass
                if not export_table.empty:
                    yvals = export_table["y"].to_numpy()
                    centers = export_table["bin_center"].to_numpy()
                    crosses = []
                    if show_intercepts:
                        for k in range(1, len(yvals)):
                            if (yvals[k-1] < y_limit <= yvals[k]) or (yvals[k-1] > y_limit >= yvals[k]):
                                crosses.append(k)
                        for idx in crosses:
                            try:
                                fig.add_vline(x=float(centers[idx]), line_color="purple", line_width=3)
                            except Exception:
                                pass
                    if show_intercepts and crosses:
                        st.write("Intercept bin centers:", ", ".join(f"{centers[idx]:.6g}" for idx in crosses))
                    elif show_intercepts:
                        st.info("No intercept with the chosen Y limit.")
            else:
                # Cumulative: probability limit
                y_limit = st.number_input("Probability limit (0–1)", value=0.5, min_value=0.0, max_value=1.0, key=f"ylcum_{i}")
                show_intercepts = st.checkbox("Show intercept guides & values", value=False, key=f"show_intercepts_cum_{i}")
                if show_intercepts:
                    try:
                        fig.add_hline(y=y_limit, line_color="purple", line_width=3)
                    except Exception:
                        pass
                if not export_table.empty:
                    cumprob = export_table["probability"].to_numpy().cumsum()
                    idx = int(np.argmax(cumprob >= y_limit)) if (cumprob >= y_limit).any() else None
                    if show_intercepts and idx is not None:
                        x_quant = float(export_table.iloc[idx]["bin_right"])  # approx
                        try:
                            fig.add_vline(x=x_quant, line_color="purple", line_width=3)
                        except Exception:
                            pass
                        st.write(
                            f"Approx. quantile at limit: x ≈ **{x_quant:.6g}** · below mass ≈ **{float(cumprob[idx]):.4f}**, above mass ≈ **{float(1 - cumprob[idx]):.4f}**"
                        )
                    elif show_intercepts:
                        st.info("Limit not reached within data range.")

        # Render chart after adding optional guide lines
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**Export:** {label} · {hist_type} · {hist_output} · {bin_mode}")
        if not export_table.empty:
            safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_") or f"hist_{i+1}"
            fname = f"hist_bins_session{i+1}_{safe_label}.csv"
            st.download_button(
                label="Download binned data (CSV)",
                data=export_table.to_csv(index=False).encode("utf-8"),
                file_name=fname,
                mime="text/csv",
                key=f"dl_{i}",
            )
        else:
            st.info("No data to export with the current settings.")

        # Optional panels (inside the session expander)
        if show_stats:
            st.markdown("**Quick stats**")
            descr = pd.Series(pd.to_numeric(df[chosen], errors="coerce"), name=chosen).describe(
                percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
            )
            st.write(descr.to_frame())
        if show_table:
            st.markdown("**Data preview**")
            st.dataframe(df.head(100))

st.caption(
    "Notes: In 'Bin width', bin size is exact. In 'Number of bins', size is fitted to START–END. In 'Auto', edges are auto‑chosen within forced limits."
)
