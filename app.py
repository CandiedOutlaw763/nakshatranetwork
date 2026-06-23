import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
from pathlib import Path
import sys

# Ensure config is accessible
sys.path.insert(0, str(Path(__file__).parent))
import config
from astropy.timeseries import BoxLeastSquares

st.set_page_config(page_title="NakshatraNetwork", layout="wide")

# Theme colors
C = {
    "bg_primary": "#0B1120",
    "bg_secondary": "#131a2a",
    "grid": "#1e293b",
    "text_primary": "#e2e8f0",
    "text_secondary": "#94a3b8",
    "accent_blue": "#00d4ff",
    "accent_purple": "#7c3aed",
    "accent_green": "#10b981",
    "accent_yellow": "#f59e0b",
    "accent_red": "#ef4444"
}

class_colors = {
    "PLANET": C["accent_green"],
    "ECLIPSING_BINARY": C["accent_yellow"],
    "BLEND": C["accent_purple"],
    "OTHER": C["accent_red"]
}

@st.cache_data
def load_results():
    results_path = config.RESULTS_DIR / "classifications.pkl"
    if results_path.exists():
        with open(results_path, "rb") as f:
            return pickle.load(f)
    return []

@st.cache_data
def get_lightcurve_data(filepath):
    # Extract filename to avoid absolute path issues across different OS/environments
    filename = Path(filepath).name
    path = config.PROCESSED_DATA_DIR / filename
    if not path.exists():
        # Fallback to absolute path just in case
        path = Path(filepath)
        if not path.exists():
            return None, None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["time"], data["flux"]
    except Exception:
        return None, None

@st.cache_data
def compute_periodogram(t, y):
    try:
        duration_grid = np.linspace(config.DURATION_MIN, config.DURATION_MAX, config.N_DURATIONS)
        bls = BoxLeastSquares(t, y)
        results = bls.autopower(duration_grid, minimum_period=config.PERIOD_MIN, maximum_period=config.PERIOD_MAX, frequency_factor=0.1)
        return results.period, results.power
    except Exception:
        return None, None

def main():
    st.markdown("""
    <style>
    .stApp { background-color: #0B1120; color: #e2e8f0; }
    div[data-testid="stMetricValue"] { color: #00d4ff; }
    </style>
    """, unsafe_allow_html=True)

    results = load_results()

    if not results:
        st.warning("No results found. Please run the detection pipeline first.")
        return

    st.title("NakshatraNetwork")
    st.caption("AI-Powered Exoplanet Transit Detection Pipeline")

    c1, c2, c3, c4 = st.columns(4)
    n_planets = sum(1 for r in results if r["classification"] == "PLANET")
    n_ebs = sum(1 for r in results if r["classification"] == "ECLIPSING_BINARY")
    c1.metric("Candidates", len(results))
    c2.metric("Planets", n_planets)
    c3.metric("EBs", n_ebs)
    c4.metric("Sector", config.TESS_SECTOR)

    st.sidebar.header("Candidates")
    
    options = [f"{r['tic_id']} - {r['classification']} ({int(r['confidence']*100)}%)" for r in results]
    selected_option = st.sidebar.selectbox("Select a Candidate", options)
    
    selected_idx = options.index(selected_option)
    result = results[selected_idx]
    
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.subheader(f"TIC {result['tic_id']}")
    col2.markdown(f"**Class:** {result['classification']}")
    col3.markdown(f"**Confidence:** {result['confidence']:.1%}")
    
    best_period = result.get('period', result.get('bls_period', 0))
    sde = result.get('sde', result.get('bls_sde', 0))
    col4.markdown(f"**Period:** {best_period:.4f} d | **SDE:** {sde:.1f}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Light Curve", "Periodogram", "Phase Folded", "Parameters", "Overview"])

    t, y = get_lightcurve_data(result['filepath'])

    with tab1:
        if t is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=y, mode="markers", marker=dict(size=2, color=C["accent_blue"], opacity=0.6)))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C["bg_secondary"],
                title="Detrended Light Curve", xaxis_title="Time (BJD)", yaxis_title="Normalized Flux",
                height=400, margin=dict(l=60, r=20, t=50, b=50)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Light curve data not available.")

    with tab2:
        if t is not None:
            with st.spinner("Computing periodogram..."):
                periods, power = compute_periodogram(t, y)
            
            if periods is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=periods, y=power, mode="lines", line=dict(color=C["accent_purple"], width=1.2), fill="tozeroy", fillcolor="rgba(124, 58, 237, 0.12)"))
                fig.add_vline(x=best_period, line_dash="dash", line_color=C["accent_green"], line_width=2, annotation_text="P")
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C["bg_secondary"],
                    title=f"BLS Periodogram (SDE = {sde:.1f})", xaxis_title="Period (days)", yaxis_title="Power",
                    height=400, margin=dict(l=60, r=20, t=50, b=50)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Periodogram computation failed.")
        else:
            st.error("Data unavailable.")

    with tab3:
        if t is not None and best_period > 0:
            epoch = result.get('epoch', result.get('bls_t0', t[0]))
            phase = ((t - epoch) % best_period) / best_period
            phase[phase > 0.5] -= 1.0
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=phase, y=y, mode="markers", marker=dict(size=3, color=C["accent_yellow"], opacity=0.7)))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C["bg_secondary"],
                title="Phase Folded Light Curve", xaxis_title="Phase", yaxis_title="Normalized Flux",
                height=400, margin=dict(l=60, r=20, t=50, b=50)
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.write("### Extracted Parameters")
        df_params = pd.DataFrame({
            "Parameter": ["TIC ID", "Period", "Epoch", "Duration", "Depth", "SDE", "SNR", "Model Confidence"],
            "Value": [
                result['tic_id'],
                f"{best_period:.4f} days",
                f"{result.get('epoch', result.get('bls_t0', 0)):.4f} BJD",
                f"{result.get('duration', result.get('bls_duration', 0)):.4f} days",
                f"{result.get('depth', result.get('bls_depth', 0)):.4f}",
                f"{sde:.2f}",
                f"{result.get('snr', 0):.2f}",
                f"{result['confidence']:.2%}"
            ]
        })
        st.table(df_params)
        
    with tab5:
        st.write("### Dashboard Overview")
        preds = [r['classification'] for r in results]
        class_counts = pd.Series(preds).value_counts()
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=class_counts.index, values=class_counts.values,
            hole=0.4, marker=dict(colors=[class_colors.get(c, C["accent_blue"]) for c in class_counts.index])
        )])
        fig_pie.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C["bg_secondary"],
            title="Classification Distribution", height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)

if __name__ == "__main__":
    main()
