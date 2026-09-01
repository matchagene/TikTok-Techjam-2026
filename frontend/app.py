"""Streamlit dashboard backed only by real experiment outputs and checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing import get_dinov2_preprocess  # noqa: E402
from evaluation.model_loading import load_adapter  # noqa: E402

st.set_page_config(
    page_title="Robust AI-Generated Image Detection",
    layout="wide",
)

st.title("Robust AI-Generated Image Detection")
st.caption("Clean baseline vs pairwise transformation-robust DINOv2 detector")


@st.cache_data
def load_csv(path: str) -> pd.DataFrame | None:
    file_path = PROJECT_ROOT / path
    return pd.read_csv(file_path) if file_path.is_file() else None


@st.cache_resource
def load_live_model(model_id: str, checkpoint_path: str, checkpoint_mtime: float):
    del checkpoint_mtime  # part of the cache key; forces reload when checkpoint changes
    checkpoint = Path(checkpoint_path)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    adapter = load_adapter(model_id, checkpoint, device=device)
    return adapter, device


def _show_missing(path: str, command: str | None = None) -> None:
    st.info(f"No real output found at `{path}`.")
    if command:
        st.code(command, language="bash")


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 Training",
        "📊 Robustness Summary",
        "🧩 Condition Breakdown",
        "🔍 Live Inference",
    ]
)

with tab1:
    st.header("Training Metrics")
    m1 = load_csv("results/training/M1_history.csv")
    m3 = load_csv("results/training/M3_history.csv")

    if m1 is None and m3 is None:
        _show_missing("results/training/M1_history.csv or M3_history.csv")
    else:
        if m1 is not None:
            st.subheader("M1 — Clean Baseline")
            st.dataframe(m1, use_container_width=True)
            if {"epoch", "clean_val_auc"}.issubset(m1.columns):
                fig = px.line(m1, x="epoch", y="clean_val_auc", markers=True, title="M1 Clean Validation ROC-AUC")
                st.plotly_chart(fig, use_container_width=True)
        if m3 is not None:
            st.subheader("M3 — Pairwise Robust Detector")
            st.dataframe(m3, use_container_width=True)
            loss_columns = [
                column
                for column in ["train_cls_loss", "train_pred_loss", "train_repr_loss", "train_total_loss"]
                if column in m3.columns
            ]
            if loss_columns and "epoch" in m3.columns:
                fig = px.line(
                    m3,
                    x="epoch",
                    y=loss_columns,
                    markers=True,
                    title="M3 Training Loss Components",
                    labels={"value": "Loss", "variable": "Component"},
                )
                st.plotly_chart(fig, use_container_width=True)
            if {"epoch", "clean_val_auc"}.issubset(m3.columns):
                fig = px.line(m3, x="epoch", y="clean_val_auc", markers=True, title="M3 Clean Validation ROC-AUC")
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Clean vs Transformed Performance")
    comparison_path = "results/evaluation/model_comparison.csv"
    comparison = load_csv(comparison_path)
    if comparison is None:
        _show_missing(
            comparison_path,
            "python -m evaluation.build_submission_results --models M1 M3",
        )
    else:
        st.dataframe(comparison, use_container_width=True)
        required = {"Model", "Clean AUC", "Robust Pooled AUC", "Worst-Case AUC"}
        if required.issubset(comparison.columns):
            fig = go.Figure()
            fig.add_trace(go.Bar(x=comparison["Model"], y=comparison["Clean AUC"], name="Clean AUC"))
            fig.add_trace(go.Bar(x=comparison["Model"], y=comparison["Robust Pooled AUC"], name="Robust Pooled AUC"))
            fig.add_trace(go.Bar(x=comparison["Model"], y=comparison["Worst-Case AUC"], name="Worst-Case AUC"))
            fig.update_layout(barmode="group", yaxis=dict(range=[0, 1.0], title="ROC-AUC"))
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Performance by Fixed Transformation Condition")
    conditions_path = "results/evaluation/condition_comparison.csv"
    conditions = load_csv(conditions_path)
    if conditions is None:
        _show_missing(
            conditions_path,
            "python -m evaluation.build_submission_results --models M1 M3",
        )
    else:
        st.dataframe(conditions, use_container_width=True)
        if {"Model", "condition_id", "roc_auc"}.issubset(conditions.columns):
            fig = px.line(
                conditions,
                x="condition_id",
                y="roc_auc",
                color="Model",
                markers=True,
                title="ROC-AUC Across Deterministic Corruption Conditions",
            )
            fig.update_yaxes(range=[0, 1.0], title="ROC-AUC")
            fig.update_xaxes(tickangle=55, title="Condition")
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Single-Image Inference")
    st.caption("This tab runs the actual trained checkpoint. No mock/random predictions are used.")

    default_m3 = PROJECT_ROOT / "checkpoints/M3_pairwise.pth"
    default_m1 = PROJECT_ROOT / "checkpoints/M1_corrected_baseline.pth"
    default_model = "M3" if default_m3.is_file() else "M1"
    model_id = st.selectbox("Model", ["M3", "M1"], index=0 if default_model == "M3" else 1)
    default_checkpoint = default_m3 if model_id == "M3" else default_m1
    checkpoint_text = st.text_input("Checkpoint", value=str(default_checkpoint))
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"])

    if uploaded_file is not None:
        checkpoint = Path(checkpoint_text).expanduser()
        if not checkpoint.is_file():
            st.error(f"Checkpoint not found: {checkpoint}")
        else:
            try:
                adapter, device = load_live_model(model_id, str(checkpoint), checkpoint.stat().st_mtime)
                preprocess = get_dinov2_preprocess()
                image = Image.open(uploaded_file).convert("RGB")
                tensor = preprocess(image).unsqueeze(0).to(device)
                with torch.inference_mode():
                    p_fake = float(adapter.predict_fake_probability(tensor).item())

                col_img, col_result = st.columns([1, 1])
                with col_img:
                    st.image(image, caption="Input image", use_container_width=True)
                with col_result:
                    st.metric("P(AI-generated)", f"{p_fake:.2%}")
                    st.write(f"Device: `{device}`")
                    if p_fake >= 0.5:
                        st.error("Prediction: AI-GENERATED / FAKE")
                    else:
                        st.success("Prediction: REAL / AUTHENTIC")
                    st.caption("Threshold shown for the demo: 0.5. The submitted `pred` value is the continuous probability.")
            except Exception as exc:
                st.exception(exc)
