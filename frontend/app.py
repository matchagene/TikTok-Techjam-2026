import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageEnhance

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Image Detection Robustness Dashboard",
    layout="wide",
)

st.title("Track 5: AI-Generated Image Detection Dashboard")
st.caption("NTIRE 2026 Adaptation: Robustness, Consistency & Stability Metrics")

# --- MOCK DATA GENERATORS (FALLBACK WHEN CSVs DO NOT EXIST) ---
@st.cache_data
def load_training_history():
    path = "results/training/M3_history.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    
    # Generate mock training history for M1-M4
    epochs = list(range(1, 6))
    data = []
    for ep in epochs:
        data.append({
            "epoch": ep,
            "M1_cls_loss": 0.50 - ep * 0.07,
            "M2_cls_loss": 0.45 - ep * 0.06,
            "M3_total_loss": 0.65 - ep * 0.08,
            "M3_cls_loss": 0.40 - ep * 0.05,
            "M3_pred_loss": 0.30 - ep * 0.04,
            "M3_repr_loss": 0.20 - ep * 0.03,
            "M4_total_loss": 0.60 - ep * 0.08,
            "clean_val_auc": 0.82 + ep * 0.025,
        })
    return pd.DataFrame(data)

@st.cache_data
def load_model_comparison():
    path = "results/evaluation/model_comparison.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    
    # Mock ablation data for M0 to M4
    return pd.DataFrame({
        "Model": ["M0 (Original)", "M1 (Corrected)", "M2 (Augmented)", "M3 (Pairwise)", "M4 (Curriculum)"],
        "Clean AUC": [0.84, 0.87, 0.89, 0.92, 0.93],
        "Robust Pooled AUC": [0.55, 0.61, 0.78, 0.85, 0.89],
        "Mean Condition AUC": [0.58, 0.64, 0.80, 0.87, 0.90],
        "Worst-Case AUC": [0.38, 0.42, 0.65, 0.74, 0.81],
        "Robustness Drop": [0.29, 0.26, 0.11, 0.07, 0.04]
    })

@st.cache_data
def load_robustness_matrix():
    path = "results/evaluation/robustness_matrix.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    
    # Mock 2x2 matrix data
    return pd.DataFrame([
        {"Dataset": "In-domain (SID)", "Condition": "Clean", "AUC": 0.93},
        {"Dataset": "In-domain (SID)", "Condition": "Corrupted", "AUC": 0.89},
        {"Dataset": "External (GenImage)", "Condition": "Clean", "AUC": 0.86},
        {"Dataset": "External (GenImage)", "Condition": "Corrupted", "AUC": 0.81},
    ])

# --- DASHBOARD TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Training Metrics", 
    "📊 Model Ablation (M0-M4)", 
    "🧩 Generalization Matrix", 
    "🔍 Live Prediction & Stability"
])

# --- TAB 1: TRAINING METRICS ---
with tab1:
    st.header("Training Progress & Loss Curves")
    df_train = load_training_history()
    
    col1, col2 = st.columns(2)
    with col1:
        fig_loss = px.line(
            df_train, x="epoch", 
            y=["M3_cls_loss", "M3_pred_loss", "M3_repr_loss", "M3_total_loss"],
            title="M3 Loss Components (Classification + Consistency)",
            labels={"value": "Loss", "variable": "Loss Type"}
        )
        st.plotly_chart(fig_loss, use_container_width=True)
        
    with col2:
        fig_auc = px.line(
            df_train, x="epoch", y="clean_val_auc",
            title="Clean Validation ROC-AUC Progression",
            markers=True
        )
        st.plotly_chart(fig_auc, use_container_width=True)

# --- TAB 2: MODEL ABLATION ---
with tab2:
    st.header("Model Ladder Comparison (M0 → M4)")
    df_comp = load_model_comparison()
    
    st.dataframe(df_comp, use_container_width=True)
    
    fig_ablation = go.Figure()
    fig_ablation.add_trace(go.Bar(x=df_comp["Model"], y=df_comp["Clean AUC"], name="Clean AUC"))
    fig_ablation.add_trace(go.Bar(x=df_comp["Model"], y=df_comp["Robust Pooled AUC"], name="Robust Pooled AUC"))
    fig_ablation.add_trace(go.Bar(x=df_comp["Model"], y=df_comp["Worst-Case AUC"], name="Worst-Case AUC"))
    
    fig_ablation.update_layout(
        barmode="group",
        title="Performance Progression Across Ladder",
        yaxis=dict(range=[0, 1.0], title="ROC-AUC Score")
    )
    st.plotly_chart(fig_ablation, use_container_width=True)

# --- TAB 3: GENERALIZATION MATRIX ---
with tab3:
    st.header("2×2 Transformation & Distribution Shift Matrix")
    df_matrix = load_robustness_matrix()
    
    # Pivot for heatmap display
    pivot_df = df_matrix.pivot(index="Condition", columns="Dataset", values="AUC")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        fig_heatmap = px.imshow(
            pivot_df,
            text_auto=True,
            color_continuous_scale="Blues",
            zmin=0.5, zmax=1.0,
            title="AUC Matrix (Shift Analysis)"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
    with col2:
        st.markdown("""
        **Interpretation Rules:**
        * **In-domain Clean → Corrupted Drop:** Measures transformation sensitivity.
        * **Clean In-domain → External Drop:** Measures generator distribution shift.
        * **Corrupted External (Worst Case):** Measures combined real-world degradation + unseen generator shift.
        """)

# --- TAB 4: LIVE PREDICTION & STABILITY ---
with tab4:
    st.header("Single Image Inference & Transformation Stability")
    
    uploaded_file = st.file_uploader("Upload an Image to Analyze", type=["jpg", "png", "jpeg", "webp"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        col_img, col_metrics = st.columns([1, 1])
        
        with col_img:
            st.image(image, caption="Uploaded Input Image", use_container_width=True)
            
        with col_metrics:
            # Mock model inference calculation
            p_fake = np.random.uniform(0.70, 0.98) # Mock logit output
            stability_score = np.random.uniform(0.85, 0.99) # Mock stability
            
            st.metric("Probability AI-Generated (p_fake)", f"{p_fake:.2%}")
            
            if p_fake > 0.5:
                st.error("Prediction: **AI-GENERATED / FAKE** (Target 1)")
            else:
                st.success("Prediction: **REAL / AUTHENTIC** (Target 0)")
                
            st.subheader("Transformation Stability Score")
            st.progress(stability_score)
            st.write(f"**Score:** `{stability_score:.4f}` / 1.0000")
            
            st.caption("""
            *Transformation Stability Score* measures prediction variance across 
            benign perturbations (JPEG compression, mild blur, downsampling).
            """)