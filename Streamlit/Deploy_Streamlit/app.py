"""
Fraud Detection App - Ready for Streamlit Cloud
Connects to deployed FastAPI on Render.com
"""

import streamlit as st
import requests
import pandas as pd
import os

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

# API URL - Use environment variable or default to deployed Render API
API_URL = os.environ.get("API_URL", "https://it5006-model-deployment.onrender.com")

# ============================================================
# HEADER
# ============================================================
st.title("🔍 Fraud Detection System")
st.markdown("*Ensemble Model: XGBoost + Random Forest Pipelines*")
st.caption(f"API: `{API_URL}`")

# ============================================================
# TABS
# ============================================================
tab1, tab2 = st.tabs(["🎯 Prediction", "📊 Model Stats"])

# ============================================================
# TAB 1: PREDICTION
# ============================================================
with tab1:
    st.header("Transaction Fraud Check")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Details")
        
        # Try to get categories from API (use cached fallback if API sleeping)
        try:
            cat_response = requests.get(f"{API_URL}/model-info", timeout=3)
            if cat_response.status_code == 200:
                categories = cat_response.json().get('category_classes', [])
            else:
                categories = []
        except:
            categories = []  # Will use fallback below
        
        # Fallback categories
        if not categories:
            categories = [
                "Grocery", "Electronics", "Clothing", "Restaurant/Cafeteria",
                "Cash Withdrawal", "Health/Beauty", "Domestic Transport",
                "Sports/Outdoors", "Holliday/Travel", "Jewelery"
            ]
        
        category = st.selectbox("Category", sorted(categories), index=0)
        amount = st.number_input("Amount ($)", min_value=0.01, max_value=50000.0, value=150.50)
        age = st.slider("Customer Age", min_value=18, max_value=90, value=35)
        days_until_expiry = st.number_input("Days Until Card Expires", min_value=0, max_value=3650, value=365)
    
    with col2:
        st.subheader("Behavioral Features")
        
        loc_delta = st.slider(
            "Location Delta (distance from last)", 
            min_value=0.0, max_value=1.0, value=0.05, step=0.01
        )
        
        loc_delta_mavg = st.slider(
            "Location Delta Avg (4h window)", 
            min_value=0.0, max_value=1.0, value=0.03, step=0.01
        )
        
        trans_volume_mavg = st.number_input(
            "Avg Transaction Amount (4h)", 
            min_value=0.0, max_value=10000.0, value=120.0
        )
        
        trans_volume_mstd = st.number_input(
            "Transaction Std Dev (4h)", 
            min_value=0.0, max_value=5000.0, value=45.0
        )
        
        trans_freq = st.number_input(
            "Transaction Frequency (4h)", 
            min_value=1, max_value=50, value=3
        )
    
    # Predict button
    st.markdown("---")
    
    if st.button("🔍 Check for Fraud", type="primary", use_container_width=True):
        payload = {
            "category": category,
            "amount": amount,
            "age_at_transaction": float(age),
            "days_until_card_expires": float(days_until_expiry),
            "loc_delta": loc_delta,
            "trans_volume_mavg": trans_volume_mavg,
            "trans_volume_mstd": trans_volume_mstd,
            "trans_freq": float(trans_freq),
            "loc_delta_mavg": loc_delta_mavg
        }
        
        try:
            with st.spinner("Analyzing transaction..."):
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                st.markdown("---")
                
                # Verdict - API returns 'verdict' not 'ensemble_verdict'
                verdict = result.get('verdict', result.get('ensemble_verdict', 'Unknown'))
                prob = result['ensemble_probability']
                
                if prob >= 0.7:
                    st.error(f"## ⚠️ {verdict}")
                elif prob >= 0.5:
                    st.warning(f"## ⚠️ {verdict}")
                elif prob >= 0.3:
                    st.info(f"## ℹ️ {verdict}")
                else:
                    st.success(f"## ✅ {verdict}")
                
                st.caption(f"Transaction ID: {result['transaction_id']}")
                
                # Model predictions - derive prediction from probability
                xgb_prob = result['xgboost_probability']
                rf_prob = result['random_forest_probability']
                ens_prob = result['ensemble_probability']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "XGBoost",
                        f"{xgb_prob*100:.1f}%",
                        delta="FRAUD" if xgb_prob >= 0.5 else "LEGIT",
                        delta_color="inverse"
                    )
                
                with col2:
                    st.metric(
                        "Random Forest",
                        f"{rf_prob*100:.1f}%",
                        delta="FRAUD" if rf_prob >= 0.5 else "LEGIT",
                        delta_color="inverse"
                    )
                
                with col3:
                    st.metric(
                        "Ensemble",
                        f"{ens_prob*100:.1f}%",
                        delta="FRAUD" if ens_prob >= 0.5 else "LEGIT",
                        delta_color="inverse"
                    )
                
                # Probability bar
                st.markdown("### Fraud Probability")
                st.progress(result['ensemble_probability'])
                
                # Drift warnings
                if result.get('drift_warnings'):
                    st.warning("**Drift Warnings:**")
                    for warning in result['drift_warnings']:
                        st.write(f"- {warning}")
                
                with st.expander("Raw API Response"):
                    st.json(result)
            else:
                st.error(f"API Error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to API at {API_URL}")
            st.info("The API may be sleeping. Try again in 30 seconds.")
        except requests.exceptions.Timeout:
            st.warning("Request timed out. The API may be waking up - try again.")
        except Exception as e:
            st.error(f"Error: {e}")

# ============================================================
# TAB 2: MODEL STATS
# ============================================================
with tab2:
    st.header("Model Performance & Monitoring")
    
    try:
        with st.spinner("Waking up API (free tier may take 30-60 seconds)..."):
            info_response = requests.get(f"{API_URL}/model-info", timeout=60)
            log_response = requests.get(f"{API_URL}/logs/summary", timeout=60)
        
        if info_response.status_code == 200:
            model_info = info_response.json()
            
            st.subheader("Model Metrics (Test Set)")
            
            metrics_data = model_info.get('metrics', {})
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data).T
                if len(metrics_df.columns) == 5:
                    metrics_df.columns = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC AUC']
                    metrics_df.index = ['XGBoost', 'Random Forest', 'Ensemble']
                    st.dataframe(
                        metrics_df.style.highlight_max(axis=0, color='lightgreen').format("{:.4f}"),
                        use_container_width=True
                    )
            
            # Training info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Samples", f"{model_info.get('training_samples', 0):,}")
            with col2:
                st.metric("Test Samples", f"{model_info.get('test_samples', 0):,}")
            with col3:
                st.metric("Features", len(model_info.get('feature_columns', [])))
            
            # Features
            st.subheader("Features Used")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Categorical:**")
                for feat in model_info.get('categorical_columns', []):
                    st.write(f"- `{feat}`")
            with col2:
                st.markdown("**Numeric:**")
                for feat in model_info.get('numeric_columns', []):
                    st.write(f"- `{feat}`")
        
        # Logs
        if log_response.status_code == 200:
            log_summary = log_response.json()
            
            st.markdown("---")
            st.subheader("Inference Monitoring")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Predictions", log_summary.get('total_predictions', 0))
            with col2:
                st.metric("Fraud Detected", log_summary.get('fraud_predictions', 0))
            with col3:
                st.metric("Fraud Rate", f"{log_summary.get('fraud_rate', 0):.1f}%")
            with col4:
                # Show count, not percentage - easier to understand
                drift_count = log_summary.get('predictions_with_drift', 0)
                st.metric("Drift Warnings", drift_count)
                
    except requests.exceptions.Timeout:
        st.warning("⏰ API is waking up (free tier cold start)")
        st.info("This takes 30-60 seconds on first request. Please refresh the page in a moment.")
    except requests.exceptions.ConnectionError:
        st.warning(f"Cannot connect to API at {API_URL}")
        st.info("The API may be sleeping (free tier). Try again in 30 seconds.")
    except Exception as e:
        st.error(f"Error: {e}")

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
**Ensemble Model** combining:
- XGBoost Pipeline
- Random Forest Pipeline

**Method**: Soft Voting (Average)

**Architecture**:
- Backend: FastAPI on Render
- Frontend: Streamlit Cloud
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Note")
st.sidebar.info("""
Free tier APIs may sleep after inactivity. 
First request may take 30-60 seconds to wake up.
""")
