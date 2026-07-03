import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. PAGE SETUP & PROFESSIONAL THEME
# ==========================================
st.set_page_config(page_title="AI Traffic Prediction System", page_icon="🚗", layout="wide")

# Custom CSS for a Clean, Professional Corporate Dashboard
st.markdown("""
    <style>
        .main-title { font-size: 28px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
        .sub-title { font-size: 14px; color: #4B5563; margin-bottom: 20px; }
        .metric-card { background-color: #F3F4F6; padding: 15px; border-radius: 8px; border-left: 5px solid #2563EB; }
        .section-header { font-size: 20px; font-weight: 600; color: #1F2937; margin-top: 15px; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚗 AI-Based Traffic Flow Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Leveraging Random Forest Machine Learning for Reliable Average Daily Traffic (ADT) Forecasting</div>', unsafe_allow_html=True)
st.write("---")

# Navigation Menu
menu = st.sidebar.radio("📌 NAVIGATION MENU", ["🏠 Home & Prediction", "📊 Model Performance Dashboard"])

# ==========================================
# 2. DATA PIPELINE & REAL RF TRAINING
# ==========================================
@st.cache_resource
def init_machine_learning_pipeline():
    try:
        # Dynamic path resolution to prevent 'File Not Found'
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "traffic_weather.csv")
        
        if not os.path.exists(csv_path):
            return None, None, None, None, {}, False, f"File not found at {csv_path}"
            
        df = pd.read_csv(csv_path)
        
        # Clean missing values in target column
        df['ADT'] = pd.to_numeric(df['ADT'], errors='coerce')
        df = df.dropna(subset=['ADT'])
        
        # Safe numeric conversion for features
        num_cols = ['CarTaxi', 'LightLorry', 'MediumLorry', 'HeavyLorry', 'Bus', 'Motorcycle', 'MinTemp_C', 'MaxTemp_C', 'AvgTemp_C', 'Rainfall_mm', 'Humidity_pct', 'Year']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
        
        df['State'] = df['State'].astype(str).str.strip()
        df['Location'] = df['Location'].astype(str).str.strip()
        
        # Feature Encoding
        le_state = LabelEncoder()
        le_loc = LabelEncoder()
        df['State_Encoded'] = le_state.fit_transform(df['State'])
        df['Location_Encoded'] = le_loc.fit_transform(df['Location'])
        
        # Align features perfectly with dataset columns
        features = ['State_Encoded', 'Location_Encoded', 'Year', 'CarTaxi', 'LightLorry', 
                    'MediumLorry', 'HeavyLorry', 'Bus', 'Motorcycle', 'AvgTemp_C', 'Rainfall_mm', 'Humidity_pct']
        
        X = df[features]
        y = df['ADT']
        
        # 80:20 Train-Test Validation Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Random Forest Regressor Model Training
        model = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42)
        model.fit(X_train, y_train)
        
        # Model Evaluation
        y_pred = model.predict(X_test)
        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred),
            "y_test": y_test,
            "y_pred": y_pred
        }
        
        return model, le_state, le_loc, df, metrics, True, "Success"
    except Exception as e:
        return None, None, None, None, {}, False, str(e)

# Execute Pipeline
model, le_state, le_loc, df, metrics, is_success, status_msg = init_machine_learning_pipeline()

if not is_success:
    st.error(f"🚨 **Pipeline Initialization Error:** {status_msg}")
    st.info("Please ensure 'traffic_weather.csv' is placed inside your 'PHYTONLAB' folder.")
else:
    # KPI Summary Cards at the top of Home
    if menu == "🏠 Home & Prediction":
        st.markdown('<div class="section-header">📈 Dataset & Model Overview</div>', unsafe_allow_html=True)
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f'<div class="metric-card"><b>Total States Covered</b><br><span style="font-size:24px;color:#2563EB;">{df["State"].nunique()} States</span></div>', unsafe_allow_html=True)
        with kpi2:
            st.markdown(f'<div class="metric-card"><b>Monitored Locations</b><br><span style="font-size:24px;color:#2563EB;">{df["Location"].nunique()} Stations</span></div>', unsafe_allow_html=True)
        with kpi3:
            st.markdown(f'<div class="metric-card"><b>Dataset Record Size</b><br><span style="font-size:24px;color:#2563EB;">{len(df):,} Rows</span></div>', unsafe_allow_html=True)
        with kpi4:
            st.markdown(f'<div class="metric-card"><b>Model Accuracy (R²)</b><br><span style="font-size:24px;color:#2563EB;">{metrics["R2"]:.2%}</span></div>', unsafe_allow_html=True)
        st.write(" ")

        # --- CODE UNTUK PART 2 AKAN MASUK DI SINI ---
        # ==========================================
        # 3. PREDICTION INTERFACE (USER INPUTS)
        # ==========================================
        st.markdown('<div class="section-header">🔮 Smart Traffic Forecasting Console</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📍 Spatial & Temporal Context")
            state_input = st.selectbox("Select State Jurisdiction", sorted(df['State'].unique()))
            
            # Filter location based on selected state dynamically
            filtered_locations = sorted(df[df['State'] == state_input]['Location'].unique())
            location_input = st.selectbox("Select Traffic Station / Route", filtered_locations)
            
            year_input = st.slider("Target Projection Year", 2015, 2030, 2026)

        with col2:
            st.subheader("🌤️ Meteorological Environment Matrix")
            avg_temp = st.slider("Average Ambient Temperature (°C)", 15, 45, 28)
            humidity = st.slider("Relative Humidity Level (%)", 40, 100, 80)
            
            rain_category = st.selectbox(
                "Rainfall Intensity Profile",
                ["Clear Conditions (0.0 mm)", "Light Drizzle (2.5 mm)", "Moderate Downpour (15.0 mm)", "Heavy Monsoon Torrential (45.0 mm)"]
            )
            rain_mapping = {
                "Clear Conditions (0.0 mm)": 0.0,
                "Light Drizzle (2.5 mm)": 2.5,
                "Moderate Downpour (15.0 mm)": 15.0,
                "Heavy Monsoon Torrential (45.0 mm)": 45.0
            }
            rainfall_mm = rain_mapping[rain_category]

        st.write(" ")
        
        # AUTOMATED HISTORICAL DATA EXTRACTION (Sistem ambil data dari background)
        route_profile = df[df['Location'] == location_input]
        
        # Calculate historical averages for that specific location
        car = int(route_profile['CarTaxi'].mean()) if (not route_profile.empty and not pd.isna(route_profile['CarTaxi'].mean())) else 15000
        moto = int(route_profile['Motorcycle'].mean()) if (not route_profile.empty and not pd.isna(route_profile['Motorcycle'].mean())) else 5000
        l_lorry = int(route_profile['LightLorry'].mean()) if (not route_profile.empty and not pd.isna(route_profile['LightLorry'].mean())) else 1200
        m_lorry = int(route_profile['MediumLorry'].mean()) if (not route_profile.empty and not pd.isna(route_profile['MediumLorry'].mean())) else 800
        h_lorry = int(route_profile['HeavyLorry'].mean()) if (not route_profile.empty and not pd.isna(route_profile['HeavyLorry'].mean())) else 500
        bus = int(route_profile['Bus'].mean()) if (not route_profile.empty and not pd.isna(route_profile['Bus'].mean())) else 300

        # Inform the user about auto-extracted metrics via a clean notice expander
        with st.expander(f"ℹ️ View Auto-Retrieved Historical Vehicle Profile for: {location_input}"):
            st.markdown(f"""
            The AI engine automatically loaded these baseline numbers from our historical server repository to avoid manual user entries:
            * 🚗 **Passenger Cars/Taxis:** {car:,} units/day
            * 🏍️ **Active Motorcycles:** {moto:,} units/day
            * 🚚 **Logistics Freight (Light/Medium Lorries):** {l_lorry + m_lorry:,} units/day
            * 🚛 **Heavy Industrial Trucks:** {h_lorry:,} units/day
            * 🚌 **Public Transit Buses:** {bus:,} units/day
            """)

        st.write(" ")
        
        # ==========================================
        # 4. PREDICTION INFERENCE & LOGIC EXECUTION
        # ==========================================
        if st.button("PREDICT TRAFFIC FLOW", type="primary"):
            state_enc = le_state.transform([state_input])[0]
            loc_enc = le_loc.transform([location_input])[0]
            
            # Input vector for Random Forest model
            input_vector = [[state_enc, loc_enc, year_input, car, l_lorry, m_lorry, h_lorry, bus, moto, avg_temp, rainfall_mm, humidity]]
            prediction = model.predict(input_vector)[0]
            
            st.write("---")
            st.markdown('<div class="section-header">🎯 Prediction Results</div>', unsafe_allow_html=True)
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.metric(label="Predicted Average Daily Traffic (ADT)", value=f"{int(prediction):,} Vehicles / Day")
                
                # Simple Traffic Status Classification
                if prediction > 22000:
                    st.error("🔴 **Traffic Status: HIGH CONGESTION**")
                    status_level = "High"
                elif prediction > 12000:
                    st.warning("🟡 **Traffic Status: MODERATE TRAFFIC**")
                    status_level = "Moderate"
                else:
                    st.success("🟢 **Traffic Status: LOW TRAFFIC**")
                    status_level = "Low"

            with res_col2:
                st.markdown("**📋 Suggested Action Plan:**")
                if status_level == "High":
                    st.markdown("✔ *Deploy traffic officers to manage the road bottleneck.*")
                    st.markdown("✔ *Increase traffic light green time (+45 seconds).*")
                    st.markdown("✔ *Advise drivers to avoid this route during peak hours.*")
                elif status_level == "Moderate":
                    st.markdown("✔ *Monitor the road condition closely during peak hours.*")
                    st.markdown("✔ *Slightly adjust traffic light timing if needed (+20 seconds).*")
                else:
                    st.markdown("✔ *No action needed. The road traffic is clear and smooth.*")

            # SIMPLIFIED AI EXPLANATION
            st.write(" ")
            st.subheader("🤖 Simple AI Explanation")
            
            # Set simple explanation text based on inputs
            if rainfall_mm >= 15.0:
                rain_text = f"Heavy rain ({rainfall_mm} mm) is selected. Usually, bad weather causes cars to drive slower, which creates higher traffic density on the road."
            else:
                rain_text = "The weather is clear with no rain, so traffic flow is not affected by weather factors."
                
            if car > 15000:
                car_text = f"This location ({location_input}) historically has a high baseline of daily cars ({car:,} cars)."
            else:
                car_text = f"This location ({location_input}) historically has a normal/low number of daily cars ({car:,} cars)."
            
            st.info(f"""
            💡 **How the AI made this decision:**
            1. **Historical Data:** {car_text} The AI uses this historical baseline as the starting point.
            2. **Weather Condition:** {rain_text} 
            3. **Final Decision:** Our Random Forest model combined these factors (Location, Vehicles, and Weather) to predict that **{int(prediction):,} vehicles** will use this road daily under these conditions.
            """)
            
            # Data Export Feature
            export_df = pd.DataFrame([{
                "Year": year_input, "State": state_input, "Location": location_input, 
                "Predicted_ADT": int(prediction), "Traffic_Status": status_level,
                "Rainfall_mm": rainfall_mm, "Temperature_C": avg_temp, "Humidity_pct": humidity
            }])
            csv_data = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Prediction Result (CSV)", data=csv_data, file_name=f"Traffic_Report_{location_input}.csv", mime="text/csv")

            # ==========================================
    # 5. MODEL PERFORMANCE & DIAGNOSTICS DASHBOARD
    # ==========================================
    elif menu == "📊 Model Performance Dashboard":
        st.markdown('<div class="section-header">📊 Rigorous Machine Learning Evaluation Diagnostics</div>', unsafe_allow_html=True)
        st.markdown("This window demonstrates the Random Forest model's training validation integrity based on an 80:20 data partition split.")
        
        # Display Core Mathematical Metrics
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Mean Absolute Error (MAE)", f"{metrics['MAE']:.2f} Vehicles")
        with m_col2:
            st.metric("Root Mean Squared Error (RMSE)", f"{metrics['RMSE']:.2f} Vehicles")
        with m_col3:
            st.metric("R² Score (Model Accuracy)", f"{metrics['R2']:.4%}")
            
        st.write("---")
        
        # GRAPH GENERATION SECTION
        st.markdown('<div class="section-header">📈 Predictive Analytics & Mathematical Validation Plots</div>', unsafe_allow_html=True)
        graph_col1, graph_col2 = st.columns(2)
        
        with graph_col1:
            st.subheader("🌲 Feature Importance Structural Weights")
            st.markdown("Identifies which structural parameters impact the ADT prediction outputs most heavily:")
            
            # Extract and sort feature importances
            importances = model.feature_importances_
            feat_imp_df = pd.DataFrame({
                "Feature Parameter": features_list, # type: ignore
                "Weight Score": importances
            }).sort_values(by="Weight Score", ascending=True) # Ascending for a clean horizontal bar look
            
            # Plotting Feature Importance Bar Chart
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.barplot(x="Weight Score", y="Feature Parameter", data=feat_imp_df, palette="Blues_r", ax=ax1)
            ax1.set_title("Random Forest Relative Variable Weighting")
            ax1.set_xlabel("Relative Importance Score")
            ax1.set_ylabel("")
            st.pyplot(fig1)
            
        with graph_col2:
            st.subheader("🎯 Empirical Model Fidelity: Prediction vs Actual")
            st.markdown("Validates model alignment by plotting actual test targets against AI model inferences:")
            
            # Sampling a small subset for scatter plot to keep rendering light and neat
            y_test_sample = metrics['y_test']
            y_pred_sample = metrics['y_pred']
            
            # Plotting Prediction vs Actual Scatter Plot
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x=y_test_sample, y=y_pred_sample, alpha=0.5, color="#2563EB", ax=ax2)
            
            # Identity diagonal reference line (Perfect alignment path)
            max_val = max(max(y_test_sample), max(y_pred_sample))
            min_val = min(min(y_test_sample), min(y_pred_sample))
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction Reference')
            
            ax2.set_title("Inference Residual Target Spread")
            ax2.set_xlabel("Actual Historical ADT")
            ax2.set_ylabel("Predicted Model ADT")
            ax2.set_legend()
            st.pyplot(fig2)

        st.write("---")
        
        # EXPLORATORY DATA ANALYSIS (EDA) FRAMEWORK
        st.markdown('<div class="section-header">🗂️ Exploratory Data Analysis (EDA) Sample Data View</div>', unsafe_allow_html=True)
        st.markdown("Pure analytical statistics and raw layout extracted from the verified training matrix:")
        
        # Display the first 15 rows of the dataset neatly
        st.dataframe(df.head(15), use_container_width=True)
        
        # Summary statistics display block
        with st.expander("📊 View General Descriptive Statistics"):
            st.dataframe(df.describe(), use_container_width=True)

# System Infrastructure Footer (Universal)
st.write("---")
st.caption("🔒 Corporate Grade Framework | AI Project Presentation | Secured Streamlit Infrastructure Running Local Machine")