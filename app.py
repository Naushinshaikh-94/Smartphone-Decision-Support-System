"""
AI-Based Smartphone Decision Support System
app.py -> Streamlit UI with two tools:
  1. Recommend Smartphones (rule-based filter, same logic as the notebook)
  2. Predict Price / Category (uses the trained ML models from train.py)

Run with:  streamlit run app.py
"""

import os
import joblib
import pandas as pd
import streamlit as st

# Get the folder where this app.py file is located.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The train.py file saves all models inside this folder.
MODEL_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(page_title="Smartphone Decision Support System", layout="wide")


# ----------------------------------------------------------------------
# Load saved artifacts (cached so they load only once)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    price_model = joblib.load(os.path.join(MODEL_DIR, "price_model.pkl"))
    category_model = joblib.load(os.path.join(MODEL_DIR, "category_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    os_encoder = joblib.load(os.path.join(MODEL_DIR, "os_encoder.pkl"))
    category_encoder = joblib.load(os.path.join(MODEL_DIR, "category_encoder.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    brands = joblib.load(os.path.join(MODEL_DIR, "brands.pkl"))
    processor_brands = joblib.load(os.path.join(MODEL_DIR, "processor_brands.pkl"))
    os_names = joblib.load(os.path.join(MODEL_DIR, "os_names.pkl"))
    phones = pd.read_csv(os.path.join(MODEL_DIR, "phones_original.csv"))
    return (price_model, category_model, scaler, os_encoder, category_encoder,
            feature_columns, brands, processor_brands, os_names, phones)


(price_model, category_model, scaler, os_encoder, category_encoder,
 feature_columns, brands, processor_brands, os_names, phones) = load_artifacts()


# ----------------------------------------------------------------------
# Priority helper functions (same logic as the notebook)
# ----------------------------------------------------------------------
def gaming_priority(priority):
    if priority == "Low":
        return {"refresh_rate_hz": 60, "clock_speed_ghz": 2.0, "core_count": 6}
    elif priority == "Medium":
        return {"refresh_rate_hz": 90, "clock_speed_ghz": 2.4, "core_count": 8}
    else:
        return {"refresh_rate_hz": 120, "clock_speed_ghz": 3.2, "core_count": 8}


def camera_priority(level):
    if level == "High":
        return {"rear_camera_main_mp": 64, "front_camera_main_mp": 16}
    elif level == "Medium":
        return {"rear_camera_main_mp": 50, "front_camera_main_mp": 8}
    else:
        return {"rear_camera_main_mp": 12, "front_camera_main_mp": 5}


def battery_priority(priority):
    if priority == "Low":
        return {"battery_mah": 4500, "charging_watt": 25}
    elif priority == "Medium":
        return {"battery_mah": 5000, "charging_watt": 45}
    else:
        return {"battery_mah": 6000, "charging_watt": 80}


st.title("📱 AI-Based Smartphone Decision Support System")

tab1, tab2 = st.tabs(["🔍 Recommend Smartphones", "🤖 Predict Price / Category"])


# ----------------------------------------------------------------------
# TAB 1 - Rule based recommender
# ----------------------------------------------------------------------
with tab1:
    st.subheader("Tell us what you need")

    col1, col2, col3 = st.columns(3)
    with col1:
        budget = st.number_input("Budget (₹)", min_value=1000, max_value=200000, value=20000, step=1000)
        ram = st.number_input("Minimum RAM (GB)", min_value=1, max_value=32, value=6)
    with col2:
        storage = st.number_input("Minimum Storage (GB)", min_value=8, max_value=1024, value=128)
        gaming_choice = st.selectbox("Gaming Priority", ["Low", "Medium", "High"])
    with col3:
        camera_choice = st.selectbox("Camera Priority", ["Low", "Medium", "High"])
        battery_choice = st.selectbox("Battery Priority", ["Low", "Medium", "High"])

    if st.button("Find Smartphones", type="primary"):
        gaming = gaming_priority(gaming_choice)
        camera = camera_priority(camera_choice)
        battery = battery_priority(battery_choice)

        user_data = {
            "price_inr": budget,
            "ram_gb": ram,
            "storage_gb": storage,
            **gaming,
            **camera,
            **battery,
        }

        recommended = phones[
            (phones["price_inr"] <= user_data["price_inr"] + 3000)
            & (phones["price_inr"] >= user_data["price_inr"] - 3000)
            & (phones["ram_gb"] >= user_data["ram_gb"])
            & (phones["storage_gb"] >= user_data["storage_gb"])
            & (phones["refresh_rate_hz"] >= user_data["refresh_rate_hz"])
            & (phones["clock_speed_ghz"] >= user_data["clock_speed_ghz"])
            & (phones["core_count"] >= user_data["core_count"])
            & (phones["battery_mah"] >= user_data["battery_mah"])
            & (phones["charging_watt"] >= user_data["charging_watt"])
            & (phones["rear_camera_main_mp"] >= user_data["rear_camera_main_mp"])
            & (phones["front_camera_main_mp"] >= user_data["front_camera_main_mp"])
        ].sort_values(by="rating_score", ascending=False)

        if recommended.empty:
            st.warning("No smartphones matched. Try relaxing your requirements.")
        else:
            st.success(f"Found {len(recommended)} matching smartphones. Showing top 5:")
            st.dataframe(
                recommended[[
                    "model", "smartphone_brand", "price_inr", "rating_score",
                    "ram_gb", "storage_gb", "refresh_rate_hz", "battery_mah",
                    "rear_camera_main_mp", "front_camera_main_mp"
                ]].head(5),
                use_container_width=True,
                hide_index=True,
            )


# ----------------------------------------------------------------------
# TAB 2 - ML model predictions
# ----------------------------------------------------------------------
with tab2:
    st.subheader("Enter the specs of a phone to predict its price & category")

    c1, c2, c3 = st.columns(3)
    with c1:
        brand = st.selectbox("Brand", brands)
        processor_brand = st.selectbox("Processor Brand", processor_brands)
        os_name = st.selectbox("OS", os_names)
        rating_score = st.slider("Rating Score", 0, 100, 75)
        core_count = st.selectbox("Core Count", [4, 6, 8], index=2)
        clock_speed_ghz = st.number_input("Clock Speed (GHz)", 1.0, 4.0, 2.4, step=0.1)
    with c2:
        ram_gb = st.selectbox("RAM (GB)", [2, 3, 4, 6, 8, 12, 16], index=4)
        storage_gb = st.selectbox("Storage (GB)", [32, 64, 128, 256, 512, 1024], index=2)
        display_inches = st.number_input("Display Size (inches)", 4.0, 8.0, 6.5, step=0.1)
        res_width_px = st.number_input("Resolution Width (px)", 480, 2000, 1080)
        res_height_px = st.number_input("Resolution Height (px)", 800, 4000, 2400)
        refresh_rate_hz = st.selectbox("Refresh Rate (Hz)", [60, 90, 120, 144], index=1)
    with c3:
        battery_mah = st.number_input("Battery (mAh)", 2000, 10000, 5000, step=100)
        charging_watt = st.number_input("Charging Speed (W)", 5.0, 250.0, 25.0, step=5.0)
        rear_camera_count = st.selectbox("Rear Camera Count", [1, 2, 3, 4], index=1)
        front_camera_count = st.selectbox("Front Camera Count", [1, 2], index=0)
        rear_camera_main_mp = st.number_input("Rear Camera (MP)", 2.0, 200.0, 50.0, step=1.0)
        front_camera_main_mp = st.number_input("Front Camera (MP)", 2.0, 60.0, 16.0, step=1.0)

    c4, c5, c6 = st.columns(3)
    with c4:
        has_5g = st.checkbox("5G Support", value=True)
    with c5:
        has_nfc = st.checkbox("NFC Support", value=False)
    with c6:
        has_ir_blaster = st.checkbox("IR Blaster", value=False)
    fast_charging = st.checkbox("Fast Charging", value=True)

    if st.button("Predict", type="primary"):
        # Start with every training column set to 0
        row = {col: 0 for col in feature_columns}

        row.update({
            "rating_score": rating_score,
            "core_count": core_count,
            "clock_speed_ghz": clock_speed_ghz,
            "ram_gb": ram_gb,
            "storage_gb": storage_gb,
            "has_5g": int(has_5g),
            "has_nfc": int(has_nfc),
            "has_ir_blaster": int(has_ir_blaster),
            "display_inches": display_inches,
            "res_width_px": res_width_px,
            "res_height_px": res_height_px,
            "refresh_rate_hz": refresh_rate_hz,
            "battery_mah": battery_mah,
            "fast_charging": int(fast_charging),
            "charging_watt": charging_watt,
            "rear_camera_count": rear_camera_count,
            "front_camera_count": front_camera_count,
            "rear_camera_main_mp": rear_camera_main_mp,
            "front_camera_main_mp": front_camera_main_mp,
            "os_name": os_encoder.transform([os_name])[0],
        })

        # One-hot columns follow the pattern smartphone_brand_<brand> /
        # processor_brand_<brand>. drop_first=True means the first
        # alphabetical category has no column (all dummies stay 0), which
        # is expected and handled automatically here.
        brand_col = f"smartphone_brand_{brand}"
        if brand_col in row:
            row[brand_col] = 1

        proc_col = f"processor_brand_{processor_brand}"
        if proc_col in row:
            row[proc_col] = 1

        input_df = pd.DataFrame([row])[feature_columns]  # keep exact training order
        input_scaled = scaler.transform(input_df)

        predicted_price = price_model.predict(input_scaled)[0]
        predicted_category_encoded = category_model.predict(input_scaled)[0]
        predicted_category = category_encoder.inverse_transform([predicted_category_encoded])[0]

        r1, r2 = st.columns(2)
        r1.metric("Predicted Price", f"₹{predicted_price:,.0f}")
        r2.metric("Predicted Category", predicted_category)
