"""Streamlit app: "what would this laptop/PC cost?"

Pick hardware specs and get a predicted fair price with a confidence range,
plus a breakdown of which specs push the price up or down.

Run from the repo root:  streamlit run app/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import resolution_to_megapixels  # noqa: E402
from src.model import load_artifacts, predict_from_specs  # noqa: E402

st.set_page_config(page_title="Computer Price Estimator", page_icon="💻")


@st.cache_resource
def get_artifacts():
    return load_artifacts(ROOT / "models")


try:
    predictor, encoder = get_artifacts()
except FileNotFoundError:
    st.error(
        "No trained model found. Run `python -m src.train` first to create "
        "models/model.pkl and models/encoders.pkl."
    )
    st.stop()

st.title("💻 Computer Price Estimator")
st.caption(
    "Estimate a fair market price for a laptop or desktop from its hardware "
    "specs — useful for sanity-checking a listing or pricing a build."
)

options = encoder.options_
ranges = encoder.numeric_ranges_

device_type = st.radio("Device type", options["device_type"], horizontal=True)
defaults = encoder.defaults_by_device_[device_type]


def default_index(field, choices):
    return choices.index(defaults[field]) if defaults[field] in choices else 0


def num_input(label, field, step=None, integer=True):
    lo, hi = ranges[field]["min"], ranges[field]["max"]
    default = defaults.get(field, ranges[field]["median"])
    if integer:
        return st.number_input(
            label, min_value=int(lo), max_value=int(hi),
            value=int(default), step=int(step or 1),
        )
    return st.number_input(
        label, min_value=float(lo), max_value=float(hi),
        value=float(default), step=float(step or 0.1),
    )


col1, col2 = st.columns(2)
with col1:
    st.subheader("Core specs")
    brand = st.selectbox("Brand", options["brand"], index=default_index("brand", options["brand"]))
    cpu_family = st.selectbox("Processor", options["cpu_family"], index=default_index("cpu_family", options["cpu_family"]))
    gpu_model = st.selectbox("Graphics card", options["gpu_model"], index=default_index("gpu_model", options["gpu_model"]))
    ram_gb = st.select_slider("RAM (GB)", [4, 8, 16, 32, 64, 96, 128], value=int(defaults["ram_gb"]))
    storage_type = st.selectbox("Storage type", options["storage_type"], index=default_index("storage_type", options["storage_type"]))
    storage_gb = st.select_slider("Storage (GB)", [128, 256, 512, 1024, 2048, 4096], value=int(defaults["storage_gb"]))
    os_choice = st.selectbox("Operating system", options["os"], index=default_index("os", options["os"]))

with col2:
    st.subheader("Details")
    form_factor = st.selectbox("Form factor", options["form_factor"], index=default_index("form_factor", options["form_factor"]))
    release_year = num_input("Release year", "release_year")
    display_type = st.selectbox("Display type", options["display_type"], index=default_index("display_type", options["display_type"]))
    display_size_in = num_input("Display size (inches)", "display_size_in", step=0.1, integer=False)
    resolution = st.selectbox("Resolution", encoder.resolutions_, index=len(encoder.resolutions_) // 2)
    refresh_hz = num_input("Refresh rate (Hz)", "refresh_hz", step=30)
    warranty_months = num_input("Warranty (months)", "warranty_months", step=12)

with st.expander("Advanced (auto-filled with typical values)"):
    a1, a2 = st.columns(2)
    with a1:
        storage_drive_count = num_input("Storage drives", "storage_drive_count")
        wifi = st.selectbox("Wi-Fi", options["wifi"], index=default_index("wifi", options["wifi"]))
        bluetooth = num_input("Bluetooth version", "bluetooth", integer=False)
        weight_kg = num_input("Weight (kg)", "weight_kg", integer=False)
    with a2:
        battery_wh = num_input("Battery (Wh, laptops)", "battery_wh", step=10)
        charger_watts = num_input("Charger (W, laptops)", "charger_watts", step=10)
        psu_watts = num_input("PSU (W, desktops)", "psu_watts", step=50)

# CPU/GPU internals (tier, cores, clocks, VRAM) follow from the chosen
# component: fill in the typical values seen for it in the training data.
# Brands follow from the component names, so invalid combos are impossible.
cpu_profile = encoder.cpu_profiles_.get(cpu_family, {})
gpu_profile = encoder.gpu_profiles_.get(gpu_model, {})
cpu_tier = cpu_profile.get("cpu_tier", defaults["cpu_tier"])
cpu_cores = cpu_profile.get("cpu_cores", defaults["cpu_cores"])
cpu_threads = cpu_profile.get("cpu_threads", defaults["cpu_threads"])
cpu_base_ghz = cpu_profile.get("cpu_base_ghz", defaults["cpu_base_ghz"])
cpu_boost_ghz = cpu_profile.get("cpu_boost_ghz", defaults["cpu_boost_ghz"])
gpu_tier = gpu_profile.get("gpu_tier", defaults["gpu_tier"])
vram_gb = gpu_profile.get("vram_gb", defaults["vram_gb"])
gpu_brand = {"RTX": "NVIDIA", "RX": "AMD", "Arc": "Intel"}.get(
    gpu_model.split()[0], "Apple"
)
cpu_brand = cpu_family.split()[0]
st.caption(
    f"Auto-filled from typical configurations: {cpu_family} "
    f"(tier {cpu_tier:.0f}, {cpu_cores:.0f} cores @ {cpu_base_ghz:.1f}–"
    f"{cpu_boost_ghz:.1f} GHz) · {gpu_model} "
    f"(tier {gpu_tier:.0f}, {vram_gb:.0f} GB VRAM)"
)

specs = {
    "device_type": device_type,
    "brand": brand,
    "release_year": release_year,
    "os": os_choice,
    "form_factor": form_factor,
    "cpu_brand": cpu_brand,
    "cpu_family": cpu_family,
    "cpu_tier": cpu_tier,
    "cpu_cores": cpu_cores,
    "cpu_threads": cpu_threads,
    "cpu_base_ghz": cpu_base_ghz,
    "cpu_boost_ghz": cpu_boost_ghz,
    "gpu_brand": gpu_brand,
    "gpu_model": gpu_model,
    "gpu_tier": gpu_tier,
    "vram_gb": vram_gb,
    "ram_gb": ram_gb,
    "storage_type": storage_type,
    "storage_gb": storage_gb,
    "storage_drive_count": storage_drive_count,
    "display_type": display_type,
    "display_size_in": display_size_in,
    "megapixels": resolution_to_megapixels(resolution),
    "refresh_hz": refresh_hz,
    "battery_wh": battery_wh,
    "charger_watts": charger_watts,
    "psu_watts": psu_watts,
    "wifi": wifi,
    "bluetooth": bluetooth,
    "weight_kg": weight_kg,
    "warranty_months": warranty_months,
}

if st.button("Estimate price", type="primary", width="stretch"):
    result = predict_from_specs(predictor, encoder, specs)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated fair price", f"${result['price']:,.0f}")
    m2.metric("Low estimate", f"${result['lower']:,.0f}")
    m3.metric("High estimate", f"${result['upper']:,.0f}")
    st.caption(
        f"90% of comparable configurations in the training data sold within "
        f"${result['lower']:,.0f}–${result['upper']:,.0f}."
    )

    st.subheader("What's driving this price?")
    st.caption(
        f"Each bar shows how your choice moves the price versus a typical "
        f"{device_type.lower()} configuration."
    )
    breakdown = predictor.explain_prediction(encoder, specs)
    if breakdown.empty:
        st.info("Your configuration matches a typical machine on every spec.")
    else:
        label_map = {"megapixels": "resolution"}
        breakdown["spec"] = breakdown["spec"].replace(label_map)
        breakdown = breakdown[breakdown["delta"].abs() >= 1]
        chart_data = breakdown.set_index("spec")["delta"]
        st.bar_chart(chart_data, horizontal=True, x_label="Price impact ($)")
        table = breakdown.assign(
            value=breakdown["value"].astype(str),
            impact=breakdown["delta"].map(lambda d: f"{'+' if d >= 0 else '−'}${abs(d):,.0f}"),
        )[["spec", "value", "impact"]]
        st.dataframe(table, hide_index=True, width="stretch")

st.divider()
st.caption(
    "⚠️ Portfolio/educational project. Estimates come from a model trained on "
    "a fixed dataset of computer configurations and are **not** a professional "
    "valuation or an offer of any kind."
)
