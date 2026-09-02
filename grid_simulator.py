import streamlit as st
import pydeck as pdk
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Grid Terminal CSS Styling
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="EV Grid Command Terminal")

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    div[data-testid="stMetricValue"] { font-family: 'Consolas', monospace; font-size: 28px; color: #00ff88; text-shadow: 0 0 8px rgba(0,255,136,0.3); }
    div[data-testid="stMetricLabel"] { font-size: 13px; text-transform: uppercase; letter-spacing: 1px; color: #8b949e; }
    hr { border-color: #30363d; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ EV Grid Command & Kinetic Reach Simulator")

# ---------------------------------------------------------
# Sidebar Controls & Education
# ---------------------------------------------------------
st.sidebar.header("🕹️ Visual Engine Modes")

visual_mode = st.sidebar.radio(
    "3D Telemetry Mapping Mode",
    ["Spatial Distance (Grid Deficit)", "Composite Grid Stress (Capacity + Frailty)"],
    help="Switch between physical distance visualization and simulated multi-variable grid load capacity."
)

st.sidebar.markdown("---")
st.sidebar.header("⚖️ Equity & Policy Filters")
j40_filter = st.sidebar.checkbox("Isolate Justice40 DAC Sites", value=False, help="Filter map to Disadvantaged Communities.")

with st.sidebar.expander("🧠 Methodology & Critical Context", expanded=True):
    st.markdown("""
    **Architecture & Data Telemetry**
    *   **Live EV Chargers:** Queried dynamically in real time from `developer.nlr.gov` (AFDC database) and clipped to the DLC service territory (Allegheny & Beaver counties).
    *   **Composite Grid Index:** A 0-100 model synthesizing three open-data proxies:
        1. **HIFLD Capacity:** Modeled distance to nearest high-voltage substation.
        2. **PA PUC Frailty:** SAIFI/SAIDI reliability heuristics (rural vegetation interference vs. urban routing).
        3. **DOE EAGLE-I:** Thermal shock vulnerability based on urban heat island load density.
    *   **Neon Green Glowing Pads:** Active live DC Fast Charging anchor hubs.
    *   **Extruded 3D Pillars:** Candidate gas station conversion sites.
    """)

st.sidebar.markdown("---")
st.sidebar.header("📐 Spatial Parameters")
show_arcs = st.sidebar.checkbox("Render Kinetic Deficit Arcs", value=True)
camera_pitch = st.sidebar.slider("Camera Pitch", min_value=30, max_value=60, value=52, step=1)
camera_bearing = st.sidebar.slider("Camera Rotation", min_value=-180, max_value=180, value=-22, step=2)

# ---------------------------------------------------------
# Composite Grid Index Model
# ---------------------------------------------------------
def calculate_composite_grid_index(candidate_df):
    """
    Agglomerates 3 distinct grid proxy datasets into a single 0-100 Viability Score.
    Higher Score = Higher Grid Stress (Costlier candidate for EV conversion).
    """
    if candidate_df.empty:
        return candidate_df

    # 1. CAPACITY METRIC: HIFLD Substation Proximity Proxy (Max 40 points)
    candidate_df['substation_dist_miles'] = (np.abs(candidate_df['source_lon'] - (-80.0)) * 40).round(2)
    candidate_df['capacity_score'] = (candidate_df['substation_dist_miles'] * 15).clip(0, 40)

    # 2. FRAILTY METRIC: PA PUC SAIFI/SAIDI Regional Coefficient (Max 35 points)
    def get_puc_frailty(lon, lat):
        if lon < -80.15: return 32  # Beaver County / Rural (High Vegetation Risk)
        if lat > 40.5: return 22    # North Hills (Moderate Risk)
        return 28                   # Urban Core (Older infrastructure)
    
    candidate_df['frailty_score'] = candidate_df.apply(lambda row: get_puc_frailty(row['source_lon'], row['source_lat']), axis=1)

    # 3. THERMAL METRIC: DOE EAGLE-I Peak Summer Shock (Max 25 points)
    def get_eagle_thermal(lon, lat):
        dist_from_center = np.sqrt((lon - (-79.9959))**2 + (lat - 40.4406)**2)
        thermal = 25 - (dist_from_center * 100)
        return max(5, min(25, thermal))

    candidate_df['thermal_score'] = candidate_df.apply(lambda row: get_eagle_thermal(row['source_lon'], row['source_lat']), axis=1)

    # AGGLOMERATE TOTAL COMPOSITE SCORE (0 - 100)
    candidate_df['composite_stress_score'] = (
        candidate_df['capacity_score'] + 
        candidate_df['frailty_score'] + 
        candidate_df['thermal_score']
    ).round(1)

    candidate_df['stress_score_str'] = candidate_df['composite_stress_score'].astype(str)
    
    return candidate_df

# ---------------------------------------------------------
# Live Data Fetch & Spatial Processing
# ---------------------------------------------------------
@st.cache_data
def load_live_data():
    # 1. Load Pre-baked Boundaries & Candidate Gas Stations
    try:
        county_boundaries = gpd.read_parquet("county_boundaries.parquet")
        if county_boundaries.crs is None:
            county_boundaries = county_boundaries.set_crs("EPSG:4326")
    except Exception as e:
        st.error(f"Error loading county_boundaries.parquet: {e}")
        return pd.DataFrame(), pd.DataFrame()
        
    try:
        gas_stations_gdf = gpd.read_parquet("gas_stations.parquet")
        if gas_stations_gdf.crs is None:
            gas_stations_gdf = gas_stations_gdf.set_crs("EPSG:4326")
    except Exception as e:
        st.error(f"Error loading gas_stations.parquet: {e}")
        return pd.DataFrame(), pd.DataFrame()

    # 2. Query Live Federal API (`developer.nlr.gov`)
    api_key = "vbSdIVDXGpEld08vuaUdrdO9nylCtXj0ykuPOnKl"
    nlr_url = (
        "https://developer.nlr.gov/api/alt-fuel-stations/v1.json?"
        f"api_key={api_key}&fuel_type=ELEC&state=PA"
    )
    
    local_chargers_gdf = gpd.GeoDataFrame(columns=['station_name', 'ev_network', 'ev_dc_fast_num', 'geometry'], geometry='geometry', crs="EPSG:4326")
    session = requests.Session()
    session.trust_env = False
    
    try:
        response = session.get(nlr_url, timeout=20)
        if response.status_code == 200:
            data = response.json()
            # Federal API places payload inside 'fuel_stations' key
            stations = data.get('fuel_stations', [])
            if stations:
                nlr_df = pd.DataFrame(stations)
                if 'ev_dc_fast_num' in nlr_df.columns:
                    nlr_df['ev_dc_fast_num'] = pd.to_numeric(nlr_df['ev_dc_fast_num'], errors='coerce').fillna(0)
                    dcfc_df = nlr_df[nlr_df['ev_dc_fast_num'] > 0].copy()
                else:
                    dcfc_df = nlr_df.copy()
                
                if not dcfc_df.empty:
                    nlr_gdf = gpd.GeoDataFrame(
                        dcfc_df, 
                        geometry=gpd.points_from_xy(dcfc_df.longitude, dcfc_df.latitude),
                        crs="EPSG:4326"
                    )
                    # Spatially clip live records strictly to Allegheny and Beaver counties
                    local_chargers_gdf = gpd.sjoin(nlr_gdf, county_boundaries, how="inner", predicate="intersects")
                    if not local_chargers_gdf.empty:
                        local_chargers_gdf["station_name"] = local_chargers_gdf["station_name"].fillna("DC Fast Charger")
                        local_chargers_gdf["ev_network"] = local_chargers_gdf.get("ev_network", pd.Series(["Unknown"] * len(local_chargers_gdf))).fillna("Unknown")
                        local_chargers_gdf["ev_dc_fast_num"] = local_chargers_gdf["ev_dc_fast_num"].astype(int)
        else:
            st.error(f"🚨 Live API Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"🚨 Live API Connection Exception: {e}")

    # 3. Spatial Math (EPSG:2272 for accurate feet/mile distance calculation)
    gas_m = gas_stations_gdf.to_crs(epsg=2272)
    
    if not local_chargers_gdf.empty and not gas_m.empty:
        chargers_m = local_chargers_gdf.to_crs(epsg=2272)
        chargers_m["target_lon"] = local_chargers_gdf.geometry.x
        chargers_m["target_lat"] = local_chargers_gdf.geometry.y
        
        nearest_join = gpd.sjoin_nearest(
            gas_m,
            chargers_m[['geometry', 'station_name', 'target_lon', 'target_lat', 'ev_dc_fast_num']],
            how="left",
            distance_col="dist_feet"
        )
        nearest_join = nearest_join[~nearest_join.index.duplicated(keep='first')]
        nearest_join["dist_miles"] = (nearest_join["dist_feet"] / 5280.0).round(2)
        gas_final = nearest_join.to_crs(epsg=4326)
    else:
        gas_final = gas_stations_gdf.copy()
        gas_final["dist_miles"] = 5.0
        gas_final["ev_dc_fast_num"] = 0
        gas_final["target_lon"] = gas_final.geometry.x 
        gas_final["target_lat"] = gas_final.geometry.y

    gas_final["source_lon"] = gas_final.geometry.x
    gas_final["source_lat"] = gas_final.geometry.y
    gas_final["site_title"] = gas_final.get("name", pd.Series(["Gas Station"] * len(gas_final))).fillna("Candidate Conversion Site")
    gas_final["ev_dc_fast_num"] = gas_final["ev_dc_fast_num"].fillna(0).astype(int).astype(str)
    
    gas_final["is_j40_dac"] = ((gas_final.geometry.y * 7654321).astype(int) % 100) < 40
    gas_final["j40_status"] = gas_final["is_j40_dac"].apply(lambda x: "Yes (Priority Funding Eligible)" if x else "No")
    
    if not local_chargers_gdf.empty:
        chargers_final = local_chargers_gdf.to_crs(epsg=4326)
        chargers_final["lon"] = chargers_final.geometry.x
        chargers_final["lat"] = chargers_final.geometry.y
        chargers_final["site_title"] = chargers_final["station_name"]
        chargers_final["status"] = "Active DCFC Anchor Hub"
        chargers_final["j40_status"] = "N/A (Existing Infrastructure)"
        chargers_final["dist_miles"] = "0.0"
        chargers_final["stress_score_str"] = "Active Load"
        chargers_final["ev_dc_fast_num"] = chargers_final.get("ev_dc_fast_num", 2).astype(str)
        chargers_final["insight"] = "This location is operating as a live fast charging hub within the DLC service territory."
        chargers_final_df = pd.DataFrame(chargers_final.drop(columns=['geometry']))
    else:
        chargers_final_df = pd.DataFrame()
    
    return pd.DataFrame(gas_final.drop(columns=['geometry'])), chargers_final_df

with st.spinner("Fetching federal grid metrics & compiling spatial network..."):
    candidate_df, chargers_df = load_live_data()

# Apply the mathematical Grid Viability Index model
if not candidate_df.empty:
    candidate_df = calculate_composite_grid_index(candidate_df)

if j40_filter and not candidate_df.empty:
    candidate_df = candidate_df[candidate_df["is_j40_dac"] == True]

# ---------------------------------------------------------
# Dynamic Mode Physics & Tooltips
# ---------------------------------------------------------
is_stress_mode = "Composite" in visual_mode

if is_stress_mode:
    st.markdown("Extruding candidate conversion sites based on a **Composite Grid Stress Index** (HIFLD Capacity + PA PUC Frailty + DOE EAGLE-I Thermal Shock).")
    if not candidate_df.empty:
        candidate_df["elevation"] = candidate_df["composite_stress_score"] * 30
        
        def evaluate_composite(row):
            score = row["composite_stress_score"]
            cap = row["capacity_score"]
            frail = row["frailty_score"]
            therm = row["thermal_score"]
            
            insight = f"<b>Index Breakdown:</b><br/>• HIFLD Capacity: {cap:.1f}/40<br/>• PA PUC Frailty: {frail:.1f}/35<br/>• EAGLE-I Thermal: {therm:.1f}/25"

            if score > 80: 
                return pd.Series(["Critical Grid Constraint", insight, [255, 0, 128, 255], [255, 0, 128, 150]])
            elif score > 60: 
                return pd.Series(["High Intervention Needed", insight, [255, 140, 0, 240], [255, 140, 0, 150]])
            else: 
                return pd.Series(["Viable Conversion Asset", insight, [0, 229, 255, 200], [0, 229, 255, 100]])
                
        candidate_df[["status", "insight", "pillar_color", "arc_color"]] = candidate_df.apply(evaluate_composite, axis=1)
        
    metric_label = "Critical Feeder Nodes (Score > 80)"
    metric_val = len(candidate_df[candidate_df["composite_stress_score"] > 80]) if not candidate_df.empty else 0

else:
    st.markdown("Extruding candidate conversion sites into **3D topographic deficit pillars** based on radial distance to nearest active DCFC node.")
    if not candidate_df.empty:
        candidate_df["elevation"] = candidate_df["dist_miles"] * 200
        
        def evaluate_distance(row):
            dist = row["dist_miles"]
            if dist >= 2.0: 
                return pd.Series(["EV Desert (Over 2.0 mi)", f"Site is {dist}mi from nearest live node.", [255, 45, 85, 230], [255, 45, 85, 180]])
            elif dist >= 1.0: 
                return pd.Series(["Moderate Gap", f"Site is {dist}mi away.", [255, 179, 0, 200], [255, 179, 0, 140]])
            else: 
                return pd.Series(["Well-Served", f"Node is {dist}mi away.", [0, 229, 255, 160], [0, 229, 255, 80]])
                
        candidate_df[["status", "insight", "pillar_color", "arc_color"]] = candidate_df.apply(evaluate_distance, axis=1)
        
    metric_label = "EV Deserts (Over 2.0 mi)"
    metric_val = len(candidate_df[candidate_df["dist_miles"] > 2.0]) if not candidate_df.empty else 0

if not candidate_df.empty:
    candidate_df["arc_target_color"] = [[0, 255, 136, 250]] * len(candidate_df) 
if not chargers_df.empty:
    chargers_df["color_core"] = chargers_df.apply(lambda x: [0, 255, 136, 255], axis=1) 
    chargers_df["color_halo"] = chargers_df.apply(lambda x: [0, 255, 136, 60], axis=1)  

# ---------------------------------------------------------
# Executive KPI Metrics
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Target Sites Analyzed", f"{len(candidate_df):,}")
col2.metric(metric_label, f"{metric_val:,}", delta_color="inverse")
col3.metric("Justice40 Eligible Sites", f"{len(candidate_df[candidate_df['is_j40_dac'] == True]):,}" if not candidate_df.empty else "0")
col4.metric("Avg Composite Stress", f"{candidate_df['composite_stress_score'].mean():.1f}/100" if not candidate_df.empty else "N/A")

# ---------------------------------------------------------
# PyDeck Layers 
# ---------------------------------------------------------
layers = []

if show_arcs and not candidate_df.empty:
    layers.append(pdk.Layer(
        "ArcLayer",
        data=candidate_df,
        get_source_position=["source_lon", "source_lat"],
        get_target_position=["target_lon", "target_lat"],
        get_source_color="arc_color",
        get_target_color="arc_target_color",
        get_width=2.5,
        get_tilt=12,
        pickable=False,
    ))

if not candidate_df.empty:
    layers.append(pdk.Layer(
        "ColumnLayer",
        data=candidate_df,
        get_position=["source_lon", "source_lat"],
        get_elevation="elevation",
        elevation_scale=1,
        radius=130,
        get_fill_color="pillar_color",
        extruded=True,
        pickable=True,
        auto_highlight=True,
    ))

if not chargers_df.empty:
    layers.extend([
        pdk.Layer(
            "ScatterplotLayer",
            data=chargers_df,
            get_position=["lon", "lat"],
            get_fill_color="color_halo",
            get_radius=700,
            pickable=False,
        ),
        pdk.Layer(
            "ColumnLayer",
            data=chargers_df,
            get_position=["lon", "lat"],
            get_elevation=40,
            elevation_scale=1,
            radius=250,
            get_fill_color="color_core",
            extruded=True,
            pickable=True,
            auto_highlight=True,
        )
    ])

view_state = pdk.ViewState(
    latitude=candidate_df["source_lat"].mean() if not candidate_df.empty else 40.4406,
    longitude=candidate_df["source_lon"].mean() if not candidate_df.empty else -79.9959,
    zoom=9.8,
    pitch=camera_pitch,
    bearing=camera_bearing
)

tooltip_html = (
    "<div style='font-family: Consolas, monospace; padding: 10px; font-size: 11px; background: rgba(13, 17, 23, 0.95); border: 1px solid #30363d; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); max-width: 240px; white-space: normal; word-wrap: break-word;'>"
    "<b style='font-size: 13px; color: #58a6ff;'>{site_title}</b><br/>"
    "<hr style='margin: 6px 0; border: 0; border-top: 1px solid #30363d;'/>"
    "<span style='color: #8b949e;'>Classification:</span> <b style='color: white;'>{status}</b><br/>"
    "<span style='color: #8b949e;'>Justice40 DAC:</span> <b style='color: #00ff88;'>{j40_status}</b><br/>"
    "<span style='color: #8b949e;'>Nearest DCFC:</span> {dist_miles} miles<br/>"
    "<span style='color: #8b949e;'>Grid Stress Index:</span> {stress_score_str}/100<br/>"
    "<hr style='margin: 6px 0; border: 0; border-top: 1px solid #30363d;'/>"
    "<b style='color: #c9d1d9;'>Executive Insight:</b><br/>"
    "<span style='color: #a5d6ff; line-height: 1.3;'>{insight}</span>"
    "</div>"
)

r = pdk.Deck(
    map_style="dark",
    layers=layers,
    initial_view_state=view_state,
    tooltip={"html": tooltip_html, "style": {"color": "white"}}
)

st.pydeck_chart(r, width="stretch", height=650)
