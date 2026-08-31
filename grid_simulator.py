import streamlit as st
import pydeck as pdk
import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests
import warnings
import os

# --- CLOUD DEPLOYMENT FIX ---
# Provide a User-Agent nametag to prevent the Overpass API from blocking the Streamlit Cloud IP.
ox.settings.requests_kwargs = {"headers": {"User-Agent": "EV-Grid-Command-Terminal/1.0"}}
ox.settings.requests_timeout = 180
# ----------------------------

if not os.path.exists(".streamlit"):
    os.makedirs(".streamlit")
if not os.path.exists(".streamlit/secrets.toml"):
    with open(".streamlit/secrets.toml", "w") as f:
        f.write('NREL_API_KEY = "ZKe4KCw4IyoPLtafYKWb6uPdDipAx9To9tOTQGry"\n')

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
    ["Spatial Distance (Grid Deficit)", "Thermal Capacity (Feeder Stress)"],
    help="Switch between physical distance visualization and simulated grid load capacity."
)

st.sidebar.markdown("---")
st.sidebar.header("⚖️ Equity & Policy Filters")
j40_filter = st.sidebar.checkbox("Isolate Justice40 DAC Sites", value=False, help="Filter the map to only show candidate sites located within Disadvantaged Communities.")

with st.sidebar.expander("🧠 Methodology & Critical Context", expanded=True):
    st.markdown("""
    **The Visual Metaphor: Pillars vs. Glowing Pads**
    *   **Neon Green Glowing Pads:** These represent the *existing* active DC Fast Charging hubs. They are rendered flat because they have a grid deficit of zero—they are the physical anchors of the current network.
    *   **Extruded 3D Pillars:** These represent *existing gas stations*, acting as our candidate conversion sites. Why gas stations? They are the ultimate "brownfield" targets for EV infrastructure. They already possess the exact physical footprint required: paved pull-through lanes, heavy-duty canopies, high-visibility lighting, and retail amenities (bathrooms, food) crucial for drivers waiting 20-30 minutes for a charge. The height of the pillar visualizes the systemic value of ripping out a gas pump and replacing it with a DCFC node at that specific location.

    **Why a 2.0 Mile Threshold?**
    In urban topologies like Allegheny County, a 2-mile spatial gap is a structural barrier. For the 30%+ of residents in multi-unit dwellings (MUDs) who cannot charge at home, driving over 2 miles exclusively to "fuel up" destroys the EV value proposition. Federal NEVI guidelines prioritize 1-mile buffers from corridors; breaching 2 miles in a metro footprint indicates a stark, unserved "EV Desert."

    **Grid Thermal Limits Explained**
    "Thermal Capacity" refers to the physical heat limit of local distribution wires. A standard 4-port 150kW DCFC station demands 600kW of instantaneous power. Forcing that load through an older commercial feeder without upgrades causes the lines to overheat and melt, blowing local transformers. "Magenta" sites require expensive utility Make-Ready Upgrades before chargers can be installed.
    
    **Justice40 Integration**
    The Justice40 Initiative mandates that 40% of federal clean energy investments flow to Disadvantaged Communities (DACs). Filtering by Justice40 isolates sites that are eligible for prioritized federal grants, merging grid equity with grid expansion. *(Note: DAC status here is modeled deterministically for demonstration).*
    """)

st.sidebar.markdown("---")
st.sidebar.header("📐 Spatial Parameters")
show_arcs = st.sidebar.checkbox("Render Kinetic Deficit Arcs", value=True)
camera_pitch = st.sidebar.slider("Camera Pitch", min_value=30, max_value=60, value=52, step=1)
camera_bearing = st.sidebar.slider("Camera Rotation", min_value=-180, max_value=180, value=-22, step=2)

@st.cache_data
def load_live_data():
    places = ["Allegheny County, Pennsylvania", "Beaver County, Pennsylvania"]
    county_boundaries = ox.geocode_to_gdf(places)
    
    # 1. Candidate Conversion Sites (Gas Stations)
    tags = {"amenity": "fuel"}
    gas_stations_gdf = ox.features_from_place(places, tags=tags)
    gas_stations_gdf = gas_stations_gdf[gas_stations_gdf.geometry.type == "Point"].copy()
    gas_stations_gdf = gas_stations_gdf.to_crs(epsg=4326)
    
    # 2. Fetch Active Fast Chargers
    api_key = st.secrets["NREL_API_KEY"]
    nlr_url = (
        "https://developer.nlr.gov/api/alt-fuel-stations/v1.json?"
        f"api_key={api_key}&fuel_type=ELEC&state=PA&ev_charging_level=dc_fast"
    )
    
    local_chargers_gdf = gpd.GeoDataFrame()
    session = requests.Session()
    session.trust_env = False
    
    try:
        response = session.get(nlr_url, timeout=5)
        if response.status_code == 200:
            stations = response.json().get('alt_fuel_stations', [])
            nlr_df = pd.DataFrame(stations)
            if not nlr_df.empty:
                nlr_gdf = gpd.GeoDataFrame(
                    nlr_df, 
                    geometry=gpd.points_from_xy(nlr_df.longitude, nlr_df.latitude),
                    crs="EPSG:4326"
                )
                local_chargers_gdf = gpd.sjoin(nlr_gdf, county_boundaries, how="inner", predicate="intersects")
                if not local_chargers_gdf.empty:
                    local_chargers_gdf["station_name"] = local_chargers_gdf["station_name"].fillna("DC Fast Charger")
                    local_chargers_gdf["ev_network"] = local_chargers_gdf.get("ev_network", pd.Series(["Unknown"] * len(local_chargers_gdf))).fillna("Unknown")
                    local_chargers_gdf["ev_dc_fast_num"] = local_chargers_gdf.get("ev_dc_fast_num", pd.Series([2] * len(local_chargers_gdf))).fillna(2).astype(int)
    except Exception:
        pass
    
    if local_chargers_gdf.empty:
        tags_ev = {"amenity": "charging_station"}
        ev_osm = ox.features_from_place(places, tags=tags_ev)
        ev_osm = ev_osm.to_crs(epsg=4326)
        ev_osm['geometry'] = ev_osm.geometry.centroid
        local_chargers_gdf = ev_osm.copy()
        local_chargers_gdf["station_name"] = local_chargers_gdf.get("name", pd.Series(["EV Charger"] * len(local_chargers_gdf))).fillna("Local EV Charger")
        local_chargers_gdf["ev_network"] = local_chargers_gdf.get("operator", pd.Series(["Independent"] * len(local_chargers_gdf))).fillna("Independent")
        local_chargers_gdf["ev_dc_fast_num"] = 2 

    # 3. Spatial Math (EPSG:2272)
    gas_m = gas_stations_gdf.to_crs(epsg=2272)
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
    gas_final["source_lon"] = gas_final.geometry.x
    gas_final["source_lat"] = gas_final.geometry.y
    gas_final["site_title"] = gas_final.get("name", pd.Series(["Gas Station"] * len(gas_final))).fillna("Candidate Conversion Site")
    gas_final["ev_dc_fast_num"] = gas_final["ev_dc_fast_num"].fillna(0).astype(int).astype(str)
    
    # Generate deterministic "Stress Score"
    gas_final["stress_score"] = ((gas_final.geometry.x * 1234567).astype(int) % 60) + 40
    gas_final["stress_score_str"] = gas_final["stress_score"].astype(str)
    
    # Simulate Justice40 DAC Status (approx 40% of sites)
    gas_final["is_j40_dac"] = ((gas_final.geometry.y * 7654321).astype(int) % 100) < 40
    gas_final["j40_status"] = gas_final["is_j40_dac"].apply(lambda x: "Yes (Priority Funding Eligible)" if x else "No")

    # Process Charger Nodes 
    chargers_final = local_chargers_gdf.to_crs(epsg=4326)
    chargers_final["lon"] = chargers_final.geometry.x
    chargers_final["lat"] = chargers_final.geometry.y
    chargers_final["site_title"] = chargers_final["station_name"]
    chargers_final["status"] = "Active DCFC Anchor Hub"
    chargers_final["j40_status"] = "N/A (Existing Infrastructure)"
    chargers_final["dist_miles"] = "0.0"
    chargers_final["stress_score_str"] = "Active Load"
    chargers_final["ev_dc_fast_num"] = chargers_final.get("ev_dc_fast_num", 2).astype(str)
    chargers_final["insight"] = "This location is currently operating as a fast charging hub. It serves as a grid anchor node; conversion metrics do not apply."
    
    return (
        pd.DataFrame(gas_final.drop(columns=['geometry'])), 
        pd.DataFrame(chargers_final.drop(columns=['geometry']))
    )

with st.spinner("Compiling 3D spatial network and intelligence briefs..."):
    candidate_df, chargers_df = load_live_data()

# Apply Justice40 Filter if toggled
if j40_filter:
    candidate_df = candidate_df[candidate_df["is_j40_dac"] == True]

# ---------------------------------------------------------
# Dynamic Mode Physics & Safe-HTML Tooltip Generation
# ---------------------------------------------------------
is_stress_mode = "Thermal" in visual_mode

if is_stress_mode:
    st.markdown("Extruding candidate conversion sites based on **simulated electrical grid load stress**. Taller magenta pillars indicate highly constrained local grid capacity.")
    candidate_df["elevation"] = candidate_df["stress_score"] * 30
    
    def evaluate_thermal(row):
        score = row["stress_score"]
        if score > 85: 
            insight = f"🛑 High Cost: Feeder load simulated at {score} percent. Adding a 600kW load will likely exceed thermal limits, triggering $100k+ in utility transformer upgrades."
            return pd.Series(["Critical Load (Over 85%)", insight, [255, 0, 128, 255], [255, 0, 128, 150]])
        elif score > 65: 
            insight = f"⚠️ Moderate Cost: Grid operating at {score} percent base capacity. May support Level 2 infrastructure, but DCFC requires a full utility interconnection study."
            return pd.Series(["High Stress", insight, [255, 140, 0, 240], [255, 140, 0, 150]])
        else: 
            insight = f"✅ Ready to Build: Local circuit has deep headroom ({score} percent baseline load). Grid architecture is plug-and-play ready for high-voltage deployment."
            return pd.Series(["Nominal Capacity", insight, [0, 229, 255, 200], [0, 229, 255, 100]])

    candidate_df[["status", "insight", "pillar_color", "arc_color"]] = candidate_df.apply(evaluate_thermal, axis=1)
    metric_label = "Critical Feeder Nodes"
    metric_val = len(candidate_df[candidate_df["stress_score"] > 85])
    
else:
    st.markdown("Extruding candidate conversion sites into **3D topographic deficit pillars**. Column height represents physical distance to the nearest fast charger.")
    candidate_df["elevation"] = candidate_df["dist_miles"] * 200
    
    def evaluate_distance(row):
        dist = row["dist_miles"]
        ports = row["ev_dc_fast_num"]
        if dist >= 2.0: 
            insight = f"⭐ High Impact: Site is {dist}mi from the nearest node. In dense urban grids, >2 miles represents a structural barrier for local residents lacking home-charging access."
            return pd.Series(["EV Desert (Over 2.0 mi)", insight, [255, 45, 85, 230], [255, 45, 85, 180]])
        elif dist >= 1.0: 
            insight = f"📊 Moderate Impact: Site is {dist}mi away, but nearest hub has only {ports} ports. High risk of queuing delays and local utilization bottlenecks during peak hours."
            return pd.Series(["Moderate Gap", insight, [255, 179, 0, 200], [255, 179, 0, 140]])
        else: 
            insight = f"📉 Low Priority: Area covered. A {ports}-port DCFC hub is just {dist}mi away. Expansion here risks cannibalizing utilization rates of existing infrastructure."
            return pd.Series(["Well-Served", insight, [0, 229, 255, 160], [0, 229, 255, 80]])

    candidate_df[["status", "insight", "pillar_color", "arc_color"]] = candidate_df.apply(evaluate_distance, axis=1)
    metric_label = "EV Deserts (Over 2.0 mi)"
    metric_val = len(candidate_df[candidate_df["dist_miles"] > 2.0])

candidate_df["arc_target_color"] = [[0, 255, 136, 250]] * len(candidate_df) 
chargers_df["color_core"] = chargers_df.apply(lambda x: [0, 255, 136, 255], axis=1) 
chargers_df["color_halo"] = chargers_df.apply(lambda x: [0, 255, 136, 60], axis=1)  

# ---------------------------------------------------------
# Executive KPI Metrics
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Target Sites Analyzed", f"{len(candidate_df):,}")
col2.metric(metric_label, f"{metric_val:,}", delta_color="inverse")
col3.metric("Justice40 Eligible Sites", f"{len(candidate_df[candidate_df['is_j40_dac'] == True]):,}")
col4.metric("Avg Feeder Stress", f"{candidate_df['stress_score'].mean():.1f}%" if len(candidate_df) > 0 else "N/A")

# ---------------------------------------------------------
# PyDeck Layers 
# ---------------------------------------------------------
layers = []

if show_arcs and not candidate_df.empty:
    layer_arcs = pdk.Layer(
        "ArcLayer",
        data=candidate_df,
        get_source_position=["source_lon", "source_lat"],
        get_target_position=["target_lon", "target_lat"],
        get_source_color="arc_color",
        get_target_color="arc_target_color",
        get_width=2.5,
        get_tilt=12,
        pickable=False,
    )
    layers.append(layer_arcs)

if not candidate_df.empty:
    layer_candidates_3d = pdk.Layer(
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
    )
    layers.append(layer_candidates_3d)

if not chargers_df.empty:
    layer_hub_halo = pdk.Layer(
        "ScatterplotLayer",
        data=chargers_df,
        get_position=["lon", "lat"],
        get_fill_color="color_halo",
        get_radius=700,
        pickable=False,
    )
    layer_hub_core = pdk.Layer(
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
    layers.extend([layer_hub_halo, layer_hub_core])

view_state = pdk.ViewState(
    latitude=candidate_df["source_lat"].mean() - 0.05 if not candidate_df.empty else 40.4406,
    longitude=candidate_df["source_lon"].mean() if not candidate_df.empty else -79.9959,
    zoom=9.8,
    pitch=camera_pitch,
    bearing=camera_bearing
)

tooltip_html = (
    "<div style='font-family: Consolas, monospace; padding: 10px; font-size: 12px; background: rgba(13, 17, 23, 0.95); border: 1px solid #30363d; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); max-width: 310px; white-space: normal;'>"
    "<b style='font-size: 14px; color: #58a6ff;'>{site_title}</b><br/>"
    "<hr/>"
    "<span style='color: #8b949e;'>Classification:</span> <b style='color: white;'>{status}</b><br/>"
    "<span style='color: #8b949e;'>Justice40 DAC:</span> <b style='color: #00ff88;'>{j40_status}</b><br/>"
    "<span style='color: #8b949e;'>Nearest DCFC:</span> {dist_miles} miles ({ev_dc_fast_num} ports)<br/>"
    "<span style='color: #8b949e;'>Grid Stress Score:</span> {stress_score_str}% capacity<br/>"
    "<hr/>"
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

st.pydeck_chart(r, use_container_width=True, height=850)
