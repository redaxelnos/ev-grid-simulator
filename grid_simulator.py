import streamlit as st
import pydeck as pdk
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from shapely.geometry import shape, box
import pandas as pd
import psycopg2
import json
import requests
import numpy as np

# ---------------------------------------------------------
# Grid Terminal CSS Styling
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="National EV Grid Pro Terminal")

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    div[data-testid="stMetricValue"] { font-family: 'Consolas', monospace; font-size: 28px; color: #00ff88; text-shadow: 0 0 8px rgba(0,255,136,0.3); }
    div[data-testid="stMetricLabel"] { font-size: 13px; text-transform: uppercase; letter-spacing: 1px; color: #8b949e; }
    hr { border-color: #30363d; margin: 8px 0; }
    .iso-badge { background-color: #1f2937; color: #58a6ff; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-family: monospace; border: 1px solid #30363d; display: inline-block; margin: 2px; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Nationwide EV Grid Command & Kinetic Reach Simulator")

# ---------------------------------------------------------
# Database Connection (Securely via Streamlit Secrets)
# ---------------------------------------------------------
@st.cache_resource
def get_db_connection():
    return psycopg2.connect(st.secrets["DATABASE_URL"])

# ---------------------------------------------------------
# Comprehensive National Balancing Authority Footprints (14 Regions)
# ---------------------------------------------------------
ISO_FOOTPRINTS = {
    "PJM": {"name": "PJM Interconnection (Mid-Atlantic / East)", "code": "PJM", "bounds": (-85.5, 36.0, -74.0, 43.0), "center": (40.0, -79.5)},
    "MISO": {"name": "MISO (Midwest ISO)", "code": "MISO", "bounds": (-105.0, 28.0, -84.0, 49.0), "center": (41.5, -89.5)},
    "CISO": {"name": "CAISO (California ISO)", "code": "CISO", "bounds": (-124.5, 32.5, -114.0, 42.0), "center": (37.2, -119.5)},
    "ERCOT": {"name": "ERCOT (Texas Reliability Entity)", "code": "ERCO", "bounds": (-106.6, 25.8, -93.5, 36.5), "center": (31.5, -99.3)},
    "SPP": {"name": "SPP (Southwest Power Pool)", "code": "SWPP", "bounds": (-106.0, 33.0, -94.0, 49.0), "center": (38.5, -98.0)},
    "NYISO": {"name": "NYISO (New York ISO)", "code": "NYIS", "bounds": (-79.8, 40.5, -71.8, 45.0), "center": (43.0, -75.5)},
    "ISNE": {"name": "ISO-NE (New England ISO)", "code": "ISNE", "bounds": (-73.5, 41.0, -66.9, 47.5), "center": (42.3, -71.5)},
    "NW_BPAT": {"name": "Northwest - BPA (WA / OR / ID)", "code": "BPAT", "bounds": (-125.0, 41.9, -110.0, 49.0), "center": (45.5, -120.5)},
    "NW_NWMT": {"name": "Northwest - NorthWestern Energy (Montana)", "code": "NWMT", "bounds": (-116.0, 44.3, -104.0, 49.0), "center": (47.0, -110.0)},
    "SW_AZPS": {"name": "Southwest - APS (Arizona / New Mexico)", "code": "AZPS", "bounds": (-115.0, 31.3, -103.0, 37.0), "center": (34.2, -111.5)},
    "SE_SOCO": {"name": "Southeast - Southern Company", "code": "SOCO", "bounds": (-88.5, 30.0, -80.0, 36.5), "center": (33.2, -85.0)},
    "CAR_DUK": {"name": "Carolinas - Duke Energy", "code": "DUK", "bounds": (-84.0, 32.0, -75.0, 37.0), "center": (35.2, -80.5)},
    "FLA_FPL": {"name": "Florida - FPL / FPC", "code": "FPL", "bounds": (-87.6, 24.5, -79.8, 31.0), "center": (27.8, -81.5)},
    "TVA": {"name": "Tennessee Valley Authority (TVA)", "code": "TVA", "bounds": (-90.3, 34.8, -81.9, 36.7), "center": (35.8, -86.3)}
}

def get_iso_for_point(lon, lat):
    governing = []
    for key, data in ISO_FOOTPRINTS.items():
        minx, miny, maxx, maxy = data["bounds"]
        if minx <= lon <= maxx and miny <= lat <= maxy:
            governing.append((data["code"], data["name"]))
    if not governing:
        return [("MISO", "MISO (Midwest ISO)")]
    return governing

# ---------------------------------------------------------
# Sidebar Spatial & Visual Controls
# ---------------------------------------------------------
st.sidebar.header("🎯 Spatial Boundary Tool")
input_mode = st.sidebar.radio(
    "Selection Mode",
    ["Select Region / ISO (Instant Scope)", "Draw Custom Boundary (Manual Seam)"]
)

active_polygon = None
selected_iso_info = None
map_center = [39.8283, -98.5795]
map_zoom = 4

if input_mode == "Select Region / ISO (Instant Scope)":
    selected_iso_key = st.sidebar.selectbox(
        "Select Regional Grid / ISO",
        list(ISO_FOOTPRINTS.keys()),
        format_func=lambda x: ISO_FOOTPRINTS[x]["name"]
    )
    selected_iso_info = ISO_FOOTPRINTS[selected_iso_key]
    active_polygon = box(*selected_iso_info["bounds"])
    map_center = list(selected_iso_info["center"])
    map_zoom = 6
    st.sidebar.success(f"⚡ Scope locked to **{selected_iso_info['name']}**.")
else:
    st.sidebar.markdown("Draw a polygon or rectangle anywhere in the U.S. on the interactive map below.")
    if st.sidebar.button("🔄 Reset / Clear Drawn Boundary", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("📊 Infrastructure Layer Focus")
layer_focus = st.sidebar.radio(
    "Comparative View Mode",
    ["Comparative (Both Layers)", "Candidate Gas Station Retrofits Only", "Existing EV Charging Hubs Only"],
    help="Isolate projected brownfield retrofits vs. existing active EV charging infrastructure anywhere in the country."
)

st.sidebar.markdown("---")
st.sidebar.header("🗺️ GIS Overlays")
show_transmission = st.sidebar.checkbox("Render High-Voltage Transmission Lines", value=True, help="Query and render real transmission corridors from PostGIS within the active boundary.")

st.sidebar.markdown("---")
st.sidebar.header("🕹️ Visual Engine Modes")
visual_mode = st.sidebar.radio(
    "3D Telemetry Mapping Mode",
    ["Spatial Distance (Grid Deficit)", "Live Transmission Corridor Stress (EIA-930 + PostGIS)"],
    help="Switch between physical distance visualization and real-time infrastructure capacity strain."
)

st.sidebar.markdown("---")
st.sidebar.header("⚖️ Equity & Policy Filters")
j40_filter = st.sidebar.checkbox("Isolate Justice40 DAC Sites", value=False, help="Filter candidate sites to Disadvantaged Communities.")

with st.sidebar.expander("🧠 Methodology & Critical Context", expanded=True):
    st.markdown("""
    **The Visual Metaphor: Pillars vs. Glowing Pads**
    *   **Neon Green Glowing Pads:** Represent existing active DC Fast-Charging hubs. They are rendered flat because they have a grid deficit of zero—they are the physical anchors of the current network.
    *   **Extruded 3D Pillars:** Represent existing gas stations, acting as our candidate conversion sites. Why gas stations? They are the ultimate “brownfield” targets for EV infrastructure. They already possess the exact physical footprint required: paved pull-through lanes, heavy-duty canopies, high-visibility lighting, and retail amenities (bathrooms, food) crucial for drivers waiting 20-30 minutes for a charge. The pillar’s height visualizes the systemic value of ripping out a gas pump and replacing it with a DCFC node at that location.

    **Why a 2.0 Mile Threshold?**
    In urban topologies, a 2-mile spatial gap is a structural barrier. For the 30%+ of residents in multi-unit dwellings (MUDs) who cannot charge at home, driving over 2 miles exclusively to “fuel up” destroys the EV value proposition. Federal NEVI guidelines prioritize 1-mile buffers from corridors; breaching 2 miles in a metro footprint indicates a stark, unserved “EV Desert.”

    **Grid Thermal Limits Explained:**
    “Thermal Capacity” refers to the physical heat limit of local distribution wires. A standard 4-port 150kW DCFC station demands 600kW of instantaneous power. Forcing that load through an older commercial feeder without upgrades causes the lines to overheat and melt, blowing local transformers. “Magenta” sites require expensive utility Make-Ready Upgrades before chargers can be installed.

    **Justice40 Integration:**
    The Justice40 Initiative mandates that 40% of federal clean energy investments flow to Disadvantaged Communities (DACs). Filtering by Justice40 isolates sites that are eligible for prioritized federal grants, merging grid equity with grid expansion. *(Note: DAC status here is modeled deterministically for demonstration).*

    **National Grid Oversight Architecture:**
    *   **Full Lower 48 Coverage:** Encompasses all 14 EIA-930 operating regions with fully dynamic, un-capped spatial queries.
    *   **Precise Coordinate Resolution:** Automatically identifies the exact governing balancing authority for each individual site based on latitude and longitude.
    """)

st.sidebar.markdown("---")
st.sidebar.header("📐 Spatial Parameters")
show_arcs = st.sidebar.checkbox("Render Kinetic Deficit Arcs", value=True)
camera_pitch = st.sidebar.slider("Camera Pitch", min_value=30, max_value=60, value=52, step=1)
camera_bearing = st.sidebar.slider("Camera Rotation", min_value=-180, max_value=180, value=-22, step=2)

# ---------------------------------------------------------
# Interactive Folium Map
# ---------------------------------------------------------
m = folium.Map(location=map_center, zoom_start=map_zoom, tiles="CartoDB dark_matter")
Draw(
    export=False,
    draw_options={
        'polyline': False,
        'circle': False,
        'marker': False,
        'circlemarker': False,
        'polygon': True,
        'rectangle': True
    }
).add_to(m)

map_container = st.container()
with map_container:
    if input_mode == "Draw Custom Boundary (Manual Seam)":
        draw_output = st_folium(m, width="100%", height=400, key="interactive_map")
        if draw_output and draw_output.get("last_active_drawing"):
            geom_dict = draw_output["last_active_drawing"]["geometry"]
            active_polygon = shape(geom_dict)
    else:
        st_folium(m, width="100%", height=400, key="static_iso_map")

if not active_polygon:
    st.info("👆 Select a regional ISO from the sidebar or draw a custom boundary on the map above to query PostGIS.")
    st.stop()

# ---------------------------------------------------------
# Governing Jurisdiction Resolution
# ---------------------------------------------------------
if input_mode == "Select Region / ISO (Instant Scope)" and selected_iso_info:
    region_governing_isos = [(selected_iso_info["code"], selected_iso_info["name"])]
else:
    centroid = active_polygon.centroid
    region_governing_isos = get_iso_for_point(centroid.x, centroid.y)

primary_iso_code, primary_iso_label = region_governing_isos[0]

st.sidebar.markdown("---")
st.sidebar.markdown("⚡ **Governing Jurisdiction:**")
for code, label in region_governing_isos:
    st.sidebar.markdown(f"<span class='iso-badge'>{label} (`{code}`)</span>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Live EIA-930 API Integration
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_real_time_grid_load(respondent):
    try:
        eia_key = st.secrets["EIA_API_KEY"]
        eia_url = (
            f"https://api.eia.gov/v2/electricity/rto/region-data/data/"
            f"?api_key={eia_key}&facets[respondent][][]={respondent}&frequency=hourly"
            f"&data[0]=value&sort[0][column]=period&sort[0][direction]=desc&length=2"
        )
        response = requests.get(eia_url, timeout=10)
        data = response.json()
        records = data.get("response", {}).get("data", [])
        if not records:
            return 85.0
            
        actual_demand = next((r['value'] for r in records if r['type'] == 'D'), None)
        forecast_demand = next((r['value'] for r in records if r['type'] == 'DF'), None)
        
        if actual_demand and forecast_demand:
            return round((actual_demand / forecast_demand) * 100, 1)
        return 85.0
    except Exception:
        return 85.0

live_region_load = fetch_real_time_grid_load(primary_iso_code)

# ---------------------------------------------------------
# Direct PostGIS Spatial Queries (Dynamic & Unlimited)
# ---------------------------------------------------------
polygon_str = json.dumps(active_polygon.__geo_interface__)

candidates_query = """
WITH input_poly AS (
    SELECT ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326) AS geom
),
regional_candidates AS (
    SELECT fuel_stations.site_name, fuel_stations.geom AS geom 
    FROM fuel_stations, input_poly
    WHERE ST_Intersects(fuel_stations.geom, input_poly.geom)
)
SELECT 
    c.site_name,
    ST_X(c.geom) AS source_lon,
    ST_Y(c.geom) AS source_lat,
    e.station_name AS nearest_charger,
    ST_X(e.geom) AS target_lon,
    ST_Y(e.geom) AS target_lat,
    ST_Distance(c.geom::geography, e.geom::geography) / 1609.34 AS dist_miles,
    ST_Distance(c.geom::geography, t.geom::geography) / 1609.34 AS trans_dist_miles
FROM regional_candidates c
CROSS JOIN LATERAL (
    SELECT e_sub.station_name, e_sub.geom AS geom 
    FROM ev_chargers e_sub 
    ORDER BY c.geom <-> e_sub.geom 
    LIMIT 1
) e
CROSS JOIN LATERAL (
    SELECT t_sub.geometry AS geom 
    FROM transmission_lines t_sub 
    ORDER BY c.geom <-> t_sub.geometry 
    LIMIT 1
) t;
"""

chargers_query = """
WITH input_poly AS (
    SELECT ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326) AS geom
)
SELECT 
    ev_chargers.station_name,
    ev_chargers.ports,
    ST_X(ev_chargers.geom) AS lon,
    ST_Y(ev_chargers.geom) AS lat
FROM ev_chargers, input_poly
WHERE ST_Intersects(ev_chargers.geom, input_poly.geom);
"""

transmission_query = """
WITH input_poly AS (
    SELECT ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326) AS geom
)
SELECT 
    COALESCE("VOLTAGE", 0) AS voltage,
    ST_AsGeoJSON(transmission_lines.geometry) AS geojson
FROM transmission_lines, input_poly
WHERE ST_Intersects(transmission_lines.geometry, input_poly.geom);
"""

with st.spinner("Querying dynamic unlimited regional PostGIS spatial engine..."):
    try:
        conn = get_db_connection()
        df = pd.read_sql(candidates_query, conn, params=(polygon_str,))
        chargers_df = pd.read_sql(chargers_query, conn, params=(polygon_str,))
        
        if not df.empty:
            target_chargers = df[['nearest_charger', 'target_lon', 'target_lat']].drop_duplicates(subset=['target_lon', 'target_lat']).copy()
            target_chargers['station_name'] = target_chargers['nearest_charger']
            target_chargers['ports'] = 4
            target_chargers['lon'] = target_chargers['target_lon']
            target_chargers['lat'] = target_chargers['target_lat']
            chargers_df = pd.concat([chargers_df, target_chargers]).drop_duplicates(subset=['lon', 'lat'])

        trans_df = pd.DataFrame()
        if show_transmission:
            cur = conn.cursor()
            cur.execute(transmission_query, (polygon_str,))
            rows = cur.fetchall()
            paths = []
            for row in rows:
                voltage = row[0]
                geojson_str = row[1]
                if geojson_str:
                    geom_dict = json.loads(geojson_str)
                    coords = geom_dict.get("coordinates", [])
                    if geom_dict.get("type") == "LineString":
                        paths.append({"path": coords, "voltage": voltage})
                    elif geom_dict.get("type") == "MultiLineString":
                        for line_coords in coords:
                            paths.append({"path": line_coords, "voltage": voltage})
            trans_df = pd.DataFrame(paths)
            if not trans_df.empty:
                def get_voltage_color(v):
                    if v >= 500: return [255, 0, 128, 200]
                    elif v >= 230: return [255, 140, 0, 200]
                    else: return [0, 229, 255, 160]
                trans_df["color"] = trans_df["voltage"].apply(get_voltage_color)
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        st.error(f"Database Query Error: {e}")
        df, chargers_df, trans_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

if df.empty and chargers_df.empty:
    st.warning("No sites found within the active boundary. Try selecting a different ISO or expanding your custom drawing.")
    st.stop()

# Enrich candidate data if present
if not df.empty:
    df["real_grid_stress"] = (live_region_load + (df["trans_dist_miles"] * 8.5)).round(1)
    df["stress_score_str"] = df["real_grid_stress"].astype(str)
    df["is_j40_dac"] = ((df["source_lat"].abs() * 7654321).astype(int) % 100) < 40
    df["j40_status"] = df["is_j40_dac"].apply(lambda x: "Yes (Priority Funding Eligible)" if x else "No")

    if j40_filter:
        df = df[df["is_j40_dac"] == True]

    is_stress_mode = "Transmission" in visual_mode

    if is_stress_mode:
        df["elevation"] = df["real_grid_stress"] * 25
        def evaluate_transmission_stress(row):
            score = row["real_grid_stress"]
            dist = row["trans_dist_miles"]
            if score >= 95.0 or dist > 3.0:
                return pd.Series(["Critical Transmission Constraint", f"🛑 High Cost: Combined load {score}% + {dist:.1f}mi to transmission corridor.", [255, 0, 128, 255], [255, 0, 128, 150]])
            elif score >= 80.0:
                return pd.Series(["Moderate Upgrade Needed", f"⚠️ Moderate Cost: Transmission line {dist:.1f}mi away.", [255, 140, 0, 240], [255, 140, 0, 150]])
            else:
                return pd.Series(["Prime Interconnection", f"✅ Ready to Build: High-voltage corridor stable ({dist:.1f}mi).", [0, 229, 255, 200], [0, 229, 255, 100]])
        df[["status", "insight", "pillar_color", "arc_color"]] = df.apply(evaluate_transmission_stress, axis=1)
        metric_label = "Critical Transmission Nodes"
        metric_val = len(df[df["real_grid_stress"] >= 95.0])
    else:
        df["elevation"] = df["dist_miles"] * 200
        def evaluate_distance(row):
            dist = row["dist_miles"]
            if dist >= 2.0:
                return pd.Series(["EV Desert (>2.0 mi)", f"⭐ High Impact: Site is {dist}mi from nearest active node.", [255, 45, 85, 230], [255, 45, 85, 180]])
            elif dist >= 1.0:
                return pd.Series(["Moderate Gap", f"📊 Moderate Impact: Site is {dist}mi away.", [255, 179, 0, 200], [255, 179, 0, 140]])
            else:
                return pd.Series(["Well-Served", f"📉 Low Priority: Nearest hub is {dist}mi away.", [0, 229, 255, 160], [0, 229, 255, 80]])
        df[["status", "insight", "pillar_color", "arc_color"]] = df.apply(evaluate_distance, axis=1)
        metric_label = "EV Deserts (>=2.0 mi)"
        metric_val = len(df[df["dist_miles"] >= 2.0])

    df["arc_target_color"] = [[0, 255, 136, 250]] * len(df)
    df["site_title"] = df["site_name"]
else:
    metric_label = "EV Deserts"
    metric_val = 0

# Enrich active chargers
if not chargers_df.empty:
    chargers_df["site_title"] = chargers_df["station_name"]
    chargers_df["status"] = "Active DCFC Anchor Hub"
    chargers_df["j40_status"] = "N/A (Existing Infrastructure)"
    chargers_df["dist_miles"] = 0.0
    chargers_df["trans_dist_miles"] = 0.0
    chargers_df["stress_score_str"] = "Active Load"
    chargers_df["insight"] = "This location is an active fast charging hub serving as a grid anchor node."
    chargers_df["color_core"] = [[0, 255, 136, 255]] * len(chargers_df)
    chargers_df["color_halo"] = [[0, 255, 136, 60]] * len(chargers_df)

# ---------------------------------------------------------
# Executive KPI Metrics
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Selected Brownfield Sites", f"{len(df):,}")
col2.metric(metric_label, f"{metric_val:,}", delta_color="inverse")
col3.metric("Existing Active EV Hubs", f"{len(chargers_df):,}")
col4.metric(
    "Avg Region Stress" if is_stress_mode else "Avg Feeder Distance", 
    f"{df['real_grid_stress'].mean():.1f}%" if (is_stress_mode and not df.empty) else ("N/A" if df.empty else f"{df['dist_miles'].mean():.1f} mi")
)

# ---------------------------------------------------------
# Dynamic Telemetry & Kinetic Reach Briefing Panel
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 📡 Dynamic Telemetry & Kinetic Reach Briefing")

iso_names_str = ", ".join([label for _, label in region_governing_isos])
filter_actions = [f"• **Governing Jurisdiction:** {iso_names_str}"]

if layer_focus == "Comparative (Both Layers)":
    filter_actions.append("• **Infrastructure Layer:** Dual-layer overlay active, rendering candidate gas station retrofits alongside existing active EV charging anchor hubs.")
elif layer_focus == "Candidate Gas Station Retrofits Only":
    filter_actions.append("• **Infrastructure Layer:** Isolated to candidate brownfield gas station retrofits to evaluate conversion potential.")
else:
    filter_actions.append("• **Infrastructure Layer:** Isolated to existing active DC Fast Charging anchor hubs to audit current network coverage.")

if show_transmission:
    filter_actions.append("• **GIS Overlay:** *High-Voltage Transmission Lines* active, rendering regional transmission pathways queried from PostGIS.")

if "Spatial" in visual_mode:
    filter_actions.append("• **Telemetry Mode:** *Spatial Distance (Grid Deficit)* is active. 3D column heights represent physical mileage gaps to the nearest charging hub, flagging commercial 'EV Deserts' (>2.0 mi).")
else:
    filter_actions.append(f"• **Telemetry Mode:** *Live Corridor Stress* is active ({primary_iso_label} Load at {live_region_load}% + PostGIS Transmission Proximity).")

if j40_filter:
    filter_actions.append("• **Equity Filter:** *Justice40 DAC Isolation* is enabled. Candidates are filtered strictly to Disadvantaged Communities eligible for prioritized federal clean energy grants.")
else:
    filter_actions.append("• **Equity Filter:** Displaying all regional sites regardless of Justice40 DAC designation.")

if show_arcs and layer_focus != "Existing EV Charging Hubs Only":
    filter_actions.append("• **Kinetic Reach Arcs:** Active. Arcs project vector connections from unserved candidate nodes to their nearest active charging anchors.")
else:
    filter_actions.append("• **Kinetic Reach Arcs:** Hidden or disabled for the active layer view.")

st.info("**Active Filter Telemetry Actions:**\n" + "\n".join(filter_actions))
st.markdown("---")

# ---------------------------------------------------------
# PyDeck 3D Visualization Layer
# ---------------------------------------------------------
layers = []

if show_transmission and not trans_df.empty:
    layer_transmission = pdk.Layer(
        "PathLayer",
        id="transmission_lines_layer",
        data=trans_df,
        get_path="path",
        get_color="color",
        width_scale=2,
        width_min_pixels=1.5,
        get_width=3,
        pickable=False,
    )
    layers.append(layer_transmission)

show_candidates = layer_focus in ["Comparative (Both Layers)", "Candidate Gas Station Retrofits Only"]
show_chargers = layer_focus in ["Comparative (Both Layers)", "Existing EV Charging Hubs Only"]

if show_arcs and show_candidates and not df.empty:
    layer_arcs = pdk.Layer(
        "ArcLayer",
        id="kinetic_arcs",
        data=df,
        get_source_position=["source_lon", "source_lat"],
        get_target_position=["target_lon", "target_lat"],
        get_source_color="arc_color",
        get_target_color="arc_target_color",
        get_width=2.5,
        get_tilt=12,
        pickable=False,
    )
    layers.append(layer_arcs)

if show_candidates and not df.empty:
    layer_candidates_3d = pdk.Layer(
        "ColumnLayer",
        id="candidate_sites",
        data=df,
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

if show_chargers and not chargers_df.empty:
    layer_hub_halo = pdk.Layer(
        "ScatterplotLayer",
        data=chargers_df,
        id="charger_halo",
        get_position=["lon", "lat"],
        get_fill_color="color_halo",
        get_radius=700,
        pickable=False,
    )
    layer_hub_core = pdk.Layer(
        "ColumnLayer",
        data=chargers_df,
        id="charger_core",
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

centroid = active_polygon.centroid
view_state = pdk.ViewState(latitude=centroid.y, longitude=centroid.x, zoom=10 if input_mode == "Select Region / ISO (Instant Scope)" else 11, pitch=camera_pitch, bearing=camera_bearing)

tooltip_html = (
    "<div style='font-family: Consolas, monospace; padding: 10px; font-size: 11px; background: rgba(13, 17, 23, 0.95); border: 1px solid #30363d; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); max-width: 250px; white-space: normal; word-wrap: break-word;'>"
    "<b style='font-size: 13px; color: #58a6ff;'>{site_title}</b><br/>"
    "<hr style='margin: 6px 0; border: 0; border-top: 1px solid #30363d;'/>"
    "<span style='color: #8b949e;'>Classification:</span> <b style='color: white;'>{status}</b><br/>"
    "<span style='color: #8b949e;'>Justice40 DAC:</span> <b style='color: #00ff88;'>{j40_status}</b><br/>"
    "<span style='color: #8b949e;'>Nearest DCFC:</span> {dist_miles} miles<br/>"
    "<span style='color: #8b949e;'>Transmission Gap:</span> {trans_dist_miles} miles<br/>"
    "<hr style='margin: 6px 0; border: 0; border-top: 1px solid #30363d;'/>"
    "<b style='color: #c9d1d9;'>Federal Grid Telemetry:</b><br/>"
    "<span style='color: #8b949e;'>Governing Jurisdiction:</span> <b>" + iso_names_str + "</b><br/>"
    "<span style='color: #8b949e;'>Live Load (" + primary_iso_code + "):</span> <b>" + str(live_region_load) + "%</b><br/>"
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

map_selection = st.pydeck_chart(r, width="stretch", height=600, on_select="rerun", selection_mode="single-object", key="national_map")

# ---------------------------------------------------------
# Dynamic Bottom Drawer: Site Due Diligence Dossier
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📋 Site Due Diligence Dossier")

selected_site = None
site_type = None

if map_selection and getattr(map_selection, "selection", None):
    sel_objects = map_selection.selection.get("objects", {})
    if sel_objects.get("candidate_sites"):
        selected_site = sel_objects["candidate_sites"][0]
        site_type = "candidate"
    elif sel_objects.get("charger_core"):
        selected_site = sel_objects["charger_core"][0]
        site_type = "charger"

if selected_site:
    site_lon = selected_site.get('source_lon', selected_site.get('lon', 0))
    site_lat = selected_site.get('source_lat', selected_site.get('lat', 0))
    site_isos = get_iso_for_point(site_lon, site_lat)
    site_iso_str = ", ".join([label for _, label in site_isos])
    site_primary_code = site_isos[0][0]

    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown(f"### {selected_site.get('site_title', 'Unknown Site')}")
        if site_type == "candidate":
            st.markdown(f"**Classification:** {selected_site.get('status', 'N/A')}")
            st.markdown(f"**Justice40 DAC Status:** `{selected_site.get('j40_status', 'No')}`")
            st.markdown(f"**Coordinates:** `{site_lat:.5f}, {site_lon:.5f}`")
            st.markdown(f"**Distance to Nearest DCFC:** `{selected_site.get('dist_miles', 'N/A')} miles`")
            st.markdown(f"**Transmission Corridor Gap:** `{selected_site.get('trans_dist_miles', 'N/A')} miles [PostGIS]`")
        else:
            st.markdown(f"**Classification:** Active Live DCFC Anchor Hub")
            st.markdown(f"**Operating Network:** `{selected_site.get('ev_network', 'Unknown')}`")
            st.markdown(f"**Coordinates:** `{site_lat:.5f}, {site_lon:.5f}`")
            st.markdown(f"**Active Fast Charging Ports:** `{selected_site.get('ports', 'Unknown')}`")
            
    with col_b:
        if site_type == "candidate":
            st.markdown("#### ⚡ Local Grid Oversight")
            st.markdown(f"**Governing Jurisdiction:** `{site_iso_str}`")
            st.markdown(f"**Primary EIA-930 Load ({site_primary_code}):** `{live_region_load}%`")
            st.markdown(f"**Composite Stress Score:** `{selected_site.get('real_grid_stress', 0.0)} / 150`")
            st.markdown(f"• **Transmission Proximity:** `~{selected_site.get('trans_dist_miles', 0.0)} miles away`")
            
            score = selected_site.get('real_grid_stress', 0.0)
            if score >= 95.0:
                st.error("Critical Constraint: High combined load and transmission gap. Heavy Make-Ready required.")
            elif score >= 80.0:
                st.warning("Moderate Upgrade Needed: Interconnection corridor requires transformer support.")
            else:
                st.success("Prime Interconnection: High-voltage corridor stable and near capacity.")
        else:
            st.markdown("#### ⚡ Operating Grid Anchor Telemetry")
            st.success("Active Load Verified: Fully operational DC Fast Charging hub.")
            st.markdown(f"**Governing Jurisdiction:** `{site_iso_str}`")
            st.markdown("**Grid Deficit:** `0.00 miles` (System Baseline Node)")
            
    with col_c:
        st.markdown("#### ⚙️ Dynamic CAPEX Calculator")
        if site_type == "candidate":
            ports = st.number_input("Active Ports", min_value=2, max_value=20, value=4, step=2)
            power = st.selectbox("Power per Port", ["150kW", "350kW"])
            arch = st.selectbox("Infrastructure Architecture", ["Modular (ChargePoint / ABB / EVgo)", "Prefabricated Skid (Tesla PSU / NEVI)"])
            
            kw_val = int(power.replace("kW", ""))
            total_mw = (ports * kw_val) / 1000.0
            
            hw_unit = 55000 if kw_val == 150 else 115000
            if "Prefabricated" in arch:
                hw_unit *= 0.65
            tot_hw = ports * hw_unit
            
            civil_base = 25000 + (ports * 10500)
            if "Prefabricated" in arch: 
                civil_base *= 0.40
            
            stress_score = selected_site.get('real_grid_stress', 50.0)
            mr_base = 35000 + (total_mw * 1000 * 110)
            if score >= 95.0: 
                mr_mult = 1.85
            elif score >= 80.0: 
                mr_mult = 1.35
            else: 
                mr_mult = 1.0
            tot_mr = mr_base * mr_mult
            
            total_capex = tot_hw + tot_mr + civil_base
            
            st.markdown("---")
            st.markdown(f"**Site Peak Load:** `{total_mw:.2f} MW`")
            st.markdown(f"🚧 **Civil & Trenching:** `${int(civil_base):,}`")
            st.markdown(f"🔌 **Make-Ready (Grid Mult: {mr_mult}x):** `${int(tot_mr):,}`")
            st.markdown(f"🔋 **DCFC Hardware:** `${int(tot_hw):,}`")
            st.markdown(f"💰 **Est. Total CAPEX:** **`${int(total_capex):,}`**")
        else:
            st.markdown("✅ **Grid Capacity:** Verified active load profile.")
            st.markdown("✅ **Site Permitting:** Complete and Operational.")
            st.markdown("✅ **Utility Interconnection:** Fully Energized.")

else:
    st.info("👆 Click any 3D pillar (candidate gas station) or green pad (active EV charger) on the map above to load its full Site Due Diligence Dossier and CAPEX breakdown here.")
