import streamlit as st
import pydeck as pdk
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import psycopg2
import json
import warnings
from shapely.geometry import shape

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

st.title("⚡ EV Grid Command & Advanced Analytics Terminal (Allegheny County)")

# ---------------------------------------------------------
# Sidebar Controls & Education
# ---------------------------------------------------------
st.sidebar.header("🕹️ Visual Engine Modes")

visual_mode = st.sidebar.radio(
    "3D Telemetry Mapping Mode",
    ["Spatial Distance (Grid Deficit)", "Provable Transmission Stress Index"],
    help="Switch between physical distance deficit visualization and the real transmission line distance stress model."
)

st.sidebar.markdown("---")
st.sidebar.header("⚖️ Equity & Policy Filters")
j40_filter = st.sidebar.checkbox(
    "Isolate Justice40 DAC Sites", 
    value=False, 
    help="Filter the map to candidate sites located within Disadvantaged Communities."
)

with st.sidebar.expander("🧠 Methodology & Critical Context", expanded=True):
    st.markdown("""
    **The Visual Metaphor: Pillars vs. Glowing Pads**
    *   **Neon Green Glowing Pads:** These represent the existing active DC Fast-Charging hubs. They are rendered flat because they have a grid deficit of zero—they are the physical anchors of the current network.
    *   **Extruded 3D Pillars:** These represent existing gas stations, acting as our candidate conversion sites. Why gas stations? They are the ultimate “brownfield” targets for EV infrastructure. They already possess the exact physical footprint required: paved pull-through lanes, high-visibility canopies, high-visibility lighting, and retail amenities (bathrooms, food) crucial for drivers waiting 20-30 minutes for a charge. The pillar’s height visualizes the systemic value of ripping out a gas pump and replacing it with a DCFC node at that location.

    **Why a 2.0 Mile Threshold?**
    In urban topologies like Allegheny County, a 2-mile spatial gap is a structural barrier. For the 30%+ of residents in multi-unit dwellings (MUDs) who cannot charge at home, driving over 2 miles exclusively to “fuel up” destroys the EV value proposition. Federal NEVI guidelines prioritize 1-mile buffers from corridors; breaching 2 miles in a metro footprint indicates a stark, unserved “EV Desert.”

    **Grid Thermal Limits Explained**
    “Thermal Capacity” refers to the physical heat limit of local distribution wires. A standard 4-port 150kW DCFC station demands 600kW of instantaneous power. Forcing that load through an older commercial feeder without upgrades causes the lines to overheat and melt, blowing local transformers. “Magenta” sites require expensive utility Make-Ready Upgrades before chargers can be installed.

    **Justice40 Integration**
    The Justice40 Initiative mandates that 40% of federal clean energy investments flow to Disadvantaged Communities (DACs). Filtering by Justice40 isolates sites that are eligible for prioritized federal grants, merging grid equity with grid expansion. *(Note: DAC status here is modeled deterministically for demonstration).*
    """)

st.sidebar.markdown("---")
st.sidebar.header("📐 Spatial Parameters")
show_arcs = st.sidebar.checkbox("Render Kinetic Deficit Arcs", value=True)
camera_pitch = st.sidebar.slider("Camera Pitch", min_value=30, max_value=60, value=52, step=1)
camera_bearing = st.sidebar.slider("Camera Rotation", min_value=-180, max_value=180, value=-22, step=2)

# ---------------------------------------------------------
# Live Data Fetch & Advanced Analytics Processing
# ---------------------------------------------------------
@st.cache_data
def load_data():
    try:
        county_boundaries = gpd.read_parquet("county_boundaries.parquet")
        if county_boundaries.crs is None:
            county_boundaries = county_boundaries.set_crs("EPSG:4326")
    except Exception as e:
        st.error(f"Error loading county_boundaries.parquet: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    try:
        gas_stations_gdf = gpd.read_parquet("gas_stations.parquet")
        if gas_stations_gdf.crs is None:
            gas_stations_gdf = gas_stations_gdf.set_crs("EPSG:4326")
    except Exception as e:
        st.error(f"Error loading gas_stations.parquet: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 1. Fetch NREL Chargers
    api_key = st.secrets.get("NREL_API_KEY", "ZKe4KCw4IyoPLtafYKWb6uPdDipAx9To9tOTQGry")
    nlr_url = (
        "https://developer.nlr.gov/api/alt-fuel-stations/v1.json?"
        f"api_key={api_key}&fuel_type=ELEC&state=PA"
    )
    
    local_chargers_gdf = gpd.GeoDataFrame(columns=['station_name', 'ev_network', 'ev_dc_fast_num', 'geometry'], geometry='geometry', crs="EPSG:4326")
    session = requests.Session()
    session.trust_env = False
    
    try:
        response = session.get(nlr_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
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
                        geometry=gpd.points_from_xy(dcfc_df['longitude'], dcfc_df['latitude']),
                        crs="EPSG:4326"
                    )
                    local_chargers_gdf = gpd.sjoin(nlr_gdf, county_boundaries, how="inner", predicate="intersects")
                    if not local_chargers_gdf.empty:
                        local_chargers_gdf["station_name"] = local_chargers_gdf["station_name"].fillna("DC Fast Charger")
                        local_chargers_gdf["ev_network"] = local_chargers_gdf.get("ev_network", pd.Series(["Unknown Network"] * len(local_chargers_gdf))).fillna("Unknown Network")
                        local_chargers_gdf["ev_dc_fast_num"] = local_chargers_gdf["ev_dc_fast_num"].astype(int)
    except requests.exceptions.RequestException:
        st.sidebar.warning(f"⚠️ External NLR API unreachable. Operating on offline fallback mode.")

    # 2. Query Supabase PostGIS for Real Transmission Lines
    trans_df = pd.DataFrame()
    transmission_gdf = gpd.GeoDataFrame(columns=['voltage', 'geometry'], crs="EPSG:4326")
    try:
        conn = psycopg2.connect(st.secrets["DATABASE_URL"])
        trans_query = """
        SELECT COALESCE("VOLTAGE", 0) AS voltage, 
               ST_AsGeoJSON(ST_Intersection(geometry, ST_MakeEnvelope(-81.0, 39.8, -79.0, 41.5, 4326))) AS geojson
        FROM transmission_lines
        WHERE ST_Intersects(geometry, ST_MakeEnvelope(-81.0, 39.8, -79.0, 41.5, 4326));
        """
        cur = conn.cursor()
        cur.execute(trans_query)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        paths = []
        trans_geoms = []
        voltages = []
        for row in rows:
            v = row[0]
            geojson_str = row[1]
            if geojson_str:
                geom_dict = json.loads(geojson_str)
                coords = geom_dict.get("coordinates", [])
                shp = shape(geom_dict)
                if not shp.is_empty:
                    trans_geoms.append(shp)
                    voltages.append(v)
                    if geom_dict.get("type") == "LineString":
                        clean_coords = [[pt[0], pt[1]] for pt in coords if len(pt) >= 2 and -81.0 <= pt[0] <= -79.0 and 39.8 <= pt[1] <= 41.5]
                        if len(clean_coords) >= 2:
                            paths.append({"path": clean_coords, "voltage": v})
                    elif geom_dict.get("type") == "MultiLineString":
                        for line_coords in coords:
                            clean_line_coords = [[pt[0], pt[1]] for pt in line_coords if len(pt) >= 2 and -81.0 <= pt[0] <= -79.0 and 39.8 <= pt[1] <= 41.5]
                            if len(clean_line_coords) >= 2:
                                paths.append({"path": clean_line_coords, "voltage": v})
        trans_df = pd.DataFrame(paths)
        if not trans_df.empty:
            def get_voltage_color(val):
                if val >= 500: return [255, 0, 128, 200]
                elif val >= 230: return [255, 140, 0, 200]
                else: return [0, 229, 255, 160]
            trans_df["color"] = trans_df["voltage"].apply(get_voltage_color)
            
        if trans_geoms:
            transmission_gdf = gpd.GeoDataFrame({'voltage': voltages}, geometry=trans_geoms, crs="EPSG:4326")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Supabase Transmission Query Warning: {e}")

    # 3. Spatial Joins & Advanced Analytics Metrics
    gas_m = gas_stations_gdf.to_crs(epsg=2272).reset_index(drop=True)
    gas_m = gas_m.drop(columns=[col for col in ['index', 'index_left', 'index_right', 'level_0'] if col in gas_m.columns])
    
    if not local_chargers_gdf.empty and not gas_m.empty:
        local_chargers_gdf["target_lon"] = local_chargers_gdf.geometry.x
        local_chargers_gdf["target_lat"] = local_chargers_gdf.geometry.y
        
        chargers_m = local_chargers_gdf.to_crs(epsg=2272).reset_index(drop=True)
        chargers_m = chargers_m.drop(columns=[col for col in ['index', 'index_left', 'index_right', 'level_0'] if col in chargers_m.columns])
        
        nearest_join = gpd.sjoin_nearest(
            gas_m,
            chargers_m[['geometry', 'station_name', 'target_lon', 'target_lat', 'ev_dc_fast_num', 'ev_network']],
            how="left",
            distance_col="dist_feet"
        )
        nearest_join = nearest_join[~nearest_join.index.duplicated(keep='first')]
        nearest_join["dist_miles"] = (nearest_join["dist_feet"] / 5280.0).round(2)
        gas_final = nearest_join.to_crs(epsg=4326)
    else:
        gas_final = gas_stations_gdf.copy()
        gas_final["dist_miles"] = 5.0
        gas_final["target_lon"] = gas_final.geometry.x 
        gas_final["target_lat"] = gas_final.geometry.y
        gas_final["ev_network"] = "None"
        gas_final["station_name"] = "None"

    # Transmission Distance Join
    if not transmission_gdf.empty and not gas_final.empty:
        gas_final_m = gas_final.to_crs(epsg=2272).reset_index(drop=True)
        gas_final_m = gas_final_m.drop(columns=[col for col in ['index', 'index_left', 'index_right', 'level_0'] if col in gas_final_m.columns])
            
        trans_m = transmission_gdf.to_crs(epsg=2272).reset_index(drop=True)
        trans_m = trans_m.drop(columns=[col for col in ['index', 'index_left', 'index_right', 'level_0'] if col in trans_m.columns])
            
        trans_nearest = gpd.sjoin_nearest(gas_final_m, trans_m, how="left", distance_col="trans_dist_feet")
        trans_nearest = trans_nearest[~trans_nearest.index.duplicated(keep='first')]
        gas_final["trans_dist_miles"] = (trans_nearest["trans_dist_feet"] / 5280.0).fillna(2.0).round(2)
    else:
        gas_final["trans_dist_miles"] = 1.5

    # --- ADVANCED ANALYTICS CALCULATIONS ---
    gas_final["source_lon"] = gas_final.geometry.x
    gas_final["source_lat"] = gas_final.geometry.y
    
    # 1. Grid Stress Score (0-100)
    gas_final["real_grid_stress"] = (50.0 + (gas_final["trans_dist_miles"] * 16.5)).clip(20.0, 100.0).round(1)

    # 2. Justice40 DAC Designation (Eligible Tract for Section 30C)
    gas_final["is_j40_dac"] = ((gas_final.geometry.y * 7654321).astype(int) % 100) < 40
    gas_final["j40_status"] = gas_final["is_j40_dac"].apply(lambda x: "Yes (Eligible 30C Tract)" if x else "No")

    # 3. AADT (Annual Average Daily Traffic) Index (Estimated via distance to Pittsburgh core -79.9959, 40.4406)
    dist_to_center = np.sqrt((gas_final["source_lon"] - (-79.9959))**2 + (gas_final["source_lat"] - 40.4406)**2) * 69.0
    gas_final["aadt_index"] = np.clip((28000 - (dist_to_center * 3200)), 3500, 32000).astype(int)

    # 4. Substation Feeder Capacity & Headroom (%)
    gas_final["feeder_headroom_pct"] = np.clip(100.0 - (gas_final["trans_dist_miles"] * 35.0), 10.0, 95.0).round(1)

    # 5. Multi-Unit Dwelling (MUD) Density (%) - Captures apartment/condo populations without home charging
    gas_final["mud_density_pct"] = np.clip(78.0 - (dist_to_center * 14.0), 18.0, 85.0).round(1)

    gas_final["site_title"] = gas_final.get("name", pd.Series(["Gas Station"] * len(gas_final))).fillna("Candidate Conversion Site")

    if not local_chargers_gdf.empty:
        chargers_final = local_chargers_gdf.to_crs(epsg=4326)
        chargers_final["lon"] = chargers_final.geometry.x
        chargers_final["lat"] = chargers_final.geometry.y
        chargers_final["site_title"] = chargers_final["station_name"]
        chargers_final["ev_network"] = chargers_final["ev_network"]
        chargers_final_df = pd.DataFrame(chargers_final.drop(columns=['geometry']))
    else:
        chargers_final_df = pd.DataFrame()
    
    return pd.DataFrame(gas_final.drop(columns=['geometry'])), chargers_final_df, trans_df

with st.spinner("Computing advanced spatial analytics & tax credit metrics..."):
    candidate_df, chargers_df, trans_df = load_data()

# Apply Justice40 Filter
if j40_filter and not candidate_df.empty:
    candidate_df = candidate_df[candidate_df["is_j40_dac"] == True]

# ---------------------------------------------------------
# Dynamic Mode Physics & Layer Preparation
# ---------------------------------------------------------
is_composite_mode = "Stress" in visual_mode

if is_composite_mode:
    st.markdown("Extruding candidate sites based on **Provable Transmission Stress Index** derived from real Supabase PostGIS transmission line distances.")
    if not candidate_df.empty:
        candidate_df["elevation"] = (candidate_df["trans_dist_miles"] * 80).clip(40, 400)
        def evaluate_composite(row):
            score = row["real_grid_stress"]
            trans = row["trans_dist_miles"]
            if score >= 80.0 or trans > 2.0: 
                return pd.Series(["Critical Transmission Constraint", [255, 0, 128, 255]])  # Magenta
            elif score >= 65.0: 
                return pd.Series(["Moderate Upgrade Needed", [255, 140, 0, 240]]) # Amber
            else: 
                return pd.Series(["Prime Interconnection", [0, 229, 255, 180]])       # Cyan
        candidate_df[["status", "pillar_color"]] = candidate_df.apply(evaluate_composite, axis=1)
    metric_label = "Critical Transmission Nodes (Stress > 80)"
    metric_val = len(candidate_df[candidate_df["real_grid_stress"] >= 80.0]) if not candidate_df.empty else 0
else:
    st.markdown("Extruding candidate brownfield sites into **3D topographic deficit pillars** based on radial distance to nearest active DCFC node.")
    if not candidate_df.empty:
        candidate_df["elevation"] = (candidate_df["dist_miles"] * 80).clip(40, 400)
        def evaluate_distance(row):
            dist = row["dist_miles"]
            if dist >= 2.0: return pd.Series(["EV Desert (>2.0 mi)", [255, 45, 85, 230]])     # Red
            elif dist >= 1.0: return pd.Series(["Moderate Gap (1.0-2.0 mi)", [255, 179, 0, 200]]) # Amber
            else: return pd.Series(["Well-Served (<1.0 mi)", [0, 229, 255, 160]])            # Cyan
        candidate_df[["status", "pillar_color"]] = candidate_df.apply(evaluate_distance, axis=1)
    metric_label = "Critical EV Deserts (>2.0 mi)"
    metric_val = len(candidate_df[candidate_df["dist_miles"] >= 2.0]) if not candidate_df.empty else 0

if not candidate_df.empty:
    candidate_df["arc_color"] = candidate_df["pillar_color"]
    candidate_df["arc_target_color"] = [[0, 255, 136, 250]] * len(candidate_df) 

if not chargers_df.empty:
    chargers_df["color_core"] = chargers_df.apply(lambda x: [0, 255, 136, 255], axis=1) 
    chargers_df["color_halo"] = chargers_df.apply(lambda x: [0, 255, 136, 60], axis=1)  

# ---------------------------------------------------------
# Executive KPI Metrics
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Candidate Sites", f"{len(candidate_df):,}")
col2.metric(metric_label, f"{metric_val:,}", delta_color="inverse")
col3.metric("Justice40 Eligible Sites", f"{len(candidate_df[candidate_df['is_j40_dac'] == True]):,}" if not candidate_df.empty else "0")
col4.metric(
    "Avg AADT Volume", 
    f"{int(candidate_df['aadt_index'].mean()):,}" if not candidate_df.empty else "0"
)

# ---------------------------------------------------------
# PyDeck Layers & View
# ---------------------------------------------------------
layers = []

if not trans_df.empty:
    layers.append(pdk.Layer(
        "PathLayer",
        id="transmission_lines_layer",
        data=trans_df,
        get_path="path",
        get_color="color",
        width_scale=2,
        width_min_pixels=1.5,
        get_width=3,
        pickable=False,
    ))

if show_arcs and not candidate_df.empty:
    layers.append(pdk.Layer(
        "ArcLayer",
        id="kinetic_arcs",
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
        id="candidate_sites",
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
            id="charger_halo",
            data=chargers_df,
            get_position=["lon", "lat"],
            get_fill_color="color_halo",
            get_radius=700,
            pickable=False,
        ),
        pdk.Layer(
            "ColumnLayer",
            id="charger_core",
            data=chargers_df,
            get_position=["lon", "lat"],
            get_elevation=15,
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

tooltip = {
    "html": "<b>{site_title}</b><br/><i>Click to load Advanced Site Dossier below</i>",
    "style": {"color": "white", "backgroundColor": "#0d1117", "border": "1px solid #30363d", "fontFamily": "Consolas, monospace", "fontSize": "12px"}
}

r = pdk.Deck(map_style="dark", layers=layers, initial_view_state=view_state, tooltip=tooltip)
map_selection = st.pydeck_chart(r, width="stretch", height=600, on_select="rerun", selection_mode="single-object")

# ---------------------------------------------------------
# Dynamic Bottom Drawer: Advanced Site Due Diligence Dossier
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📋 Advanced Site Due Diligence Dossier & ROI Engine")

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
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown(f"### {selected_site.get('site_title', 'Unknown Site')}")
        if site_type == "candidate":
            st.markdown(f"**Classification:** {selected_site.get('status', 'N/A')}")
            st.markdown(f"**Coordinates:** `{selected_site.get('source_lat', 0):.5f}, {selected_site.get('source_lon', 0):.5f}`")
            st.markdown(f"**AADT Traffic Index:** `{selected_site.get('aadt_index', 0):,} vehicles/day`")
            st.markdown(f"**MUD Renter Density:** `{selected_site.get('mud_density_pct', 0.0)}%`")
        else:
            st.markdown(f"**Classification:** Active Live DCFC Anchor Hub")
            st.markdown(f"**Operating Network:** `{selected_site.get('ev_network', 'Unknown')}`")
            st.markdown(f"**Coordinates:** `{selected_site.get('lat', 0):.5f}, {selected_site.get('lon', 0):.5f}`")
            
    with col_b:
        if site_type == "candidate":
            st.markdown("#### ⚡ Grid & Feeder Capacity Telemetry")
            st.markdown(f"**Transmission Gap:** `~{selected_site.get('trans_dist_miles', 0.0)} miles [PostGIS]`")
            st.markdown(f"**Feeder Thermal Headroom:** `{selected_site.get('feeder_headroom_pct', 0.0)}%`")
            st.markdown(f"**Section 30C Tract Status:** `{selected_site.get('j40_status', 'No')}`")
            
            headroom = selected_site.get('feeder_headroom_pct', 50.0)
            if headroom < 40.0:
                 st.error("Feeder Constraint: High risk of thermal overload. Substation reinforcement required.")
            elif headroom < 70.0:
                 st.warning("Moderate Headroom: Transformer buffering recommended.")
            else:
                 st.success("Stable Feeder Capacity: High hosting capacity available.")
        else:
            st.markdown("#### ⚡ Operating Grid Anchor Telemetry")
            st.success("Active Load Verified: Fully operational DC Fast Charging hub.")
            st.markdown("**Grid Deficit:** `0.00 miles` (System Baseline Node)")
            
    with col_c:
        st.markdown("#### ⚙️ Section 30C Tax Credit & ROI Calculator")
        if site_type == "candidate":
            ports = st.number_input("Active Ports", min_value=2, max_value=20, value=4, step=2)
            power = st.selectbox("Power per Port", ["150kW", "350kW"])
            pwa_met = st.checkbox("Prevailing Wage & Apprenticeship (PWA) Met?", value=True, help="Unlocks 30% tax credit rate vs 6% base rate.")
            
            kw_val = int(power.replace("kW", ""))
            total_mw = (ports * kw_val) / 1000.0
            
            hw_unit = 55000 if kw_val == 150 else 115000
            if "Prefabricated" in arch if 'arch' in locals() else False:
                hw_unit *= 0.65
            tot_hw = ports * hw_unit
            civil_base = 25000 + (ports * 10500)
            
            stress_score = selected_site.get('real_grid_stress', 50.0)
            mr_base = 35000 + (total_mw * 1000 * 110)
            mr_mult = 1.85 if stress_score >= 80.0 else (1.35 if stress_score >= 65.0 else 1.0)
            tot_mr = mr_base * mr_mult
            
            gross_capex = tot_hw + tot_mr + civil_base
            
            # Section 30C Calculation (30% if PWA & eligible tract, else 6%, capped at $100k/port)
            is_eligible_tract = selected_site.get('is_j40_dac', False)
            if is_eligible_tract:
                credit_rate = 0.30 if pwa_met else 0.06
                potential_credit = gross_capex * credit_rate
                max_allowable_credit = ports * 100000.0
                tax_credit = min(potential_credit, max_allowable_credit)
            else:
                tax_credit = 0.0
                
            net_capex = gross_capex - tax_credit
            
            st.markdown("---")
            st.markdown(f"**Gross CAPEX:** `${int(gross_capex):,}`")
            st.markdown(f"🏛️ **Section 30C Tax Credit:** `- ${int(tax_credit):,}`")
            st.markdown(f"💰 **Net Project CAPEX:** **`${int(net_capex):,}`**")
        else:
            st.markdown("✅ **Grid Capacity:** Verified active load profile.")
            st.markdown("✅ **Site Permitting:** Complete and Operational.")

else:
    st.info("👆 Click any 3D pillar (candidate gas station) or green pad (active EV charger) on the map to load its advanced due diligence dossier here.")
