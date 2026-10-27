import streamlit as st
from pathlib import Path
import math
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="DA2GC 5G NR Planning", page_icon="📡", layout="wide")

NAVY = "#17365D"
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background:#f4f7fb; border:1px solid #dce5f0; padding:12px; border-radius:10px;}
.small-note {font-size:0.86rem;color:#5b6573;}
</style>
""", unsafe_allow_html=True)

logo_path = Path("assets/Tel-U logo_1.PRIMER Utama_png.png")

col_logo, col_info = st.columns(
    [1.5, 5],
    vertical_alignment="center"
)

with col_logo:
    if logo_path.exists():
        st.image(str(logo_path), width=200)

with col_info:
    st.title("5G NR Direct Air-to-Ground Communication for In-Flight Connectivity")
    st.caption("Radio Network Planning Dashboard")
    st.markdown(
        "**Tim Penelitian Wireless Communication - InFlight Connectivity**  \n"
        "Telkom University"
    )

st.divider()

with st.sidebar:
    st.header("Scenario inputs")
    freq_mhz = st.number_input("Carrier frequency (MHz)", min_value=100.0, max_value=30000.0, value=3500.0, step=100.0)
    bandwidth_mhz = st.number_input("Channel bandwidth (MHz)", min_value=1.0, max_value=400.0, value=100.0, step=10.0)
    tx_dbm = st.number_input("gNB transmit power per sector (dBm)", min_value=0.0, max_value=80.0, value=46.0, step=1.0)
    tx_gain = st.number_input("Ground antenna gain toward aircraft (dBi)", min_value=-10.0, max_value=45.0, value=17.0, step=1.0)
    rx_gain = st.number_input("Aircraft antenna gain (dBi)", min_value=-10.0, max_value=35.0, value=5.0, step=1.0)
    tx_loss = st.number_input("Ground-side losses (dB)", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
    rx_loss = st.number_input("Aircraft-side losses (dB)", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
    rx_threshold = st.number_input("Minimum received-power threshold (dBm)", min_value=-150.0, max_value=-30.0, value=-90.0, step=1.0)
    altitude_km = st.number_input("Aircraft altitude above local ground (km)", min_value=0.1, max_value=20.0, value=10.0, step=0.5)
    gs_height_m = st.number_input("Ground antenna height above ground (m)", min_value=1.0, max_value=300.0, value=30.0, step=5.0)
    overlap_pct = st.slider("Linear overlap allowance (%)", min_value=0, max_value=40, value=10, step=5)
    route_length_km = st.number_input("Route/corridor length (km)", min_value=10.0, max_value=5000.0, value=800.0, step=25.0)
    st.divider()
    st.subheader("Capacity assumptions")
    traffic_bw_fraction = st.slider("Bandwidth available for traffic (%)", 10, 100, 80, 5) / 100
    spectral_eff = st.number_input("Average spectral efficiency (bit/s/Hz/layer)", min_value=0.1, max_value=10.0, value=3.0, step=0.1)
    mimo_layers = st.number_input("Effective MIMO layers", min_value=1, max_value=8, value=2, step=1)
    protocol_eff = st.slider("Protocol/scheduling efficiency (%)", 10, 100, 75, 5) / 100
    util_ceiling = st.slider("Capacity utilization ceiling (%)", 10, 95, 70, 5) / 100
    concurrent_aircraft = st.number_input("Concurrent aircraft served", min_value=1, max_value=100, value=3, step=1)
    passengers = st.number_input("Passengers per aircraft", min_value=1, max_value=600, value=180, step=10)
    active_fraction = st.slider("Active-user fraction (%)", 1, 100, 35, 1) / 100
    user_mbps = st.number_input("Target throughput per active user (Mbps)", min_value=0.1, max_value=100.0, value=2.0, step=0.5)
    site_reserve_pct = st.slider("Operational site reserve (%)", 0, 50, 10, 5) / 100

st.subheader("Route and map")
route_options = {
    "Jakarta – Surabaya (illustrative)": {"origin": (-6.1256, 106.6559), "dest": (-7.3798, 112.7870), "length": 660.0},
    "Jakarta – Denpasar (illustrative)": {"origin": (-6.1256, 106.6559), "dest": (-8.7482, 115.1672), "length": 965.0},
    "Surabaya – Makassar (illustrative)": {"origin": (-7.3798, 112.7870), "dest": (-5.0616, 119.5540), "length": 790.0},
    "Custom coordinates": {"origin": (-6.1256, 106.6559), "dest": (-7.3798, 112.7870), "length": 800.0},
}
route_choice = st.selectbox("Route preset", list(route_options.keys()))
preset = route_options[route_choice]
c1, c2, c3 = st.columns(3)
with c1:
    origin_name = st.text_input("Origin label", "Jakarta")
    origin_lat = st.number_input("Origin latitude", min_value=-11.0, max_value=6.0, value=float(preset["origin"][0]), step=0.01, format="%.4f")
    origin_lon = st.number_input("Origin longitude", min_value=95.0, max_value=141.0, value=float(preset["origin"][1]), step=0.01, format="%.4f")
with c2:
    dest_name = st.text_input("Destination label", "Surabaya")
    dest_lat = st.number_input("Destination latitude", min_value=-11.0, max_value=6.0, value=float(preset["dest"][0]), step=0.01, format="%.4f")
    dest_lon = st.number_input("Destination longitude", min_value=95.0, max_value=141.0, value=float(preset["dest"][1]), step=0.01, format="%.4f")
with c3:
    st.metric("Preset route distance (reference)", f"{preset['length']:.0f} km")
    st.caption("Distances are illustrative corridor lengths; edit route length in the sidebar to match your study.")

# Geodesic distance using haversine
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2-lat1)
    dl = math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(min(1, math.sqrt(a)))

def interpolate_route(lat1, lon1, lat2, lon2, n=100):
    # Linear interpolation for display only; use geodesic interpolation for rigorous GIS analysis.
    return [(lat1+(lat2-lat1)*i/(n-1), lon1+(lon2-lon1)*i/(n-1)) for i in range(n)]

geo_dist = haversine_km(origin_lat, origin_lon, dest_lat, dest_lon)
st.caption(f"Great-circle distance between entered coordinates: **{geo_dist:.1f} km**. Route/corridor length used in site dimensioning remains the editable value in the sidebar.")

# Link budget and coverage screening
eirp = tx_dbm + tx_gain - tx_loss
allowable_pathloss = tx_dbm + tx_gain + rx_gain - tx_loss - rx_loss - rx_threshold
fspl_radius_link = 10 ** ((allowable_pathloss - 32.45 - 20*math.log10(freq_mhz))/20)
earth_r_km = 6371.0
radio_horizon = math.sqrt(2*earth_r_km*altitude_km) + math.sqrt(2*earth_r_km*(gs_height_m/1000))
fspl_radius = min(fspl_radius_link, radio_horizon)
overlap_factor = 1 - overlap_pct/100
fspl_isd = 2*fspl_radius*overlap_factor
fspl_sites = math.ceil(route_length_km/fspl_isd) + 1 if fspl_isd > 0 else None
fspl_sites_res = math.ceil(fspl_sites*(1+site_reserve_pct)) if fspl_sites else None

# Capacity model
usable_bw_mhz = bandwidth_mhz * traffic_bw_fraction
raw_cell_mbps = usable_bw_mhz * spectral_eff * mimo_layers
effective_cell_mbps = raw_cell_mbps * protocol_eff
planning_site_mbps = effective_cell_mbps * util_ceiling
active_users_per_aircraft = passengers * active_fraction
demand_per_aircraft_mbps = active_users_per_aircraft * user_mbps
aggregate_demand_mbps = demand_per_aircraft_mbps * concurrent_aircraft
capacity_sites = math.ceil(aggregate_demand_mbps/planning_site_mbps) if planning_site_mbps > 0 else None
capacity_sites_res = math.ceil(capacity_sites*(1+site_reserve_pct)) if capacity_sites else None

# P.528 data upload
st.subheader("ITU-R P.528 data integration")
st.markdown(
    "Upload a CSV exported from a validated ITU-R P.528 implementation. Required columns: "
    "`distance_km` and `basic_transmission_loss_db`. Optional columns: `time_percentage`, `frequency_mhz`, `source`."
)
sample = pd.DataFrame({
    "distance_km": [10, 20, 30, 50, 75, 100, 150, 200],
    "basic_transmission_loss_db": [np.nan]*8,
    "time_percentage": [5]*8,
    "frequency_mhz": [freq_mhz]*8,
    "source": ["Replace with validated P.528 output"]*8
})
st.download_button("Download P.528 CSV template", sample.to_csv(index=False).encode("utf-8"), "p528_input_template.csv", "text/csv")
p528_file = st.file_uploader("Upload P.528 CSV (optional)", type=["csv"])
p528_df = None
p528_radius = None
p528_sites_res = None
if p528_file is not None:
    try:
        p528_df = pd.read_csv(p528_file)
        required = {"distance_km", "basic_transmission_loss_db"}
        if not required.issubset(set(p528_df.columns)):
            st.error("CSV must include distance_km and basic_transmission_loss_db columns.")
            p528_df = None
        else:
            p528_df = p528_df.dropna(subset=["distance_km", "basic_transmission_loss_db"]).sort_values("distance_km")
            if len(p528_df):
                p528_df["margin_db"] = allowable_pathloss - p528_df["basic_transmission_loss_db"]
                covered = p528_df[p528_df["margin_db"] >= 0]
                if len(covered):
                    p528_radius = float(covered["distance_km"].max())
                    p528_isd = 2*p528_radius*overlap_factor
                    p528_sites = math.ceil(route_length_km/p528_isd)+1 if p528_isd > 0 else None
                    p528_sites_res = math.ceil(p528_sites*(1+site_reserve_pct)) if p528_sites else None
                st.success(f"Loaded {len(p528_df)} valid P.528 rows. Radius estimate uses the farthest uploaded distance meeting the simplified allowable-path-loss threshold.")
            else:
                st.warning("No numeric P.528 rows found.")
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

# KPI cards
st.subheader("Key performance indicators")
m1,m2,m3,m4 = st.columns(4)
m1.metric("Allowable path loss", f"{allowable_pathloss:.1f} dB")
m2.metric("FSPL radius (screening)", f"{fspl_radius:.1f} km")
m3.metric("Effective cell throughput", f"{effective_cell_mbps:,.0f} Mbps")
m4.metric("Aggregate traffic demand", f"{aggregate_demand_mbps:,.0f} Mbps")

m5,m6,m7,m8 = st.columns(4)
m5.metric("FSPL coverage sites", f"{fspl_sites_res}")
m6.metric("Capacity-driven sites", f"{capacity_sites_res}")
m7.metric("P.528 radius", f"{p528_radius:.1f} km" if p528_radius is not None else "Not loaded")
combined_fspl = max(fspl_sites_res or 0, capacity_sites_res or 0)
combined_p528 = max(p528_sites_res or 0, capacity_sites_res or 0) if p528_sites_res is not None else None
m8.metric("Combined sites (FSPL)", f"{combined_fspl}")

tab1, tab2, tab3, tab4 = st.tabs(["Coverage & link budget", "Capacity", "Site comparison", "Map & assumptions"])
with tab1:
    a,b = st.columns(2)
    with a:
        st.markdown("#### Link budget")
        lb = pd.DataFrame([
            ["Transmit power", tx_dbm, "dBm"],
            ["Ground antenna gain", tx_gain, "dBi"],
            ["Aircraft antenna gain", rx_gain, "dBi"],
            ["Ground-side losses", -tx_loss, "dB"],
            ["Aircraft-side losses", -rx_loss, "dB"],
            ["EIRP (simplified)", eirp, "dBm"],
            ["Minimum received-power threshold", rx_threshold, "dBm"],
            ["Allowable path loss", allowable_pathloss, "dB"],
            ["FSPL-derived radius before horizon cap", fspl_radius_link, "km"],
            ["Geometric radio horizon approximation", radio_horizon, "km"],
            ["FSPL effective radius", fspl_radius, "km"],
            ["FSPL ISD after overlap", fspl_isd, "km"],
        ], columns=["Parameter","Value","Unit"])
        st.dataframe(lb, use_container_width=True, hide_index=True)
        st.latex(r"PL_{FSPL}(dB)=32.45+20\log_{10}(f_{MHz})+20\log_{10}(d_{km})")
        st.latex(r"PL_{allow}=P_{TX}+G_{TX}+G_{RX}-L_{TX}-L_{RX}-P_{RX,min}")
    with b:
        st.markdown("#### FSPL loss versus distance")
        maxd = max(10, min(600, route_length_km))
        dvals = np.linspace(1, maxd, 150)
        losses = 32.45 + 20*np.log10(freq_mhz) + 20*np.log10(dvals)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dvals, y=losses, mode="lines", name="FSPL"))
        fig.add_hline(y=allowable_pathloss, line_dash="dash", annotation_text="Allowable path loss")
        if p528_df is not None and len(p528_df):
            fig.add_trace(go.Scatter(x=p528_df["distance_km"], y=p528_df["basic_transmission_loss_db"],
                                     mode="markers+lines", name="ITU-R P.528 imported"))
        fig.update_layout(xaxis_title="Distance (km)", yaxis_title="Path loss / basic transmission loss (dB)", height=420)
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Caution: a generic received-power threshold is not interchangeable with SS-RSRP, SINR, or a 3GPP coverage criterion. Include antenna patterns, interference, fading margin, and link direction in a publication-grade link budget.")

with tab2:
    st.markdown("#### Capacity calculation")
    cap = pd.DataFrame([
        ["Bandwidth", bandwidth_mhz, "MHz"],
        ["Traffic bandwidth fraction", traffic_bw_fraction*100, "%"],
        ["Usable bandwidth", usable_bw_mhz, "MHz"],
        ["Average spectral efficiency", spectral_eff, "bit/s/Hz/layer"],
        ["Effective MIMO layers", mimo_layers, "layers"],
        ["Raw cell throughput", raw_cell_mbps, "Mbps"],
        ["Protocol/scheduling efficiency", protocol_eff*100, "%"],
        ["Effective cell throughput", effective_cell_mbps, "Mbps"],
        ["Utilization ceiling", util_ceiling*100, "%"],
        ["Planning capacity per site", planning_site_mbps, "Mbps"],
        ["Active users per aircraft", active_users_per_aircraft, "users"],
        ["Traffic demand per aircraft", demand_per_aircraft_mbps, "Mbps"],
        ["Concurrent aircraft", concurrent_aircraft, "aircraft"],
        ["Aggregate demand", aggregate_demand_mbps, "Mbps"],
        ["Capacity-driven sites before reserve", capacity_sites, "sites"],
        ["Capacity-driven sites with reserve", capacity_sites_res, "sites"],
    ], columns=["Metric","Value","Unit"])
    st.dataframe(cap, use_container_width=True, hide_index=True)
    st.latex(r"C_{effective}=B_{usable}\times SE\times N_{layers}\times \eta_{protocol}")
    st.latex(r"N_{capacity}=\left\lceil\frac{D_{aggregate}}{C_{site}\times U_{max}}\right\rceil")
    st.info("This is a first-order capacity model. For publication, derive spectral efficiency from a defensible SINR/MCS model or measured/vendor data, and model TDD, PRB allocation, BLER, Doppler, mobility, and per-sector resource limits.")

with tab3:
    st.markdown("#### Coverage-driven vs capacity-driven dimensioning")
    rows = [
        ["FSPL", fspl_sites_res, capacity_sites_res, combined_fspl],
    ]
    if p528_sites_res is not None:
        rows.append(["ITU-R P.528 (imported)", p528_sites_res, capacity_sites_res, combined_p528])
    comp = pd.DataFrame(rows, columns=["Propagation model","Coverage sites","Capacity sites","Combined sites"])
    st.dataframe(comp, use_container_width=True, hide_index=True)
    fig2 = px.bar(comp.melt(id_vars="Propagation model", value_vars=["Coverage sites","Capacity sites","Combined sites"],
                           var_name="Dimensioning basis", value_name="Sites"),
                  x="Propagation model", y="Sites", color="Dimensioning basis", barmode="group",
                  title="Required sites by planning constraint")
    st.plotly_chart(fig2, use_container_width=True)
    st.latex(r"N_{required}=\max(N_{coverage},N_{capacity})")
    st.caption("Coverage site count uses a one-dimensional corridor approximation: ISD = 2 × radius × (1 − overlap). Actual site layout depends on route geometry, antenna sectorization, terrain, and handover design.")
    export_df = comp.copy()
    st.download_button("Download site comparison CSV", export_df.to_csv(index=False).encode("utf-8"), "site_comparison.csv", "text/csv")

with tab4:
    st.markdown("#### Route map and indicative ground-station placements")
    route = interpolate_route(origin_lat, origin_lon, dest_lat, dest_lon, 100)
    mid = ((origin_lat+dest_lat)/2, (origin_lon+dest_lon)/2)
    fmap = folium.Map(location=mid, zoom_start=5, tiles="CartoDB positron")
    folium.PolyLine(route, color="#1f77b4", weight=4, opacity=0.85, tooltip="Illustrative straight-line route").add_to(fmap)
    folium.Marker([origin_lat, origin_lon], tooltip=origin_name, popup=f"Origin: {origin_name}",
                  icon=folium.Icon(color="green", icon="plane", prefix="fa")).add_to(fmap)
    folium.Marker([dest_lat, dest_lon], tooltip=dest_name, popup=f"Destination: {dest_name}",
                  icon=folium.Icon(color="red", icon="plane", prefix="fa")).add_to(fmap)
    # Place illustrative sites along straight-line route. Not optimized and not operationally vetted.
    if fspl_sites_res and fspl_sites_res <= 80:
        n_sites = max(2, fspl_sites_res)
        for i in range(n_sites):
            t = i/(n_sites-1) if n_sites > 1 else 0
            lat = origin_lat + (dest_lat-origin_lat)*t
            lon = origin_lon + (dest_lon-origin_lon)*t
            folium.CircleMarker([lat,lon], radius=4, color="#0f6b78", fill=True, fill_opacity=0.9,
                                tooltip=f"Indicative site {i+1}").add_to(fmap)
            folium.Circle([lat,lon], radius=fspl_radius*1000, color="#0f6b78", weight=1, fill=True,
                          fill_opacity=0.08, tooltip=f"FSPL screening radius ≈ {fspl_radius:.1f} km").add_to(fmap)
    else:
        st.info("Map site markers are suppressed when the estimated site count is very large; review site count and scenario inputs.")
    st_folium(fmap, use_container_width=True, height=560, returned_objects=[])
    st.caption("Map is illustrative: route interpolation is straight-line, and site locations are not optimized or checked for terrain, airport restrictions, land access, sector pattern, interference, or regulatory feasibility.")
    st.markdown("#### Assumptions and validation checklist")
    checklist = [
        "Confirm legal spectrum allocation and coexistence/interference constraints for airborne use at 3.5 GHz.",
        "Use a validated ITU-R P.528 implementation/version and document antenna heights, frequency, polarization, time percentage, and output loss.",
        "Validate antenna gains and patterns, feeder losses, receiver threshold, and both uplink/downlink link budgets.",
        "Separate received power / RSRP from SINR and derive coverage criteria consistently.",
        "Validate traffic demand, spectral efficiency, MIMO layers, TDD configuration, overhead, Doppler, and aircraft concurrency.",
        "Replace illustrative route and station locations with a real route geometry and GIS/site feasibility analysis.",
        "Run sensitivity analysis across altitude, bandwidth, traffic demand, overlap, and availability targets.",
    ]
    for item in checklist:
        st.checkbox(item, value=False, key="check_"+str(abs(hash(item))))

st.divider()
st.markdown("**Model notes**")
st.markdown("""
- **FSPL:** `32.45 + 20 log10(f_MHz) + 20 log10(d_km)`.
- **Coverage dimensioning:** `ISD = 2 × effective radius × (1 − overlap fraction)`; `N_coverage = ceil(route length / ISD) + 1`.
- **Capacity:** first-order bandwidth × spectral efficiency × MIMO layers × efficiency model.
- **P.528:** this app only uses P.528 values imported by the researcher. It does not contain a proprietary, interpolated, or fabricated substitute for the ITU-R method.
- **Research use:** record app version, source data, parameter set, P.528 implementation and version, and export the tables used in the manuscript.
""")
st.caption("DA2GC planning research prototype • Review all assumptions before using outputs in a paper or operational decision.")
