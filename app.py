import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="Community Data – MVP (OSM)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------
# DEFAULT DUMMY DATA
# --------------------------------

def get_default_buildings_df():
    buildings_data = [
        {"building_id": 1, "name": "Waterloo City Hall", "building_type": "Municipal", "neighbourhood": "Central",
         "latitude": 43.466, "longitude": -80.522, "floor_area_m2": 12000, "elec_kwh": 1500000,
         "gas_m3": 90000, "baseline_tco2e": 420, "solar_kw_potential": 300},
        {"building_id": 2, "name": "King Street Library", "building_type": "Institutional", "neighbourhood": "Central",
         "latitude": 43.468, "longitude": -80.523, "floor_area_m2": 6000, "elec_kwh": 400000,
         "gas_m3": 20000, "baseline_tco2e": 95, "solar_kw_potential": 120},
        {"building_id": 3, "name": "Lincoln Heights School", "building_type": "Educational", "neighbourhood": "North",
         "latitude": 43.485, "longitude": -80.528, "floor_area_m2": 11000, "elec_kwh": 500000,
         "gas_m3": 35000, "baseline_tco2e": 160, "solar_kw_potential": 250},
        {"building_id": 4, "name": "Tech Hub Offices", "building_type": "Commercial", "neighbourhood": "South",
         "latitude": 43.455, "longitude": -80.515, "floor_area_m2": 9000, "elec_kwh": 800000,
         "gas_m3": 25000, "baseline_tco2e": 190, "solar_kw_potential": 210},
        {"building_id": 5, "name": "Maple Apartments", "building_type": "Residential", "neighbourhood": "East",
         "latitude": 43.462, "longitude": -80.510, "floor_area_m2": 15000, "elec_kwh": 1200000,
         "gas_m3": 100000, "baseline_tco2e": 380, "solar_kw_potential": 180},
        {"building_id": 6, "name": "Kingsview Community Centre", "building_type": "Municipal", "neighbourhood": "North",
         "latitude": 43.482, "longitude": -80.520, "floor_area_m2": 7000, "elec_kwh": 300000,
         "gas_m3": 15000, "baseline_tco2e": 70, "solar_kw_potential": 90},
    ]
    return pd.DataFrame(buildings_data)


def get_default_ev_df():
    ev_data = [
        {"site_id": 1, "name": "City Hall EV Lot", "neighbourhood": "Central",
         "latitude": 43.4665, "longitude": -80.5215, "num_ports": 8, "power_kw": 150},
        {"site_id": 2, "name": "Tech Hub EV Station", "neighbourhood": "South",
         "latitude": 43.4545, "longitude": -80.516, "num_ports": 6, "power_kw": 120},
        {"site_id": 3, "name": "Library EV Parking", "neighbourhood": "Central",
         "latitude": 43.4682, "longitude": -80.5225, "num_ports": 4, "power_kw": 60},
        {"site_id": 4, "name": "North Community EV Hub", "neighbourhood": "North",
         "latitude": 43.484, "longitude": -80.527, "num_ports": 10, "power_kw": 200},
    ]
    return pd.DataFrame(ev_data)


def get_default_waste_df():
    waste_data = [
        {"site_id": 1, "name": "North Transfer Station", "neighbourhood": "North",
         "latitude": 43.488, "longitude": -80.53, "annual_waste_tonnes": 12000, "waste_tco2e": 5500},
        {"site_id": 2, "name": "South Organics Depot", "neighbourhood": "South",
         "latitude": 43.452, "longitude": -80.518, "annual_waste_tonnes": 6000, "waste_tco2e": 2100},
    ]
    return pd.DataFrame(waste_data)


def get_neighbourhood_polygons():
    # NOTE: folium expects [lat, lon]; earlier we used [lon, lat], so we flip.
    return [
        {
            "neighbourhood": "Central",
            "coords": [
                (43.464, -80.526),
                (43.464, -80.516),
                (43.472, -80.516),
                (43.472, -80.526),
            ],
        },
        {
            "neighbourhood": "North",
            "coords": [
                (43.48, -80.532),
                (43.48, -80.522),
                (43.49, -80.522),
                (43.49, -80.532),
            ],
        },
        {
            "neighbourhood": "South",
            "coords": [
                (43.452, -80.52),
                (43.452, -80.51),
                (43.46, -80.51),
                (43.46, -80.52),
            ],
        },
        {
            "neighbourhood": "East",
            "coords": [
                (43.46, -80.514),
                (43.46, -80.504),
                (43.468, -80.504),
                (43.468, -80.514),
            ],
        },
    ]


# Start with defaults
buildings_df = get_default_buildings_df()
ev_df = get_default_ev_df()
waste_df = get_default_waste_df()
neighbourhood_polygons = get_neighbourhood_polygons()

# --------------------------------
# HELPER FUNCTIONS
# --------------------------------

ELEC_EMISSION_FACTOR = 0.0002   # tCO2e per kWh (example)
GAS_EMISSION_FACTOR = 0.0019    # tCO2e per m3 (example)
ELEC_COST = 0.15                # $/kWh (example)
GAS_COST = 0.40                 # $/m3 (example)


def apply_filters(buildings, neighbourhood, building_type):
    df = buildings.copy()
    if neighbourhood != "All":
        df = df[df["neighbourhood"] == neighbourhood]
    if building_type != "All":
        df = df[df["building_type"] == building_type]
    return df


def calculate_scenario(df, retrofit_pct, solar_pct):
    """
    Calculate scenario emissions and costs, using gas vs electricity split where possible.
    """
    df = df.copy()

    df["elec_kwh"] = df.get("elec_kwh", 0).fillna(0)
    df["gas_m3"] = df.get("gas_m3", 0).fillna(0)

    est_elec = df["elec_kwh"] * ELEC_EMISSION_FACTOR
    est_gas = df["gas_m3"] * GAS_EMISSION_FACTOR
    total_est = est_elec + est_gas

    df["baseline_elec_t"] = 0.0
    df["baseline_gas_t"] = 0.0

    mask = total_est > 0
    df.loc[mask, "baseline_elec_t"] = df.loc[mask, "baseline_tco2e"] * (est_elec[mask] / total_est[mask])
    df.loc[mask, "baseline_gas_t"] = df.loc[mask, "baseline_tco2e"] * (est_gas[mask] / total_est[mask])

    mask_no_energy = ~mask
    df.loc[mask_no_energy, "baseline_gas_t"] = df.loc[mask_no_energy, "baseline_tco2e"]

    df["baseline_cost"] = df["elec_kwh"] * ELEC_COST + df["gas_m3"] * GAS_COST

    retrofit_reduction_factor = 0.30  # on gas emissions
    solar_offset_factor = 0.20        # on electricity emissions

    gas_factor = 1 - (retrofit_pct / 100) * retrofit_reduction_factor
    elec_factor = 1 - (solar_pct / 100) * solar_offset_factor

    df["scenario_gas_t"] = df["baseline_gas_t"] * gas_factor
    df["scenario_elec_t"] = df["baseline_elec_t"] * elec_factor
    df["scenario_tco2e"] = df["scenario_gas_t"] + df["scenario_elec_t"]

    df["scenario_cost"] = (
        df["elec_kwh"] * elec_factor * ELEC_COST
        + df["gas_m3"] * gas_factor * GAS_COST
    )

    total_baseline = df["baseline_tco2e"].sum()
    total_scenario = df["scenario_tco2e"].sum()
    reduction = total_baseline - total_scenario
    reduction_pct = (reduction / total_baseline) * 100 if total_baseline else 0

    total_cost_baseline = df["baseline_cost"].sum()
    total_cost_scenario = df["scenario_cost"].sum()
    cost_savings = total_cost_baseline - total_cost_scenario
    cost_savings_pct = (cost_savings / total_cost_baseline) * 100 if total_cost_baseline else 0

    return (
        df,
        total_baseline,
        total_scenario,
        reduction,
        reduction_pct,
        total_cost_baseline,
        total_cost_scenario,
        cost_savings,
        cost_savings_pct,
    )


def generate_ai_style_summary(df, total_baseline, total_scenario, reduction, reduction_pct):
    if df.empty or total_baseline == 0:
        return "No buildings match the current filters. Try selecting a different neighbourhood or building type."

    top_neighbourhoods = df.groupby("neighbourhood")["baseline_tco2e"].sum().sort_values(ascending=False)
    main_neighbourhood = top_neighbourhoods.index[0]
    main_emissions = top_neighbourhoods.iloc[0]

    top_building = df.sort_values("baseline_tco2e", ascending=False).iloc[0]

    text = (
        f"Under the current filters, the buildings shown emit about {total_baseline:.0f} tonnes of CO₂ per year. "
        f"If the selected retrofit and solar measures were implemented, this could fall to roughly {total_scenario:.0f} tonnes, "
        f"a reduction of around {reduction_pct:.1f}% or {reduction:.0f} tonnes. "
        f"In this view, the highest-emitting neighbourhood is {main_neighbourhood}, at approximately {main_emissions:.0f} tonnes per year. "
        f"The single largest building is {top_building['name']}, which alone accounts for about "
        f"{top_building['baseline_tco2e']:.0f} tonnes annually. "
        f"For planners and sustainability staff, this suggests that early action in {main_neighbourhood} and targeted retrofits "
        f"at high-impact sites such as {top_building['name']} would deliver the quickest emission reductions."
    )
    return text


def split_text(text, max_chars):
    words = text.split()
    lines = []
    current = []
    for w in words:
        if sum(len(x) for x in current) + len(current) + len(w) > max_chars:
            lines.append(" ".join(current))
            current = [w]
        else:
            current.append(w)
    if current:
        lines.append(" ".join(current))
    return lines


def create_pdf(summary_text, scenario_df, include_images=False):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    margin = 50
    y = height - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Community Data – Buildings Scenario Report")
    y -= 30

    c.setFont("Helvetica", 9)
    c.drawString(margin, y, datetime.now().strftime("Generated on %Y-%m-%d %H:%M"))
    y -= 20

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Narrative summary")
    y -= 16

    c.setFont("Helvetica", 10)
    for line in split_text(summary_text, 90):
        c.drawString(margin, y, line)
        y -= 14
        if y < margin:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 10)

    if include_images:
        grouped = scenario_df.groupby("neighbourhood")[["baseline_tco2e", "scenario_tco2e"]].sum()
        grouped = grouped.reset_index()

        if not grouped.empty:
            if y < 200:
                c.showPage()
                y = height - margin

            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, y, "Emissions by neighbourhood – baseline vs scenario")
            y -= 20

            max_val = float(max(grouped["baseline_tco2e"].max(), grouped["scenario_tco2e"].max(), 1))
            chart_width = width - 2 * margin
            bar_height = 12
            gap = 10

            for _, row in grouped.iterrows():
                base_len = chart_width * (row["baseline_tco2e"] / max_val) * 0.8
                scen_len = chart_width * (row["scenario_tco2e"] / max_val) * 0.8

                c.setFont("Helvetica-Bold", 10)
                c.drawString(margin, y, row["neighbourhood"])
                y -= bar_height

                c.setFillGray(0.7)
                c.rect(margin, y, base_len, bar_height, fill=True, stroke=False)

                c.setFillGray(0.3)
                c.rect(margin, y - bar_height - 2, scen_len, bar_height, fill=True, stroke=False)

                y -= (bar_height * 2 + gap)
                c.setFillGray(0.0)

                if y < margin + 100:
                    c.showPage()
                    y = height - margin

    c.showPage()
    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


# --------------------------------
# SIDEBAR – FILTERS & UPLOADS
# --------------------------------

st.sidebar.title("Community Data – MVP")
st.sidebar.markdown(
    "Explore building, EV, and waste data at the neighbourhood level, and test simple retrofit and solar scenarios."
)

# Filters
st.sidebar.markdown("### 1. Filters")

neighbourhood_choice = st.sidebar.selectbox(
    "Neighbourhood",
    ["All"] + sorted(buildings_df["neighbourhood"].unique()),
)

btype_choice = st.sidebar.selectbox(
    "Building type",
    ["All"] + sorted(buildings_df["building_type"].unique()),
)

# Scenario sliders
st.sidebar.markdown("### 2. Scenario settings")

retrofit_choice = st.sidebar.slider(
    "Retrofit adoption (%)",
    0, 60, 20,
    help="Approximate share of buildings that receive energy retrofits (affects gas use)."
)

solar_choice = st.sidebar.slider(
    "Solar on suitable roofs (%)",
    0, 80, 30,
    help="Approximate share of suitable roofs with solar installed (affects electricity use)."
)

# Layer toggles
st.sidebar.markdown("### 3. Map layers")

show_buildings = st.sidebar.checkbox("Show buildings", True)
show_ev = st.sidebar.checkbox("Show EV chargers", True)
show_waste = st.sidebar.checkbox("Show waste sites", True)
show_neighbourhoods = st.sidebar.checkbox("Show neighbourhood boundaries", True)

# Optional data upload
st.sidebar.markdown("### 4. Optional: upload your own data")
st.sidebar.caption("Upload CSVs with the same column names as the default templates.")

b_file = st.sidebar.file_uploader("Buildings CSV", type="csv")
if b_file is not None:
    try:
        tmp = pd.read_csv(b_file)
        required_cols = {"name", "building_type", "neighbourhood", "latitude", "longitude", "baseline_tco2e"}
        if required_cols.issubset(tmp.columns):
            buildings_df = tmp
            st.sidebar.success("Using uploaded buildings data.")
        else:
            st.sidebar.error("Buildings CSV missing required columns. Using default sample instead.")
    except Exception:
        st.sidebar.error("Could not read buildings CSV. Using default sample instead.")

ev_file = st.sidebar.file_uploader("EV chargers CSV", type="csv")
if ev_file is not None:
    try:
        tmp = pd.read_csv(ev_file)
        required_cols = {"name", "neighbourhood", "latitude", "longitude", "num_ports"}
        if required_cols.issubset(tmp.columns):
            ev_df = tmp
            st.sidebar.success("Using uploaded EV chargers data.")
        else:
            st.sidebar.error("EV CSV missing required columns. Using default sample instead.")
    except Exception:
        st.sidebar.error("Could not read EV CSV. Using default sample instead.")

w_file = st.sidebar.file_uploader("Waste sites CSV", type="csv")
if w_file is not None:
    try:
        tmp = pd.read_csv(w_file)
        required_cols = {"name", "neighbourhood", "latitude", "longitude", "annual_waste_tonnes", "waste_tco2e"}
        if required_cols.issubset(tmp.columns):
            waste_df = tmp
            st.sidebar.success("Using uploaded waste data.")
        else:
            st.sidebar.error("Waste CSV missing required columns. Using default sample instead.")
    except Exception:
        st.sidebar.error("Could not read waste CSV. Using default sample instead.")


# --------------------------------
# APPLY FILTERS + SCENARIOS
# --------------------------------

filtered_buildings = apply_filters(buildings_df, neighbourhood_choice, btype_choice)
(
    scenario_buildings,
    total_baseline,
    total_scenario,
    reduction,
    reduction_pct,
    total_cost_baseline,
    total_cost_scenario,
    cost_savings,
    cost_savings_pct,
) = calculate_scenario(filtered_buildings, retrofit_choice, solar_choice)

ai_summary = generate_ai_style_summary(
    scenario_buildings,
    total_baseline,
    total_scenario,
    reduction,
    reduction_pct,
)


# --------------------------------
# MAIN LAYOUT
# --------------------------------

st.title("Community Data – Neighbourhood-level MVP ")
st.markdown(
    "This prototype uses OpenStreetMap as the basemap and overlays buildings, EV chargers, "
    "waste sites, and neighbourhood boundaries. You can filter by neighbourhood and building type, test simple scenarios, "
    "and generate lightweight reports."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Interactive Map", "Scenario Dashboard", "Insights & Narrative", "Reports"]
)

# --------------------------------
# TAB 1 – INTERACTIVE MAP (FOLIUM + OSM)
# --------------------------------
with tab1:
    st.subheader("Interactive OpenStreetMap view")

    if not scenario_buildings.empty:
        center_lat = scenario_buildings["latitude"].mean()
        center_lon = scenario_buildings["longitude"].mean()
    else:
        center_lat = buildings_df["latitude"].mean()
        center_lon = buildings_df["longitude"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")

    # Neighbourhood polygons
    if show_neighbourhoods:
        for poly in neighbourhood_polygons:
            coords = poly["coords"]
            folium.Polygon(
                locations=coords,
                color="#666666",
                weight=1,
                fill=True,
                fill_opacity=0.15,
                popup=f"Neighbourhood: {poly['neighbourhood']}",
            ).add_to(m)

    # Buildings
    if show_buildings and not scenario_buildings.empty:
        max_em = scenario_buildings["baseline_tco2e"].max() or 1
        for _, row in scenario_buildings.iterrows():
            intensity = row["baseline_tco2e"] / max_em
            r = int(160 + 95 * intensity)
            g = 90
            b = 140
            color = f"#{r:02x}{g:02x}{b:02x}"

            popup = (
                f"<b>{row['name']}</b><br>"
                f"Type: {row['building_type']}<br>"
                f"Neighbourhood: {row['neighbourhood']}<br>"
                f"Baseline: {row['baseline_tco2e']:.1f} tCO₂e<br>"
                f"Scenario: {row['scenario_tco2e']:.1f} tCO₂e"
            )

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=7,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                popup=popup,
            ).add_to(m)

    # EV chargers
    if show_ev and not ev_df.empty:
        if neighbourhood_choice != "All":
            ev_filtered = ev_df[ev_df["neighbourhood"] == neighbourhood_choice]
        else:
            ev_filtered = ev_df

        for _, row in ev_filtered.iterrows():
            popup = (
                f"<b>{row['name']}</b><br>"
                f"Neighbourhood: {row['neighbourhood']}<br>"
                f"Ports: {row['num_ports']}"
            )
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=5,
                color="#008000",
                fill=True,
                fill_color="#00a000",
                fill_opacity=0.9,
                popup=popup,
            ).add_to(m)

    # Waste sites
    if show_waste and not waste_df.empty:
        if neighbourhood_choice != "All":
            waste_filtered = waste_df[waste_df["neighbourhood"] == neighbourhood_choice]
        else:
            waste_filtered = waste_df

        for _, row in waste_filtered.iterrows():
            popup = (
                f"<b>{row['name']}</b><br>"
                f"Neighbourhood: {row['neighbourhood']}<br>"
                f"Waste: {row['annual_waste_tonnes']} t/yr<br>"
                f"Emissions: {row['waste_tco2e']} tCO₂e/yr"
            )
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=6,
                color="#8B4513",
                fill=True,
                fill_color="#A0522D",
                fill_opacity=0.9,
                popup=popup,
            ).add_to(m)

    st_data = st_folium(m, width="100%", height=600)
    st.caption(
        "Basemap: OpenStreetMap. Purple circles = buildings (darker = higher emissions), "
        "green circles = EV chargers, brown circles = waste sites, grey polygons = neighbourhoods."
    )

# --------------------------------
# TAB 2 – SCENARIO DASHBOARD
# --------------------------------
with tab2:
    st.subheader("Scenario dashboard – emissions and costs under current filters")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Baseline emissions (tCO₂e)", f"{total_baseline:,.1f}")
    col2.metric("Scenario emissions (tCO₂e)", f"{total_scenario:,.1f}")
    col3.metric("Reduction (tCO₂e)", f"{reduction:,.1f}")
    col4.metric("Reduction (%)", f"{reduction_pct:,.1f}%")
    col5.metric("Annual cost savings ($)", f"{cost_savings:,.0f}")

    st.markdown("#### Emissions by building (baseline vs scenario)")
    if not scenario_buildings.empty:
        chart_df = scenario_buildings[["name", "baseline_tco2e", "scenario_tco2e"]].set_index("name")
        st.bar_chart(chart_df)
    else:
        st.info("No buildings to display for the current filters.")

    with st.expander("Assumptions used in this simple scenario model"):
        st.write(
            "- Retrofits are assumed to cut **natural gas-related emissions** from affected buildings by about **30%**.\n"
            "- Solar is assumed to offset about **20%** of **electricity-related emissions** on participating roofs.\n"
            "- Energy cost assumptions are illustrative only (e.g., about $0.15/kWh for electricity and $0.40/m³ for natural gas). "
            "These can be updated to match your actual tariffs."
        )

# --------------------------------
# TAB 3 – INSIGHTS & NARRATIVE
# --------------------------------
with tab3:
    st.subheader("Insights for planners and sustainability staff")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Highest-emitting buildings")
        if not scenario_buildings.empty:
            st.dataframe(
                scenario_buildings.sort_values("baseline_tco2e", ascending=False)[
                    ["name", "neighbourhood", "building_type", "baseline_tco2e", "scenario_tco2e"]
                ].head(5)
            )
        else:
            st.info("No building emissions to display under the current filters.")

        st.markdown("#### EV charger coverage by neighbourhood")
        if not ev_df.empty:
            ev_counts = ev_df.groupby("neighbourhood")["num_ports"].sum().reset_index()
            st.dataframe(ev_counts)
        else:
            st.info("No EV charger data available.")

    with col2:
        st.markdown("#### Waste sites")
        if not waste_df.empty:
            st.dataframe(waste_df[["name", "neighbourhood", "annual_waste_tonnes", "waste_tco2e"]])
        else:
            st.info("No waste site data available.")

        st.markdown("#### Emissions and cost per neighbourhood")
        if not scenario_buildings.empty:
            nh_summary = scenario_buildings.groupby("neighbourhood").agg(
                baseline_emissions=("baseline_tco2e", "sum"),
                scenario_emissions=("scenario_tco2e", "sum"),
                baseline_cost=("baseline_cost", "sum"),
                scenario_cost=("scenario_cost", "sum"),
            ).reset_index()
            st.dataframe(nh_summary)
        else:
            st.info("No neighbourhood summary available under current filters.")

        st.markdown("#### Narrative summary")
        st.write(ai_summary)

# --------------------------------
# TAB 4 – REPORTS
# --------------------------------
with tab4:
    st.subheader("Downloadable reports")

    st.markdown(
        "Generate simple PDF reports based on the current filters and scenario assumptions. "
        "These can be used as starting points for staff updates or briefing notes."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Generate text-only PDF report"):
            pdf_bytes = create_pdf(ai_summary, scenario_buildings, include_images=False)
            st.download_button(
                label="Download text-only PDF",
                data=pdf_bytes,
                file_name="community_data_report_text_only.pdf",
                mime="application/pdf",
            )

    with col_b:
        if st.button("Generate PDF report with simple visuals"):
            pdf_bytes_img = create_pdf(ai_summary, scenario_buildings, include_images=True)
            st.download_button(
                label="Download PDF with visuals",
                data=pdf_bytes_img,
                file_name="community_data_report_with_visuals.pdf",
                mime="application/pdf",
            )

    st.markdown(
        "_In a future version, these templates could be aligned with your specific staff and council reporting formats._"
    )
