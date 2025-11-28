import os
from io import BytesIO
from datetime import datetime

import pandas as pd
import pydeck as pdk
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

st.set_page_config(page_title="Community Data – MVP", layout="wide")

# ------------------------
# DUMMY DATA
# ------------------------

# Buildings data
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

buildings_df = pd.DataFrame(buildings_data)

# EV chargers data
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

ev_df = pd.DataFrame(ev_data)

# Waste sites data
waste_data = [
    {"site_id": 1, "name": "North Transfer Station", "neighbourhood": "North",
     "latitude": 43.488, "longitude": -80.53, "annual_waste_tonnes": 12000, "waste_tco2e": 5500},
    {"site_id": 2, "name": "South Organics Depot", "neighbourhood": "South",
     "latitude": 43.452, "longitude": -80.518, "annual_waste_tonnes": 6000, "waste_tco2e": 2100},
]

waste_df = pd.DataFrame(waste_data)

# Neighbourhood boundaries (simple dummy polygons)
neighbourhood_polygons = [
    {
        "neighbourhood": "Central",
        "polygon": [
            [-80.526, 43.464],
            [-80.516, 43.464],
            [-80.516, 43.472],
            [-80.526, 43.472],
        ],
    },
    {
        "neighbourhood": "North",
        "polygon": [
            [-80.532, 43.48],
            [-80.522, 43.48],
            [-80.522, 43.49],
            [-80.532, 43.49],
        ],
    },
    {
        "neighbourhood": "South",
        "polygon": [
            [-80.52, 43.452],
            [-80.51, 43.452],
            [-80.51, 43.46],
            [-80.52, 43.46],
        ],
    },
    {
        "neighbourhood": "East",
        "polygon": [
            [-80.514, 43.46],
            [-80.504, 43.46],
            [-80.504, 43.468],
            [-80.514, 43.468],
        ],
    },
]

neighbourhood_df = pd.DataFrame(neighbourhood_polygons)

# ------------------------
# HELPER FUNCTIONS
# ------------------------

def apply_filters(neighbourhood, building_type):
    df = buildings_df.copy()
    if neighbourhood != "All":
        df = df[df["neighbourhood"] == neighbourhood]
    if building_type != "All":
        df = df[df["building_type"] == building_type]
    return df


def calculate_scenario(df, retrofit_pct, solar_pct):
    retrofit_reduction_factor = 0.30    # 30% emissions reduction for retrofitted buildings
    solar_offset_factor = 0.20          # 20% electricity offset from solar installs

    df = df.copy()
    df["scenario_tco2e"] = df["baseline_tco2e"] * (
        1 - (retrofit_pct / 100) * retrofit_reduction_factor
    ) * (
        1 - (solar_pct / 100) * solar_offset_factor
    )

    total_baseline = df["baseline_tco2e"].sum()
    total_scenario = df["scenario_tco2e"].sum()
    reduction = total_baseline - total_scenario
    reduction_pct = (reduction / total_baseline) * 100 if total_baseline else 0

    return df, total_baseline, total_scenario, reduction, reduction_pct


def generate_fallback_summary(df, total_baseline, total_scenario, reduction, reduction_pct):
    if df.empty:
        return "No buildings match the current filters. Try selecting a different neighbourhood or building type."

    top_neighbourhoods = df.groupby("neighbourhood")["baseline_tco2e"].sum().sort_values(ascending=False)
    main_neighbourhood = top_neighbourhoods.index[0]
    main_emissions = top_neighbourhoods.iloc[0]
    top_building = df.sort_values("baseline_tco2e", ascending=False).iloc[0]

    return (
        f"Under the current filters, the buildings shown emit about {total_baseline:.0f} tonnes of CO2 per year. "
        f"With the selected retrofit and solar assumptions, this could drop to roughly {total_scenario:.0f} tonnes, "
        f"a reduction of about {reduction_pct:.1f} percent, or {reduction:.0f} tonnes. "
        f"The neighbourhood contributing most of these emissions is {main_neighbourhood}, at around {main_emissions:.0f} tonnes. "
        f"The single largest building in this view is {top_building['name']}, which alone accounts for about "
        f"{top_building['baseline_tco2e']:.0f} tonnes per year. "
        f"This suggests that early action in {main_neighbourhood} and targeted retrofits at high-impact sites such as "
        f"{top_building['name']} will deliver the quickest emission reductions."
    )


def generate_ai_summary(df, total_baseline, total_scenario, reduction, reduction_pct):
    """
    If OPENAI_API_KEY + openai library are available, use the OpenAI API.
    Otherwise fall back to a simple internal summary so the app never breaks.
    """
    try:
        from openai import OpenAI  # requires `openai` in requirements.txt
        client = OpenAI()  # uses OPENAI_API_KEY from env or Streamlit secrets

        # Build a compact prompt
        prompt = (
            "You are a municipal climate and sustainability analyst. "
            "Write a short, plain-language paragraph (max 220 words) summarizing the current emissions scenario "
            "for a city planner and sustainability manager.\n\n"
            f"Total baseline emissions (tCO2e): {total_baseline:.1f}\n"
            f"Scenario emissions (tCO2e): {total_scenario:.1f}\n"
            f"Reduction (tCO2e): {reduction:.1f}\n"
            f"Reduction (%): {reduction_pct:.1f}\n\n"
            "Dataset (each row = building):\n"
        )

        small_df = df[["name", "neighbourhood", "building_type", "baseline_tco2e", "scenario_tco2e"]].head(10)
        prompt += small_df.to_csv(index=False)

        prompt += (
            "\nFocus on: which neighbourhoods or building types stand out; "
            "what this means for near-term retrofit and solar decisions; "
            "and how this view could support staff or council reporting. "
            "Avoid jargon and keep the tone clear and practical."
        )

        # Use Responses API (current recommended pattern) :contentReference[oaicite:0]{index=0}
        response = client.responses.create(
            model="gpt-5.1-mini",
            input=prompt,
            max_output_tokens=300,
        )

        # Extract plain text
        first_output = response.output[0]
        first_content = first_output.content[0]
        return getattr(first_content, "text", str(first_content))

    except Exception:
        # Any issue (no key, library missing, etc.) → safe fallback
        return generate_fallback_summary(df, total_baseline, total_scenario, reduction, reduction_pct)


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
    c.drawString(margin, y, "AI-generated narrative summary")
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

# ------------------------
# SIDEBAR CONTROLS
# ------------------------
st.sidebar.title("Community Data – MVP")
st.sidebar.markdown("Filter community-level data and test scenarios across buildings, EV chargers, and waste sites.")

neighbourhood_choice = st.sidebar.selectbox("Neighbourhood", ["All"] + sorted(buildings_df["neighbourhood"].unique()))
btype_choice = st.sidebar.selectbox("Building type", ["All"] + sorted(buildings_df["building_type"].unique()))

retrofit_choice = st.sidebar.slider("Retrofit adoption (%)", 0, 60, 20)
solar_choice = st.sidebar.slider("Solar on suitable roofs (%)", 0, 80, 30)

st.sidebar.markdown("---")
show_buildings = st.sidebar.checkbox("Show buildings", True)
show_ev = st.sidebar.checkbox("Show EV chargers", True)
show_waste = st.sidebar.checkbox("Show waste sites", True)
show_neighbourhoods = st.sidebar.checkbox("Show neighbourhood boundaries", True)
show_heatmap = st.sidebar.checkbox("Show emissions heatmap (buildings)", False)

# ------------------------
# APPLY FILTERS + SCENARIOS
# ------------------------
filtered_buildings = apply_filters(neighbourhood_choice, btype_choice)
scenario_buildings, total_baseline, total_scenario, reduction, reduction_pct = calculate_scenario(
    filtered_buildings, retrofit_choice, solar_choice
)

ai_summary = generate_ai_summary(scenario_buildings, total_baseline, total_scenario, reduction, reduction_pct)

# ------------------------
# LAYOUT
# ------------------------
st.title("Community Data – Buildings, EV, Waste, and Scenarios")

tab1, tab2, tab3, tab4 = st.tabs(["Interactive Map", "Scenario Dashboard", "Insights & AI Summary", "Reports"])

# ------------------------
# TAB 1 – INTERACTIVE MAP
# ------------------------
with tab1:
    st.subheader("Community map: buildings, EV chargers, waste, neighbourhoods, and heatmap")

    layers = []

    # Neighbourhood polygons
    if show_neighbourhoods:
        poly_df = neighbourhood_df.copy()
        poly_df["elevation"] = 10

        poly_layer = pdk.Layer(
            "PolygonLayer",
            data=poly_df,
            get_polygon="polygon",
            get_fill_color=[200, 200, 200, 40],
            get_line_color=[80, 80, 80],
            line_width_min_pixels=1,
            pickable=True,
        )
        layers.append(poly_layer)

    # Heatmap of building emissions (overlays on top of polygons)
    if show_heatmap and not scenario_buildings.empty:
        heat_layer = pdk.Layer(
            "HeatmapLayer",
            data=scenario_buildings,
            get_position=["longitude", "latitude"],
            get_weight="baseline_tco2e",
            radiusPixels=60,
        )
        layers.append(heat_layer)

    # Building points
    if show_buildings and not scenario_buildings.empty:
        max_em = scenario_buildings["baseline_tco2e"].max() or 1
        scenario_buildings = scenario_buildings.copy()
        scenario_buildings["color_r"] = (255 * scenario_buildings["baseline_tco2e"] / max_em).clip(50, 255)
        scenario_buildings["color"] = scenario_buildings["color_r"].apply(lambda v: [int(v), 80, 120, 200])

        bldg_layer = pdk.Layer(
            "ScatterplotLayer",
            data=scenario_buildings,
            get_position=["longitude", "latitude"],
            get_radius=80,
            get_fill_color="color",
            pickable=True,
        )
        layers.append(bldg_layer)

    # EV chargers
    if show_ev:
        if neighbourhood_choice != "All":
            ev_filtered = ev_df[ev_df["neighbourhood"] == neighbourhood_choice]
        else:
            ev_filtered = ev_df

        if not ev_filtered.empty:
            ev_layer = pdk.Layer(
                "ScatterplotLayer",
                data=ev_filtered,
                get_position=["longitude", "latitude"],
                get_radius=60,
                get_fill_color=[0, 150, 0, 200],
                pickable=True,
            )
            layers.append(ev_layer)

    # Waste sites
    if show_waste:
        if neighbourhood_choice != "All":
            waste_filtered = waste_df[waste_df["neighbourhood"] == neighbourhood_choice]
        else:
            waste_filtered = waste_df

        if not waste_filtered.empty:
            waste_layer = pdk.Layer(
                "ScatterplotLayer",
                data=waste_filtered,
                get_position=["longitude", "latitude"],
                get_radius=100,
                get_fill_color=[150, 75, 0, 220],
                pickable=True,
            )
            layers.append(waste_layer)

    # Map centre
    if not scenario_buildings.empty:
        center_lat = scenario_buildings["latitude"].mean()
        center_lon = scenario_buildings["longitude"].mean()
    else:
        center_lat = buildings_df["latitude"].mean()
        center_lon = buildings_df["longitude"].mean()

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=12
    )

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={"text": "{name}"},
        map_style="mapbox://styles/mapbox/light-v9",
    )

    st.pydeck_chart(deck)
    st.caption("Use the checkboxes on the left to toggle buildings, heatmap, EV chargers, waste sites, and neighbourhood boundaries.")

# ------------------------
# TAB 2 – SCENARIO DASHBOARD
# ------------------------
with tab2:
    st.subheader("Scenario dashboard – buildings only")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Baseline emissions (tCO2e)", f"{total_baseline:,.1f}")
    col2.metric("Scenario emissions (tCO2e)", f"{total_scenario:,.1f}")
    col3.metric("Reduction (tCO2e)", f"{reduction:,.1f}")
    col4.metric("Reduction (%)", f"{reduction_pct:,.1f}%")

    st.markdown("#### Emissions by building (baseline vs scenario)")
    if not scenario_buildings.empty:
        chart_df = scenario_buildings[["name", "baseline_tco2e", "scenario_tco2e"]].set_index("name")
        st.bar_chart(chart_df)
    else:
        st.info("No buildings to display for the current filters.")

# ------------------------
# TAB 3 – INSIGHTS & AI SUMMARY
# ------------------------
with tab3:
    st.subheader("Insights for planners and sustainability teams")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Highest-emitting buildings")
        st.dataframe(
            scenario_buildings.sort_values("baseline_tco2e", ascending=False)[
                ["name", "neighbourhood", "building_type", "baseline_tco2e", "scenario_tco2e"]
            ].head(5)
        )

        st.markdown("#### EV charger coverage by neighbourhood")
        ev_counts = ev_df.groupby("neighbourhood")["num_ports"].sum().reset_index()
        st.dataframe(ev_counts)

    with col2:
        st.markdown("#### Waste sites")
        st.dataframe(waste_df[["name", "neighbourhood", "annual_waste_tonnes", "waste_tco2e"]])

        st.markdown("#### AI-style narrative summary")
        st.write(ai_summary)

# ------------------------
# TAB 4 – REPORTS
# ------------------------
with tab4:
    st.subheader("Downloadable reports")

    st.markdown("Generate simple PDF reports based on the current filters and scenario assumptions.")

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

    st.markdown("---")
    st.markdown(
        "These PDFs are intentionally lightweight for a pilot. "
        "In a production version, they could be expanded into full council-ready or staff-ready reporting templates."
    )

