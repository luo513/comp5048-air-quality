from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, ctx, dcc, html

from data_pipeline import SELECTED_ATTRIBUTES


DATA_PATH = Path(__file__).parent / "data" / "air_quality_clean.csv"
NUMERIC_ATTRIBUTES = SELECTED_ATTRIBUTES[1:]
ATTRIBUTE_LABELS = {
    "CO(GT)": "CO reference",
    "PT08.S1(CO)": "CO sensor",
    "C6H6(GT)": "Benzene",
    "NOx(GT)": "NOx reference",
    "NO2(GT)": "NO₂ reference",
    "PT08.S5(O3)": "O₃ sensor",
    "T": "Temperature",
    "RH": "Relative humidity",
    "AH": "Absolute humidity",
}
PLOT_TEMPLATE = "plotly_white"
HOUR_PRESETS = {
    "preset-all": [0, 23],
    "preset-morning": [7, 10],
    "preset-midday": [12, 15],
    "preset-evening": [17, 20],
}


def load_data() -> pd.DataFrame:
    frame = pd.read_csv(DATA_PATH, parse_dates=["DateTime"])
    frame["row_id"] = frame.index.astype(str)
    return frame


df = load_data()
month_values = sorted(df["Month"].unique())
ALL_MONTHS = "__ALL_MONTHS__"
month_options = [
    {"label": "All months", "value": ALL_MONTHS},
    *[{"label": value, "value": value} for value in month_values],
]
attribute_options = [
    {"label": ATTRIBUTE_LABELS[name], "value": name} for name in NUMERIC_ATTRIBUTES
]


def prepare_data(months: list[str], hours: list[int], grain: str) -> pd.DataFrame:
    selected_months = month_values if not months or ALL_MONTHS in months else months
    filtered = df[
        df["Month"].isin(selected_months)
        & df["Hour"].between(hours[0], hours[1], inclusive="both")
    ].copy()

    if grain == "daily":
        daily = (
            filtered.set_index("DateTime")[NUMERIC_ATTRIBUTES]
            .resample("D")
            .mean()
            .dropna(how="all")
            .reset_index()
        )
        daily["Hour"] = -1
        daily["Weekday"] = daily["DateTime"].dt.day_name()
        daily["Month"] = daily["DateTime"].dt.to_period("M").astype(str)
        daily["row_id"] = daily["DateTime"].dt.strftime("%Y-%m-%d")
        return daily

    return filtered


def selected_ids(selected_data: dict | None) -> set[str]:
    if not selected_data or not selected_data.get("points"):
        return set()
    return {
        str(point["customdata"][0])
        for point in selected_data["points"]
        if point.get("customdata")
    }


def empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15, "color": "#52606D"},
    )
    figure.update_layout(template=PLOT_TEMPLATE, xaxis={"visible": False}, yaxis={"visible": False})
    return figure


def build_scatter(
    frame: pd.DataFrame,
    x_attribute: str,
    y_attribute: str,
    colour_attribute: str,
    chosen_ids: set[str],
) -> go.Figure:
    plotted = frame.dropna(subset=[x_attribute, y_attribute, colour_attribute]).copy()
    if plotted.empty:
        return empty_figure("No observations match the current filters.")

    plotted["selected"] = plotted["row_id"].isin(chosen_ids)
    figure = px.scatter(
        plotted,
        x=x_attribute,
        y=y_attribute,
        color=colour_attribute,
        custom_data=["row_id", "DateTime"],
        color_continuous_scale="Viridis",
        labels=ATTRIBUTE_LABELS,
        template=PLOT_TEMPLATE,
        render_mode="webgl",
    )
    figure.update_traces(
        marker={"size": 7, "opacity": 0.62, "line": {"width": 0}},
        selected={"marker": {"size": 10, "opacity": 1, "color": "#102A43"}},
        unselected={"marker": {"opacity": 0.16}},
    )
    if chosen_ids:
        figure.update_traces(selectedpoints=[
            index for index, value in enumerate(plotted["selected"].tolist()) if value
        ])
    figure.update_layout(
        title="Projection view — drag a box or lasso around a candidate group",
        dragmode="lasso",
        margin={"l": 55, "r": 25, "t": 70, "b": 50},
        coloraxis_colorbar={"title": ATTRIBUTE_LABELS[colour_attribute]},
        uirevision="scatter-controls",
    )
    return figure


def build_parallel(frame: pd.DataFrame, chosen_ids: set[str]) -> tuple[go.Figure, pd.DataFrame]:
    focus = frame[frame["row_id"].isin(chosen_ids)].copy() if chosen_ids else frame.copy()
    plotted = focus.dropna(subset=NUMERIC_ATTRIBUTES)
    if plotted.empty:
        return empty_figure("The selected observations do not have complete values across all axes."), plotted

    dimensions = [
        {
            "label": ATTRIBUTE_LABELS[name],
            "values": plotted[name],
            "range": [float(frame[name].min()), float(frame[name].max())],
        }
        for name in NUMERIC_ATTRIBUTES
    ]
    figure = go.Figure(
        data=go.Parcoords(
            line={
                "color": plotted["CO(GT)"],
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": "CO"},
            },
            dimensions=dimensions,
            labelfont={"size": 12, "color": "#243B53"},
            tickfont={"size": 10, "color": "#52606D"},
        )
    )
    title = "Parallel coordinates — selected group" if chosen_ids else "Parallel coordinates — filtered observations"
    figure.update_layout(
        title=title,
        template=PLOT_TEMPLATE,
        height=550,
        margin={"l": 45, "r": 65, "t": 85, "b": 45},
    )
    return figure, plotted


def build_time_distribution(frame: pd.DataFrame, chosen_ids: set[str], grain: str) -> go.Figure:
    focus = frame[frame["row_id"].isin(chosen_ids)].copy() if chosen_ids else frame.copy()
    if focus.empty:
        return empty_figure("No observations to summarise.")

    if grain == "hourly":
        counts = focus.groupby("Hour", as_index=False).size()
        figure = px.bar(counts, x="Hour", y="size", template=PLOT_TEMPLATE)
        figure.update_xaxes(dtick=2, title="Hour of day")
    else:
        counts = focus.groupby("Month", as_index=False).size()
        figure = px.bar(counts, x="Month", y="size", template=PLOT_TEMPLATE)
        figure.update_xaxes(title="Month")
    figure.update_traces(marker_color="#2F80ED")
    figure.update_yaxes(title="Observations")
    figure.update_layout(
        title="When the current observations occur",
        showlegend=False,
        margin={"l": 50, "r": 20, "t": 65, "b": 55},
    )
    return figure


def build_attribute_profile(frame: pd.DataFrame, chosen_ids: set[str]) -> go.Figure:
    if not chosen_ids:
        return empty_figure("Select a candidate group in the projection to compare its attribute profile.")

    selected = frame[frame["row_id"].isin(chosen_ids)]
    rows = []
    for attribute in NUMERIC_ATTRIBUTES:
        population = frame[attribute].dropna()
        group_values = selected[attribute].dropna()
        if population.empty or group_values.empty:
            continue
        group_median = float(group_values.median())
        percentile = float((population <= group_median).mean() * 100)
        rows.append(
            {
                "attribute": ATTRIBUTE_LABELS[attribute],
                "percentile": percentile,
                "direction": "Above typical" if percentile >= 50 else "Below typical",
            }
        )

    if not rows:
        return empty_figure("The selected group has no comparable attribute values.")

    profile = pd.DataFrame(rows).sort_values("percentile")
    colours = ["#D95F59" if value >= 50 else "#2F80ED" for value in profile["percentile"]]
    figure = go.Figure(
        go.Bar(
            x=profile["percentile"],
            y=profile["attribute"],
            orientation="h",
            marker_color=colours,
            text=[f"{value:.0f}th" for value in profile["percentile"]],
            textposition="outside",
            cliponaxis=False,
            customdata=profile["direction"],
            hovertemplate="%{y}<br>%{x:.1f}th percentile<br>%{customdata}<extra></extra>",
        )
    )
    figure.add_vline(x=50, line_dash="dash", line_color="#7B8794")
    figure.update_xaxes(range=[0, 108], ticksuffix="th", dtick=25, title="Percentile within filtered data")
    figure.update_layout(
        title="Selected group attribute profile",
        template=PLOT_TEMPLATE,
        height=440,
        showlegend=False,
        margin={"l": 40, "r": 55, "t": 65, "b": 55},
    )
    return figure


app = Dash(__name__)
server = app.server
app.title = "Air Quality Visual Analytics"
app.layout = html.Main(
    [
        html.Header(
            [
                html.P("INTERACTIVE EXPLORATION", className="eyebrow"),
                html.H1("Air Quality Visual Analytics"),
                html.P(
                    "Filter the observation period, inspect a projection, and select a candidate group. "
                    "The linked views reveal its multivariate profile and temporal distribution.",
                    className="subtitle",
                ),
            ],
            className="page-header",
        ),
        html.Section(
            [
                html.Div(
                    [
                        html.Label("Observation level"),
                        dcc.RadioItems(
                            id="grain",
                            options=[
                                {"label": " Daily", "value": "daily"},
                                {"label": " Hourly", "value": "hourly"},
                            ],
                            value="daily",
                            inline=True,
                            className="radio-row",
                        ),
                    ],
                    className="control",
                ),
                html.Div(
                    [
                        html.Label("Months"),
                        dcc.Dropdown(
                            id="months",
                            options=month_options,
                            value=[ALL_MONTHS],
                            multi=True,
                            placeholder="Select months",
                        ),
                    ],
                    className="control control-wide",
                ),
                html.Div(
                    [
                        html.Label("Hours included"),
                        dcc.RangeSlider(
                            id="hours",
                            min=0,
                            max=23,
                            step=1,
                            value=[0, 23],
                            marks={0: "00", 6: "06", 12: "12", 18: "18", 23: "23"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        html.Div(
                            [
                                html.Button("All day 00–23", id="preset-all", n_clicks=0),
                                html.Button("Morning peak 07–10", id="preset-morning", n_clicks=0),
                                html.Button("Midday 12–15", id="preset-midday", n_clicks=0),
                                html.Button("Evening peak 17–20", id="preset-evening", n_clicks=0),
                            ],
                            className="preset-buttons",
                        ),
                        html.P(id="hours-help", className="control-help"),
                    ],
                    className="control control-wide",
                ),
            ],
            className="controls panel",
        ),
        html.Section(
            [
                html.Div([html.Span("Filtered", className="card-label"), html.Strong(id="filtered-count")], className="metric-card"),
                html.Div([html.Span("Selected", className="card-label"), html.Strong(id="selected-count")], className="metric-card"),
                html.Div([html.Span("Complete profile", className="card-label"), html.Strong(id="complete-count")], className="metric-card"),
                html.Div([html.Span("Date range", className="card-label"), html.Strong(id="date-range")], className="metric-card metric-card-wide"),
            ],
            className="metrics",
        ),
        html.Section(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("X axis"),
                                dcc.Dropdown(id="x-attribute", options=attribute_options, value="T", clearable=False),
                            ],
                            className="mini-control",
                        ),
                        html.Div(
                            [
                                html.Label("Y axis"),
                                dcc.Dropdown(id="y-attribute", options=attribute_options, value="CO(GT)", clearable=False),
                            ],
                            className="mini-control",
                        ),
                        html.Div(
                            [
                                html.Label("Colour"),
                                dcc.Dropdown(id="colour", options=attribute_options, value="NOx(GT)", clearable=False),
                            ],
                            className="mini-control",
                        ),
                    ],
                    className="chart-controls",
                ),
                dcc.Graph(id="projection", config={"displaylogo": False, "modeBarButtonsToAdd": ["select2d", "lasso2d"]}),
                html.Div(
                    [
                        html.P(
                            "Drag around a dense or visually separated region, then inspect the linked views.",
                            className="chart-note",
                        ),
                        html.Button("Clear selection", id="clear-selection", n_clicks=0, className="clear-button"),
                    ],
                    className="chart-actions",
                ),
            ],
            className="panel projection-panel",
        ),
        html.Section(
            [
                html.Div(dcc.Graph(id="parallel-coordinates", config={"displaylogo": False}), className="panel parallel-panel"),
                html.Div(
                    [
                        html.Div(dcc.Graph(id="time-distribution", config={"displaylogo": False}), className="panel time-panel"),
                        html.Div(dcc.Graph(id="attribute-profile", config={"displaylogo": False}), className="panel profile-panel"),
                    ],
                    className="supporting-grid",
                ),
            ],
            className="linked-grid",
        ),
        html.Section(
            [
                html.H2("How to read this view"),
                html.P(
                    "Each line in the parallel coordinates represents one hour or one daily average. "
                    "Similar line paths indicate similar multivariate profiles. The time chart shows when "
                    "the currently selected observations occur. The attribute profile positions each selected-group "
                    "median within the filtered population: values near 100 are unusually high and values near 0 "
                    "are unusually low. Values coded as −200 are treated as missing and are never imputed."
                ),
            ],
            className="method-note",
        ),
    ]
)


@app.callback(
    Output("projection", "figure"),
    Output("parallel-coordinates", "figure"),
    Output("time-distribution", "figure"),
    Output("attribute-profile", "figure"),
    Output("filtered-count", "children"),
    Output("selected-count", "children"),
    Output("complete-count", "children"),
    Output("date-range", "children"),
    Input("months", "value"),
    Input("hours", "value"),
    Input("grain", "value"),
    Input("x-attribute", "value"),
    Input("y-attribute", "value"),
    Input("colour", "value"),
    Input("projection", "selectedData"),
)
def update_views(
    months: list[str],
    hours: list[int],
    grain: str,
    x_attribute: str,
    y_attribute: str,
    colour_attribute: str,
    selected_data: dict | None,
):
    frame = prepare_data(months, hours, grain)
    chosen_ids = selected_ids(selected_data)
    chosen_ids &= set(frame["row_id"].astype(str))

    scatter = build_scatter(frame, x_attribute, y_attribute, colour_attribute, chosen_ids)
    parallel, complete = build_parallel(frame, chosen_ids)
    time_distribution = build_time_distribution(frame, chosen_ids, grain)
    attribute_profile = build_attribute_profile(frame, chosen_ids)

    if frame.empty:
        date_text = "No data"
    else:
        date_text = f"{frame['DateTime'].min():%d %b %Y} – {frame['DateTime'].max():%d %b %Y}"

    return (
        scatter,
        parallel,
        time_distribution,
        attribute_profile,
        f"{len(frame):,}",
        f"{len(chosen_ids):,}" if chosen_ids else "All",
        f"{len(complete):,}",
        date_text,
    )


@app.callback(
    Output("hours", "value"),
    Input("preset-all", "n_clicks"),
    Input("preset-morning", "n_clicks"),
    Input("preset-midday", "n_clicks"),
    Input("preset-evening", "n_clicks"),
    prevent_initial_call=True,
)
def apply_hour_preset(*_clicks: int) -> list[int]:
    return HOUR_PRESETS.get(ctx.triggered_id, [0, 23])


@app.callback(Output("hours-help", "children"), Input("grain", "value"))
def explain_hour_filter(grain: str) -> str:
    if grain == "daily":
        return "Daily values are averages calculated from the selected hours."
    return "Only individual hourly observations within the selected range are shown."


@app.callback(
    Output("projection", "selectedData"),
    Input("clear-selection", "n_clicks"),
    prevent_initial_call=True,
)
def clear_projection_selection(_n_clicks: int) -> None:
    return None


if __name__ == "__main__":
    app.run(
        debug=os.getenv("DASH_DEBUG", "false").lower() == "true",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8050")),
    )
