from __future__ import annotations

import os
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

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
GROUP_COLOURS = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02"]
ANALYSIS_SCENARIOS = {
    "scenario-pollution": ("T", "CO(GT)", "NOx(GT)", "Pollution peaks"),
    "scenario-humidity": ("RH", "NO2(GT)", "AH", "Humidity effect"),
    "scenario-sensors": ("CO(GT)", "PT08.S1(CO)", "C6H6(GT)", "Sensor consistency"),
    "scenario-weather": ("T", "RH", "CO(GT)", "Weather and pollution"),
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


def default_saved_groups() -> list[dict]:
    """Return the two manually validated groups used by the shared application."""
    daily = prepare_data([ALL_MONTHS], [0, 23], "daily").dropna(
        subset=["T", "CO(GT)", "NOx(GT)"]
    )
    high_pollution = daily[
        daily["T"].between(5, 23, inclusive="both")
        & daily["CO(GT)"].between(3, 5.65, inclusive="both")
    ]
    hot_low_pollution = daily[
        daily["T"].between(25, 33, inclusive="both")
        & daily["CO(GT)"].between(0.5, 1.7, inclusive="both")
    ]
    return [
        {
            "name": "High pollution",
            "colour": GROUP_COLOURS[0],
            "ids": high_pollution["row_id"].astype(str).tolist(),
        },
        {
            "name": "Hot low-pollution",
            "colour": GROUP_COLOURS[1],
            "ids": hot_low_pollution["row_id"].astype(str).tolist(),
        },
    ]


def with_default_groups(saved_groups: list[dict] | None) -> list[dict]:
    """Keep the two validated groups present in every browser/session state."""
    defaults = default_saved_groups()
    default_names = {str(group["name"]) for group in defaults}
    extras = [
        group
        for group in (saved_groups or [])
        if str(group.get("name", "")) not in default_names
    ]
    return defaults + extras


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
    selection_revision: int,
    saved_groups: list[dict],
    hidden_groups: list[str] | None = None,
) -> go.Figure:
    plotted = frame.dropna(subset=[x_attribute, y_attribute, colour_attribute]).copy()
    if plotted.empty:
        return empty_figure("No observations match the current filters.")

    plotted["selected"] = plotted["row_id"].isin(chosen_ids)
    active_groups = [group for group in saved_groups if group.get("ids")]
    if active_groups:
        plotted["saved_group"] = "Ungrouped"
        colour_map = {"Ungrouped": "#B8C2CC"}
        group_order = ["Ungrouped"]
        for group in active_groups:
            name = str(group["name"])
            plotted.loc[plotted["row_id"].isin(set(group["ids"])), "saved_group"] = name
            colour_map[name] = group["colour"]
            group_order.append(name)
        figure = px.scatter(
            plotted,
            x=x_attribute,
            y=y_attribute,
            color="saved_group",
            custom_data=["row_id", "DateTime"],
            color_discrete_map=colour_map,
            category_orders={"saved_group": group_order},
            labels={**ATTRIBUTE_LABELS, "saved_group": "Saved group"},
            template=PLOT_TEMPLATE,
            render_mode="webgl",
        )
    else:
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

    hidden = set(hidden_groups or [])
    for trace in figure.data:
        if trace.name in hidden:
            trace.visible = "legendonly"
    figure.update_traces(
        marker={"size": 7, "opacity": 0.62, "line": {"width": 0}},
        selected={"marker": {"size": 10, "opacity": 1}},
        unselected={"marker": {"opacity": 0.16}},
    )
    for trace in figure.data:
        if chosen_ids:
            trace.selectedpoints = [
                index
                for index, customdata in enumerate(trace.customdata)
                if str(customdata[0]) in chosen_ids
            ]
        else:
            trace.selectedpoints = None
    figure.update_layout(
        title="Projection view — drag a box or lasso around a candidate group",
        dragmode="lasso",
        margin={"l": 55, "r": 25, "t": 70, "b": 50},
        coloraxis_colorbar={"title": ATTRIBUTE_LABELS[colour_attribute]},
        uirevision="scatter-controls",
        selectionrevision=str(selection_revision),
        selections=[],
    )
    return figure


def build_parallel(frame: pd.DataFrame, chosen_ids: set[str] | None) -> tuple[go.Figure, pd.DataFrame]:
    focus = frame.copy() if chosen_ids is None else frame[frame["row_id"].isin(chosen_ids)].copy()
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
    title = "Parallel coordinates — selected group" if chosen_ids is not None else "Parallel coordinates — filtered observations"
    figure.update_layout(
        title=title,
        template=PLOT_TEMPLATE,
        height=550,
        margin={"l": 45, "r": 65, "t": 85, "b": 45},
    )
    return figure, plotted


def build_time_distribution(frame: pd.DataFrame, chosen_ids: set[str] | None, grain: str) -> go.Figure:
    focus = frame.copy() if chosen_ids is None else frame[frame["row_id"].isin(chosen_ids)].copy()
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


def build_attribute_profile(frame: pd.DataFrame, chosen_ids: set[str] | None) -> go.Figure:
    if chosen_ids is None:
        return empty_figure("Select a candidate group in the projection to compare its attribute profile.")
    if not chosen_ids:
        return empty_figure("Both saved groups are hidden. Show a group or make a new selection.")

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
        dcc.Store(id="saved-groups", storage_type="memory", data=default_saved_groups()),
        dcc.Store(id="hidden-groups", storage_type="memory", data=[]),
        dcc.Download(id="download-groups"),
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
                html.Div([html.Span("1"), html.Strong("Choose a question"), html.Small("Use a recommended view")], className="guide-step active-step"),
                html.Div([html.Span("2"), html.Strong("Circle a pattern"), html.Small("Box or lasso a local region")], className="guide-step"),
                html.Div([html.Span("3"), html.Strong("Save and compare"), html.Small("Confirm it in the linked views")], className="guide-step"),
            ],
            className="workflow-guide",
        ),
        html.Section(
            [
                html.Button(
                    [
                        html.Div([html.Strong("Pollution peaks"), html.Small("Temperature × CO, coloured by NOx")]),
                    ],
                    id="scenario-pollution",
                    n_clicks=0,
                    className="scenario-card selected-scenario",
                ),
                html.Button([html.Strong("Humidity effect"), html.Small("Humidity × NO₂")], id="scenario-humidity", n_clicks=0, className="scenario-card"),
                html.Button([html.Strong("Sensor consistency"), html.Small("Reference × sensor")], id="scenario-sensors", n_clicks=0, className="scenario-card"),
                html.Button([html.Strong("Weather and pollution"), html.Small("Temperature × humidity")], id="scenario-weather", n_clicks=0, className="scenario-card"),
            ],
            className="scenario-grid",
        ),
        html.P(["Current view: ", html.Strong("Pollution peaks", id="scenario-label")], className="scenario-label"),
        html.Details(
            [
                html.Summary("Advanced filters and manual attribute choices"),
                html.Section(
                    [
                        html.Div(
                            [
                                html.Label("Observation level"),
                                dcc.RadioItems(
                                    id="grain",
                                    options=[{"label": " Daily", "value": "daily"}, {"label": " Hourly", "value": "hourly"}],
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
                                dcc.Dropdown(id="months", options=month_options, value=[ALL_MONTHS], multi=True, placeholder="Select months"),
                            ],
                            className="control control-wide",
                        ),
                        html.Div(
                            [
                                html.Label("Hours included"),
                                dcc.RangeSlider(
                                    id="hours", min=0, max=23, step=1, value=[0, 23],
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
                    className="controls",
                ),
                html.Div(
                    [
                        html.Div([html.Label("X axis"), dcc.Dropdown(id="x-attribute", options=attribute_options, value="T", clearable=False)], className="mini-control"),
                        html.Div([html.Label("Y axis"), dcc.Dropdown(id="y-attribute", options=attribute_options, value="CO(GT)", clearable=False)], className="mini-control"),
                        html.Div([html.Label("Colour"), dcc.Dropdown(id="colour", options=attribute_options, value="NOx(GT)", clearable=False)], className="mini-control"),
                    ],
                    className="chart-controls advanced-chart-controls",
                ),
            ],
            className="advanced-panel panel",
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
                dcc.Graph(id="projection", config={"displaylogo": False, "modeBarButtonsToAdd": ["select2d", "lasso2d"]}),
                html.Details(
                    [
                        html.Summary("Select by value range"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(id="helper-x-label"),
                                        dcc.RangeSlider(id="helper-x-range", min=0, max=1, value=[0, 1], marks={0: "0", 1: "1"}, dots=False, tooltip={"placement": "bottom", "always_visible": True}),
                                    ],
                                    className="range-control",
                                ),
                                html.Div(
                                    [
                                        html.Label(id="helper-y-label"),
                                        dcc.RangeSlider(id="helper-y-range", min=0, max=1, value=[0, 1], marks={0: "0", 1: "1"}, dots=False, tooltip={"placement": "bottom", "always_visible": True}),
                                    ],
                                    className="range-control",
                                ),
                                html.Div(
                                    [
                                        html.P(id="range-selection-count", className="range-count"),
                                        html.Button("Preview selection", id="preview-range-selection", n_clicks=0, className="primary-button"),
                                    ],
                                    className="range-action",
                                ),
                            ],
                            className="range-helper-body",
                        ),
                    ],
                    className="range-helper",
                ),
                html.Div(
                    [
                        html.P(
                            "Drag around a dense or visually separated region, then inspect the linked views.",
                            className="chart-note",
                        ),
                        html.Div(
                            [
                                dcc.Input(
                                    id="group-name",
                                    type="text",
                                    placeholder="Group name (optional)",
                                    className="group-name-input",
                                ),
                                html.Button("Save as group", id="save-group", n_clicks=0, className="primary-button"),
                                html.Button("Clear selection", id="clear-selection", n_clicks=0, className="clear-button"),
                            ],
                            className="selection-buttons",
                        ),
                    ],
                    className="chart-actions",
                ),
                html.P(id="selection-guidance", className="selection-guidance"),
                html.Div(
                    [
                        html.Div(id="saved-group-list", className="saved-group-list"),
                        html.Div(
                            [
                                html.Button("Download groups.csv", id="download-groups-button", n_clicks=0, className="clear-button"),
                                html.Button("Remove all groups", id="remove-all-groups", n_clicks=0, className="danger-button"),
                            ],
                            className="group-file-actions",
                        ),
                    ],
                    className="group-manager",
                ),
                html.P(id="group-status", className="group-status"),
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
    Input("clear-selection", "n_clicks"),
    Input("saved-groups", "data"),
    Input("hidden-groups", "data"),
)
def update_views(
    months: list[str],
    hours: list[int],
    grain: str,
    x_attribute: str,
    y_attribute: str,
    colour_attribute: str,
    selected_data: dict | None,
    clear_clicks: int,
    saved_groups: list[dict] | None,
    hidden_groups: list[str] | None,
):
    frame = prepare_data(months, hours, grain)
    manual_ids = set() if ctx.triggered_id == "clear-selection" else selected_ids(selected_data)
    manual_ids &= set(frame["row_id"].astype(str))
    linked_ids: set[str] | None = manual_ids if manual_ids else None

    # When one saved group is hidden, use the remaining visible group as the
    # focus for all linked views. A manual box/lasso selection still takes
    # precedence when present.
    if not manual_ids and hidden_groups:
        visible_groups = [
            group
            for group in with_default_groups(saved_groups)
            if group["name"] not in set(hidden_groups)
        ]
        linked_ids = {
            str(row_id)
            for group in visible_groups
            for row_id in group.get("ids", [])
        }
        linked_ids &= set(frame["row_id"].astype(str))

    scatter_focus_ids = linked_ids or set()

    scatter = build_scatter(
        frame,
        x_attribute,
        y_attribute,
        colour_attribute,
        scatter_focus_ids,
        clear_clicks or 0,
        with_default_groups(saved_groups),
        hidden_groups,
    )
    parallel, complete = build_parallel(frame, linked_ids)
    time_distribution = build_time_distribution(frame, linked_ids, grain)
    attribute_profile = build_attribute_profile(frame, linked_ids)

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
        "All" if linked_ids is None else f"{len(linked_ids):,}",
        f"{len(complete):,}",
        date_text,
    )


@app.callback(
    Output("x-attribute", "value"),
    Output("y-attribute", "value"),
    Output("colour", "value"),
    Output("scenario-label", "children"),
    Output("scenario-pollution", "className"),
    Output("scenario-humidity", "className"),
    Output("scenario-sensors", "className"),
    Output("scenario-weather", "className"),
    Input("scenario-pollution", "n_clicks"),
    Input("scenario-humidity", "n_clicks"),
    Input("scenario-sensors", "n_clicks"),
    Input("scenario-weather", "n_clicks"),
    prevent_initial_call=True,
)
def choose_scenario(*_clicks: int):
    selected_scenario = ctx.triggered_id
    x_attribute, y_attribute, colour_attribute, label = ANALYSIS_SCENARIOS[selected_scenario]
    classes = [
        "scenario-card selected-scenario" if scenario_id == selected_scenario else "scenario-card"
        for scenario_id in ANALYSIS_SCENARIOS
    ]
    return x_attribute, y_attribute, colour_attribute, label, *classes


@app.callback(Output("selection-guidance", "children"), Input("projection", "selectedData"))
def explain_selection(selected_data: dict | None) -> str:
    count = len(selected_ids(selected_data))
    if not count:
        return "Next: use Box Select or Lasso Select to circle one local pattern — not the whole chart."
    return f"{count:,} observations selected. Check the linked views below; save only if they share a meaningful profile."


def slider_bounds(frame: pd.DataFrame, attribute: str) -> tuple[float, float, float]:
    values = frame[attribute].dropna()
    if values.empty:
        return 0.0, 1.0, 0.01
    raw_minimum = float(values.min())
    raw_maximum = float(values.max())
    span = raw_maximum - raw_minimum
    if span >= 1000:
        step = 10.0
    elif span >= 100:
        step = 1.0
    elif span >= 10:
        step = 0.1
    elif span >= 1:
        step = 0.05
    else:
        step = 0.01
    minimum = math.floor(raw_minimum / step) * step
    maximum = math.ceil(raw_maximum / step) * step
    if minimum == maximum:
        maximum = minimum + step
    return minimum, maximum, step


@app.callback(
    Output("helper-x-range", "min"), Output("helper-x-range", "max"),
    Output("helper-x-range", "step"), Output("helper-x-range", "marks"), Output("helper-x-range", "value"),
    Output("helper-y-range", "min"), Output("helper-y-range", "max"),
    Output("helper-y-range", "step"), Output("helper-y-range", "marks"), Output("helper-y-range", "value"),
    Output("helper-x-label", "children"), Output("helper-y-label", "children"),
    Input("months", "value"), Input("hours", "value"), Input("grain", "value"),
    Input("x-attribute", "value"), Input("y-attribute", "value"),
    Input("clear-selection", "n_clicks"),
)
def configure_range_helper(months, hours, grain, x_attribute, y_attribute, _clear_clicks):
    frame = prepare_data(months, hours, grain)
    x_min, x_max, x_step = slider_bounds(frame, x_attribute)
    y_min, y_max, y_step = slider_bounds(frame, y_attribute)
    x_marks = {x_min: f"{x_min:g}", x_max: f"{x_max:g}"}
    y_marks = {y_min: f"{y_min:g}", y_max: f"{y_max:g}"}
    return (
        x_min, x_max, x_step, x_marks, [x_min, x_max],
        y_min, y_max, y_step, y_marks, [y_min, y_max],
        f"X range — {ATTRIBUTE_LABELS[x_attribute]}",
        f"Y range — {ATTRIBUTE_LABELS[y_attribute]}",
    )


@app.callback(
    Output("range-selection-count", "children"),
    Input("helper-x-range", "value"), Input("helper-y-range", "value"),
    Input("months", "value"), Input("hours", "value"), Input("grain", "value"),
    Input("x-attribute", "value"), Input("y-attribute", "value"), Input("colour", "value"),
)
def describe_range_selection(x_range, y_range, months, hours, grain, x_attribute, y_attribute, colour_attribute):
    frame = prepare_data(months, hours, grain).dropna(subset=[x_attribute, y_attribute, colour_attribute])
    count = int(
        (frame[x_attribute].between(x_range[0], x_range[1], inclusive="both")
         & frame[y_attribute].between(y_range[0], y_range[1], inclusive="both")).sum()
    )
    if count == 0:
        assessment = "No observations — widen a range."
    elif count > 100:
        assessment = "Broad selection — narrow one or both ranges."
    elif count < 10:
        assessment = "Very small selection — widen it slightly."
    else:
        assessment = "Useful size for inspection."
    return f"{count:,} observations in this rectangle. {assessment}"


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
    Input("preview-range-selection", "n_clicks"),
    State("helper-x-range", "value"), State("helper-y-range", "value"),
    State("months", "value"), State("hours", "value"), State("grain", "value"),
    State("x-attribute", "value"), State("y-attribute", "value"), State("colour", "value"),
    prevent_initial_call=True,
)
def set_projection_selection(
    _clear_clicks, _preview_clicks, x_range, y_range, months, hours, grain,
    x_attribute, y_attribute, colour_attribute,
):
    if ctx.triggered_id == "clear-selection":
        return None
    frame = prepare_data(months, hours, grain).dropna(subset=[x_attribute, y_attribute, colour_attribute])
    chosen = frame[
        frame[x_attribute].between(x_range[0], x_range[1], inclusive="both")
        & frame[y_attribute].between(y_range[0], y_range[1], inclusive="both")
    ]
    return {
        "points": [
            {"x": row[x_attribute], "y": row[y_attribute], "customdata": [str(row["row_id"]), str(row["DateTime"])]}
            for _, row in chosen.iterrows()
        ]
    }


@app.callback(
    Output("saved-groups", "data"),
    Output("group-name", "value"),
    Output("group-status", "children"),
    Input("save-group", "n_clicks"),
    Input("remove-all-groups", "n_clicks"),
    Input({"type": "remove-group", "index": ALL}, "n_clicks"),
    State("projection", "selectedData"),
    State("group-name", "value"),
    State("saved-groups", "data"),
    prevent_initial_call=True,
)
def manage_groups(
    _save_clicks: int,
    _remove_all_clicks: int,
    _remove_clicks: list[int],
    selected_data: dict | None,
    requested_name: str | None,
    saved_groups: list[dict] | None,
):
    groups = with_default_groups(saved_groups)
    trigger = ctx.triggered_id

    if trigger == "remove-all-groups":
        return default_saved_groups(), "", "Additional groups were removed; the two validated groups were kept."

    if isinstance(trigger, dict) and trigger.get("type") == "remove-group":
        index = int(trigger["index"])
        if 0 <= index < len(groups):
            if groups[index]["name"] in {"High pollution", "Hot low-pollution"}:
                return groups, no_update, "The two validated groups are fixed and cannot be removed."
            removed = groups.pop(index)
            return groups, no_update, f"{removed['name']} was removed."
        return no_update, no_update, no_update

    ids = sorted(selected_ids(selected_data))
    if not ids:
        return no_update, no_update, "Select points in the projection before saving a group."

    existing_ids = {row_id for group in groups for row_id in group.get("ids", [])}
    ids = [row_id for row_id in ids if row_id not in existing_ids]
    if not ids:
        return no_update, no_update, "Every selected observation is already in a saved group."

    next_number = 1
    existing_names = {str(group["name"]).casefold() for group in groups}
    while f"group {next_number}" in existing_names:
        next_number += 1
    default_name = f"Group {next_number}"
    name = (requested_name or "").strip() or default_name
    if name.casefold() in existing_names:
        return no_update, no_update, f'A group named "{name}" already exists.'

    colour = GROUP_COLOURS[len(groups) % len(GROUP_COLOURS)]
    groups.append({"name": name, "colour": colour, "ids": ids})
    return groups, "", f"Saved {len(ids):,} observations as {name}. Clear the selection to see its group colour."


@app.callback(
    Output("saved-group-list", "children"),
    Input("saved-groups", "data"),
)
def show_saved_groups(saved_groups: list[dict] | None):
    groups = with_default_groups(saved_groups)
    if not groups:
        return html.P("No groups saved yet.", className="empty-groups")
    return [
        html.Div(
            [
                html.Span(className="group-swatch", style={"backgroundColor": group["colour"]}),
                html.Strong(group["name"]),
                html.Span(f"{len(group.get('ids', [])):,} observations", className="group-count"),
                html.Button(
                    "Show / Hide",
                    id={"type": "toggle-group", "index": group["name"]},
                    n_clicks=0,
                    className="visibility-group-button",
                    title="Temporarily show or hide this group in the chart",
                ),
                (
                    html.Span("Fixed", className="group-count")
                    if group["name"] in {"High pollution", "Hot low-pollution"}
                    else html.Button(
                        "Remove",
                        id={"type": "remove-group", "index": index},
                        n_clicks=0,
                        className="remove-group-button",
                    )
                ),
            ],
            className="saved-group-item",
        )
        for index, group in enumerate(groups)
    ]


@app.callback(
    Output("hidden-groups", "data"),
    Input({"type": "toggle-group", "index": ALL}, "n_clicks"),
    State("hidden-groups", "data"),
    prevent_initial_call=True,
)
def toggle_group_visibility(_clicks: list[int], hidden_groups: list[str] | None):
    if not any(_clicks or []):
        return no_update
    trigger = ctx.triggered_id
    if not isinstance(trigger, dict):
        return no_update
    name = str(trigger.get("index", ""))
    hidden = set(hidden_groups or [])
    if name in hidden:
        hidden.remove(name)
    else:
        hidden.add(name)
    return sorted(hidden)


@app.callback(
    Output("download-groups", "data"),
    Input("download-groups-button", "n_clicks"),
    State("saved-groups", "data"),
    prevent_initial_call=True,
)
def download_saved_groups(_n_clicks: int, saved_groups: list[dict] | None):
    groups = with_default_groups(saved_groups)
    if not groups:
        return no_update

    hourly_dates = df.set_index("row_id")["DateTime"].astype(str).to_dict()
    records = []
    for group in groups:
        for row_id in group.get("ids", []):
            if row_id in hourly_dates:
                date_time = hourly_dates[row_id]
                date = date_time[:10]
            else:
                date = row_id[:10]
                date_time = row_id[:10]
            records.append(
                {
                    "date": date,
                    "group": group["name"],
                    "observation_id": row_id,
                    "datetime": date_time,
                }
            )
    export = pd.DataFrame(records).sort_values(["group", "datetime"])
    return dcc.send_data_frame(export.to_csv, "groups.csv", index=False)


if __name__ == "__main__":
    app.run(
        debug=os.getenv("DASH_DEBUG", "false").lower() == "true",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8050")),
    )
