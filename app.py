from __future__ import annotations

from pathlib import Path
import os

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html

from data_pipeline import SELECTED_ATTRIBUTES


DATA_PATH = Path(__file__).parent / "data" / "air_quality_clean.csv"
NUMERIC_ATTRIBUTES = SELECTED_ATTRIBUTES[1:]


def load_data() -> pd.DataFrame:
    frame = pd.read_csv(DATA_PATH, parse_dates=["DateTime"])
    return frame


df = load_data()
month_options = [{"label": value, "value": value} for value in sorted(df["Month"].unique())]

app = Dash(__name__)
server = app.server
app.title = "Air Quality Visual Analytics"
app.layout = html.Main(
    [
        html.H1("Air Quality Visual Analytics"),
        html.P(
            "Task 1 workspace: visually inspect groups using the same selected attributes. "
            "No clustering algorithm is used."
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Months"),
                        dcc.Dropdown(
                            id="months",
                            options=month_options,
                            value=[option["value"] for option in month_options],
                            multi=True,
                        ),
                    ],
                    className="control control-wide",
                ),
                html.Div(
                    [
                        html.Label("Hours"),
                        dcc.RangeSlider(id="hours", min=0, max=23, step=1, value=[0, 23]),
                    ],
                    className="control",
                ),
                html.Div(
                    [
                        html.Label("Colour by"),
                        dcc.Dropdown(
                            id="colour",
                            options=[{"label": name, "value": name} for name in NUMERIC_ATTRIBUTES],
                            value="CO(GT)",
                            clearable=False,
                        ),
                    ],
                    className="control",
                ),
            ],
            className="controls",
        ),
        html.Div(id="coverage", className="coverage"),
        dcc.Graph(id="parallel-coordinates", config={"displaylogo": False}),
        html.P(
            "Brush ranges directly on the axes. Treat visible bundles, separations and transitions "
            "as candidate evidence, then verify them in linked views before reporting a group.",
            className="note",
        ),
    ]
)


@app.callback(
    Output("parallel-coordinates", "figure"),
    Output("coverage", "children"),
    Input("months", "value"),
    Input("hours", "value"),
    Input("colour", "value"),
)
def update_parallel_coordinates(months: list[str], hours: list[int], colour: str):
    selected_months = months or []
    filtered = df[
        df["Month"].isin(selected_months)
        & df["Hour"].between(hours[0], hours[1], inclusive="both")
    ].copy()

    # Plotly parallel coordinates requires complete rows for the displayed dimensions.
    # Rows are deleted only for this view; the cleaned dataset retains every observation.
    plotted = filtered.dropna(subset=NUMERIC_ATTRIBUTES)
    figure = px.parallel_coordinates(
        plotted,
        dimensions=NUMERIC_ATTRIBUTES,
        color=colour,
        color_continuous_scale=px.colors.diverging.Tealrose,
        title="Task 1 — parallel coordinates for visual grouping",
    )
    figure.update_layout(margin={"l": 55, "r": 55, "t": 75, "b": 40}, height=650)

    coverage = (
        f"Filtered observations: {len(filtered):,}. "
        f"Shown with complete values across all selected numeric attributes: {len(plotted):,}. "
        "Missing observations were not imputed."
    )
    return figure, coverage


if __name__ == "__main__":
    app.run(
        debug=os.getenv("DASH_DEBUG", "false").lower() == "true",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8050")),
    )
