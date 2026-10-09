# Air Quality Visual Analytics

An interactive Dash application for exploring the UCI Air Quality dataset with parallel coordinates, time filters, and selectable colour encoding.

## Requirements

- Python 3.11 or later
- Dash
- Plotly
- Pandas
- Gunicorn for deployment

Install the Python packages with:

```bash
python -m pip install -r requirements.txt
```

## Run locally

The cleaned dataset is included in `data/`. Start the application with:

```bash
python app.py
```

Then open `http://127.0.0.1:8050/` in a browser.

To rebuild the cleaned dataset from `AirQualityUCI.csv`, place the source file in the project directory and run:

```bash
python data_pipeline.py
```

Values coded as `-200` are treated as missing values and are not imputed.
