# COMP5048 Assignment 2 — A workspace

This workspace implements A's initial responsibilities: reproducible data cleaning, the Dash application shell, and the Task 1 parallel-coordinates view.

## Data policy

- The source file is never overwritten.
- Every `-200` value in the selected numeric attributes becomes a missing value.
- Missing values are not imputed.
- The same selected attributes are defined once in `data_pipeline.py` and reused by the application.
- `Hour`, `Weekday`, and `Month` are derived from `DateTime` for filtering and colouring; they do not replace the selected attributes.

## Run

```powershell
& 'D:\anaconda3\envs\comp5310\python.exe' data_pipeline.py
& 'D:\anaconda3\envs\comp5310\python.exe' -m pip install -r requirements.txt
& 'D:\anaconda3\envs\comp5310\python.exe' app.py
```

Open the local address printed by Dash. Brush ranges directly on parallel-coordinate axes to identify candidate visual groups. The application does not run a clustering algorithm.

## Public deployment

The repository includes `render.yaml`. Push the project to GitHub, create a Render Web Service from that repository, and Render can use the included build and start commands. The deployed service receives a public `onrender.com` address that can be shared with the group.

## Current scope

The first working view provides month and hour filters, a colour-variable control, missing-data coverage text, and parallel-axis brushing. Linked T2/T3 views will be connected to the same cleaned dataset later.
