from importlib.resources import files

import pandas as pd


def load_example_data():
    """Load the bundled example dataset as a pandas DataFrame."""
    resource = files("hiermeta").joinpath("data", "example_df.csv")
    with resource.open("r", encoding="utf-8") as csv_file:
        return pd.read_csv(csv_file)
