import os

import pandas as pd

DATASET_FOLDER = os.path.join(os.path.dirname(__file__), "datasets")
COMPANY_LIST = [
    entry.name.split("_")[0]
    for entry in os.scandir(DATASET_FOLDER)
    if entry.is_file() and entry.name.endswith("_hourly.csv")
]


def get_dataset(company: str) -> pd.DataFrame:
    """Get the dataset for a specific company.

    Parameters
    ----------
    company : str
        The name of the company for which to retrieve the dataset.

    Returns
    -------
    pd.DataFrame
        The dataset containing hourly energy consumption data for the specified company.
    """

    if company not in COMPANY_LIST:
        raise ValueError(f"No dataset available for Company '{company}'.")

    filename = f"{company}_hourly.csv"

    return pd.read_csv(
        os.path.join(DATASET_FOLDER, filename),
        parse_dates=["Datetime"],
        index_col="Datetime",
    )


def main() -> None:
    hourly_consumption_df = get_dataset("COMED")

    print(hourly_consumption_df.head())

    return


if __name__ == "__main__":
    main()
