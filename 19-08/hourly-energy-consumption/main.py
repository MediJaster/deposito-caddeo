import os
from typing import Literal

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

    raw_df = pd.read_csv(
        os.path.join(DATASET_FOLDER, filename),
        parse_dates=["Datetime"],
        index_col="Datetime",
    )

    renamed_df = raw_df.rename(columns={f"{company}_MW": "Consumption"})

    return renamed_df


def calculate_high_or_low_consumption(
    df: pd.DataFrame, reference: Literal["daily", "weekly", "total"] = "total"
) -> pd.DataFrame:
    df = df.copy()

    match reference:
        case "daily":
            df["High Consumption"] = (
                df["Consumption"] > df["Consumption"].rolling(window=24).mean()
            )
        case "weekly":
            df["High Consumption"] = (
                df["Consumption"] > df["Consumption"].rolling(window=(24 * 7)).mean()
            )
        case "total":
            df["High Consumption"] = df["Consumption"] > df["Consumption"].mean()
        case _:
            raise ValueError(
                f"Invalid reference '{reference}'. Choose from 'daily', 'weekly', or 'total'."
            )

    return df


def main() -> None:
    print("Available companies:\n", "\n".join(COMPANY_LIST))

    selected_company = (
        input(f"Select a company from the list above (default: {COMPANY_LIST[0]}): ")
        or COMPANY_LIST[0]
    )

    hourly_consumption_df = get_dataset(company=selected_company)

    processed_df = calculate_high_or_low_consumption(
        df=hourly_consumption_df,
        reference="daily",  # Change to "weekly" or "total" as needed
    )

    print(processed_df.head())

    return


if __name__ == "__main__":
    main()
