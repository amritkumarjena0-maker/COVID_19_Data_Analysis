
import os
import urllib.request
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ==========================================================
#             COVID-19 DATA ANALYSIS SYSTEM
# ==========================================================

print("=" * 65)
print("                 COVID-19 DATA ANALYSIS")
print("=" * 65)

# Project directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, "owid-covid-data.csv")

# Historical COVID-19 dataset
DATA_URL = (
    "https://covid.ourworldindata.org/data/"
    "owid-covid-data.csv"
)

# Required columns
REQUIRED_COLUMNS = [
    "iso_code",
    "continent",
    "location",
    "date",
    "total_cases",
    "new_cases",
    "total_deaths",
    "new_deaths",
    "population"
]


# ==========================================================
# 1. DOWNLOAD DATASET
# ==========================================================

def download_dataset():

    if os.path.exists(DATA_FILE):
        print("\nDataset already exists.")
        return

    print("\nDownloading COVID-19 dataset...")

    try:
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        print("Dataset downloaded successfully.")

    except Exception as error:
        print("Dataset download failed:", error)
        raise SystemExit(
            "Please download the CSV dataset manually."
        )


# ==========================================================
# 2. LOAD AND CLEAN DATA
# ==========================================================

def load_data():

    print("\nLoading COVID-19 dataset...")

    try:
        df = pd.read_csv(
            DATA_FILE,
            usecols=REQUIRED_COLUMNS,
            low_memory=False
        )

        df["date"] = pd.to_datetime(
            df["date"], errors="coerce"
        )

        numeric_columns = [
            "total_cases",
            "new_cases",
            "total_deaths",
            "new_deaths",
            "population"
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

        df = df.dropna(
            subset=["location", "date"]
        )

        df = df.sort_values(
            ["location", "date"]
        )

        print("Dataset loaded successfully.")

        return df

    except Exception as error:
        print("Data loading error:", error)
        raise SystemExit(1)


# ==========================================================
# 3. DATA CLEANING AND PREPARATION
# ==========================================================

def prepare_data(df):

    print("\nPreparing COVID-19 data...")

    # Select country-level observations
    countries = df[
        df["continent"].notna()
    ].copy()

    # Fill missing numerical values
    metrics = [
        "total_cases",
        "new_cases",
        "total_deaths",
        "new_deaths"
    ]

    countries[metrics] = countries[metrics].fillna(0)

    # Remove negative corrections from daily counts
    # for this analysis only
    countries["new_cases"] = (
        countries["new_cases"].clip(lower=0)
    )

    countries["new_deaths"] = (
        countries["new_deaths"].clip(lower=0)
    )

    # Calculate death percentage
    countries["death_rate"] = np.where(
        countries["total_cases"] > 0,
        (
            countries["total_deaths"]
            / countries["total_cases"]
        ) * 100,
        0
    )

    return countries


# ==========================================================
# 4. GLOBAL COVID-19 SUMMARY
# ==========================================================

def global_analysis(df):

    latest = (
        df.sort_values("date")
        .groupby("location")
        .tail(1)
    )

    total_cases = latest["total_cases"].sum()
    total_deaths = latest["total_deaths"].sum()

    total_countries = latest["location"].nunique()

    death_rate = (
        (total_deaths / total_cases) * 100
        if total_cases > 0 else 0
    )

    print("\nGLOBAL COVID-19 SUMMARY")
    print("-" * 65)

    print(f"Countries/regions analyzed : {total_countries}")
    print(f"Total confirmed cases      : {total_cases:,.0f}")
    print(f"Total confirmed deaths     : {total_deaths:,.0f}")
    print(f"Calculated death rate      : {death_rate:.2f}%")

    summary = pd.DataFrame({
        "Metric": [
            "Countries/Regions",
            "Total Cases",
            "Total Deaths",
            "Death Rate (%)"
        ],
        "Value": [
            total_countries,
            total_cases,
            total_deaths,
            death_rate
        ]
    })

    summary.to_csv(
        os.path.join(OUTPUT_DIR, "covid_summary.csv"),
        index=False
    )

    return latest


# ==========================================================
# 5. TOP 10 COUNTRIES ANALYSIS
# ==========================================================

def country_analysis(latest):

    print("\nTOP 10 COUNTRIES BY COVID-19 CASES")
    print("-" * 65)

    top_countries = (
        latest.sort_values(
            "total_cases", ascending=False
        ).head(10)
    )

    result = top_countries[
        [
            "location",
            "total_cases",
            "total_deaths",
            "death_rate"
        ]
    ].copy()

    print(
        result.to_string(
            index=False,
            formatters={
                "total_cases": "{:,.0f}".format,
                "total_deaths": "{:,.0f}".format,
                "death_rate": "{:.2f}%".format
            }
        )
    )

    result.to_csv(
        os.path.join(OUTPUT_DIR, "country_analysis.csv"),
        index=False
    )

    return top_countries


# ==========================================================
# 6. INDIA COVID-19 ANALYSIS
# ==========================================================

def india_analysis(df):

    india = df[
        df["location"] == "India"
    ].copy()

    print("\nINDIA COVID-19 ANALYSIS")
    print("-" * 65)

    if india.empty:
        print("India data is unavailable.")
        return

    latest = india.iloc[-1]

    print(f"Latest available date : {latest['date'].date()}")
    print(f"Total cases           : {latest['total_cases']:,.0f}")
    print(f"Total deaths          : {latest['total_deaths']:,.0f}")
    print(f"New cases             : {latest['new_cases']:,.0f}")
    print(f"New deaths            : {latest['new_deaths']:,.0f}")

    plt.figure(figsize=(12, 6))

    plt.plot(
        india["date"],
        india["total_cases"],
        label="Total Cases",
        linewidth=2
    )

    plt.title("India - Total COVID-19 Cases")
    plt.xlabel("Date")
    plt.ylabel("Confirmed Cases")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "india_covid_analysis.png"
        ),
        dpi=300
    )

    plt.close()


# ==========================================================
# 7. TOP 10 COUNTRIES GRAPH
# ==========================================================

def plot_top_countries(top_countries):

    plt.figure(figsize=(12, 7))

    sns.barplot(
        data=top_countries,
        x="total_cases",
        y="location"
    )

    plt.title("Top 10 Countries - COVID-19 Cases")
    plt.xlabel("Total Confirmed Cases")
    plt.ylabel("Country")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "top_10_countries.png"
        ),
        dpi=300
    )

    plt.close()


# ==========================================================
# 8. GLOBAL DAILY CASES GRAPH
# ==========================================================

def daily_cases_analysis(df):

    # Use World aggregate when available
    world = df[
        df["location"] == "World"
    ].copy()

    if world.empty:
        world = (
            df.groupby("date", as_index=False)
            [["new_cases"]].sum()
        )

    world = world.sort_values("date")

    world["new_cases_7day_average"] = (
        world["new_cases"]
        .rolling(window=7, min_periods=1)
        .mean()
    )

    plt.figure(figsize=(13, 6))

    plt.plot(
        world["date"],
        world["new_cases"],
        label="Daily New Cases",
        alpha=0.4
    )

    plt.plot(
        world["date"],
        world["new_cases_7day_average"],
        label="7-Day Average",
        linewidth=2
    )

    plt.title("Global Daily COVID-19 Cases")
    plt.xlabel("Date")
    plt.ylabel("New Cases")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "daily_cases_analysis.png"
        ),
        dpi=300
    )

    plt.close()


# ==========================================================
# 9. GLOBAL DAILY DEATHS GRAPH
# ==========================================================

def daily_deaths_analysis(df):

    world = df[
        df["location"] == "World"
    ].copy()

    if world.empty:
        world = (
            df.groupby("date", as_index=False)
            [["new_deaths"]].sum()
        )

    world = world.sort_values("date")

    world["new_deaths_7day_average"] = (
        world["new_deaths"]
        .rolling(window=7, min_periods=1)
        .mean()
    )

    plt.figure(figsize=(13, 6))

    plt.plot(
        world["date"],
        world["new_deaths"],
        label="Daily Deaths",
        alpha=0.4
    )

    plt.plot(
        world["date"],
        world["new_deaths_7day_average"],
        label="7-Day Average",
        linewidth=2
    )

    plt.title("Global Daily COVID-19 Deaths")
    plt.xlabel("Date")
    plt.ylabel("New Deaths")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "daily_deaths_analysis.png"
        ),
        dpi=300
    )

    plt.close()


# ==========================================================
# 10. MAIN PROGRAM
# ==========================================================

def main():

    download_dataset()

    df = load_data()

    countries = prepare_data(df)

    latest = global_analysis(countries)

    top_countries = country_analysis(latest)

    india_analysis(countries)

    plot_top_countries(top_countries)

    daily_cases_analysis(df)

    daily_deaths_analysis(df)

    print("\n" + "=" * 65)
    print("          COVID-19 ANALYSIS COMPLETED")
    print("=" * 65)

    print("\nGenerated reports:")
    print("1. covid_summary.csv")
    print("2. country_analysis.csv")
    print("3. top_10_countries.png")
    print("4. daily_cases_analysis.png")
    print("5. daily_deaths_analysis.png")
    print("6. india_covid_analysis.png")

    print("\nOutput folder:", OUTPUT_DIR)

    print("\nThank you for using COVID-19 Data Analysis!")


if __name__ == "__main__":
    main()
