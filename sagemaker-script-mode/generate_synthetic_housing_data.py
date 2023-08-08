from random import choice
import numpy as np
import pandas as pd


NUM_HOUSES_PER_LOCATION = 1000
LOCATIONS = [
    "NewYork_NY",
    "LosAngeles_CA",
    "Chicago_IL",
    "Houston_TX",
    "Dallas_TX",
    "Phoenix_AZ",
    "Philadelphia_PA",
    "SanAntonio_TX",
    "SanDiego_CA",
    "SanFrancisco_CA",
]
MAX_YEAR = 2019


def generate_price(house):
    """Generate price based on features of the house"""

    if house["FRONT_PORCH"] == "y":
        garage = 1
    else:
        garage = 0

    if house["FRONT_PORCH"] == "y":
        front_porch = 1
    else:
        front_porch = 0

    price = int(
        150 * house["SQUARE_FEET"]
        + 10000 * house["NUM_BEDROOMS"]
        + 15000 * house["NUM_BATHROOMS"]
        + 15000 * house["LOT_ACRES"]
        + 10000 * garage
        + 10000 * front_porch
        + 15000 * house["GARAGE_SPACES"]
        - 5000 * (MAX_YEAR - house["YEAR_BUILT"])
    )
    return price


def generate_yes_no():
    """Generate values (y/n) for categorical features"""
    answer = choice([1, 0])
    return answer


def generate_random_house():
    """Generate a row of data (single house information)"""
    house = {
        "SQUARE_FEET": np.random.normal(3000, 750),
        "NUM_BEDROOMS": np.random.randint(2, 7),
        "NUM_BATHROOMS": np.random.randint(2, 7) / 2,
        "LOT_ACRES": round(np.random.normal(1.0, 0.25), 2),
        "GARAGE_SPACES": np.random.randint(0, 4),
        "YEAR_BUILT": min(MAX_YEAR, int(np.random.normal(1995, 10))),
        "FRONT_PORCH": generate_yes_no(),
        "DECK": generate_yes_no(),
    }

    price = generate_price(house)

    return [
        house["YEAR_BUILT"],
        house["SQUARE_FEET"],
        house["NUM_BEDROOMS"],
        house["NUM_BATHROOMS"],
        house["LOT_ACRES"],
        house["GARAGE_SPACES"],
        house["FRONT_PORCH"],
        house["DECK"],
        price,
    ]


def generate_houses(num_houses):
    """Generate housing dataset"""
    house_list = []

    for _ in range(num_houses):
        house_list.append(generate_random_house())

    df = pd.DataFrame(
        house_list,
        columns=[
            "YEAR_BUILT",
            "SQUARE_FEET",
            "NUM_BEDROOMS",
            "NUM_BATHROOMS",
            "LOT_ACRES",
            "GARAGE_SPACES",
            "FRONT_PORCH",
            "DECK",
            "PRICE",
        ],
    )
    return df
