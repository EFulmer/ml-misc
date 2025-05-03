import datetime

import numpy as np
import pandas as pd

COUNTRIES = ["US", "CA", "UK", "CN", "MX", "BR", "DE", "CH", "PL", "JP"]
USER_COUNT = 10
ITEM_COUNT = 15
CATEGORIES = ["electronics", "books", "clothing", "kitchen", "sports", "music"]
RELEASE_YEARS = np.arange(2000, 2025)

def main():
    # TODO: seed RNG.
    user_data = pd.DataFrame({
        "user_id": range(USER_COUNT),
        "user_age": np.random.randint(low=18, high=75, size=USER_COUNT),
        "user_account_age_days": np.random.randint(low=0, high=365*5, size=USER_COUNT),
        "user_avg_rating_given": np.random.uniform(1.0, 5.0, size=USER_COUNT),
        "user_tier": np.random.randint(0, 3, size=USER_COUNT),
        "country": np.random.choice(COUNTRIES, size=USER_COUNT),
        "event_timestamp": pd.Timestamp(datetime.datetime.utcnow()),
        "created": pd.Timestamp(datetime.date.today()),
    })
    user_data.to_parquet("data/user.parquet")
    item_data = pd.DataFrame({
        "item_id": range(ITEM_COUNT),
        "item_category": np.random.choice(CATEGORIES, size=ITEM_COUNT),
        "item_price": np.round(np.random.uniform(5.0, 500.0, size=ITEM_COUNT), 2),
        "item_avg_rating": np.round(np.random.uniform(1.0, 5.0, size=ITEM_COUNT), 2),
        "item_popularity_30d": np.random.randint(0, 10000, size=ITEM_COUNT),
        "item_release_year": np.random.choice(RELEASE_YEARS, size=ITEM_COUNT),
        "event_timestamp": pd.Timestamp(datetime.datetime.utcnow()),
        "created": pd.Timestamp(datetime.date.today()),
    })
    item_data.to_parquet("data/item.parquet")


if __name__ == "__main__":
    main()
