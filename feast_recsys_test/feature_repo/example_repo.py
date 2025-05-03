from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource, Project
from feast.feature_logging import LoggingConfig
from feast.infra.offline_stores.file_source import FileLoggingDestination
from feast.types import Float32, Float64, Int32, String


project = Project(
    name="feast_recsys_test",
    description="Personal learning project to use Feast + Redis for a recommender system.",
)
user = Entity(name="user", join_keys=["user_id"])
user_stats_source = FileSource(
    name="user_stats_source",
    path="data/user.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)
user_stats_fv = FeatureView(
    name="user_stats",
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name="user_age", dtype=Int32),
        Field(name="user_account_age_days", dtype=Int32),
        Field(name="user_avg_rating_given", dtype=Float32),
        Field(name="user_tier", dtype=Int32),
        Field(name="country", dtype=String),
    ],
    online=True,
    source=user_stats_source,
    tags={},
)
user_stats_v1 = FeatureService(
    name="user_stats_v1",
    features=[
        user_stats_fv[["user_age", "user_account_age_days", "user_avg_rating_given", "user_tier", "country"]],
    ],
    logging_config=LoggingConfig(
        destination=FileLoggingDestination(path="data")
    )
)
