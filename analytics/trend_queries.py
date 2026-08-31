"""
analytics/trend_queries.py

MongoDB aggregation-pipeline equivalents of the earlier Snowflake
trend_queries.sql. Each function returns a list of dicts (already
JSON-serializable, aside from datetime objects which FastAPI's
jsonable_encoder handles automatically).

NOTE ON FIELD NAMES:
These assume the shared `cases` collection schema uses:
  - received_at      (datetime the email/case was received)
  - classification    ("phishing" | "bec" | "legitimate" | "suspicious")
  - country            (sender/reported country, if enriched upstream)
  - campaign_id        (grouping id if campaign detection is done elsewhere)
  - infrastructure_type (e.g. "compromised_host", "bulletproof_hosting", etc.)
  - risk_score          (0-1 float)
If your team's actual case schema uses different field names, update the
$group/$match keys below to match — the pipeline shapes stay the same.
"""

from datetime import datetime, timedelta

from analytics.mongo_analytics import run_aggregation, cached_query


@cached_query(ttl_seconds=120)
def phishing_bec_volume_by_country_weekly(months_back=6):
    """
    Weekly phishing/BEC volume broken down by country, for the last N months.
    Equivalent to trend_queries.sql query #1.
    """
    since = datetime.utcnow() - timedelta(days=30 * months_back)

    pipeline = [
        {
            "$match": {
                "received_at": {"$gte": since},
                "classification": {"$in": ["phishing", "bec"]},
            }
        },
        {
            "$group": {
                "_id": {
                    "country": "$country",
                    "week": {"$dateTrunc": {"date": "$received_at", "unit": "week"}},
                    "classification": "$classification",
                },
                "case_count": {"$sum": 1},
            }
        },
        {
            "$project": {
                "_id": 0,
                "country": "$_id.country",
                "week": "$_id.week",
                "classification": "$_id.classification",
                "case_count": 1,
            }
        },
        {"$sort": {"week": 1, "country": 1}},
    ]
    return run_aggregation(pipeline)


@cached_query(ttl_seconds=120)
def fastest_growing_campaigns(days_this_period=45, days_previous_period=45):
    """
    Compares campaign case counts between this period and the prior period
    of equal length, returns campaigns sorted by growth rate descending.
    Equivalent to trend_queries.sql query #2 (CTE-based growth calc).
    Requires a `campaign_id` field to already exist on each case document.
    """
    now = datetime.utcnow()
    current_start = now - timedelta(days=days_this_period)
    previous_start = current_start - timedelta(days=days_previous_period)

    pipeline = [
        {
            "$match": {
                "received_at": {"$gte": previous_start},
                "campaign_id": {"$ne": None},
            }
        },
        {
            "$group": {
                "_id": "$campaign_id",
                "current_count": {
                    "$sum": {
                        "$cond": [{"$gte": ["$received_at", current_start]}, 1, 0]
                    }
                },
                "previous_count": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$gte": ["$received_at", previous_start]},
                                    {"$lt": ["$received_at", current_start]},
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "campaign_id": "$_id",
                "current_count": 1,
                "previous_count": 1,
                "growth_rate": {
                    "$cond": [
                        {"$eq": ["$previous_count", 0]},
                        None,
                        {
                            "$divide": [
                                {"$subtract": ["$current_count", "$previous_count"]},
                                "$previous_count",
                            ]
                        },
                    ]
                },
            }
        },
        {"$match": {"current_count": {"$gt": 0}}},
        {"$sort": {"growth_rate": -1}},
        {"$limit": 20},
    ]
    return run_aggregation(pipeline)


@cached_query(ttl_seconds=120)
def high_risk_infrastructure_breakdown(risk_threshold=0.7):
    """
    Breaks down high-risk cases (risk_score >= threshold) by infrastructure_type.
    Equivalent to trend_queries.sql query #3.
    """
    pipeline = [
        {"$match": {"risk_score": {"$gte": risk_threshold}}},
        {
            "$group": {
                "_id": "$infrastructure_type",
                "case_count": {"$sum": 1},
                "avg_risk_score": {"$avg": "$risk_score"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "infrastructure_type": "$_id",
                "case_count": 1,
                "avg_risk_score": {"$round": ["$avg_risk_score", 3]},
            }
        },
        {"$sort": {"case_count": -1}},
    ]
    return run_aggregation(pipeline)