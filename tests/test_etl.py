"""
tests/test_etl.py
──────────────────
Unit tests for ETL pipeline modules.

Run:
    pytest tests/ -v
    pytest tests/test_etl.py -v --tb=short
"""

import os
import sys
import sqlite3

import numpy as np
import pandas as pd
import pytest

# Make sure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.etl.transform import (
    _clean_orders,
    _clean_order_items,
    _clean_payments,
    _clean_reviews,
    _build_fact_orders,
    _build_weekly_agg,
    _build_seller_scorecard,
)
from src.utils.config_loader import load_config


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_orders():
    return pd.DataFrame({
        "order_id":                      ["o1","o2","o3","o1"],  # o1 duplicate
        "customer_id":                   ["c1","c2","c3","c1"],
        "order_status":                  ["delivered","delivered","canceled","delivered"],
        "order_purchase_timestamp":      ["2017-06-01 10:00:00","2017-07-15 14:00:00","2018-01-01 09:00:00","2017-06-01 10:00:00"],
        "order_approved_at":             ["2017-06-01 10:30:00","2017-07-15 14:30:00",None,None],
        "order_delivered_carrier_date":  ["2017-06-02 00:00:00","2017-07-17 00:00:00",None,None],
        "order_delivered_customer_date": ["2017-06-10 00:00:00","2017-07-30 00:00:00",None,None],
        "order_estimated_delivery_date": ["2017-06-12 00:00:00","2017-07-25 00:00:00","2018-01-15 00:00:00",None],
    })


@pytest.fixture
def sample_items():
    return pd.DataFrame({
        "order_id":           ["o1","o1","o2"],
        "order_item_id":      [1, 2, 1],
        "product_id":         ["p1","p2","p1"],
        "seller_id":          ["s1","s1","s2"],
        "shipping_limit_date":["2017-06-03","2017-06-03","2017-07-18"],
        "price":              [100.0, 50.0, 200.0],
        "freight_value":      [10.0,  5.0,  20.0],
    })


@pytest.fixture
def sample_payments():
    return pd.DataFrame({
        "order_id":             ["o1","o2","o3_garbage"],
        "payment_sequential":   [1, 1, 1],
        "payment_type":         ["credit_card","boleto","credit_card"],
        "payment_installments": [3, 1, 1],
        "payment_value":        [165.0, 220.0, 0.0],   # o3_garbage has 0 → should be dropped
    })


@pytest.fixture
def sample_reviews():
    return pd.DataFrame({
        "review_id":               ["r1","r2","r_bad"],
        "order_id":                ["o1","o2","o3"],
        "review_score":            [5, 3, None],        # None should be dropped
        "review_comment_title":    ["","",""],
        "review_comment_message":  ["Good","Ok",""],
        "review_creation_date":    ["2017-06-15","2017-08-01","2018-01-10"],
        "review_answer_timestamp": ["2017-06-16","2017-08-02","2018-01-11"],
    })


# ── Tests: _clean_orders ──────────────────────────────────────────────────────

class TestCleanOrders:

    def test_deduplication(self, sample_orders):
        result = _clean_orders(sample_orders.copy())
        assert result["order_id"].nunique() == len(result), "Duplicate order_ids remain"
        assert len(result) == 3

    def test_date_parsing(self, sample_orders):
        result = _clean_orders(sample_orders.copy())
        assert pd.api.types.is_datetime64_any_dtype(result["order_purchase_timestamp"])

    def test_delivery_delay_calculation(self, sample_orders):
        result = _clean_orders(sample_orders.copy())
        # o1: delivered 2017-06-10, estimated 2017-06-12 → delay = -2 (early)
        o1 = result[result["order_id"] == "o1"].iloc[0]
        assert o1["delivery_delay_days"] == -2

    def test_is_late_flag(self, sample_orders):
        result = _clean_orders(sample_orders.copy())
        # o2: delivered 2017-07-30, estimated 2017-07-25 → 5 days late
        o2 = result[result["order_id"] == "o2"].iloc[0]
        assert o2["is_late_delivery"] == 1

    def test_early_delivery_not_late(self, sample_orders):
        result = _clean_orders(sample_orders.copy())
        o1 = result[result["order_id"] == "o1"].iloc[0]
        assert o1["is_late_delivery"] == 0

    def test_time_features_created(self, sample_orders):
        result = _clean_orders(sample_orders.copy())
        for col in ["order_year","order_month","order_week","order_dow","order_hour"]:
            assert col in result.columns, f"Missing column: {col}"


# ── Tests: _clean_order_items ─────────────────────────────────────────────────

class TestCleanOrderItems:

    def test_negative_price_removed(self):
        df = pd.DataFrame({
            "order_id":           ["o1","o2","o3"],
            "order_item_id":      [1,1,1],
            "product_id":         ["p1","p2","p3"],
            "seller_id":          ["s1","s2","s3"],
            "shipping_limit_date":["2017-01-01"]*3,
            "price":              [100.0, -5.0, 0.0],
            "freight_value":      [10.0, 5.0, 5.0],
        })
        result = _clean_order_items(df.copy(), min_price=0.01)
        assert len(result) == 1
        assert result.iloc[0]["order_id"] == "o1"

    def test_total_value_computed(self, sample_items):
        result = _clean_order_items(sample_items.copy(), min_price=0.01)
        assert "total_value" in result.columns
        assert result.iloc[0]["total_value"] == pytest.approx(110.0)

    def test_freight_ratio_bounded(self, sample_items):
        result = _clean_order_items(sample_items.copy(), min_price=0.01)
        assert result["freight_ratio"].between(0, 1).all()


# ── Tests: _clean_payments ────────────────────────────────────────────────────

class TestCleanPayments:

    def test_zero_value_removed(self, sample_payments):
        result = _clean_payments(sample_payments.copy())
        assert (result["payment_value"] > 0).all()
        assert len(result) == 2

    def test_payment_value_numeric(self, sample_payments):
        result = _clean_payments(sample_payments.copy())
        assert pd.api.types.is_float_dtype(result["payment_value"])


# ── Tests: _clean_reviews ─────────────────────────────────────────────────────

class TestCleanReviews:

    def test_null_score_dropped(self, sample_reviews):
        result = _clean_reviews(sample_reviews.copy())
        assert result["review_score"].notna().all()
        assert len(result) == 2

    def test_is_negative_flag(self, sample_reviews):
        result = _clean_reviews(sample_reviews.copy())
        # score 5 → not negative; score 3 → not negative (threshold is <=2)
        assert result[result["review_score"] == 5]["is_negative"].iloc[0] == 0

    def test_is_positive_flag(self, sample_reviews):
        result = _clean_reviews(sample_reviews.copy())
        assert result[result["review_score"] == 5]["is_positive"].iloc[0] == 1
        assert result[result["review_score"] == 3]["is_positive"].iloc[0] == 0


# ── Tests: _build_fact_orders ─────────────────────────────────────────────────

class TestBuildFactOrders:

    def test_one_row_per_order(self, sample_orders, sample_items, sample_payments, sample_reviews):
        orders   = _clean_orders(sample_orders.copy())
        items    = _clean_order_items(sample_items.copy(), 0.01)
        payments = _clean_payments(sample_payments.copy())
        reviews  = _clean_reviews(sample_reviews.copy())
        customers = pd.DataFrame({
            "customer_id":    ["c1","c2","c3"],
            "customer_state": ["SP","RJ","MG"],
            "customer_city":  ["sao_paulo","rio_de_janeiro","belo_horizonte"],
        })
        fact = _build_fact_orders(orders, customers, items, payments, reviews)
        assert fact["order_id"].nunique() == len(fact), "Fact table has duplicate order_ids"

    def test_revenue_aggregated_correctly(self, sample_orders, sample_items, sample_payments, sample_reviews):
        orders   = _clean_orders(sample_orders.copy())
        items    = _clean_order_items(sample_items.copy(), 0.01)
        payments = _clean_payments(sample_payments.copy())
        reviews  = _clean_reviews(sample_reviews.copy())
        customers = pd.DataFrame({
            "customer_id":    ["c1","c2","c3"],
            "customer_state": ["SP","RJ","MG"],
            "customer_city":  ["sp","rj","mg"],
        })
        fact = _build_fact_orders(orders, customers, items, payments, reviews)
        o1 = fact[fact["order_id"] == "o1"].iloc[0]
        # o1 has items priced 100 + 50 = 150
        assert o1["order_revenue"] == pytest.approx(150.0)


# ── Tests: Config loader ──────────────────────────────────────────────────────

class TestConfigLoader:

    def test_config_loads(self):
        cfg = load_config()
        assert "paths" in cfg
        assert "etl" in cfg
        assert "anomaly" in cfg
        assert "reports" in cfg

    def test_paths_are_absolute(self):
        cfg = load_config()
        for key, path in cfg["paths"].items():
            assert os.path.isabs(path), f"Path '{key}' is not absolute: {path}"

    def test_required_keys_present(self):
        cfg = load_config()
        assert "database" in cfg["paths"]
        assert "raw_data" in cfg["paths"]
        assert "zscore_threshold" in cfg["anomaly"]
        assert "health_score_weights" in cfg["etl"]
