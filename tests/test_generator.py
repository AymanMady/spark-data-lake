"""Tests of the data generator (phase 15)."""

from __future__ import annotations

import numpy as np
import pytest
from faker import Faker

from src.generator.generate_data import (
    PRESETS,
    RATE_DUPLICATE,
    build_pools,
    generate_customers,
    generate_orders_chunk,
    generate_products,
)


@pytest.fixture(scope="module")
def pools():
    fake = Faker("fr_FR")
    Faker.seed(1)
    return build_pools(fake, np.random.default_rng(1), n_first=50, n_last=50, n_products=80)


def test_same_seed_produces_the_same_dataset(pools):
    """Without reproducibility, two runs cannot be compared."""
    a = generate_orders_chunk(500, 1, 100, 20, np.random.default_rng(42))
    b = generate_orders_chunk(500, 1, 100, 20, np.random.default_rng(42))
    assert a.equals(b)


def test_different_seeds_produce_different_datasets():
    a = generate_orders_chunk(500, 1, 100, 20, np.random.default_rng(1))
    b = generate_orders_chunk(500, 1, 100, 20, np.random.default_rng(2))
    assert not a.equals(b)


def test_expected_columns(pools):
    rng = np.random.default_rng(0)
    assert list(generate_customers(50, pools, rng).columns) == [
        "customer_id", "first_name", "last_name", "email", "country", "created_at"]
    assert list(generate_products(20, pools, rng).columns) == [
        "product_id", "product_name", "category", "price"]
    assert list(generate_orders_chunk(50, 1, 50, 20, rng).columns) == [
        "order_id", "customer_id", "product_id", "quantity", "order_date", "status"]


def test_duplicates_are_injected(pools):
    """The dataset MUST be dirty: otherwise the Silver layer proves nothing."""
    df = generate_orders_chunk(10_000, 1, 500, 100, np.random.default_rng(7))
    assert df.duplicated().sum() > 0
    # The row count exceeds what was asked for, because of the duplicates.
    assert len(df) > 10_000
    assert len(df) == pytest.approx(10_000 * (1 + RATE_DUPLICATE), rel=0.5)


def test_anomalies_are_injected():
    df = generate_orders_chunk(20_000, 1, 1_000, 200, np.random.default_rng(3))
    quantities = df["quantity"]
    assert quantities.isna().sum() > 0, "there must be missing quantities"
    numeric = quantities.dropna().astype(int)
    assert (numeric <= 0).sum() > 0, "there must be negative or zero quantities"
    assert df["customer_id"].isna().sum() > 0, "there must be missing keys"
    assert df["order_date"].isna().sum() > 0, "there must be missing dates"


def test_emails_are_partially_broken(pools):
    df = generate_customers(5_000, pools, np.random.default_rng(11))
    emails = df["email"].dropna()
    invalid = (~emails.str.contains("@")).sum()
    assert invalid > 0, "there must be emails without an at-sign"
    assert invalid < len(emails) * 0.2, "but they must stay a minority"


def test_several_date_formats(pools):
    """
    The two formats generated for created_at ('2024-05-01' and '01/05/2024')
    have the SAME length: len() cannot tell them apart. So the test looks at
    the separator, which is the real marker.
    """
    df = generate_customers(5_000, pools, np.random.default_rng(13))
    dates = df["created_at"].dropna()
    iso = dates.str.contains("-").sum()
    european = dates.str.contains("/").sum()
    assert iso > 0, "some dates must still be in ISO format"
    assert european > 0, "there must be dates in DD/MM/YYYY format"


def test_ids_are_unique_before_duplicate_injection(pools):
    """The ids are generated sequentially: duplicates really do come from the injection."""
    df = generate_customers(100, pools, np.random.default_rng(5))
    unique = df["customer_id"].nunique()
    assert unique == 100
    assert len(df) >= 100


def test_consecutive_chunks_do_not_reuse_ids():
    """Generating in batches must not recreate the same order_id."""
    rng = np.random.default_rng(9)
    c1 = generate_orders_chunk(100, 1, 50, 10, rng)
    c2 = generate_orders_chunk(100, 101, 50, 10, rng)
    assert c1["order_id"].max() < c2["order_id"].min()


def test_presets_are_increasing():
    sizes = [PRESETS[k] for k in ["10mb", "100mb", "1gb", "10gb"]]
    assert sizes == sorted(sizes)
    assert PRESETS["100mb"] == PRESETS["10mb"] * 10
