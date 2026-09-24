"""
Realistic e-commerce data generator (phase 2).

Why not use Faker row by row?
Faker generates roughly 20,000 values per second. For 10 million orders that
would take hours. The strategy used here:

  1. Faker builds POOLS of realistic values (a few thousand).
  2. numpy then draws from those pools in a VECTORISED way (millions/second).

This is exactly Spark's logic: avoid row-by-row Python loops.

The data is deliberately DIRTY. A pipeline that only ever receives clean data
proves nothing: the Silver and Gold layers would have no work to do.

Usage:
    python3 -m src.generator.generate_data --rows 1000000
    python3 -m src.generator.generate_data --preset 100mb
    python3 -m src.generator.generate_data --rows 5000000 --seed 42
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# Parameters of the e-commerce "model"
# ---------------------------------------------------------------------------

COUNTRIES = [
    "France", "Germany", "Spain", "Italy", "United Kingdom", "Belgium",
    "Netherlands", "Portugal", "Poland", "Sweden", "Morocco", "Tunisia",
    "Canada", "United States", "Brazil", "Japan",
]
# Weights: a real e-commerce site does not have a uniform split by country.
COUNTRY_WEIGHTS = np.array(
    [0.22, 0.14, 0.09, 0.09, 0.10, 0.04, 0.04, 0.03, 0.03, 0.03,
     0.05, 0.03, 0.04, 0.04, 0.02, 0.01]
)

CATEGORIES = [
    "Electronics", "Clothing", "Home & Kitchen", "Books", "Sports",
    "Beauty", "Toys", "Automotive", "Garden", "Grocery",
]

# Statuses are not equally likely: most orders go through.
STATUSES = ["delivered", "shipped", "pending", "cancelled", "returned"]
STATUS_WEIGHTS = np.array([0.62, 0.15, 0.12, 0.07, 0.04])

# ---------------------------------------------------------------------------
# Injected anomaly rates (documented: the tests rely on them)
# ---------------------------------------------------------------------------

RATE_DUPLICATE = 0.015        # strictly duplicated rows
RATE_BAD_EMAIL = 0.030        # email without @, with spaces, in upper case
RATE_NULL_COUNTRY = 0.020     # missing country
RATE_BAD_PRICE = 0.020        # negative or zero price
RATE_BAD_QUANTITY = 0.012     # quantity <= 0
RATE_NULL_FK = 0.006          # missing customer_id or product_id
RATE_ORPHAN_FK = 0.004        # reference to a non-existent id
RATE_NULL_DATE = 0.005        # missing date
RATE_ALT_DATE_FMT = 0.10      # date in a different format

# Presets calibrated on a real MEASUREMENT: 1,000,000 orders produce 49.2 MB of
# CSV in total (customers + products + orders), i.e. ~51.6 bytes per order.
# These are not made-up values.
PRESETS = {
    "10mb": 200_000,
    "100mb": 2_000_000,
    "1gb": 21_000_000,
    "10gb": 210_000_000,
}


def log(msg: str) -> None:
    print(f"[GENERATOR] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Faker pools
# ---------------------------------------------------------------------------

def build_pools(fake: Faker, rng: np.random.Generator, n_first: int = 3000,
                n_last: int = 3000, n_products: int = 4000) -> dict:
    """Build the pools of realistic values, once."""
    log(f"Building the Faker pools ({n_first} first names, {n_last} last names,"
        f" {n_products} products)...")
    first_names = np.array([fake.first_name() for _ in range(n_first)])
    last_names = np.array([fake.last_name() for _ in range(n_last)])

    # Product names: "Adjective Material Object" -> realistic and varied.
    adjectives = ["Premium", "Classic", "Ultra", "Eco", "Smart", "Pro", "Compact",
                  "Deluxe", "Essential", "Vintage", "Modern", "Portable"]
    materials = ["Wireless", "Cotton", "Steel", "Wooden", "Leather", "Ceramic",
                 "Bamboo", "Aluminum", "Glass", "Silicone"]
    objects = ["Headphones", "T-Shirt", "Blender", "Notebook", "Backpack", "Lamp",
               "Keyboard", "Chair", "Bottle", "Mug", "Sneakers", "Camera",
               "Monitor", "Jacket", "Watch", "Speaker", "Desk", "Pillow"]
    products = np.array([
        f"{rng.choice(adjectives)} {rng.choice(materials)} {rng.choice(objects)}"
        for _ in range(n_products)
    ])
    return {"first": first_names, "last": last_names, "product": products}


# ---------------------------------------------------------------------------
# Dirtying helpers
# ---------------------------------------------------------------------------

def dirty_text(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Add stray whitespace and inconsistent casing, like real data entry.

    The "if mask.any()" guards are not defensive on principle: the np.char
    functions raise a ValueError on an EMPTY array. Without them the generator
    crashes as soon as n is small (a case hit in the tests, where no row is
    drawn for a given anomaly).
    """
    out = values.astype(object).copy()
    n = len(out)
    idx_upper = rng.random(n) < 0.05
    idx_lower = rng.random(n) < 0.05
    idx_space = rng.random(n) < 0.06
    if idx_upper.any():
        out[idx_upper] = np.char.upper(values[idx_upper].astype(str))
    if idx_lower.any():
        out[idx_lower] = np.char.lower(values[idx_lower].astype(str))
    if idx_space.any():
        out[idx_space] = np.char.add("  ", np.char.add(values[idx_space].astype(str), " "))
    return out


def make_emails(first: np.ndarray, last: np.ndarray, ids: np.ndarray,
                rng: np.random.Generator) -> np.ndarray:
    """Mostly valid emails, with a deliberately broken minority."""
    domains = np.array(["gmail.com", "yahoo.fr", "outlook.com", "orange.fr", "proton.me"])
    dom = rng.choice(domains, size=len(ids))
    base = np.char.add(
        np.char.add(
            np.char.add(np.char.lower(first.astype(str)), "."),
            np.char.lower(last.astype(str)),
        ),
        np.char.add(ids.astype(str), np.char.add("@", dom)),
    )
    emails = base.astype(object)
    n = len(emails)

    # 1) missing at-sign  2) surrounding spaces  3) upper case  4) empty value
    broken = rng.random(n) < RATE_BAD_EMAIL
    kind = rng.integers(0, 4, size=n)
    # Same precaution as in dirty_text: np.char refuses empty arrays.
    m0, m1, m2 = (broken & (kind == k) for k in (0, 1, 2))
    if m0.any():
        emails[m0] = np.char.replace(base[m0].astype(str), "@", ".")
    if m1.any():
        emails[m1] = np.char.add(" ", np.char.add(base[m1].astype(str), "  "))
    if m2.any():
        emails[m2] = np.char.upper(base[m2].astype(str))
    emails[broken & (kind == 3)] = None
    return emails


def format_dates(ts: np.ndarray, rng: np.random.Generator,
                 with_time: bool = False) -> np.ndarray:
    """
    Format dates, deliberately mixing several conventions:
      - 'YYYY-MM-DD'  (ISO, the majority)
      - 'DD/MM/YYYY'  (European convention)
      - 'YYYY-MM-DD HH:MM:SS'
    The Silver layer will have to reconcile all of them.
    """
    s = pd.Series(pd.to_datetime(ts))
    iso = s.dt.strftime("%Y-%m-%d %H:%M:%S" if with_time else "%Y-%m-%d").to_numpy(dtype=object)
    eu = s.dt.strftime("%d/%m/%Y").to_numpy(dtype=object)

    out = iso.copy()
    alt = rng.random(len(out)) < RATE_ALT_DATE_FMT
    out[alt] = eu[alt]
    out[rng.random(len(out)) < RATE_NULL_DATE] = None
    return out


def inject_duplicates(df: pd.DataFrame, rng: np.random.Generator,
                      rate: float = RATE_DUPLICATE) -> pd.DataFrame:
    """Re-inject existing rows verbatim (pure duplicates)."""
    n_dup = int(len(df) * rate)
    if n_dup == 0:
        return df
    dup_idx = rng.integers(0, len(df), size=n_dup)
    return pd.concat([df, df.iloc[dup_idx]], ignore_index=True)


# ---------------------------------------------------------------------------
# Generating the three tables
# ---------------------------------------------------------------------------

def generate_customers(n: int, pools: dict, rng: np.random.Generator) -> pd.DataFrame:
    ids = np.arange(1, n + 1)
    first = rng.choice(pools["first"], size=n)
    last = rng.choice(pools["last"], size=n)

    # Sign-ups spread over 3 years.
    start = np.datetime64("2022-01-01")
    created = start + rng.integers(0, 365 * 3, size=n).astype("timedelta64[D]")

    weights = COUNTRY_WEIGHTS / COUNTRY_WEIGHTS.sum()
    country = rng.choice(COUNTRIES, size=n, p=weights).astype(object)
    country[rng.random(n) < RATE_NULL_COUNTRY] = None

    df = pd.DataFrame({
        "customer_id": ids,
        "first_name": dirty_text(first, rng),
        "last_name": dirty_text(last, rng),
        "email": make_emails(first, last, ids, rng),
        "country": country,
        "created_at": format_dates(created, rng),
    })
    return inject_duplicates(df, rng)


def generate_products(n: int, pools: dict, rng: np.random.Generator) -> pd.DataFrame:
    ids = np.arange(1, n + 1)
    names = rng.choice(pools["product"], size=n)
    category = rng.choice(CATEGORIES, size=n)

    # Log-normal prices: many cheap items, a few expensive ones.
    price = np.round(np.exp(rng.normal(3.1, 0.9, size=n)), 2)
    price = np.clip(price, 0.99, 4999.0)

    price = price.astype(object)
    bad = rng.random(n) < RATE_BAD_PRICE
    kind = rng.integers(0, 3, size=n)
    price[bad & (kind == 0)] = -1.0           # negative price
    price[bad & (kind == 1)] = 0.0            # zero price
    price[bad & (kind == 2)] = None           # missing price

    df = pd.DataFrame({
        "product_id": ids,
        "product_name": dirty_text(names, rng),
        "category": dirty_text(category, rng),
        "price": price,
    })
    return inject_duplicates(df, rng)


def generate_orders_chunk(n: int, start_id: int, n_customers: int, n_products: int,
                          rng: np.random.Generator) -> pd.DataFrame:
    order_id = np.arange(start_id, start_id + n)

    # Pareto distribution: a minority of customers place most of the orders.
    customer_id = np.minimum(
        (rng.pareto(1.6, size=n) * n_customers / 12).astype(np.int64) + 1, n_customers
    ).astype(object)
    product_id = rng.integers(1, n_products + 1, size=n).astype(object)

    # Broken foreign keys: missing or orphaned.
    customer_id[rng.random(n) < RATE_NULL_FK] = None
    product_id[rng.random(n) < RATE_NULL_FK] = None
    customer_id[rng.random(n) < RATE_ORPHAN_FK] = n_customers + 999_999
    product_id[rng.random(n) < RATE_ORPHAN_FK] = n_products + 999_999

    quantity = rng.integers(1, 6, size=n).astype(object)
    bad_q = rng.random(n) < RATE_BAD_QUANTITY
    kind = rng.integers(0, 3, size=n)
    quantity[bad_q & (kind == 0)] = 0
    quantity[bad_q & (kind == 1)] = -2
    quantity[bad_q & (kind == 2)] = None

    # Orders over 2 years, with a seasonal peak at the end of the year.
    start = np.datetime64("2024-01-01")
    day_offsets = rng.integers(0, 730, size=n)
    seasonal = rng.random(n) < 0.15
    day_offsets[seasonal] = rng.integers(320, 360, size=seasonal.sum())
    # Random time of day: otherwise every order would be at midnight.
    seconds = rng.integers(0, 86_400, size=n)
    order_ts = (start.astype("datetime64[s]")
                + day_offsets.astype("timedelta64[D]").astype("timedelta64[s]")
                + seconds.astype("timedelta64[s]"))

    status = rng.choice(STATUSES, size=n, p=STATUS_WEIGHTS / STATUS_WEIGHTS.sum())

    df = pd.DataFrame({
        "order_id": order_id,
        "customer_id": customer_id,
        "product_id": product_id,
        "quantity": quantity,
        "order_date": format_dates(order_ts, rng, with_time=True),
        "status": dirty_text(status, rng),
    })
    return inject_duplicates(df, rng)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_csv(df: pd.DataFrame, path: Path, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, mode="a" if append else "w", header=not append)


def human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a realistic (and deliberately dirty) e-commerce dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rows", type=int, default=1_000_000,
                        help="Number of orders to generate")
    parser.add_argument("--preset", choices=sorted(PRESETS),
                        help="Approximate target size (overrides --rows)")
    parser.add_argument("--customers", type=int, default=None,
                        help="Number of customers (default: rows/20, min 1000)")
    parser.add_argument("--products", type=int, default=None,
                        help="Number of products (default: rows/200, min 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed: same seed = same dataset")
    parser.add_argument("--output", type=Path, default=Path("data/raw"),
                        help="Output directory")
    parser.add_argument("--chunk-size", type=int, default=1_000_000,
                        help="Write batch size (caps the RAM used)")
    parser.add_argument("--clean", action="store_true",
                        help="Empty the output directory before generating")
    args = parser.parse_args(argv)

    rows = PRESETS[args.preset] if args.preset else args.rows
    n_customers = args.customers or max(1_000, rows // 20)
    n_products = args.products or max(100, rows // 200)

    rng = np.random.default_rng(args.seed)
    fake = Faker("fr_FR")
    Faker.seed(args.seed)

    out = args.output
    if args.clean and out.exists():
        shutil.rmtree(out)

    log(f"seed={args.seed} | orders={rows:,} | customers={n_customers:,} | products={n_products:,}")
    t0 = time.perf_counter()

    pools = build_pools(fake, rng)

    log(f"Generating {n_customers:,} customers...")
    customers = generate_customers(n_customers, pools, rng)
    write_csv(customers, out / "customers" / "customers.csv")
    log(f"  -> {len(customers):,} rows (duplicates included)")

    log(f"Generating {n_products:,} products...")
    products = generate_products(n_products, pools, rng)
    write_csv(products, out / "products" / "products.csv")
    log(f"  -> {len(products):,} rows (duplicates included)")

    log(f"Generating {rows:,} orders in batches of {args.chunk_size:,}...")
    orders_path = out / "orders" / "orders.csv"
    if orders_path.exists():
        orders_path.unlink()

    written = 0
    next_id = 1
    while written < rows:
        n = min(args.chunk_size, rows - written)
        chunk = generate_orders_chunk(n, next_id, n_customers, n_products, rng)
        write_csv(chunk, orders_path, append=written > 0)
        written += n
        next_id += n
        pct = 100 * written / rows
        log(f"  {written:,}/{rows:,} ({pct:.0f}%)")

    elapsed = time.perf_counter() - t0
    total = dir_size(out)
    log("-" * 58)
    log(f"Done in {elapsed:.1f}s | volume generated: {human(total)}")
    for name in ("customers", "products", "orders"):
        f = out / name / f"{name}.csv"
        if f.exists():
            log(f"  {name:<10} {human(f.stat().st_size):>10}  {f}")
    log(f"Throughput: {rows / elapsed:,.0f} orders/second")
    return 0


if __name__ == "__main__":
    sys.exit(main())
