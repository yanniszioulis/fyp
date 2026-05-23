"""
Coverage / filter-attrition counts for the SPX options sample used in the
write-up. Mirrors preprocess_optionmetrics.OptionMetricsPreprocess up to (but
not including) the per-day kernel smoothing.

Window: filter_moneyness in [-0.25, 0.25], filter_ttm in [1/365, 1.75],
through 2023-12-29 inclusive.
"""
import numpy as np
import pandas as pd

OPTIONS = "/Users/yanniszioulis/Documents/FYP_real/fyp/_data_prep/SPX_options.csv"
FORWARD = "/Users/yanniszioulis/Documents/FYP_real/fyp/_data_prep/SPX_forward.csv"
END_DATE = pd.Timestamp("2023-12-29")

M_LO, M_HI = -0.25, 0.25
TTM_LO, TTM_HI = 1.0 / 365.0, 1.75
IV_MIN, IV_MAX = 0.01, 3.0
MIN_VEGA = 0.5
ATM_THR = 0.02


def load_forward():
    fwd = pd.read_csv(FORWARD).rename(columns={
        "expiration": "exdate", "AMSettlement": "am_settlement",
        "ForwardPrice": "forward_price",
    })[["date", "exdate", "am_settlement", "forward_price"]]
    fwd["date"] = pd.to_datetime(fwd["date"])
    fwd["exdate"] = pd.to_datetime(fwd["exdate"])
    fwd = (fwd.sort_values(["date", "exdate", "am_settlement"])
              .drop_duplicates(subset=["date", "exdate"], keep="first")
              .drop(columns=["am_settlement"]))
    return fwd


def main():
    fwd = load_forward()
    print(f"forward rows after PM-preferred dedup: {len(fwd):,}")

    counters = {
        "raw_to_2023_12_29": 0,
        "after_forward_join": 0,
        "after_ttm_window":   0,
        "after_moneyness_win": 0,
        "after_iv_vega_quality": 0,
        "after_otm_atm_band":  0,
    }
    calls_kept, puts_kept = 0, 0
    per_day_rows = []

    usecols = ["date", "exdate", "cp_flag", "strike_price",
               "impl_volatility", "volume", "vega"]
    dtypes  = {"cp_flag": "category", "strike_price": "float64",
               "impl_volatility": "float32", "volume": "int64",
               "vega": "float32"}

    for chunk in pd.read_csv(OPTIONS, chunksize=1_000_000,
                              usecols=usecols, dtype=dtypes):
        chunk["date"]   = pd.to_datetime(chunk["date"])
        chunk = chunk[chunk["date"] <= END_DATE]
        if not len(chunk):
            continue
        chunk["exdate"] = pd.to_datetime(chunk["exdate"])

        counters["raw_to_2023_12_29"] += len(chunk)

        chunk = chunk.merge(fwd, on=["date", "exdate"], how="left")
        chunk = chunk[chunk["forward_price"] > 0]
        counters["after_forward_join"] += len(chunk)

        chunk["ttm"] = (chunk["exdate"] - chunk["date"]).dt.days / 365.0
        chunk = chunk[(chunk["ttm"] >= TTM_LO) & (chunk["ttm"] <= TTM_HI)]
        counters["after_ttm_window"] += len(chunk)

        strike = chunk["strike_price"] / 1000.0
        chunk["moneyness"] = np.log(strike / chunk["forward_price"])
        chunk = chunk[(chunk["moneyness"] >= M_LO)
                      & (chunk["moneyness"] <= M_HI)]
        counters["after_moneyness_win"] += len(chunk)

        iv = chunk["impl_volatility"]
        base_ok = (np.isfinite(iv) & (iv > IV_MIN) & (iv < IV_MAX)
                   & np.isfinite(chunk["vega"]))
        is_wing     = chunk["moneyness"].abs() > 0.07
        vega_ok     = np.where(is_wing, chunk["vega"] >= 0.1,
                               chunk["vega"] >= MIN_VEGA)
        vol_ok      = (chunk["volume"] > 0) | is_wing
        chunk = chunk[base_ok & vega_ok & vol_ok]
        counters["after_iv_vega_quality"] += len(chunk)

        calls = chunk[chunk["cp_flag"] == "C"]
        puts  = chunk[chunk["cp_flag"] == "P"]
        calls = calls[(calls["moneyness"] > 0.0)
                      | (calls["moneyness"].abs() < ATM_THR)]
        puts  = puts[(puts["moneyness"] < 0.0)
                     | (puts["moneyness"].abs() < ATM_THR)]
        kept = pd.concat([calls, puts])
        counters["after_otm_atm_band"] += len(kept)
        calls_kept += len(calls)
        puts_kept  += len(puts)

        if len(kept):
            grp = kept.groupby("date").agg(
                total=("cp_flag", "size"),
                calls=("cp_flag", lambda s: (s == "C").sum()),
                puts =("cp_flag", lambda s: (s == "P").sum()),
            )
            per_day_rows.append(grp)

    print("\n--- attrition (rows = option quotes) ---")
    raw = counters["raw_to_2023_12_29"]
    for k, v in counters.items():
        print(f"{k:30s} {v:>14,}   {100*v/max(raw,1):6.2f}%")

    per_day = pd.concat(per_day_rows).groupby(level=0).sum()
    print(f"\ntrading days in sample: {len(per_day):,}")
    print(f"date range: {per_day.index.min().date()}  ->  {per_day.index.max().date()}")
    print(f"\ncall/put split of kept quotes:")
    print(f"  calls kept: {calls_kept:>14,}")
    print(f"  puts  kept: {puts_kept:>14,}")

    desc = per_day.describe(percentiles=[0.05, 0.5, 0.95]).round(0)
    print("\n--- per-day quote counts (kept quotes that feed the kernel) ---")
    print(desc.to_string())

    per_day.to_csv(
        "/Users/yanniszioulis/Documents/FYP_real/fyp/_data_prep/per_day_quote_counts.csv"
    )
    print("\nwrote per_day_quote_counts.csv")


if __name__ == "__main__":
    main()
