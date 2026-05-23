"""
Full descriptive-stats table for the SPX options sample.
Same filters as the surface preprocessor up to (but not including) the
per-day kernel smoothing. Through 2023-12-29.
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

    usecols = ["date", "exdate", "cp_flag", "strike_price",
               "impl_volatility", "volume", "vega"]
    dtypes  = {"cp_flag": "category", "strike_price": "float64",
               "impl_volatility": "float32", "volume": "int64",
               "vega": "float32"}

    kept_chunks = []
    for chunk in pd.read_csv(OPTIONS, chunksize=1_000_000,
                              usecols=usecols, dtype=dtypes):
        chunk["date"]   = pd.to_datetime(chunk["date"])
        chunk = chunk[chunk["date"] <= END_DATE]
        if not len(chunk):
            continue
        chunk["exdate"] = pd.to_datetime(chunk["exdate"])

        chunk = chunk.merge(fwd, on=["date", "exdate"], how="left")
        chunk = chunk[chunk["forward_price"] > 0]

        chunk["ttm"] = (chunk["exdate"] - chunk["date"]).dt.days / 365.0
        chunk = chunk[(chunk["ttm"] >= TTM_LO) & (chunk["ttm"] <= TTM_HI)]

        strike = chunk["strike_price"] / 1000.0
        chunk["moneyness"] = np.log(strike / chunk["forward_price"])
        chunk = chunk[(chunk["moneyness"] >= M_LO) & (chunk["moneyness"] <= M_HI)]

        iv = chunk["impl_volatility"]
        base_ok = (np.isfinite(iv) & (iv > IV_MIN) & (iv < IV_MAX)
                   & np.isfinite(chunk["vega"]))
        is_wing = chunk["moneyness"].abs() > 0.07
        vega_ok = np.where(is_wing, chunk["vega"] >= 0.1,
                           chunk["vega"] >= MIN_VEGA)
        vol_ok  = (chunk["volume"] > 0) | is_wing
        chunk = chunk[base_ok & vega_ok & vol_ok]

        calls = chunk[chunk["cp_flag"] == "C"]
        puts  = chunk[chunk["cp_flag"] == "P"]
        calls = calls[(calls["moneyness"] > 0.0)
                      | (calls["moneyness"].abs() < ATM_THR)]
        puts  = puts[(puts["moneyness"] < 0.0)
                     | (puts["moneyness"].abs() < ATM_THR)]
        kept = pd.concat([calls, puts])
        if len(kept):
            kept_chunks.append(
                kept[["date", "cp_flag", "ttm", "moneyness", "impl_volatility"]]
            )

    df = pd.concat(kept_chunks, ignore_index=True)
    df["cp_flag"] = df["cp_flag"].astype(str)
    print(f"total kept: {len(df):,}")
    print(f"dates: {df['date'].min().date()}  ->  {df['date'].max().date()}")

    def split(g):
        n = len(g)
        nc = (g["cp_flag"] == "C").sum()
        npu = (g["cp_flag"] == "P").sum()
        miv = g["impl_volatility"].mean()
        return pd.Series({"N": n, "calls": nc, "puts": npu, "mean_iv": miv})

    # Maturity buckets
    ttm_edges  = [0, 30/365, 90/365, 180/365, 365/365, 1.75]
    ttm_labels = ["<=1m", "1-3m", "3-6m", "6-12m", "12-21m"]
    df["ttm_bucket"] = pd.cut(df["ttm"], bins=ttm_edges, labels=ttm_labels,
                              include_lowest=True)

    # Moneyness buckets
    m_edges  = [-0.25, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 0.25]
    m_labels = ["deep ITM P / OTM C (-0.25,-0.10]",
                "OTM put (-0.10,-0.05]",
                "near-ATM put (-0.05,-0.02]",
                "ATM (-0.02,0.02]",
                "near-ATM call (0.02,0.05]",
                "OTM call (0.05,0.10]",
                "deep OTM call (0.10,0.25]"]
    df["m_bucket"] = pd.cut(df["moneyness"], bins=m_edges, labels=m_labels,
                            include_lowest=True)

    # Sub-period
    cov = pd.Timestamp("2020-02-29")
    df["period"] = np.where(df["date"] <= cov, "pre-COVID", "post-COVID")

    print("\n=== Panel A: by maturity ===")
    print(df.groupby("ttm_bucket", observed=True).apply(split).round(3).to_string())

    print("\n=== Panel B: by moneyness ===")
    print(df.groupby("m_bucket", observed=True).apply(split).round(3).to_string())

    print("\n=== Panel C: by sub-period ===")
    pc = df.groupby("period", observed=True).apply(split)
    pc["days"] = df.groupby("period", observed=True)["date"].nunique()
    print(pc.round(3).to_string())

    print("\n=== Panel D: by year ===")
    df["year"] = df["date"].dt.year
    pd_y = df.groupby("year").apply(split)
    pd_y["days"] = df.groupby("year")["date"].nunique()
    print(pd_y.round(3).to_string())


if __name__ == "__main__":
    main()
