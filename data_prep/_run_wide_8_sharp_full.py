"""
Run the proposed `wide_8_sharp` smoothing config on the FULL OptionMetrics
dataset and write a sidecar log of per-date ffill counts so we can see which
days had insufficient raw quote support.

Outputs (under _wide_8_sharp_full/):
  - SPX_surfaces.csv   : 8x8 surfaces, m in [-0.20, 0.20], tau in [0.04, 1.0]
  - daily_ffill.csv    : date, n_ffill_cells, n_total_cells

Run from data_prep/.
"""
import os, sys
from os.path import dirname, abspath, join

import numpy as np
import pandas as pd

sys.path.insert(0, dirname(abspath(__file__)))
from preprocess_optionmetrics import OptionMetricsPreprocess


class TrackedPreprocess(OptionMetricsPreprocess):
    """Subclass that records per-date ffill counts during the daily loop."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._daily_ffill = []

    def process_daily_surfaces(self):
        # mirror the parent, but also record per-date ffill
        self.options_df["ttm"] = (
            self.options_df["exdate"] - self.options_df["date"]
        ).dt.days / 365.0
        eps_t = 1.0 / 365.0
        self.options_df = self.options_df[
            (self.options_df["ttm"] >= max(self.filter_ttm_min, eps_t))
            & (self.options_df["ttm"] <= self.filter_ttm_max)
        ].copy()
        self.options_df["strike"] = self.options_df["strike_price"] / 1000.0
        self.options_df["moneyness"] = np.log(
            self.options_df["strike"] / self.options_df["forward_price"]
        )

        n_dates = self.options_df["date"].nunique()
        print(f"Processing {n_dates} unique dates...")

        combined_data = []
        total_cells = 0
        total_nan_cells = 0
        n_axis = self.n_axis_points
        cells_per_day = n_axis * n_axis

        for i, (date, day_options) in enumerate(
            self.options_df.groupby("date", sort=True)
        ):
            if (i + 1) % 250 == 0:
                print(f"  {i+1}/{n_dates}  {date.strftime('%Y-%m-%d')}")
            do = day_options.copy()
            do = do[
                (do["moneyness"] >= self.filter_moneyness_min)
                & (do["moneyness"] <= self.filter_moneyness_max)
            ].copy()
            do = do[
                np.isfinite(do["impl_volatility"])
                & (do["impl_volatility"] > self.iv_min)
                & (do["impl_volatility"] < self.iv_max)
            ].copy()
            do = do[
                np.isfinite(do["vega"])
                & (do["vega"] >= self.min_vega)
            ].copy()
            if self.require_volume:
                do = do[do["volume"] > 0].copy()

            calls = do[do["cp_flag"] == "C"]
            puts = do[do["cp_flag"] == "P"]
            calls_keep = calls[
                (calls["moneyness"] > 0.0)
                | (np.abs(calls["moneyness"]) < self.atm_threshold)
            ]
            puts_keep = puts[
                (puts["moneyness"] < 0.0)
                | (np.abs(puts["moneyness"]) < self.atm_threshold)
            ]
            combined = pd.concat([calls_keep, puts_keep]).reset_index(drop=True)
            surface, n_nan = self.create_surface(combined, date)

            self._daily_ffill.append({
                "date": date.strftime("%Y-%m-%d"),
                "n_obs_used": len(combined),
                "n_ffill_cells": int(n_nan),
                "n_total_cells": cells_per_day,
                "surface_produced": surface is not None,
            })
            total_cells += cells_per_day
            total_nan_cells += n_nan

            if surface is not None:
                fwd_ref = float(np.median(combined["forward_price"].values))
                combined_data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "forward_ref": round(fwd_ref, 4),
                    "surface": surface,
                })

        if total_cells > 0:
            pct = 100.0 * total_nan_cells / total_cells
            print(f"\nGrid cells filled by ffill/bfill: "
                  f"{total_nan_cells:,}/{total_cells:,} ({pct:.2f}%)")

        # write the sidecar
        ffill_path = join(self.output_dir, "daily_ffill.csv")
        pd.DataFrame(self._daily_ffill).to_csv(ffill_path, index=False)
        print(f"Wrote per-date ffill log: {ffill_path}")

        self.save_combined_surface(combined_data)
        print(f"Saved {len(combined_data)} surfaces.")


def main():
    out_dir = "_wide_8_sharp_full"
    os.makedirs(out_dir, exist_ok=True)
    pre = TrackedPreprocess(
        options_file="SPX_options.csv",
        forward_file="SPX_forward.csv",
        output_dir=out_dir,
        # wide_8_sharp config
        n_axis_points=8,
        moneyness_min=-0.20, moneyness_max=0.20,
        filter_moneyness_min=-0.40, filter_moneyness_max=0.40,
        h_m=0.001,
        h_tau=0.10,
    )
    pre.load_data()
    pre.process_daily_surfaces()


if __name__ == "__main__":
    main()
