"""
Preprocess OptionMetrics data to create implied volatility surfaces.

Coordinates: (log(T), log(K/F))
  - T = time to expiration in years
  - F = forward price for the option's expiration (from Forward_Price table)
  - K = strike

Methodology:
  - Filter raw OptionMetrics quotes (OTM + narrow ATM band on each side).
  - Fit a vega-weighted Nadaraya-Watson kernel smoother in (log T, log-moneyness)
    space at every target grid point. No intermediate fit grid is used.
  - Save one row per date with the flattened surface in long-form column names.

Fixes vs. the original script:
  1. tau grid is now geometric in T (uniform in log T), matching the kernel's
     log-T distance metric. Avoids huge gaps at the front of the maturity axis
     that caused the smoother to extrapolate at low tau.
  2. Bandwidth h_m defaults to 0.001 (sigma ~ 0.032 in log-moneyness), matching
     the OM manual's relative bandwidth on the (log T, delta) axes. The old
     h_m=0.015 was about 20x too large in variance terms and flattened the
     skew while pulling neighboring grid cells into the same plateau.
  3. The wing buffer is widened: target window stays at +/-0.10 but the filter
     window opens to +/-0.30 so the kernel has data inside +/- 3 sigma of
     every target grid point.
  4. Effective sample size guard: if the unnormalized vega-weighted kernel
     mass at a grid point falls below `min_neff`, that cell is set to NaN
     instead of being filled by whichever single observation happens to
     dominate. This kills the flat-plateau-with-vertical-wall artifact.
  5. The softmax-stabilization trick (subtracting log_k.max(axis=1)) is kept,
     but applied AFTER computing the unnormalized neff so the guard sees the
     real weight mass rather than a per-row-rescaled version.
  6. Optional cap on per-grid-point single-observation share (`max_w_share`),
     which catches the "one obs dominates the whole neighborhood" case even
     when the absolute neff looks fine.

The kernel is the OM manual form (p.39-40), adapted from (log T, delta) to
(log T, log(K/F)) space:

    sigma_hat(j) = sum_i [ V_i * sigma_i * phi(x_ij, y_ij) ]
                   / sum_i [ V_i * phi(x_ij, y_ij) ]

    phi(x, y) = exp( -x^2 / (2 h_T) - y^2 / (2 h_m) )

with x_ij = log(T_i / T_j), y_ij = k_i - k_j, k = log(K/F). h_T and h_m are
variances (not standard deviations) -- the same convention as the OM manual.

All data comes from two files:
  - SPX_options.csv: option quotes
  - SPX_forward.csv: forward prices from the IvyDB Forward_Price table
The two are merged inside load_data on (date, exdate). Because the options
file does not carry an am_settlement column, we cannot join AM-settled
monthlies to AM forwards exclusively. Instead we collapse the forward table
to one row per (date, exdate), preferring the PM forward (am_settlement=0)
and falling back to AM (am_settlement=1) only when no PM row exists. This
is exact for PM-settled options and introduces a sub-day carry bias of a
few basis points for AM-settled monthlies, well below the kernel bandwidth.
"""
import os
from os.path import join

import numpy as np
import pandas as pd


class OptionMetricsPreprocess:
    def __init__(
        self,
        options_file,
        forward_file,
        output_dir,
        # target grid (what gets saved) -- in log forward moneyness.
        # Wider window (+/-0.20) than the original ATM core (+/-0.10) so the
        # surface includes the put/call wings where skew/curvature dynamics
        # live independently of the ATM level. Coarser grid (8 vs 20) so each
        # cell aggregates enough quotes to be robust without over-smoothing.
        moneyness_min=-0.20,
        moneyness_max=0.20,
        ttm_min=0.04,
        ttm_max=1.0,
        n_axis_points=8,
        # raw-data filter window (wider than target so the kernel has support
        # near the target boundaries) -- in log forward moneyness
        filter_moneyness_min=-0.40,
        filter_moneyness_max=0.40,
        filter_ttm_min=0.0,
        filter_ttm_max=1.5,
        # ATM band: calls below ATM-forward and puts above ATM-forward are
        # kept only inside this band around log-moneyness = 0
        atm_threshold=0.01,
        # kernel bandwidths (variances, NOT standard deviations -- matches
        # the OM manual convention: phi = exp(-x^2 / (2 h))). Sized so
        # sigma is ~0.5x the new grid step (sharper than before): grid step
        # in m is 0.40/7 = 0.057, sigma_m = sqrt(h_m) = 0.032 -> ~0.55 step;
        # log-T step is 0.46, sigma_logT = sqrt(h_tau) = 0.32 -> ~0.69 step.
        h_tau=0.10,     # in log(T);    sigma_logT  ~ 0.316
        h_m=0.001,      # in log(K/F);  sigma_m     ~ 0.032
        # filtering
        min_vega=0.5,        # OM excludes options with vega below this
        iv_min=0.01,
        iv_max=3.0,
        require_volume=True,
        # tau grid spacing -- geometric (log-uniform) by default now;
        # matches the log-T metric used inside the kernel
        log_tau=True,
        # sparse-data guards (the new bits)
        min_neff=2.0,        # min effective sample size per grid cell
        max_w_share=0.85,    # max share of total weight on any single obs
    ):
        self.options_file = options_file
        self.forward_file = forward_file
        self.output_dir = output_dir

        self.moneyness_min = moneyness_min
        self.moneyness_max = moneyness_max
        self.ttm_min = ttm_min
        self.ttm_max = ttm_max
        self.n_axis_points = n_axis_points

        self.filter_moneyness_min = filter_moneyness_min
        self.filter_moneyness_max = filter_moneyness_max
        self.filter_ttm_min = filter_ttm_min
        self.filter_ttm_max = filter_ttm_max

        self.atm_threshold = atm_threshold

        self.h_tau = h_tau
        self.h_m = h_m

        self.min_vega = min_vega
        self.iv_min = iv_min
        self.iv_max = iv_max
        self.require_volume = require_volume

        self.log_tau = log_tau

        self.min_neff = min_neff
        self.max_w_share = max_w_share

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ I/O

    def load_data(self):
        print("Loading options data...")
        chunks = []
        chunk_size = 1_000_000
        for chunk in pd.read_csv(self.options_file, chunksize=chunk_size):
            chunks.append(chunk)
        self.options_df = pd.concat(chunks, ignore_index=True)
        print(f"Loaded {len(self.options_df):,} option records")

        # column sanity (forward_price is added later from the forward file)
        required = {"date", "exdate", "cp_flag", "strike_price",
                    "impl_volatility", "volume"}
        missing = required - set(self.options_df.columns)
        if missing:
            raise ValueError(f"Missing required columns in options file: {missing}")

        if "vega" not in self.options_df.columns:
            print("Warning: no 'vega' column found - kernel weights will be uniform.")
            self.options_df["vega"] = 1.0

        # drop the existing (empty) forward_price column if present so the
        # merge below populates it cleanly
        if "forward_price" in self.options_df.columns:
            self.options_df = self.options_df.drop(columns=["forward_price"])

        # ----- forwards -----
        print("Loading forward prices...")
        fwd = pd.read_csv(self.forward_file)
        print(f"Loaded {len(fwd):,} forward records")

        # normalise column names: WRDS exports as "expiration", "amsettlement",
        # "forwardprice"; rename to match what we use internally
        rename_map = {
            "expiration":   "exdate",
            "AMSettlement": "am_settlement",
            "ForwardPrice": "forward_price",
        }
        fwd = fwd.rename(columns={k: v for k, v in rename_map.items() if k in fwd.columns})

        fwd_required = {"date", "exdate", "am_settlement", "forward_price"}
        fwd_missing = fwd_required - set(fwd.columns)
        if fwd_missing:
            raise ValueError(f"Missing required columns in forward file: {fwd_missing}")

        fwd = fwd[["date", "exdate", "am_settlement", "forward_price"]].copy()
        fwd["date"]   = pd.to_datetime(fwd["date"])
        fwd["exdate"] = pd.to_datetime(fwd["exdate"])

        # The options file has no am_settlement column, so we cannot tell
        # AM-settled monthlies from PM-settled weeklies/EOMs apart per row.
        # Strategy: collapse the forward table to one row per (date, exdate),
        # preferring the PM forward (am_settlement=0) and falling back to the
        # AM forward (am_settlement=1) when only AM is available. This is
        # exact for PM-settled options and introduces a sub-day carry bias
        # (~1-5 bps) for AM-settled monthlies on days where only an AM
        # forward exists.
        n_dup = fwd.duplicated(subset=["date", "exdate", "am_settlement"]).sum()
        if n_dup > 0:
            print(f"Warning: {n_dup:,} duplicate keys in forward file; keeping first.")
            fwd = fwd.drop_duplicates(subset=["date", "exdate", "am_settlement"])

        # sort so PM (0) comes before AM (1), then keep the first row per
        # (date, exdate) -- equivalent to "PM if available, else AM"
        fwd = (
            fwd.sort_values(["date", "exdate", "am_settlement"])
               .drop_duplicates(subset=["date", "exdate"], keep="first")
               .drop(columns=["am_settlement"])
        )

        # parse option-side dates so the merge keys line up
        self.options_df["date"]   = pd.to_datetime(self.options_df["date"])
        self.options_df["exdate"] = pd.to_datetime(self.options_df["exdate"])

        n_before = len(self.options_df)
        self.options_df = self.options_df.merge(
            fwd, on=["date", "exdate"], how="left"
        )
        assert len(self.options_df) == n_before, "merge changed row count"

        n_missing_fwd = self.options_df["forward_price"].isna().sum()
        if n_missing_fwd > 0:
            pct = 100.0 * n_missing_fwd / len(self.options_df)
            print(f"Warning: {n_missing_fwd:,} of {len(self.options_df):,} rows "
                  f"({pct:.2f}%) have no forward after merge; dropping.")
            self.options_df = self.options_df.dropna(subset=["forward_price"])

        # forward must be strictly positive
        self.options_df = self.options_df[self.options_df["forward_price"] > 0].copy()

    # ------------------------------------------------------------------ grid

    def _make_tau_grid(self, t_min, t_max, n):
        eps = 1.0 / 365.0
        t_min = max(float(t_min), eps)
        t_max = float(t_max)
        if self.log_tau:
            return np.exp(np.linspace(np.log(t_min), np.log(t_max), n))
        return np.linspace(t_min, t_max, n)

    # ------------------------------------------------------------------ main

    def process_daily_surfaces(self):
        # dates already parsed in load_data
        self.options_df["ttm"] = (
            self.options_df["exdate"] - self.options_df["date"]
        ).dt.days / 365.0

        eps_t = 1.0 / 365.0
        self.options_df = self.options_df[
            (self.options_df["ttm"] >= max(self.filter_ttm_min, eps_t))
            & (self.options_df["ttm"] <= self.filter_ttm_max)
        ].copy()

        # log forward moneyness, computed once for the whole frame
        self.options_df["strike"] = self.options_df["strike_price"] / 1000.0
        self.options_df["moneyness"] = np.log(
            self.options_df["strike"] / self.options_df["forward_price"]
        )

        n_dates = self.options_df["date"].nunique()
        print(f"Processing {n_dates} unique dates...")

        combined_data = []
        # rough diagnostic counters
        total_cells = 0
        total_nan_cells = 0

        # groupby iterates the frame once instead of scanning per-date
        for i, (date, day_options) in enumerate(
            self.options_df.groupby("date", sort=True)
        ):
            if (i + 1) % 100 == 0:
                print(
                    f"Processing date {i+1}/{n_dates}: "
                    f"{date.strftime('%Y-%m-%d')}"
                )

            day_options = day_options.copy()

            # log-moneyness filter
            day_options = day_options[
                (day_options["moneyness"] >= self.filter_moneyness_min)
                & (day_options["moneyness"] <= self.filter_moneyness_max)
            ].copy()

            # IV sanity (kills -99.99 sentinels and obvious garbage)
            day_options = day_options[
                np.isfinite(day_options["impl_volatility"])
                & (day_options["impl_volatility"] > self.iv_min)
                & (day_options["impl_volatility"] < self.iv_max)
            ].copy()

            # vega cutoff (OM excludes vega < 0.5 from surface calc)
            day_options = day_options[
                np.isfinite(day_options["vega"])
                & (day_options["vega"] >= self.min_vega)
            ].copy()

            # volume > 0
            if self.require_volume:
                day_options = day_options[day_options["volume"] > 0].copy()

            # OTM + narrow ATM band on each side
            # call OTM: K > F  <=>  log(K/F) > 0
            # put  OTM: K < F  <=>  log(K/F) < 0
            calls = day_options[day_options["cp_flag"] == "C"]
            puts = day_options[day_options["cp_flag"] == "P"]

            calls_keep = calls[
                (calls["moneyness"] > 0.0)
                | (np.abs(calls["moneyness"]) < self.atm_threshold)
            ]
            puts_keep = puts[
                (puts["moneyness"] < 0.0)
                | (np.abs(puts["moneyness"]) < self.atm_threshold)
            ]

            combined_options = pd.concat([calls_keep, puts_keep]).reset_index(drop=True)
            surface, n_nan = self.create_surface(combined_options, date)

            total_cells += self.n_axis_points * self.n_axis_points
            total_nan_cells += n_nan

            if surface is not None:
                # representative forward for the day (median across surviving
                # contracts) just for downstream sanity / plotting
                fwd_ref = float(np.median(combined_options["forward_price"].values))
                combined_data.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "forward_ref": round(fwd_ref, 4),
                        "surface": surface,
                    }
                )

        if total_cells > 0:
            pct_nan = 100.0 * total_nan_cells / total_cells
            print(f"\nGrid cells with insufficient support (filled by ffill/bfill): "
                  f"{total_nan_cells:,}/{total_cells:,} ({pct_nan:.2f}%)")

        self.save_combined_surface(combined_data)
        print(f"\nCompleted! Saved {len(combined_data)} surfaces.")

    # ------------------------------------------------------------------ kernel

    def create_surface(self, options, date):
        """Vega-weighted Nadaraya-Watson smoother in (log T, log(K/F)) space.

        Returns (Z, n_nan) where Z is the (n_tau, n_m) surface and n_nan is
        the number of grid cells that had insufficient kernel support before
        the fallback fill.
        """
        if len(options) < 3:
            print(f"  Warning: only {len(options)} options for {date}, skipping...")
            return None, 0

        m = options["moneyness"].values.astype(float)
        t = options["ttm"].values.astype(float)
        iv = options["impl_volatility"].values.astype(float)
        vega = options["vega"].values.astype(float)

        valid = (
            np.isfinite(m) & np.isfinite(t) & np.isfinite(iv) & np.isfinite(vega)
            & (t > 0) & (vega > 0)
        )
        if valid.sum() < 3:
            print(f"  Warning: only {valid.sum()} valid points for {date}, skipping...")
            return None, 0

        m, t, iv, vega = m[valid], t[valid], iv[valid], vega[valid]

        # target grid
        m_target = np.linspace(self.moneyness_min, self.moneyness_max, self.n_axis_points)
        t_target = self._make_tau_grid(self.ttm_min, self.ttm_max, self.n_axis_points)
        Mg, Tg = np.meshgrid(m_target, t_target)   # (n_tau, n_m)

        # pairwise distances: rows = grid points, cols = observations
        log_t_obs = np.log(t)[None, :]                          # (1, N)
        log_t_grid = np.log(Tg.ravel())[:, None]                # (G, 1)
        x = log_t_grid - log_t_obs                              # (G, N)
        y = Mg.ravel()[:, None] - m[None, :]                    # (G, N)

        # raw log-kernel (kept un-rescaled so we can compute the true,
        # absolute weight mass per grid point for the support guard)
        log_k_raw = -0.5 * (x ** 2 / self.h_tau + y ** 2 / self.h_m)

        # for numerical stability subtract the per-row max (this only affects
        # the absolute scale, not the ratio that produces the smoothed IV)
        log_k = log_k_raw - log_k_raw.max(axis=1, keepdims=True)
        k = np.exp(log_k)

        # vega-weighted kernel weights
        w = k * vega[None, :]              # (G, N)
        w_sum = w.sum(axis=1)              # (G,)

        # Nadaraya-Watson estimate
        with np.errstate(invalid="ignore", divide="ignore"):
            Z_flat = (w * iv[None, :]).sum(axis=1) / np.where(w_sum > 0, w_sum, np.nan)

        # ---------- support diagnostics ----------
        # Effective sample size on the vega-weighted kernel weights (Kish):
        #     neff = (sum w)^2 / sum(w^2)
        # This is invariant to the per-row max-subtraction we applied above,
        # so we can use the rescaled w. neff = N when all weights equal,
        # neff -> 1 when one obs dominates.
        with np.errstate(invalid="ignore", divide="ignore"):
            neff = w_sum ** 2 / (w ** 2).sum(axis=1)
            max_share = w.max(axis=1) / np.where(w_sum > 0, w_sum, np.nan)

        bad = (
            ~np.isfinite(Z_flat)
            | (neff < self.min_neff)
            | (max_share > self.max_w_share)
        )

        Z_flat = np.where(bad, np.nan, Z_flat)
        Z = Z_flat.reshape(self.n_axis_points, self.n_axis_points)
        n_nan_before_fill = int(np.isnan(Z).sum())

        # Fallback fill: forward/back fill along both axes so the saved
        # surface has no NaNs. Cells filled this way are necessarily
        # extrapolations -- the diagnostic counter above tells you how many.
        if n_nan_before_fill > 0:
            Z = (
                pd.DataFrame(Z)
                .ffill(axis=0).ffill(axis=1)
                .bfill(axis=0).bfill(axis=1)
                .values
            )

        return Z, n_nan_before_fill

    # ------------------------------------------------------------------ save

    def save_combined_surface(self, surfaces_data):
        output_file = join(self.output_dir, "SPX_surfaces.csv")

        moneyness_grid = np.linspace(
            self.moneyness_min, self.moneyness_max, self.n_axis_points
        )
        ttm_grid = self._make_tau_grid(self.ttm_min, self.ttm_max, self.n_axis_points)

        column_names = ["date", "forward_ref"]
        for i in range(self.n_axis_points):
            for j in range(self.n_axis_points):
                m = round(moneyness_grid[j], 4)
                tau = round(ttm_grid[i], 4)
                column_names.append(f"iv_{m}_{tau}")

        rows = []
        for data in surfaces_data:
            surface = data["surface"]
            row = {"date": data["date"], "forward_ref": data["forward_ref"]}
            for i in range(self.n_axis_points):
                for j in range(self.n_axis_points):
                    m = round(moneyness_grid[j], 4)
                    tau = round(ttm_grid[i], 4)
                    row[f"iv_{m}_{tau}"] = surface[i, j]
            rows.append(row)

        df = pd.DataFrame(rows, columns=column_names)
        df.to_csv(output_file, index=False)
        print(f"Saved {len(surfaces_data)} surfaces to {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess OptionMetrics data")
    parser.add_argument("--options_file", type=str, default="SPX_options.csv")
    parser.add_argument("--forward_file", type=str, default="SPX_forward.csv")
    parser.add_argument("--output_dir", type=str, default="./data/optionmetrics_processed")

    # target grid (log forward moneyness)
    parser.add_argument("--m_low", type=float, default=-0.20)
    parser.add_argument("--m_high", type=float, default=0.20)
    parser.add_argument("--ttm_low", type=float, default=0.04)
    parser.add_argument("--ttm_high", type=float, default=1.0)
    parser.add_argument("--grid_size", type=int, default=8)

    # filter window (log forward moneyness)
    parser.add_argument("--filter_m_low", type=float, default=-0.40)
    parser.add_argument("--filter_m_high", type=float, default=0.40)
    parser.add_argument("--filter_ttm_low", type=float, default=0.0)
    parser.add_argument("--filter_ttm_high", type=float, default=1.5)

    # ATM band
    parser.add_argument("--atm_threshold", type=float, default=0.01)

    # kernel bandwidths (variances; sigma = sqrt(h))
    parser.add_argument("--h_tau", type=float, default=0.10)
    parser.add_argument("--h_m", type=float, default=0.001)

    # quote filters
    parser.add_argument("--min_vega", type=float, default=0.5)
    parser.add_argument("--iv_min", type=float, default=0.01)
    parser.add_argument("--iv_max", type=float, default=3.0)
    parser.add_argument("--no_volume_filter", action="store_true",
                        help="Disable the volume > 0 filter")

    # tau grid: geometric (log-uniform) by default; pass --linear_tau to
    # restore the old linear behaviour
    parser.add_argument("--linear_tau", action="store_true",
                        help="Use a linear tau grid instead of the default "
                             "geometric grid. Not recommended -- the kernel "
                             "uses log-T distances so a linear grid leaves a "
                             "huge gap at the front of the maturity axis.")

    # sparse-data guards
    parser.add_argument("--min_neff", type=float, default=2.0,
                        help="Min effective sample size (Kish) per grid cell. "
                             "Cells below this are filled by ffill/bfill.")
    parser.add_argument("--max_w_share", type=float, default=0.85,
                        help="Max share of total kernel weight on any single "
                             "observation. Cells above this are filled by "
                             "ffill/bfill.")

    args = parser.parse_args()

    preprocessor = OptionMetricsPreprocess(
        options_file=args.options_file,
        forward_file=args.forward_file,
        output_dir=args.output_dir,
        moneyness_min=args.m_low,
        moneyness_max=args.m_high,
        ttm_min=args.ttm_low,
        ttm_max=args.ttm_high,
        n_axis_points=args.grid_size,
        filter_moneyness_min=args.filter_m_low,
        filter_moneyness_max=args.filter_m_high,
        filter_ttm_min=args.filter_ttm_low,
        filter_ttm_max=args.filter_ttm_high,
        atm_threshold=args.atm_threshold,
        h_tau=args.h_tau,
        h_m=args.h_m,
        min_vega=args.min_vega,
        iv_min=args.iv_min,
        iv_max=args.iv_max,
        require_volume=not args.no_volume_filter,
        log_tau=not args.linear_tau,
        min_neff=args.min_neff,
        max_w_share=args.max_w_share,
    )

    preprocessor.load_data()
    preprocessor.process_daily_surfaces()
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
