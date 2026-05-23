"""
Preprocess OptionMetrics SPX quotes into a daily implied-volatility surface.

Coordinates: (log T, log(K/F)). Default grid 10 tau x 15 moneyness covering
log(K/F) in [-0.10, +0.10] (odd count -> centre cell at ATM-forward) and
T in [30/365, 1] year (geometric, matching the kernel's log-T metric).

Surface is a vega-weighted Nadaraya-Watson smooth of OTM + narrow-ATM-band
quotes. Bandwidths h_tau, h_m are kernel *variances* (OM manual convention).
h_m optionally ramps linearly in log(tau) from h_m at tau_min to h_m_long at
tau_max (widens the smile-direction smoothing where quote density is sparse).

The options file's am_set_flag column ships empty from WRDS, so the forward
join collapses to one row per (date, exdate) preferring PM over AM. Exact
for PM-settled options; sub-day carry bias on AM monthlies on PM-missing days.

Output: SPX_surfaces.csv with `iv_{m}_{tau}` columns ordered tau-major
(k = i_t * n_moneyness + i_m); downstream consumers reshape C-order to
(n_tau, n_moneyness).
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
        moneyness_min=-0.10,
        moneyness_max=0.10,
        ttm_min=0.0,
        ttm_max=1.0,
        n_moneyness=15,
        n_tau=10,
        filter_moneyness_min=-0.25,
        filter_moneyness_max=0.25,
        filter_ttm_min=0.0,
        filter_ttm_max=1.75,
        atm_threshold=0.02,
        h_tau=0.16,
        h_m=2.0e-4,
        h_m_long=6.0e-4,
        min_vega=0.5,
        iv_min=0.01,
        iv_max=3.0,
        require_volume=True,
        log_tau=True,
        min_neff=2.0,
        max_w_share=0.85,
    ):
        self.options_file = options_file
        self.forward_file = forward_file
        self.output_dir = output_dir

        self.moneyness_min = moneyness_min
        self.moneyness_max = moneyness_max
        self.ttm_min = ttm_min
        self.ttm_max = ttm_max
        self.n_moneyness = n_moneyness
        self.n_tau = n_tau

        if self.n_moneyness % 2 == 0:
            raise ValueError(
                f"n_moneyness must be odd; got {self.n_moneyness}."
            )
        if abs(0.5 * (self.moneyness_min + self.moneyness_max)) > 1e-12:
            raise ValueError(
                f"moneyness window must be symmetric around 0; got "
                f"[{self.moneyness_min}, {self.moneyness_max}]."
            )

        self.filter_moneyness_min = filter_moneyness_min
        self.filter_moneyness_max = filter_moneyness_max
        self.filter_ttm_min = filter_ttm_min
        self.filter_ttm_max = filter_ttm_max

        self.atm_threshold = atm_threshold

        self.h_tau = h_tau
        self.h_m = h_m
        self.h_m_long = h_m_long

        self.min_vega = min_vega
        self.iv_min = iv_min
        self.iv_max = iv_max
        self.require_volume = require_volume

        self.log_tau = log_tau

        self.min_neff = min_neff
        self.max_w_share = max_w_share

        shape = (self.n_tau, self.n_moneyness)
        self.diag = {
            "n_days":         0,
            "n_fail_total":   np.zeros(shape, dtype=np.int64),
            "n_fail_neff":    np.zeros(shape, dtype=np.int64),
            "n_fail_share":   np.zeros(shape, dtype=np.int64),
            "n_fail_div":     np.zeros(shape, dtype=np.int64),
            "neff_sum":       np.zeros(shape, dtype=np.float64),
            "neff_n":         np.zeros(shape, dtype=np.int64),
            "share_sum":      np.zeros(shape, dtype=np.float64),
            "share_n":        np.zeros(shape, dtype=np.int64),
            "nobs_in_band":   np.zeros(shape, dtype=np.int64),
        }
        # per-day fit error of the gridded surface vs the raw quotes
        self.fit_rows = []

        # Per-tau moneyness bandwidth. If h_m_long is set, ramp linearly in
        # log(tau) from h_m at tau_min to h_m_long at tau_max; this widens
        # the smile-direction smoothing in deep maturities where the quote
        # density is sparse. Flat order matches the (n_tau, n_moneyness)
        # grid in tau-major C-order (cell k = i_t * n_moneyness + i_m).
        tau_grid = self._make_tau_grid(self.ttm_min, self.ttm_max, self.n_tau)
        if self.h_m_long is None:
            self._h_m_per_tau = np.full(self.n_tau, self.h_m, dtype=np.float64)
        else:
            lt   = np.log(tau_grid)
            frac = (lt - lt.min()) / (lt.max() - lt.min())
            self._h_m_per_tau = (
                self.h_m + frac * (self.h_m_long - self.h_m)
            ).astype(np.float64)
        self._h_m_flat = np.repeat(self._h_m_per_tau, self.n_moneyness)
        sched = ", ".join(f"{h:.2e}" for h in self._h_m_per_tau)
        print(f"h_m schedule (per tau): [{sched}]")

        os.makedirs(output_dir, exist_ok=True)

    def load_data(self):
        # Forward file is small; load and dedup it up front so each options
        # chunk can be merged against it in the streaming loop below.
        print("Loading forward prices...")
        fwd = pd.read_csv(self.forward_file)
        print(f"Loaded {len(fwd):,} forward records")

        fwd = fwd.rename(columns={
            "expiration":   "exdate",
            "AMSettlement": "am_settlement",
            "ForwardPrice": "forward_price",
        })

        fwd_required = {"date", "exdate", "am_settlement", "forward_price"}
        fwd_missing = fwd_required - set(fwd.columns)
        if fwd_missing:
            raise ValueError(f"Missing required columns in forward file: {fwd_missing}")

        fwd = fwd[["date", "exdate", "am_settlement", "forward_price"]].copy()
        fwd["date"]   = pd.to_datetime(fwd["date"])
        fwd["exdate"] = pd.to_datetime(fwd["exdate"])

        n_dup = fwd.duplicated(subset=["date", "exdate", "am_settlement"]).sum()
        if n_dup > 0:
            print(f"Warning: {n_dup:,} duplicate keys in forward file; keeping first.")
            fwd = fwd.drop_duplicates(subset=["date", "exdate", "am_settlement"])

        # PM (0) sorts before AM (1) -> keep first = "PM if available, else AM"
        fwd = (
            fwd.sort_values(["date", "exdate", "am_settlement"])
               .drop_duplicates(subset=["date", "exdate"], keep="first")
               .drop(columns=["am_settlement"])
        )

        # Stream the (multi-GB) options file: each chunk is merged, range- and
        # forward-filtered and column-pruned *before* it is retained, so peak
        # memory tracks the filtered surface support, not the raw file. The
        # ttm / moneyness windows are date-independent, so applying them here
        # is equivalent to the per-day filtering done later.
        print("Loading options data in chunks...")
        usecols = ["date", "exdate", "cp_flag", "strike_price",
                   "impl_volatility", "volume", "vega"]
        dtypes  = {"cp_flag": "category", "strike_price": "float64",
                   "impl_volatility": "float32", "volume": "int64",
                   "vega": "float32"}
        keep_cols = ["date", "cp_flag", "impl_volatility", "volume", "vega",
                     "forward_price", "ttm", "moneyness"]
        eps_t  = 1.0 / 365.0
        ttm_lo = max(self.filter_ttm_min, eps_t)

        kept = []
        n_raw = n_no_fwd = 0
        for chunk in pd.read_csv(
            self.options_file, chunksize=1_000_000,
            usecols=usecols, dtype=dtypes,
        ):
            n_raw += len(chunk)
            chunk["date"]   = pd.to_datetime(chunk["date"])
            chunk["exdate"] = pd.to_datetime(chunk["exdate"])

            n_before = len(chunk)
            chunk = chunk.merge(fwd, on=["date", "exdate"], how="left")
            assert len(chunk) == n_before, "merge changed row count"

            n_no_fwd += int(chunk["forward_price"].isna().sum())
            chunk = chunk[chunk["forward_price"] > 0]  # NaN > 0 is False too

            chunk["ttm"] = (chunk["exdate"] - chunk["date"]).dt.days / 365.0
            chunk = chunk[(chunk["ttm"] >= ttm_lo)
                          & (chunk["ttm"] <= self.filter_ttm_max)]

            strike = chunk["strike_price"] / 1000.0
            chunk["moneyness"] = np.log(strike / chunk["forward_price"])
            chunk = chunk[(chunk["moneyness"] >= self.filter_moneyness_min)
                          & (chunk["moneyness"] <= self.filter_moneyness_max)]

            if len(chunk):
                kept.append(chunk[keep_cols].copy())

        self.options_df = (pd.concat(kept, ignore_index=True) if kept
                           else pd.DataFrame(columns=keep_cols))
        print(f"Loaded {n_raw:,} option records; {len(self.options_df):,} "
              f"retained after forward/ttm/moneyness filters")
        if n_no_fwd > 0:
            pct = 100.0 * n_no_fwd / max(n_raw, 1)
            print(f"Note: {n_no_fwd:,} of {n_raw:,} rows ({pct:.2f}%) had no "
                  f"forward after merge; dropped.")

    def _make_tau_grid(self, t_min, t_max, n):
        eps = 30.0 / 365.0
        t_min = max(float(t_min), eps)
        t_max = float(t_max)
        if self.log_tau:
            return np.exp(np.linspace(np.log(t_min), np.log(t_max), n))
        return np.linspace(t_min, t_max, n)

    def process_daily_surfaces(self):
        # ttm / moneyness and their range filters are applied per-chunk in
        # load_data; here we only group the retained quotes into surfaces.
        n_dates = self.options_df["date"].nunique()
        print(f"Processing {n_dates} unique dates...")

        combined_data = []
        total_cells = 0
        total_nan_cells = 0

        for i, (date, day_options) in enumerate(
            self.options_df.groupby("date", sort=True)
        ):
            if (i + 1) % 100 == 0:
                print(f"Processing date {i+1}/{n_dates}: "
                      f"{date.strftime('%Y-%m-%d')}")

            # Wing-relaxed filters: |m| > 0.07 keeps thinly-traded OTM quotes
            # (looser vega cutoff, volume not required) to feed the otherwise
            # sparse wing cells. Dense ATM core keeps the strict filters.
            base_ok = (
                np.isfinite(day_options["impl_volatility"])
                & (day_options["impl_volatility"] > self.iv_min)
                & (day_options["impl_volatility"] < self.iv_max)
                & np.isfinite(day_options["vega"])
            )
            in_window = ((day_options["moneyness"] >= self.filter_moneyness_min)
                         & (day_options["moneyness"] <= self.filter_moneyness_max))

            is_wing     = day_options["moneyness"].abs() > 0.07
            strict_vega = day_options["vega"] >= self.min_vega
            loose_vega  = day_options["vega"] >= 0.1
            vega_ok     = np.where(is_wing, loose_vega, strict_vega)

            vol_ok = (~self.require_volume) | (day_options["volume"] > 0) | is_wing

            day = day_options[base_ok & in_window & vega_ok & vol_ok]

            # OTM only, plus a narrow ATM band on each side
            calls = day[day["cp_flag"] == "C"]
            puts  = day[day["cp_flag"] == "P"]
            calls_keep = calls[(calls["moneyness"] > 0.0)
                               | (np.abs(calls["moneyness"]) < self.atm_threshold)]
            puts_keep  = puts[(puts["moneyness"] < 0.0)
                              | (np.abs(puts["moneyness"]) < self.atm_threshold)]
            combined = pd.concat([calls_keep, puts_keep]).reset_index(drop=True)

            surface, n_nan = self.create_surface(combined, date)
            total_cells += self.n_tau * self.n_moneyness
            total_nan_cells += n_nan

            if surface is not None:
                fwd_ref = float(np.median(combined["forward_price"].values))
                combined_data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "forward_ref": round(fwd_ref, 4),
                    "surface": surface,
                })
                self._accumulate_fit(date, combined, surface)

        if total_cells > 0:
            pct_nan = 100.0 * total_nan_cells / total_cells
            print(f"\nGrid cells filled by ffill/bfill (insufficient support): "
                  f"{total_nan_cells:,}/{total_cells:,} ({pct_nan:.2f}%)")

        self.save_combined_surface(combined_data)
        self.save_diagnostics()
        self.save_fit_diagnostics()
        print(f"\nCompleted! Saved {len(combined_data)} surfaces.")

    def create_surface(self, options, date):
        """Vega-weighted Nadaraya-Watson smooth at every target grid point.

        sigma_hat(j) = sum_i V_i sigma_i phi(x_ij, y_ij) / sum_i V_i phi(...)
        phi(x, y) = exp(-x^2 / (2 h_tau) - y^2 / (2 h_m))  with h's as variances.
        """
        if len(options) < 3:
            print(f"  Warning: only {len(options)} options for {date}, skipping...")
            return None, 0

        m    = options["moneyness"].values.astype(float)
        t    = options["ttm"].values.astype(float)
        iv   = options["impl_volatility"].values.astype(float)
        vega = options["vega"].values.astype(float)

        valid = (np.isfinite(m) & np.isfinite(t) & np.isfinite(iv) & np.isfinite(vega)
                 & (t > 0) & (vega > 0))
        if valid.sum() < 3:
            print(f"  Warning: only {valid.sum()} valid points for {date}, skipping...")
            return None, 0
        m, t, iv, vega = m[valid], t[valid], iv[valid], vega[valid]

        m_target = np.linspace(self.moneyness_min, self.moneyness_max, self.n_moneyness)
        t_target = self._make_tau_grid(self.ttm_min, self.ttm_max, self.n_tau)
        Mg, Tg = np.meshgrid(m_target, t_target)

        x = np.log(Tg.ravel())[:, None] - np.log(t)[None, :]
        y = Mg.ravel()[:, None] - m[None, :]

        # h_m varies per tau row; broadcast against the flat grid axis.
        h_m_col = self._h_m_flat[:, None]
        log_k = -0.5 * (x ** 2 / self.h_tau + y ** 2 / h_m_col)
        # softmax-stabilise; ratios (and Kish neff) are unaffected.
        log_k -= log_k.max(axis=1, keepdims=True)
        w = np.exp(log_k) * vega[None, :]
        w_sum = w.sum(axis=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            Z_flat = (w * iv[None, :]).sum(axis=1) / np.where(w_sum > 0, w_sum, np.nan)
            # Guards: Kish neff catches "one obs dominates"; max_share is the
            # stricter single-obs cap. Cells failing either get ffill/bfill'd.
            neff      = w_sum ** 2 / (w ** 2).sum(axis=1)
            max_share = w.max(axis=1) / np.where(w_sum > 0, w_sum, np.nan)

        fail_div   = ~np.isfinite(Z_flat)
        fail_neff  = np.where(np.isfinite(neff),      neff < self.min_neff,        False)
        fail_share = np.where(np.isfinite(max_share), max_share > self.max_w_share, False)
        bad = fail_div | fail_neff | fail_share

        # accumulate per-cell diagnostics
        shape = (self.n_tau, self.n_moneyness)
        self.diag["n_days"]       += 1
        self.diag["n_fail_total"] += bad.reshape(shape).astype(np.int64)
        self.diag["n_fail_div"]   += fail_div.reshape(shape).astype(np.int64)
        self.diag["n_fail_neff"]  += fail_neff.reshape(shape).astype(np.int64)
        self.diag["n_fail_share"] += fail_share.reshape(shape).astype(np.int64)
        good = ~bad
        if good.any():
            self.diag["neff_sum"]  += np.where(good, neff,      0.0).reshape(shape)
            self.diag["neff_n"]    += good.reshape(shape).astype(np.int64)
            self.diag["share_sum"] += np.where(good, max_share, 0.0).reshape(shape)
            self.diag["share_n"]   += good.reshape(shape).astype(np.int64)
        # observations falling inside ~2 sigma of each cell (raw support count)
        sig_logT = np.sqrt(self.h_tau)
        sig_m    = np.sqrt(self._h_m_flat)[:, None]
        in_band  = (np.abs(x) <= 2.0 * sig_logT) & (np.abs(y) <= 2.0 * sig_m)
        self.diag["nobs_in_band"] += in_band.sum(axis=1).reshape(shape)

        Z = np.where(bad, np.nan, Z_flat).reshape(self.n_tau, self.n_moneyness)
        n_nan = int(np.isnan(Z).sum())
        if n_nan > 0:
            Z = (pd.DataFrame(Z)
                   .ffill(axis=0).ffill(axis=1)
                   .bfill(axis=0).bfill(axis=1)
                   .values)
        return Z, n_nan

    def save_combined_surface(self, surfaces_data):
        output_file = join(self.output_dir, "SPX_surfaces.csv")

        m_grid = np.linspace(self.moneyness_min, self.moneyness_max, self.n_moneyness)
        t_grid = self._make_tau_grid(self.ttm_min, self.ttm_max, self.n_tau)

        # Column order is part of the contract with downstream consumers:
        # tau outer, moneyness inner (k = i_t * n_moneyness + i_m).
        columns = ["date", "forward_ref"]
        for i in range(self.n_tau):
            for j in range(self.n_moneyness):
                columns.append(f"iv_{round(m_grid[j], 4)}_{round(t_grid[i], 4)}")

        rows = []
        for d in surfaces_data:
            row = {"date": d["date"], "forward_ref": d["forward_ref"]}
            surf = d["surface"]
            for i in range(self.n_tau):
                for j in range(self.n_moneyness):
                    row[f"iv_{round(m_grid[j], 4)}_{round(t_grid[i], 4)}"] = surf[i, j]
            rows.append(row)

        pd.DataFrame(rows, columns=columns).to_csv(output_file, index=False)
        print(f"Saved {len(surfaces_data)} surfaces to {output_file}")

    def save_diagnostics(self):
        m_grid = np.linspace(self.moneyness_min, self.moneyness_max, self.n_moneyness)
        t_grid = self._make_tau_grid(self.ttm_min, self.ttm_max, self.n_tau)
        d = self.diag
        n_days = max(d["n_days"], 1)
        rows = []
        for i in range(self.n_tau):
            for j in range(self.n_moneyness):
                neff_n  = d["neff_n"][i, j]
                share_n = d["share_n"][i, j]
                rows.append({
                    "i_tau": i, "j_moneyness": j,
                    "tau": round(float(t_grid[i]), 6),
                    "moneyness": round(float(m_grid[j]), 6),
                    "fill_rate":       d["n_fail_total"][i, j] / n_days,
                    "fail_neff_rate":  d["n_fail_neff"][i, j]  / n_days,
                    "fail_share_rate": d["n_fail_share"][i, j] / n_days,
                    "fail_div_rate":   d["n_fail_div"][i, j]   / n_days,
                    "mean_neff":  (d["neff_sum"][i, j]  / neff_n)  if neff_n  > 0 else float("nan"),
                    "mean_share": (d["share_sum"][i, j] / share_n) if share_n > 0 else float("nan"),
                    "mean_nobs_in_2sigma_band": d["nobs_in_band"][i, j] / n_days,
                })
        out = join(self.output_dir, "support_diagnostics.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"Saved per-cell support diagnostics to {out}")

    def _accumulate_fit(self, date, options, surface):
        """Per-day fit error: bilinearly interpolate the gridded surface (in the
        kernel's own log-T / m coordinates) to each raw quote and compare to the
        quoted IV. Restricted to quotes inside the grid box, since the surface
        only spans the target window."""
        m_t  = np.linspace(self.moneyness_min, self.moneyness_max, self.n_moneyness)
        t_t  = self._make_tau_grid(self.ttm_min, self.ttm_max, self.n_tau)
        lt_t = np.log(t_t)

        m  = options["moneyness"].values.astype(float)
        t  = options["ttm"].values.astype(float)
        iv = options["impl_volatility"].values.astype(float)
        vg = options["vega"].values.astype(float)
        lt = np.log(np.where(t > 0, t, np.nan))

        inbox = (np.isfinite(m) & np.isfinite(lt) & np.isfinite(iv)
                 & (m >= m_t[0]) & (m <= m_t[-1])
                 & (lt >= lt_t[0]) & (lt <= lt_t[-1]))
        n_in = int(inbox.sum())
        if n_in == 0:
            return
        m, lt, iv, vg = m[inbox], lt[inbox], iv[inbox], vg[inbox]

        jm = np.clip(np.searchsorted(m_t, m) - 1, 0, self.n_moneyness - 2)
        jt = np.clip(np.searchsorted(lt_t, lt) - 1, 0, self.n_tau - 2)
        wm = (m - m_t[jm])  / (m_t[jm + 1]  - m_t[jm])
        wt = (lt - lt_t[jt]) / (lt_t[jt + 1] - lt_t[jt])
        z = (surface[jt,     jm    ] * (1 - wt) * (1 - wm)
             + surface[jt,     jm + 1] * (1 - wt) * wm
             + surface[jt + 1, jm    ] * wt       * (1 - wm)
             + surface[jt + 1, jm + 1] * wt       * wm)
        r = z - iv

        atm    = np.abs(m) < 0.03
        vg_sum = float(vg.sum())
        self.fit_rows.append({
            "date":      date.strftime("%Y-%m-%d"),
            "n_quotes":  n_in,
            "rmse":      float(np.sqrt(np.mean(r ** 2))),
            "mae":       float(np.mean(np.abs(r))),
            "bias":      float(np.mean(r)),
            "rmse_vw":   float(np.sqrt(np.sum(vg * r ** 2) / vg_sum)) if vg_sum > 0 else float("nan"),
            "rmse_atm":  float(np.sqrt(np.mean(r[atm] ** 2)))   if atm.any()    else float("nan"),
            "rmse_wing": float(np.sqrt(np.mean(r[~atm] ** 2)))  if (~atm).any() else float("nan"),
        })

    def save_fit_diagnostics(self):
        if not self.fit_rows:
            return
        out = join(self.output_dir, "fit_diagnostics.csv")
        df = pd.DataFrame(self.fit_rows)
        df.to_csv(out, index=False)
        print(f"Saved per-day fit diagnostics to {out}")
        print(f"  mean daily RMSE vs quotes: {df['rmse'].mean():.5f}  "
              f"(ATM {df['rmse_atm'].mean():.5f}, wing {df['rmse_wing'].mean():.5f})")


def main():
    import argparse

    p = argparse.ArgumentParser(description="Preprocess OptionMetrics data")
    p.add_argument("--options_file", type=str, default="SPX_options.csv")
    p.add_argument("--forward_file", type=str, default="SPX_forward.csv")
    p.add_argument("--output_dir",   type=str, default="./data/optionmetrics_processed")

    p.add_argument("--m_low",   type=float, default=-0.10)
    p.add_argument("--m_high",  type=float, default=0.10)
    p.add_argument("--ttm_low", type=float, default=0.0)
    p.add_argument("--ttm_high", type=float, default=1.0)
    p.add_argument("--n_moneyness", type=int, default=15,
                   help="Must be odd so the centre cell sits at ATM-forward.")
    p.add_argument("--n_tau",       type=int, default=10)

    p.add_argument("--filter_m_low",    type=float, default=-0.25)
    p.add_argument("--filter_m_high",   type=float, default=0.25)
    p.add_argument("--filter_ttm_low",  type=float, default=0.0)
    p.add_argument("--filter_ttm_high", type=float, default=1.75)

    p.add_argument("--atm_threshold", type=float, default=0.02)

    p.add_argument("--h_tau", type=float, default=0.16)
    p.add_argument("--h_m",   type=float, default=2.0e-4,
                   help="Moneyness bandwidth at the short maturity (tau_min).")
    p.add_argument("--h_m_long", type=float, default=6.0e-4,
                   help="h_m at tau_max; ramped linearly in log(tau) from "
                        "--h_m. Pass --h_m_long <same as --h_m> for constant.")

    p.add_argument("--min_vega", type=float, default=0.5)
    p.add_argument("--iv_min",   type=float, default=0.01)
    p.add_argument("--iv_max",   type=float, default=3.0)
    p.add_argument("--no_volume_filter", action="store_true")
    p.add_argument("--linear_tau",       action="store_true",
                   help="Use a linear tau grid (kernel uses log-T, so not recommended).")

    p.add_argument("--min_neff",    type=float, default=2.0)
    p.add_argument("--max_w_share", type=float, default=0.85)

    args = p.parse_args()

    pre = OptionMetricsPreprocess(
        options_file=args.options_file,
        forward_file=args.forward_file,
        output_dir=args.output_dir,
        moneyness_min=args.m_low,
        moneyness_max=args.m_high,
        ttm_min=args.ttm_low,
        ttm_max=args.ttm_high,
        n_moneyness=args.n_moneyness,
        n_tau=args.n_tau,
        filter_moneyness_min=args.filter_m_low,
        filter_moneyness_max=args.filter_m_high,
        filter_ttm_min=args.filter_ttm_low,
        filter_ttm_max=args.filter_ttm_high,
        atm_threshold=args.atm_threshold,
        h_tau=args.h_tau,
        h_m=args.h_m,
        h_m_long=args.h_m_long,
        min_vega=args.min_vega,
        iv_min=args.iv_min,
        iv_max=args.iv_max,
        require_volume=not args.no_volume_filter,
        log_tau=not args.linear_tau,
        min_neff=args.min_neff,
        max_w_share=args.max_w_share,
    )
    pre.load_data()
    pre.process_daily_surfaces()
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
