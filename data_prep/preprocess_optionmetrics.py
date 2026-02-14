"""
Preprocess OptionMetrics data to create volatility surfaces
Fit on wider domain, sample on target grid
"""
import numpy as np
import pandas as pd
from scipy import interpolate
from numpy import meshgrid, linspace
import os
from os.path import join
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator



class OptionMetricsPreprocess:
    def __init__(
        self,
        price_file,
        options_file,
        output_dir,
        moneyness_min=0.9,
        moneyness_max=1.1,
        ttm_min=0.04,
        ttm_max=1.0,
        n_axis_points=20,
        fit_moneyness_min=0.85,
        fit_moneyness_max=1.15,
        fit_ttm_min=0.0,
        fit_ttm_max=1.5,
        fit_grid_mult=4,
        gauss_sigma_tau=1.0,
        gauss_sigma_m=1.0,
        log_tau=True,

    ):
        self.price_file = price_file
        self.options_file = options_file
        self.output_dir = output_dir

        self.moneyness_min = moneyness_min
        self.moneyness_max = moneyness_max
        self.ttm_min = ttm_min
        self.ttm_max = ttm_max
        self.n_axis_points = n_axis_points

        self.fit_moneyness_min = fit_moneyness_min
        self.fit_moneyness_max = fit_moneyness_max
        self.fit_ttm_min = fit_ttm_min
        self.fit_ttm_max = fit_ttm_max
        self.fit_grid_mult = fit_grid_mult
        self.gauss_sigma_tau = gauss_sigma_tau
        self.gauss_sigma_m = gauss_sigma_m
        self.log_tau = log_tau


        os.makedirs(output_dir, exist_ok=True)

    def load_data(self):
        print("Loading price data...")
        self.price_df = pd.read_csv(self.price_file)
        self.price_df["date"] = pd.to_datetime(self.price_df["date"])
        self.price_df["price"] = (self.price_df["high"] + self.price_df["low"]) / 2
        self.price_dict = dict(zip(self.price_df["date"], self.price_df["price"]))

        print("Loading options data...")
        chunks = []
        chunk_size = 1000000
        for chunk in pd.read_csv(self.options_file, chunksize=chunk_size):
            chunks.append(chunk)
        self.options_df = pd.concat(chunks, ignore_index=True)
        print(f"Loaded {len(self.options_df):,} option records")

    def process_daily_surfaces(self):
        self.options_df["date"] = pd.to_datetime(self.options_df["date"])
        self.options_df["exdate"] = pd.to_datetime(self.options_df["exdate"])
        self.options_df["ttm"] = (self.options_df["exdate"] - self.options_df["date"]).dt.days / 365.0

        eps_t = 1.0 / 365.0
        self.options_df = self.options_df[
            (self.options_df["ttm"] >= max(self.fit_ttm_min, eps_t)) & (self.options_df["ttm"] <= self.fit_ttm_max)
        ].copy()

        unique_dates = sorted(self.options_df["date"].unique())
        print(f"Processing {len(unique_dates)} unique dates...")

        combined_data = []

        for i, date in enumerate(unique_dates):
            if (i + 1) % 100 == 0:
                print(f"Processing date {i+1}/{len(unique_dates)}: {date.strftime('%Y-%m-%d')}")

            if date not in self.price_dict:
                print(f"Warning: No price data for {date}, skipping...")
                continue

            underlying_price = round(self.price_dict[date], 3)
            day_options = self.options_df[self.options_df["date"] == date].copy()

            day_options["strike"] = day_options["strike_price"] / 1000.0
            day_options["moneyness"] = day_options["strike"] / underlying_price

            day_options = day_options[
                (day_options["moneyness"] >= self.fit_moneyness_min) & (day_options["moneyness"] <= self.fit_moneyness_max)
            ].copy()

            day_options = day_options[day_options["volume"] > 0].copy()

            atm_threshold = 0.01
            calls = day_options[day_options["cp_flag"] == "C"].copy()
            puts = day_options[day_options["cp_flag"] == "P"].copy()

            if len(calls) > 0:
                calls_otm = calls[calls["moneyness"] > 1.0].copy()
                calls_atm = calls[np.abs(calls["moneyness"] - 1.0) < atm_threshold].copy()
                calls_filtered = pd.concat([calls_otm, calls_atm]).drop_duplicates()
            else:
                calls_filtered = calls.copy()

            if len(puts) > 0:
                puts_otm = puts[puts["moneyness"] < 1.0].copy()
                puts_atm = puts[np.abs(puts["moneyness"] - 1.0) < atm_threshold].copy()
                puts_filtered = pd.concat([puts_otm, puts_atm]).drop_duplicates()
            else:
                puts_filtered = puts.copy()

            combined_options = pd.concat([calls_filtered, puts_filtered]).reset_index(drop=True)
            combined_surface = self.create_surface(combined_options, date, underlying_price, "Combined")

            if combined_surface is not None:
                combined_data.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "underlying_price": underlying_price,
                        "surface": combined_surface,
                        "calls": calls_filtered,
                        "puts": puts_filtered,
                    }
                )

        self.save_combined_surface(combined_data)
        print(f"\nCompleted! Saved {len(combined_data)} combined surfaces to SPX_surfaces.csv")

    def _make_tau_grid(self, t_min, t_max, n):
        eps = 1.0 / 365.0
        t_min = max(float(t_min), eps)
        t_max = float(t_max)
        if self.log_tau:
            return np.exp(np.linspace(np.log(t_min), np.log(t_max), n))
        return np.linspace(t_min, t_max, n)

    
    def create_surface(self, options, date, underlying_price, option_type):
        if len(options) < 3:
            print(f"  Warning: Only {len(options)} {option_type} options for {date}, skipping...")
            return None

        m = options["moneyness"].values
        t = options["ttm"].values
        iv = options["impl_volatility"].values

        valid = np.isfinite(iv) & np.isfinite(m) & np.isfinite(t)
        if valid.sum() < 3:
            print(f"  Warning: Only {valid.sum()} valid IVs for {option_type} on {date}, skipping...")
            return None

        m = m[valid]
        t = t[valid]
        iv = iv[valid]

        # --- target grid (what you save) ---
        m_target = np.linspace(self.moneyness_min, self.moneyness_max, self.n_axis_points)
        t_target = self._make_tau_grid(self.ttm_min, self.ttm_max, self.n_axis_points)
        Xt, Yt = np.meshgrid(m_target, t_target)   # shape (n_tau, n_m)

        # --- wide fit grid (what you smooth) ---
        n_fit = int(self.n_axis_points * self.fit_grid_mult)
        m_fit = np.linspace(self.fit_moneyness_min, self.fit_moneyness_max, n_fit)
        t_fit = self._make_tau_grid(self.fit_ttm_min, self.fit_ttm_max, n_fit)
        Xf, Yf = np.meshgrid(m_fit, t_fit)         # shape (n_fit_tau, n_fit_m)

        pts = np.column_stack([m, t])

        try:
            # 1) wide grid from linear interpolation
            Zf = interpolate.griddata(
                pts, iv, (Xf, Yf),
                method="linear",
                fill_value=np.nan
            )

            # 2) fill remaining holes on wide grid using nearest
            if np.isnan(Zf).any():
                Zf_nn = interpolate.griddata(
                    pts, iv, (Xf, Yf),
                    method="nearest"
                )
                Zf = np.where(np.isnan(Zf), Zf_nn, Zf)

            # 3) gaussian smooth on the wide grid
            Zf_smooth = gaussian_filter(
                Zf,
                sigma=(self.gauss_sigma_tau, self.gauss_sigma_m),
                mode="nearest"
            )

            # 4) sample the smoothed wide grid at target grid points
            rgi = RegularGridInterpolator(
                (t_fit, m_fit),
                Zf_smooth,
                bounds_error=False,
                fill_value=None
            )
            sample_pts = np.column_stack([Yt.ravel(), Xt.ravel()])  # (tau, moneyness)
            Zt = rgi(sample_pts).reshape(self.n_axis_points, self.n_axis_points)

            # final safety fill if any nans remain (should be rare)
            if np.isnan(Zt).any():
                # nearest on *target* grid using rgi by expanding to nearest with nan_to_num fallback
                Zt = pd.DataFrame(Zt).ffill(axis=0).ffill(axis=1).bfill(axis=0).bfill(axis=1).values

            return Zt

        except Exception as e:
            print(f"  Error interpolating {option_type} surface for {date}: {e}")
            return None


    def save_combined_surface(self, surfaces_data):
        output_file = join(self.output_dir, "SPX_surfaces.csv")

        moneyness_grid = linspace(self.moneyness_min, self.moneyness_max, self.n_axis_points)
        ttm_grid = self._make_tau_grid(self.ttm_min, self.ttm_max, self.n_axis_points)


        column_names = ["date", "underlying_price"]
        for i in range(self.n_axis_points):
            for j in range(self.n_axis_points):
                m = round(moneyness_grid[j], 4)
                tau = round(ttm_grid[i], 4)
                column_names.append(f"iv_{m}_{tau}")

        rows = []
        for data in surfaces_data:
            surface = data["surface"]
            row = {"date": data["date"], "underlying_price": data["underlying_price"]}
            for i in range(self.n_axis_points):
                for j in range(self.n_axis_points):
                    m = round(moneyness_grid[j], 4)
                    tau = round(ttm_grid[i], 4)
                    row[f"iv_{m}_{tau}"] = surface[i, j]
            rows.append(row)

        df = pd.DataFrame(rows, columns=column_names)
        df.to_csv(output_file, index=False)
        print(f"Saved {len(surfaces_data)} combined surfaces to {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess OptionMetrics data")
    parser.add_argument("--price_file", type=str, default="SPX_Price.csv")
    parser.add_argument("--options_file", type=str, default="SPX_options.csv")
    parser.add_argument("--output_dir", type=str, default="./data/optionmetrics_processed")

    parser.add_argument("--m_low", type=float, default=0.9)
    parser.add_argument("--m_high", type=float, default=1.1)
    parser.add_argument("--ttm_low", type=float, default=0.04)
    parser.add_argument("--ttm_high", type=float, default=1.0)
    parser.add_argument("--grid_size", type=int, default=20)

    parser.add_argument("--fit_m_low", type=float, default=0.85)
    parser.add_argument("--fit_m_high", type=float, default=1.15)
    parser.add_argument("--fit_ttm_low", type=float, default=0.0)
    parser.add_argument("--fit_ttm_high", type=float, default=1.5)

    parser.add_argument("--fit_grid_mult", type=int, default=4)
    parser.add_argument("--gauss_sigma_tau", type=float, default=1.0)
    parser.add_argument("--gauss_sigma_m", type=float, default=1.0)
    parser.add_argument("--log_tau", action="store_true")


    args = parser.parse_args()

    preprocessor = OptionMetricsPreprocess(
        price_file=args.price_file,
        options_file=args.options_file,
        output_dir=args.output_dir,
        moneyness_min=args.m_low,
        moneyness_max=args.m_high,
        ttm_min=args.ttm_low,
        ttm_max=args.ttm_high,
        n_axis_points=args.grid_size,
        fit_moneyness_min=args.fit_m_low,
        fit_moneyness_max=args.fit_m_high,
        fit_ttm_min=args.fit_ttm_low,
        fit_ttm_max=args.fit_ttm_high,
        fit_grid_mult=args.fit_grid_mult,
        gauss_sigma_tau=args.gauss_sigma_tau,
        gauss_sigma_m=args.gauss_sigma_m,
        log_tau=args.log_tau,

    )

    preprocessor.load_data()
    preprocessor.process_daily_surfaces()
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
