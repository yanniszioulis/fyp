"""
Preprocess OptionMetrics data to create volatility surfaces
Adapted for SPX_Price.csv and SPX_options.csv format
"""
import numpy as np
import pandas as pd
from scipy import interpolate
from numpy import meshgrid, linspace
import os
from os.path import join
from datetime import datetime as dt


class OptionMetricsPreprocess:
    def __init__(self, price_file, options_file, output_dir, 
                 moneyness_min=0.9, moneyness_max=1.1, 
                 ttm_min=0.04, ttm_max=1.0, 
                 n_axis_points=20):
        """
        Args:
            price_file: Path to SPX_Price.csv
            options_file: Path to SPX_options.csv
            output_dir: Directory to save output CSV files
            moneyness_min, moneyness_max: Moneyness range [0.9, 1.1]
            ttm_min, ttm_max: Time to maturity range [0.04, 1.0] years
            n_axis_points: Grid size (default 20x20)
        """
        self.price_file = price_file
        self.options_file = options_file
        self.output_dir = output_dir
        self.moneyness_min = moneyness_min
        self.moneyness_max = moneyness_max
        self.ttm_min = ttm_min
        self.ttm_max = ttm_max
        self.n_axis_points = n_axis_points
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self):
        """Load and prepare price and options data"""
        print("Loading price data...")
        self.price_df = pd.read_csv(self.price_file)
        self.price_df['date'] = pd.to_datetime(self.price_df['date'])
        # Calculate mid price as (high + low) / 2
        self.price_df['price'] = (self.price_df['high'] + self.price_df['low']) / 2
        # Create lookup dictionary for fast access
        self.price_dict = dict(zip(self.price_df['date'], self.price_df['price']))
        
        print("Loading options data...")
        # Read options file in chunks if it's large
        chunks = []
        chunk_size = 1000000  # 1M rows at a time
        for chunk in pd.read_csv(self.options_file, chunksize=chunk_size):
            chunks.append(chunk)
        
        self.options_df = pd.concat(chunks, ignore_index=True)
        print(f"Loaded {len(self.options_df):,} option records")
        
    def process_daily_surfaces(self):
        """Process each day and create volatility surfaces for calls and puts"""
        # Convert dates
        self.options_df['date'] = pd.to_datetime(self.options_df['date'])
        self.options_df['exdate'] = pd.to_datetime(self.options_df['exdate'])
        
        # Calculate TTM in years
        self.options_df['ttm'] = (self.options_df['exdate'] - self.options_df['date']).dt.days / 365.0
        
        # Filter by TTM range
        ttm_filter = (self.options_df['ttm'] >= self.ttm_min) & (self.options_df['ttm'] <= self.ttm_max)
        self.options_df = self.options_df[ttm_filter].copy()
        
        # Get unique dates
        unique_dates = sorted(self.options_df['date'].unique())
        print(f"Processing {len(unique_dates)} unique dates...")
        
        # Prepare output dataframes
        calls_data = []
        puts_data = []
        
        for i, date in enumerate(unique_dates):
            if (i + 1) % 100 == 0:
                print(f"Processing date {i+1}/{len(unique_dates)}: {date.strftime('%Y-%m-%d')}")
            
            # Get underlying price for this date
            if date not in self.price_dict:
                print(f"Warning: No price data for {date}, skipping...")
                continue
            
            underlying_price = round(self.price_dict[date], 3)
            
            # Filter options for this date
            day_options = self.options_df[self.options_df['date'] == date].copy()
            
            # Convert strike from multiplied by 1000 to actual strike
            day_options['strike'] = day_options['strike_price'] / 1000.0
            
            # Calculate moneyness
            day_options['moneyness'] = day_options['strike'] / underlying_price
            
            # Filter by moneyness range
            moneyness_filter = (day_options['moneyness'] >= self.moneyness_min) & \
                              (day_options['moneyness'] <= self.moneyness_max)
            day_options = day_options[moneyness_filter].copy()
            
            # Filter out zero volume (optional - you may want to keep this)
            day_options = day_options[day_options['volume'] > 0].copy()
            
            # Process calls and puts separately
            calls = day_options[day_options['cp_flag'] == 'C'].copy()
            puts = day_options[day_options['cp_flag'] == 'P'].copy()
            
            # Create surfaces
            calls_surface = self.create_surface(calls, date, underlying_price, 'C')
            puts_surface = self.create_surface(puts, date, underlying_price, 'P')
            
            if calls_surface is not None:
                calls_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'underlying_price': underlying_price,
                    'surface': calls_surface
                })
            
            if puts_surface is not None:
                puts_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'underlying_price': underlying_price,
                    'surface': puts_surface
                })
        
        # Save to CSV files
        self.save_surfaces(calls_data, 'calls')
        self.save_surfaces(puts_data, 'puts')
        
        print(f"\nCompleted! Saved {len(calls_data)} call surfaces and {len(puts_data)} put surfaces")
        
    def create_surface(self, options, date, underlying_price, option_type):
        """
        Create interpolated volatility surface for a given day
        
        Args:
            options: DataFrame with options for this day and type
            date: Trading date
            underlying_price: Underlying price on this date
            option_type: 'C' or 'P'
            
        Returns:
            Interpolated surface as 2D numpy array (n_axis_points x n_axis_points)
        """
        if len(options) < 3:  # Need at least 3 points for interpolation
            print(f"  Warning: Only {len(options)} {option_type} options for {date}, skipping...")
            return None
        
        # Get moneyness, ttm, and IV
        moneyness = options['moneyness'].values
        ttm = options['ttm'].values
        iv = options['impl_volatility'].values
        
        # Filter out NaN IVs
        valid_mask = ~np.isnan(iv)
        if valid_mask.sum() < 3:
            print(f"  Warning: Only {valid_mask.sum()} valid IVs for {option_type} on {date}, skipping...")
            return None
        
        moneyness = moneyness[valid_mask]
        ttm = ttm[valid_mask]
        iv = iv[valid_mask]
        
        # Create grid for interpolation
        moneyness_grid = linspace(self.moneyness_min, self.moneyness_max, self.n_axis_points)
        ttm_grid = linspace(self.ttm_min, self.ttm_max, self.n_axis_points)
        X, Y = meshgrid(moneyness_grid, ttm_grid)
        
        # Interpolate using cubic spline (griddata with cubic method)
        try:
            Z = interpolate.griddata(
                np.array([moneyness, ttm]).T, 
                iv, 
                (X, Y), 
                method='cubic',
                fill_value=np.nan  # Fill missing with NaN
            )
            
            # Handle NaN values: forward fill, backward fill
            Z_df = pd.DataFrame(Z)
            Z_df = Z_df.ffill(axis=0).ffill(axis=1).bfill(axis=0).bfill(axis=1)
            Z = Z_df.values
            
            # If still NaN, use nearest neighbor for remaining
            if np.isnan(Z).any():
                Z_nearest = interpolate.griddata(
                    np.array([moneyness, ttm]).T,
                    iv,
                    (X, Y),
                    method='nearest'
                )
                Z = np.where(np.isnan(Z), Z_nearest, Z)
            
            return Z
            
        except Exception as e:
            print(f"  Error interpolating {option_type} surface for {date}: {e}")
            return None
    
    def save_surfaces(self, surfaces_data, option_type):
        """
        Save surfaces to CSV file
        
        Args:
            surfaces_data: List of dicts with 'date', 'underlying_price', 'surface'
            option_type: 'calls' or 'puts'
        """
        output_file = join(self.output_dir, f'SPX_{option_type}_surfaces.csv')
        
        # Calculate moneyness and ttm grid values
        moneyness_grid = linspace(self.moneyness_min, self.moneyness_max, self.n_axis_points)
        ttm_grid = linspace(self.ttm_min, self.ttm_max, self.n_axis_points)
        
        # Generate column names
        column_names = ['date', 'underlying_price']
        for i in range(self.n_axis_points):  # TTM axis (rows)
            for j in range(self.n_axis_points):  # Moneyness axis (columns)
                m = round(moneyness_grid[j], 4)  # Round moneyness to 4dp
                tau = round(ttm_grid[i], 4)  # Round ttm to 4dp
                column_names.append(f'iv_{m}_{tau}')
        
        rows = []
        for data in surfaces_data:
            surface = data['surface']  # shape: (n_axis_points, n_axis_points)
            
            # Create row with date, underlying_price, and flattened surface
            row = {
                'date': data['date'],
                'underlying_price': data['underlying_price']
            }
            
            # Add surface values as columns (flatten row-wise: i=ttm, j=moneyness)
            for i in range(self.n_axis_points):
                for j in range(self.n_axis_points):
                    m = round(moneyness_grid[j], 4)
                    tau = round(ttm_grid[i], 4)
                    row[f'iv_{m}_{tau}'] = surface[i, j]
            
            rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows, columns=column_names)
        df.to_csv(output_file, index=False)
        print(f"Saved {len(surfaces_data)} {option_type} surfaces to {output_file}")


def main():
    """Main processing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess OptionMetrics data')
    parser.add_argument('--price_file', type=str, default='SPX_Price.csv',
                       help='Path to SPX_Price.csv')
    parser.add_argument('--options_file', type=str, default='SPX_options.csv',
                       help='Path to SPX_options.csv')
    parser.add_argument('--output_dir', type=str, default='./data/optionmetrics_processed',
                       help='Output directory for processed surfaces')
    parser.add_argument('--m_low', type=float, default=0.9, help='Moneyness minimum')
    parser.add_argument('--m_high', type=float, default=1.1, help='Moneyness maximum')
    parser.add_argument('--ttm_low', type=float, default=0.04, help='TTM minimum (years)')
    parser.add_argument('--ttm_high', type=float, default=1.0, help='TTM maximum (years)')
    parser.add_argument('--grid_size', type=int, default=20, help='Grid size (n_axis_points)')
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = OptionMetricsPreprocess(
        price_file=args.price_file,
        options_file=args.options_file,
        output_dir=args.output_dir,
        moneyness_min=args.m_low,
        moneyness_max=args.m_high,
        ttm_min=args.ttm_low,
        ttm_max=args.ttm_high,
        n_axis_points=args.grid_size
    )
    
    # Process
    preprocessor.load_data()
    preprocessor.process_daily_surfaces()
    
    print("\nPreprocessing complete!")


if __name__ == '__main__':
    main()
