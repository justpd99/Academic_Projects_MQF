import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_options_data(file_path='MARUTI_OPTIONS_MERGED.csv'):
    print("Loading merged options data...")
    df = pd.read_csv(file_path, sep=',', skipinitialspace=True)
    df.columns = [col.strip() for col in df.columns]
    if 'Option type' in df.columns:
        df = df.rename(columns={'Option type': 'Option Type'})
    if 'Open Int' in df.columns:
        df = df.rename(columns={'Open Int': 'Open Interest'})
    print("Cleaned column names:", df.columns.tolist())
    return df

def convert_dates(df):
    print("Converting dates...")
    df_copy = df.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'].astype(str).str.strip(), errors='coerce', dayfirst=False)
    if 'Expiry' in df_copy.columns:
        df_copy['Expiry'] = pd.to_datetime(df_copy['Expiry'].astype(str).str.strip(), errors='coerce', dayfirst=False)
    return df_copy

def get_earnings_dates():
    # Extracted from opstra (11).csv
    return [
        pd.Timestamp('2025-04-25'),
        pd.Timestamp('2025-01-29'),
        pd.Timestamp('2024-10-29'),
        pd.Timestamp('2024-08-01'),
        pd.Timestamp('2024-04-27'),
        pd.Timestamp('2024-01-31'),
        pd.Timestamp('2023-10-27'),
        pd.Timestamp('2023-07-31'),
        pd.Timestamp('2023-04-26'),
        pd.Timestamp('2023-01-24'),
        pd.Timestamp('2022-10-28'),
        pd.Timestamp('2022-07-27'),
        pd.Timestamp('2022-04-29'),
        pd.Timestamp('2022-01-25'),
        pd.Timestamp('2021-10-27'),
        pd.Timestamp('2021-07-28'),
        pd.Timestamp('2021-04-27'),
        pd.Timestamp('2021-01-28'),
        pd.Timestamp('2020-10-29'),
        pd.Timestamp('2020-07-29'),
        pd.Timestamp('2020-05-13'),
        pd.Timestamp('2020-01-28'),
        pd.Timestamp('2019-10-24'),
        pd.Timestamp('2019-07-26'),
        pd.Timestamp('2019-04-25'),
        pd.Timestamp('2019-01-25'),
        pd.Timestamp('2018-10-25'),
        pd.Timestamp('2018-07-26'),
        pd.Timestamp('2018-04-27'),
        pd.Timestamp('2018-01-25'),
        pd.Timestamp('2017-10-27'),
        pd.Timestamp('2017-07-27'),
        pd.Timestamp('2017-04-27'),
        pd.Timestamp('2017-01-25'),
        pd.Timestamp('2016-10-27'),
        pd.Timestamp('2016-07-26'),
        pd.Timestamp('2016-04-26'),
        pd.Timestamp('2016-01-28'),
        pd.Timestamp('2015-10-27'),
        pd.Timestamp('2015-07-28'),
        pd.Timestamp('2015-04-27'),
        pd.Timestamp('2015-01-27'),
    ]

def get_stock_price_mapping(options_df):
    print("Getting stock prices from underlying values in options data...")
    mapping = options_df[['Date', 'Underlying Value']].drop_duplicates().copy()
    mapping = mapping.dropna(subset=['Date', 'Underlying Value'])
    mapping['Date'] = pd.to_datetime(mapping['Date'])
    mapping = mapping.set_index('Date').sort_index()
    return mapping

def analyze_earnings_options(options_df, earnings_dates, date_price_mapping):
    print("Analyzing earnings options...")
    results = []
    for earnings_date in earnings_dates:
        for offset, tlabel in zip([-1, 0, 1], ['T-1', 'T', 'T+1']):
            trade_date = earnings_date + pd.Timedelta(days=offset)
            for direction, opt_type in [('UP', 'CE'), ('DOWN', 'PE')]:
                row = {
                    'Earnings_Date': earnings_date.strftime('%Y-%m-%d'),
                    'Trading_Day': tlabel,
                    'Trading_Date': trade_date.strftime('%Y-%m-%d'),
                    'Stock_Close': '-',
                    'Target_Strike': '-',
                    'Actual_Strike': '-',
                    'Direction': direction,
                    'Option_Type': opt_type,
                    'Expiry': '-',
                    'Open': '-',
                    'High': '-',
                    'Low': '-',
                    'Close': '-',
                    'Volume': '-',
                    'Open_Interest': '-',
                }
                # Get stock close
                if trade_date in date_price_mapping.index:
                    val = date_price_mapping.loc[trade_date, 'Underlying Value']
                    if isinstance(val, pd.Series):
                        val = val.iloc[0]
                    if pd.notnull(val) and val != '-':
                        row['Stock_Close'] = f"{float(val):.2f}"
                        # Target strike: round to nearest 50 or 100 (as per Maruti's lot size, adjust if needed)
                        close = float(val)
                        if close > 1000:
                            target_strike = round(close / 100) * 100
                        else:
                            target_strike = round(close / 50) * 50
                        row['Target_Strike'] = f"{target_strike:.2f}"
                        # Find expiry after trade_date
                        exp_dates = options_df['Expiry'].dropna().unique()
                        exp_dates = [d for d in exp_dates if d >= trade_date]
                        if exp_dates:
                            expiry = min(exp_dates)
                            row['Expiry'] = pd.to_datetime(expiry).strftime('%Y-%m-%d')
                            # Find closest actual strike
                            strikes = options_df[(options_df['Expiry'] == expiry) & (options_df['Option Type'] == opt_type)]['Strike Price'].dropna().unique()
                            if len(strikes) > 0:
                                actual_strike = min(strikes, key=lambda x: abs(float(x) - target_strike))
                                row['Actual_Strike'] = f"{float(actual_strike):.2f}"
                                # Get option row
                                opt_row = options_df[(options_df['Date'] == trade_date) & (options_df['Expiry'] == expiry) & (options_df['Option Type'] == opt_type) & (options_df['Strike Price'] == actual_strike)]
                                if not opt_row.empty:
                                    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Open Interest']:
                                        val2 = opt_row.iloc[0][col] if col in opt_row.columns else '-'
                                        if pd.notnull(val2) and val2 != '-':
                                            if col in ['Open', 'High', 'Low', 'Close']:
                                                row[col] = f"{float(val2):.2f}"
                                            elif col in ['Volume', 'Open Interest']:
                                                row[col] = f"{float(val2):.2f}"
                                        else:
                                            row[col] = '-'
                results.append(row)
    result_df = pd.DataFrame(results)
    # Rename columns for output
    result_df = result_df.rename(columns={'Option Type': 'Option_Type', 'Open Interest': 'Open_Interest'})
    col_order = ['Earnings_Date','Trading_Day','Trading_Date','Stock_Close','Target_Strike','Actual_Strike','Direction','Option_Type','Expiry','Open','High','Low','Close','Volume','Open_Interest']
    result_df = result_df[col_order]
    # Sort by Earnings_Date, Option_Type, and Trading_Day (T-1, T, T+1)
    trading_day_order = {'T-1': 0, 'T': 1, 'T+1': 2}
    result_df['Trading_Day_Order'] = result_df['Trading_Day'].map(trading_day_order)
    result_df = result_df.sort_values(['Earnings_Date', 'Option_Type', 'Trading_Day_Order'])
    result_df = result_df.drop(columns=['Trading_Day_Order'])
    # Filter to only include earnings dates from 2015-05-18 onwards
    result_df = result_df[result_df['Earnings_Date'] >= '2015-05-18']
    # Replace any remaining NaN or None with '-'
    result_df = result_df.fillna('-').replace({None: '-'})
    result_df.to_csv('MARUTI_EARNINGS_OPTIONS_ANALYSIS_UNDERLYING_PNL.csv', index=False)
    print("Analysis complete. Output saved to MARUTI_EARNINGS_OPTIONS_ANALYSIS_UNDERLYING_PNL.csv in strict Dr. Reddy's format.")

def main():
    options_df = load_options_data()
    options_df = convert_dates(options_df)
    earnings_dates = get_earnings_dates()
    date_price_mapping = get_stock_price_mapping(options_df)
    analyze_earnings_options(options_df, earnings_dates, date_price_mapping)

if __name__ == "__main__":
    main() 