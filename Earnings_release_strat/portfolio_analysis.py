import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Company lot sizes and cash requirements
COMPANY_LOT_SIZES = {
    'ASIANPAINT_EARNINGS_OPTIONS_ANA': 200,
    'BHARTIARTL_EARNINGS_OPTIONS_ANA': 475,
    'DRREDDY_EARNINGS_OPTIONS_ANALYS': 625,
    'GRASIM_EARNINGS_OPTIONS_ANALYSI': 250,
    'POWERGRID_EARNINGS_OPTIONS_ANAL': 1800,
    'TATASTEEL_EARNINGS_OPTIONS_ANAL': 5500,
    'TECHM_EARNINGS_OPTIONS_ANALYSIS': 600,
    'RELIANCE_EARNINGS_OPTIONS_ANALY': 500,
    'TCS_EARNINGS_OPTIONS_ANALYSIS_P': 175,
    'HDFCBANK_EARNINGS_OPTIONS_ANALY': 550,
    'HINDUNILVR_EARNINGS_OPTIONS_ANA': 300,
    'ICICIBANK_EARNINGS_OPTIONS_ANAL': 700,
    'INFY_EARNINGS_OPTIONS_ANALYSIS_': 400,
    'LT_EARNINGS_OPTIONS_ANALYSIS_PN': 150,
    'AXISBANK_EARNINGS_OPTIONS_ANALY': 625
}

# Constants
CASH_PER_LOT = 100000  # 1 Lakh per lot
TOTAL_NOTIONAL = 150000000  # 15 Crore

def read_excel_sheets(file_path):
    """Read all sheets from the Excel file"""
    try:
        # Read all sheet names
        xl_file = pd.ExcelFile(file_path)
        print(f"Found {len(xl_file.sheet_names)} sheets: {xl_file.sheet_names}")
        
        # Read all sheets into a dictionary
        sheets = {}
        for sheet_name in xl_file.sheet_names:
            if sheet_name.lower() != 'p&l':  # Skip the total PNL sheet
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    sheets[sheet_name] = df
                    print(f"Successfully read sheet: {sheet_name} with shape {df.shape}")
                except Exception as e:
                    print(f"Error reading sheet {sheet_name}: {e}")
        
        return sheets
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

def find_pnl_columns(df):
    """Find the PNL columns (OO, OC, CO, CC) in the dataframe"""
    pnl_columns = []
    for col in df.columns:
        if str(col).strip().upper() in ['OO', 'OC', 'CO', 'CC']:
            pnl_columns.append(col)
    return pnl_columns

def get_best_pnl_strategy(df, pnl_columns):
    """Find the best performing PNL strategy"""
    if not pnl_columns:
        return None, None
    
    # Get the actual last row (total PNL) - make sure we get the last non-null row
    last_row = df.iloc[-1]
    
    # If the last row has NaN values, try to find the last row with actual data
    if last_row.isna().all():
        # Find the last row that has at least one non-null value in PNL columns
        for i in range(len(df) - 1, -1, -1):
            row = df.iloc[i]
            if any(pd.notna(row[col]) for col in pnl_columns):
                last_row = row
                break
    
    # Calculate total PNL for each strategy
    strategy_totals = {}
    for col in pnl_columns:
        try:
            total_pnl = last_row[col]
            if pd.notna(total_pnl) and isinstance(total_pnl, (int, float)):
                strategy_totals[col] = total_pnl
        except:
            continue
    
    if not strategy_totals:
        return None, None
    
    # Find the best strategy
    best_strategy = max(strategy_totals.items(), key=lambda x: x[1])
    return best_strategy[0], best_strategy[1]

def extract_date_and_pnl(df, best_strategy_col):
    """Extract date and PNL data for the best strategy"""
    # Look for date column
    date_col = None
    for col in df.columns:
        if 'date' in str(col).lower() or 'time' in str(col).lower():
            date_col = col
            break
    
    if date_col is None:
        # Try to find any column that might contain dates
        for col in df.columns:
            try:
                pd.to_datetime(df[col].iloc[0])
                date_col = col
                break
            except:
                continue
    
    if date_col is None:
        print("Could not find date column")
        return None, None
    
    # Extract date and PNL data
    try:
        dates = pd.to_datetime(df[date_col], errors='coerce')
        pnl_values = pd.to_numeric(df[best_strategy_col], errors='coerce')
        
        # Remove rows with NaN values
        valid_data = pd.DataFrame({
            'date': dates,
            'pnl': pnl_values
        }).dropna()
        
        return valid_data['date'], valid_data['pnl']
    except Exception as e:
        print(f"Error extracting data: {e}")
        return None, None

def calculate_exact_profits(pnl_values, lot_size, cash_per_lot):
    """Calculate exact profits based on lot size and cash per lot"""
    # The PNL values are already the total accumulated profits for the strategy
    # So we return them directly without additional calculations
    return pnl_values

def create_equal_weighted_portfolio(sheets_data):
    """Create equal weighted portfolio from all companies with exact profit calculations"""
    portfolio_data = {}
    
    for company, df in sheets_data.items():
        print(f"\nProcessing {company}...")
        
        # Check if company has lot size defined
        if company not in COMPANY_LOT_SIZES:
            print(f"No lot size defined for {company}, skipping...")
            continue
        
        lot_size = COMPANY_LOT_SIZES[company]
        print(f"Lot size for {company}: {lot_size}")
        
        # Find PNL columns
        pnl_columns = find_pnl_columns(df)
        print(f"Found PNL columns: {pnl_columns}")
        
        if not pnl_columns:
            print(f"No PNL columns found for {company}")
            continue
        
        # Get best strategy
        best_strategy, total_pnl = get_best_pnl_strategy(df, pnl_columns)
        print(f"Best strategy for {company}: {best_strategy} with total PNL: {total_pnl}")
        
        if best_strategy is None:
            print(f"Could not determine best strategy for {company}")
            continue
        
        # Extract date and PNL data
        dates, pnl_values = extract_date_and_pnl(df, best_strategy)
        
        if dates is not None and pnl_values is not None:
            # Calculate exact profits for individual trades
            exact_profits = calculate_exact_profits(pnl_values, lot_size, CASH_PER_LOT)
            
            # Use the total accumulated profit directly
            total_exact_profit = total_pnl
            
            portfolio_data[company] = {
                'dates': dates,
                'pnl': pnl_values,
                'exact_profits': exact_profits,
                'strategy': best_strategy,
                'total_pnl': total_pnl,
                'total_exact_profit': total_exact_profit,
                'lot_size': lot_size,
                'lots_per_trade': CASH_PER_LOT / lot_size
            }
            print(f"Successfully extracted data for {company}")
            print(f"Lots per trade: {CASH_PER_LOT / lot_size:.2f}")
            print(f"Total exact profit: ₹{total_exact_profit:,.2f}")
        else:
            print(f"Failed to extract data for {company}")
    
    return portfolio_data

def calculate_portfolio_returns(portfolio_data):
    """Calculate equal weighted portfolio returns with exact profits"""
    if not portfolio_data:
        return None, None
    
    # Get all unique dates
    all_dates = set()
    for company_data in portfolio_data.values():
        all_dates.update(company_data['dates'])
    
    all_dates = sorted(list(all_dates))
    
    # Calculate daily portfolio returns
    portfolio_returns = []
    portfolio_dates = []
    
    for date in all_dates:
        daily_return = 0
        valid_companies = 0
        
        for company_data in portfolio_data.values():
            if date in company_data['dates'].values:
                # Find the exact profit value for this date
                date_idx = company_data['dates'] == date
                if date_idx.any():
                    exact_profit = company_data['exact_profits'][date_idx].iloc[0]
                    daily_return += exact_profit
                    valid_companies += 1
        
        if valid_companies > 0:
            # Use total daily return (sum) instead of average
            portfolio_returns.append(daily_return)
            portfolio_dates.append(date)
    
    return portfolio_dates, portfolio_returns

def plot_portfolio(portfolio_dates, portfolio_returns, portfolio_data):
    """Plot the equal weighted portfolio with exact profits"""
    if not portfolio_dates or not portfolio_returns:
        print("No data to plot")
        return
    
    # Convert to pandas Series for easier manipulation
    portfolio_series = pd.Series(portfolio_returns, index=portfolio_dates)
    
    # Calculate cumulative returns
    cumulative_returns = portfolio_series.cumsum()
    
    # Create the plot
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Daily Returns
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_dates, portfolio_returns, linewidth=1, alpha=0.7, color='blue')
    plt.title('Equal Weighted Portfolio - Daily Exact Profits (₹)', fontsize=14, fontweight='bold')
    plt.ylabel('Daily Profit (₹)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative Returns
    plt.subplot(2, 1, 2)
    plt.plot(portfolio_dates, cumulative_returns, linewidth=2, color='green')
    plt.title('Equal Weighted Portfolio - Cumulative Exact Profits (₹)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Profit (₹)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add summary statistics
    total_return = cumulative_returns.iloc[-1] if len(cumulative_returns) > 0 else 0
    avg_daily_return = np.mean(portfolio_returns) if portfolio_returns else 0
    std_daily_return = np.std(portfolio_returns) if portfolio_returns else 0
    
    # Calculate total investment and ROI using total accumulated profits
    total_investment = len(portfolio_data) * CASH_PER_LOT
    total_profit = sum(data['total_exact_profit'] for data in portfolio_data.values())
    roi_percentage = (total_profit / total_investment) * 100 if total_investment > 0 else 0
    
    plt.figtext(0.02, 0.02, 
                f'Total Profit: ₹{total_profit:,.2f}\n'
                f'Avg Daily Profit: ₹{avg_daily_return:,.2f}\n'
                f'Std Daily Profit: ₹{std_daily_return:,.2f}\n'
                f'Total Investment: ₹{total_investment:,.2f}\n'
                f'ROI: {roi_percentage:.2f}%', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n=== PORTFOLIO SUMMARY ===")
    print(f"Number of companies: {len(portfolio_data)}")
    print(f"Total Investment: ₹{total_investment:,.2f}")
    print(f"Total Profit: ₹{total_profit:,.2f}")
    print(f"ROI: {roi_percentage:.2f}%")
    print(f"Average Daily Profit: ₹{avg_daily_return:,.2f}")
    print(f"Standard Deviation of Daily Profit: ₹{std_daily_return:,.2f}")
    
    # Print individual company performance
    print(f"\n=== INDIVIDUAL COMPANY PERFORMANCE ===")
    for company, data in portfolio_data.items():
        company_name = company.replace('_EARNINGS_OPTIONS_ANALYSIS', '').replace('_EARNINGS_OPTIONS_ANA', '').replace('_EARNINGS_OPTIONS_ANALY', '').replace('_EARNINGS_OPTIONS_ANALYS', '').replace('_EARNINGS_OPTIONS_ANAL', '').replace('_PN', '')
        print(f"{company_name}: {data['strategy']} strategy, Lot Size: {data['lot_size']}, "
              f"Lots per Trade: {data['lots_per_trade']:.2f}, Total Profit: ₹{data['total_exact_profit']:,.2f}")

def main():
    # File path
    file_path = "Net_PNL New.xlsx"
    
    print("Reading Excel file...")
    sheets = read_excel_sheets(file_path)
    
    if sheets is None:
        print("Failed to read Excel file")
        return
    
    print(f"\nProcessing {len(sheets)} company sheets...")
    print(f"Cash per lot: ₹{CASH_PER_LOT:,.2f}")
    print(f"Total notional: ₹{TOTAL_NOTIONAL:,.2f}")
    
    # Create portfolio data
    portfolio_data = create_equal_weighted_portfolio(sheets)
    
    if not portfolio_data:
        print("No valid portfolio data found")
        return
    
    print(f"\nSuccessfully processed {len(portfolio_data)} companies")
    
    # Calculate portfolio returns
    portfolio_dates, portfolio_returns = calculate_portfolio_returns(portfolio_data)
    
    # Plot the portfolio
    plot_portfolio(portfolio_dates, portfolio_returns, portfolio_data)

if __name__ == "__main__":
    main() 