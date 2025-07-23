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
        xl_file = pd.ExcelFile(file_path)
        sheets = {}
        for sheet_name in xl_file.sheet_names:
            if sheet_name.lower() != 'p&l':
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    sheets[sheet_name] = df
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
    
    last_row = df.iloc[-1]
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
    
    best_strategy = max(strategy_totals.items(), key=lambda x: x[1])
    return best_strategy[0], best_strategy[1]

def extract_date_and_pnl(df, best_strategy_col):
    """Extract date and PNL data for the best strategy"""
    date_col = None
    for col in df.columns:
        if 'date' in str(col).lower() or 'time' in str(col).lower():
            date_col = col
            break
    
    if date_col is None:
        for col in df.columns:
            try:
                pd.to_datetime(df[col].iloc[0])
                date_col = col
                break
            except:
                continue
    
    if date_col is None:
        return None, None
    
    try:
        dates = pd.to_datetime(df[date_col], errors='coerce')
        pnl_values = pd.to_numeric(df[best_strategy_col], errors='coerce')
        
        valid_data = pd.DataFrame({
            'date': dates,
            'pnl': pnl_values
        }).dropna()
        
        return valid_data['date'], valid_data['pnl']
    except Exception as e:
        return None, None

def calculate_exact_profits(pnl_values, lot_size, cash_per_lot):
    """Calculate exact profits based on lot size and cash per lot"""
    # The PNL values are already the total accumulated profits for the strategy
    # So we return them directly without additional calculations
    return pnl_values

def analyze_portfolio():
    """Analyze the portfolio and create summary"""
    file_path = "Net_PNL New.xlsx"
    sheets = read_excel_sheets(file_path)
    
    if sheets is None:
        return None
    
    portfolio_data = {}
    summary_data = []
    
    for company, df in sheets.items():
        if company not in COMPANY_LOT_SIZES:
            continue
        
        lot_size = COMPANY_LOT_SIZES[company]
        pnl_columns = find_pnl_columns(df)
        
        if not pnl_columns:
            continue
        
        best_strategy, total_pnl = get_best_pnl_strategy(df, pnl_columns)
        
        if best_strategy is None:
            continue
        
        dates, pnl_values = extract_date_and_pnl(df, best_strategy)
        
        if dates is not None and pnl_values is not None:
            exact_profits = calculate_exact_profits(pnl_values, lot_size, CASH_PER_LOT)
            # Use the total_pnl directly as the correct profit (already accumulated)
            total_exact_profit = total_pnl
            lots_per_trade = CASH_PER_LOT / lot_size
            
            # Clean company name
            company_name = company.replace('_EARNINGS_OPTIONS_ANALYSIS', '').replace('_EARNINGS_OPTIONS_ANA', '').replace('_EARNINGS_OPTIONS_ANALY', '').replace('_EARNINGS_OPTIONS_ANALYS', '').replace('_EARNINGS_OPTIONS_ANAL', '').replace('_PN', '')
            
            portfolio_data[company] = {
                'dates': dates,
                'exact_profits': exact_profits,
                'strategy': best_strategy,
                'total_exact_profit': total_exact_profit,
                'lot_size': lot_size,
                'lots_per_trade': lots_per_trade
            }
            
            summary_data.append({
                'Company': company_name,
                'Strategy': best_strategy,
                'Lot_Size': lot_size,
                'Lots_per_Trade': round(lots_per_trade, 2),
                'Total_PNL': round(total_pnl, 2),
                'Total_Exact_Profit': round(total_exact_profit, 2),
                'Investment': CASH_PER_LOT,
                'ROI_%': round((total_exact_profit / CASH_PER_LOT) * 100, 2)
            })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Total_Exact_Profit', ascending=False)
    
    # Calculate portfolio statistics
    total_investment = len(portfolio_data) * CASH_PER_LOT
    total_profit = summary_df['Total_Exact_Profit'].sum()
    portfolio_roi = (total_profit / total_investment) * 100
    
    # Add portfolio summary row
    portfolio_summary = {
        'Company': 'PORTFOLIO TOTAL',
        'Strategy': 'EQUAL WEIGHTED',
        'Lot_Size': '-',
        'Lots_per_Trade': '-',
        'Total_PNL': '-',
        'Total_Exact_Profit': round(total_profit, 2),
        'Investment': total_investment,
        'ROI_%': round(portfolio_roi, 2)
    }
    
    summary_df = pd.concat([summary_df, pd.DataFrame([portfolio_summary])], ignore_index=True)
    
    return portfolio_data, summary_df

def create_detailed_analysis():
    """Create detailed analysis with daily profits"""
    portfolio_data, summary_df = analyze_portfolio()
    
    if portfolio_data is None:
        return None, None
    
    # Create daily profit analysis
    all_dates = set()
    for company_data in portfolio_data.values():
        all_dates.update(company_data['dates'])
    
    all_dates = sorted(list(all_dates))
    
    daily_profits = []
    for date in all_dates:
        daily_profit = 0
        valid_companies = 0
        
        for company, company_data in portfolio_data.items():
            if date in company_data['dates'].values:
                date_idx = company_data['dates'] == date
                if date_idx.any():
                    exact_profit = company_data['exact_profits'][date_idx].iloc[0]
                    daily_profit += exact_profit
                    valid_companies += 1
        
        if valid_companies > 0:
            avg_daily_profit = daily_profit / valid_companies
            daily_profits.append({
                'Date': date,
                'Total_Daily_Profit': round(daily_profit, 2),
                'Avg_Daily_Profit': round(avg_daily_profit, 2),
                'Active_Companies': valid_companies
            })
    
    daily_df = pd.DataFrame(daily_profits)
    daily_df['Cumulative_Profit'] = daily_df['Total_Daily_Profit'].cumsum()
    
    return summary_df, daily_df

def save_results():
    """Save all results to files"""
    summary_df, daily_df = create_detailed_analysis()
    
    if summary_df is not None:
        # Save summary
        summary_df.to_csv('portfolio_summary.csv', index=False)
        print("Portfolio summary saved to 'portfolio_summary.csv'")
        
        # Save daily analysis
        daily_df.to_csv('daily_profit_analysis.csv', index=False)
        print("Daily profit analysis saved to 'daily_profit_analysis.csv'")
        
        # Print summary
        print("\n=== PORTFOLIO SUMMARY ===")
        print(summary_df.to_string(index=False))
        
        return summary_df, daily_df
    
    return None, None

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

if __name__ == "__main__":
    summary_df, daily_df = save_results() 