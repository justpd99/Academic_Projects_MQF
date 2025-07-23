import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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
CASH_PER_LOT = 100000

# Helper functions for best-case daily/cumulative PNL

def get_best_strategy_and_pnl(df):
    pnl_columns = [col for col in df.columns if str(col).strip().upper() in ['OO', 'OC', 'CO', 'CC']]
    last_row = df.iloc[-1]
    strategy_totals = {col: last_row[col] for col in pnl_columns if pd.notna(last_row[col]) and isinstance(last_row[col], (int, float))}
    if not strategy_totals:
        return None, None
    best_strategy = max(strategy_totals.items(), key=lambda x: x[1])[0]
    return best_strategy, df[best_strategy]

def get_worst_strategy_and_pnl(df):
    pnl_columns = [col for col in df.columns if str(col).strip().upper() in ['OO', 'OC', 'CO', 'CC']]
    last_row = df.iloc[-1]
    strategy_totals = {col: last_row[col] for col in pnl_columns if pd.notna(last_row[col]) and isinstance(last_row[col], (int, float))}
    if not strategy_totals:
        return None, None
    worst_strategy = min(strategy_totals.items(), key=lambda x: x[1])[0]
    return worst_strategy, df[worst_strategy]

def get_date_column(df):
    for col in df.columns:
        if 'date' in str(col).lower():
            return col
    for col in df.columns:
        try:
            pd.to_datetime(df[col].iloc[0])
            return col
        except:
            continue
    return None

def get_all_company_daily_pnls():
    file_path = "Net_PNL New.xlsx"
    all_dates = set()
    company_daily_pnl = {}
    for company, lot_size in COMPANY_LOT_SIZES.items():
        df = pd.read_excel(file_path, sheet_name=company)
        best_strategy, pnl_series = get_best_strategy_and_pnl(df)
        if best_strategy is None:
            continue
        date_col = get_date_column(df)
        if date_col is None:
            continue
        dates = pd.to_datetime(df[date_col], errors='coerce')
        pnl = pd.to_numeric(pnl_series, errors='coerce')
        valid = (~dates.isna()) & (~pnl.isna())
        dates = dates[valid]
        pnl = pnl[valid]
        scaled_pnl = pnl * lot_size
        daily_pnl = scaled_pnl.groupby(dates).sum()
        company_daily_pnl[company] = daily_pnl
        all_dates.update(daily_pnl.index)
    all_dates = sorted(all_dates)
    return company_daily_pnl, all_dates

def get_portfolio_daily_and_cumulative():
    company_daily_pnl, all_dates = get_all_company_daily_pnls()
    portfolio_daily = []
    for date in all_dates:
        daily_sum = sum(company_daily_pnl[company].get(date, 0) for company in company_daily_pnl)
        portfolio_daily.append(daily_sum)
    portfolio_cum = np.cumsum(portfolio_daily)
    return all_dates, portfolio_daily, portfolio_cum, company_daily_pnl

# 1. Per-Company Best-Case Bar Chart
def plot_company_bestcase_bar():
    company_daily_pnl, _ = get_all_company_daily_pnls()
    company_totals = {k: v.sum() for k, v in company_daily_pnl.items()}
    names = [k.replace('_EARNINGS_OPTIONS_ANALYSIS', '').replace('_EARNINGS_OPTIONS_ANA', '').replace('_EARNINGS_OPTIONS_ANALY', '').replace('_EARNINGS_OPTIONS_ANALYS', '').replace('_EARNINGS_OPTIONS_ANAL', '').replace('_PN', '') for k in company_totals.keys()]
    plt.figure(figsize=(12,6))
    sns.barplot(x=names, y=list(company_totals.values()))
    plt.title('Best-Case Total Profit per Company')
    plt.ylabel('Total Profit (₹)')
    plt.xlabel('Company')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('company_bestcase_bar.png', dpi=200)
    plt.show()

# 2. Per-Company Cumulative PNL Curves
def plot_company_cum_curves():
    company_daily_pnl, all_dates = get_all_company_daily_pnls()
    plt.figure(figsize=(14,8))
    for k, v in company_daily_pnl.items():
        v = v.reindex(all_dates, fill_value=0)
        cum = v.cumsum()
        plt.plot(all_dates, cum, label=k.split('_')[0])
    plt.title('Cumulative PNL per Company (Best-Case)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Profit (₹)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('company_cum_curves.png', dpi=200)
    plt.show()

# 3. Portfolio Drawdown Curve
def plot_portfolio_drawdown():
    dates, _, cum_pnl, _ = get_portfolio_daily_and_cumulative()
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max
    plt.figure(figsize=(14,6))
    plt.plot(dates, drawdown, color='red')
    plt.title('Portfolio Drawdown Over Time')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (₹)')
    plt.tight_layout()
    plt.savefig('portfolio_drawdown.png', dpi=200)
    plt.show()

# 4. Monthly/Quarterly/Yearly PNL Bar Chart
def plot_agg_pnl_bar(freq='M'):
    dates, daily, _, _ = get_portfolio_daily_and_cumulative()
    df = pd.DataFrame({'date': dates, 'pnl': daily})
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    agg = df['pnl'].resample(freq).sum()
    plt.figure(figsize=(14,6))
    agg.plot(kind='bar')
    plt.title(f'Portfolio PNL by {"Month" if freq=="M" else "Quarter" if freq=="Q" else "Year"}')
    plt.ylabel('Total PNL (₹)')
    plt.xlabel('Period')
    plt.tight_layout()
    plt.savefig(f'portfolio_pnl_{freq}_bar.png', dpi=200)
    plt.show()

# 5. Distribution of Daily Returns (Histogram)
def plot_daily_pnl_hist():
    _, daily, _, _ = get_portfolio_daily_and_cumulative()
    plt.figure(figsize=(10,6))
    plt.hist(daily, bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Daily Portfolio PNL')
    plt.xlabel('Daily PNL (₹)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('portfolio_daily_pnl_hist.png', dpi=200)
    plt.show()

# 6. Heatmap of PNL by Company and Year
def plot_company_year_heatmap():
    company_daily_pnl, _ = get_all_company_daily_pnls()
    heatmap_data = {}
    for k, v in company_daily_pnl.items():
        years = v.index.year
        yearly = v.groupby(years).sum()
        heatmap_data[k.split('_')[0]] = yearly
    df = pd.DataFrame(heatmap_data).T
    plt.figure(figsize=(12,8))
    sns.heatmap(df, annot=True, fmt='.0f', cmap='YlGnBu')
    plt.title('Company-Year PNL Heatmap')
    plt.ylabel('Company')
    plt.xlabel('Year')
    plt.tight_layout()
    plt.savefig('company_year_heatmap.png', dpi=200)
    plt.show()

# 7. Rolling Sharpe Ratio and Volatility
def plot_rolling_sharpe_vol(window=30):
    _, daily, _, _ = get_portfolio_daily_and_cumulative()
    daily = pd.Series(daily)
    roll_mean = daily.rolling(window).mean()
    roll_std = daily.rolling(window).std()
    sharpe = roll_mean / (roll_std + 1e-9)
    plt.figure(figsize=(14,6))
    plt.plot(sharpe, label='Rolling Sharpe')
    plt.plot(roll_std, label='Rolling Volatility')
    plt.title(f'Rolling {window}-Day Sharpe Ratio and Volatility')
    plt.xlabel('Day')
    plt.legend()
    plt.tight_layout()
    plt.savefig('rolling_sharpe_vol.png', dpi=200)
    plt.show()

# 8. Best vs. Worst Strategy Comparison (per company)
def plot_best_vs_worst_company():
    file_path = "Net_PNL New.xlsx"
    all_dates = set()
    best_cum = {}
    worst_cum = {}
    for company, lot_size in COMPANY_LOT_SIZES.items():
        df = pd.read_excel(file_path, sheet_name=company)
        best_strategy, best_pnl = get_best_strategy_and_pnl(df)
        worst_strategy, worst_pnl = get_worst_strategy_and_pnl(df)
        date_col = get_date_column(df)
        if best_strategy is None or worst_strategy is None or date_col is None:
            continue
        dates = pd.to_datetime(df[date_col], errors='coerce')
        best = pd.to_numeric(best_pnl, errors='coerce') * lot_size
        worst = pd.to_numeric(worst_pnl, errors='coerce') * lot_size
        valid_b = (~dates.isna()) & (~best.isna())
        valid_w = (~dates.isna()) & (~worst.isna())
        best = best[valid_b]
        worst = worst[valid_w]
        dates_b = dates[valid_b]
        dates_w = dates[valid_w]
        best = best.groupby(dates_b).sum().cumsum()
        worst = worst.groupby(dates_w).sum().cumsum()
        best_cum[company.split('_')[0]] = best
        worst_cum[company.split('_')[0]] = worst
        all_dates.update(best.index)
        all_dates.update(worst.index)
    all_dates = sorted(all_dates)
    plt.figure(figsize=(14,8))
    for k in best_cum:
        plt.plot(all_dates, best_cum[k].reindex(all_dates, method='ffill', fill_value=0), label=f'{k} Best', linestyle='-')
        plt.plot(all_dates, worst_cum[k].reindex(all_dates, method='ffill', fill_value=0), label=f'{k} Worst', linestyle='--')
    plt.title('Best vs. Worst Strategy Cumulative PNL (per company)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Profit (₹)')
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig('best_vs_worst_company.png', dpi=200)
    plt.show()

# 9. Correlation Matrix of Daily PNLs
def plot_correlation_matrix():
    company_daily_pnl, all_dates = get_all_company_daily_pnls()
    df = pd.DataFrame({k.split('_')[0]: v.reindex(all_dates, fill_value=0) for k, v in company_daily_pnl.items()}, index=all_dates)
    corr = df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Daily PNLs')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=200)
    plt.show()

# 10. Calendar Heatmap of Portfolio PNL
def plot_calendar_heatmap():
    import calmap
    dates, daily, _, _ = get_portfolio_daily_and_cumulative()
    df = pd.Series(daily, index=pd.to_datetime(dates))
    plt.figure(figsize=(16,6))
    calmap.calendarplot(df, fillcolor='lightgray', cmap='YlGn', linewidth=0.5, yearlabels=True, daylabels='MTWTFSS', dayticks=[0, 1, 2, 3, 4, 5, 6])
    plt.title('Calendar Heatmap of Portfolio Daily PNL')
    plt.tight_layout()
    plt.savefig('calendar_heatmap.png', dpi=200)
    plt.show()

# 11. Portfolio vs. Nifty 50
def plot_portfolio_vs_nifty():
    dates, _, cum_pnl, _ = get_portfolio_daily_and_cumulative()
    start = pd.to_datetime(dates[0]).strftime('%Y-%m-%d')
    end = pd.to_datetime(dates[-1]).strftime('%Y-%m-%d')
    nifty_df = yf.download('^NSEI', start=start, end=end)
    # Use 'Adj Close' if available, else 'Close'
    if 'Adj Close' in nifty_df.columns:
        nifty = nifty_df['Adj Close']
    elif 'Close' in nifty_df.columns:
        nifty = nifty_df['Close']
    else:
        raise ValueError('Nifty data does not have Adj Close or Close columns!')
    nifty = nifty.reindex(pd.to_datetime(dates), method='ffill')
    nifty_norm = (nifty / nifty.iloc[0]) * 1_500_000  # Normalize to same initial capital
    plt.figure(figsize=(14,7))
    plt.plot(dates, cum_pnl + 1_500_000, label='Portfolio (Best-Case)', linewidth=2)
    plt.plot(dates, nifty_norm, label='Nifty 50 (₹1.5L start)', linewidth=2)
    plt.title('Portfolio vs. Nifty 50 (Cumulative)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (₹)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('portfolio_vs_nifty.png', dpi=200)
    plt.show()

if __name__ == "__main__":
    plot_company_bestcase_bar()
    plot_company_cum_curves()
    plot_portfolio_drawdown()
    plot_agg_pnl_bar('M')
    plot_agg_pnl_bar('Q')
    plot_agg_pnl_bar('A')
    plot_daily_pnl_hist()
    plot_company_year_heatmap()
    plot_rolling_sharpe_vol(30)
    plot_best_vs_worst_company()
    plot_correlation_matrix()
    # plot_calendar_heatmap()  # Uncomment if calmap is installed
    plot_portfolio_vs_nifty() 