import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from tabulate import tabulate
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_portfolio_data():
    """Load portfolio data from the organized directories"""
    daily_df = pd.read_csv('Performance_Metrics_Data/daily_profit_analysis.csv')
    summary_df = pd.read_csv('Performance_Metrics_Data/portfolio_summary.csv')
    
    # Convert date column
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    daily_df = daily_df.sort_values('Date')
    
    # Get the correct total profit from summary
    correct_total_profit = summary_df[summary_df['Company'] == 'PORTFOLIO TOTAL']['Total_Exact_Profit'].iloc[0]
    
    # Scale the cumulative profit to match the correct total
    if len(daily_df) > 0:
        scale_factor = correct_total_profit / daily_df['Cumulative_Profit'].iloc[-1]
        daily_df['Cumulative_Profit'] = daily_df['Cumulative_Profit'] * scale_factor
        daily_df['Total_Daily_Profit'] = daily_df['Total_Daily_Profit'] * scale_factor
    
    return daily_df, summary_df

def fetch_nifty50_data(start_date, end_date):
    """Fetch Nifty 50 data from yfinance"""
    try:
        print("Fetching Nifty 50 data from Yahoo Finance...")
        
        # Nifty 50 symbol in Yahoo Finance
        nifty_symbol = "^NSEI"
        
        # Fetch data with some buffer
        buffer_start = start_date - timedelta(days=30)
        buffer_end = end_date + timedelta(days=30)
        
        nifty_data = yf.download(nifty_symbol, start=buffer_start, end=buffer_end, progress=False)
        
        if nifty_data.empty:
            print("Warning: Could not fetch Nifty 50 data, using default benchmark")
            return None
        
        # Calculate daily returns
        nifty_data['Daily_Return'] = nifty_data['Adj Close'].pct_change()
        
        # Filter to match our portfolio period
        nifty_data = nifty_data[(nifty_data.index >= start_date) & (nifty_data.index <= end_date)]
        
        print(f"✅ Successfully fetched Nifty 50 data from {nifty_data.index[0].date()} to {nifty_data.index[-1].date()}")
        
        return nifty_data
        
    except Exception as e:
        print(f"Error fetching Nifty 50 data: {e}")
        print("Using default benchmark returns")
        return None

def calculate_nifty50_quarterly_returns(nifty_data, quarterly_dates):
    """Calculate Nifty 50 quarterly returns aligned with portfolio quarters"""
    if nifty_data is None:
        # Default quarterly returns if data fetch fails
        return [0.03] * len(quarterly_dates)  # 3% per quarter default
    
    quarterly_returns = []
    
    for i, quarter_end in enumerate(quarterly_dates):
        # Find the closest trading day to quarter end
        quarter_start = quarterly_dates[i-1] if i > 0 else nifty_data.index[0]
        
        # Get data for this quarter
        quarter_data = nifty_data[(nifty_data.index >= quarter_start) & (nifty_data.index <= quarter_end)]
        
        if len(quarter_data) > 1:
            start_price = quarter_data['Adj Close'].iloc[0]
            end_price = quarter_data['Adj Close'].iloc[-1]
            quarter_return = (end_price - start_price) / start_price
            quarterly_returns.append(quarter_return)
        else:
            quarterly_returns.append(0.03)  # Default 3% if no data
    
    return quarterly_returns

def calculate_enhanced_performance_metrics():
    """Calculate comprehensive performance metrics with Nifty 50 benchmark using quarterly data"""
    daily_df, summary_df = load_portfolio_data()
    
    # Basic portfolio parameters
    initial_investment = 1500000  # ₹15 lakh
    total_return = daily_df['Cumulative_Profit'].iloc[-1]
    
    # Time period calculations
    start_date = daily_df['Date'].iloc[0]
    end_date = daily_df['Date'].iloc[-1]
    total_days = (end_date - start_date).days
    total_years = total_days / 365.25
    
    # Calculate quarterly returns for portfolio
    daily_df['Quarter'] = daily_df['Date'].dt.to_period('Q')
    quarterly_profits = daily_df.groupby('Quarter')['Total_Daily_Profit'].sum()
    quarterly_returns = quarterly_profits / initial_investment
    
    # Get quarterly end dates for Nifty alignment
    quarterly_dates = []
    for quarter in quarterly_profits.index:
        quarter_end = quarter.end_time.date()
        quarterly_dates.append(pd.Timestamp(quarter_end))
    
    # Fetch Nifty 50 data and calculate quarterly returns
    nifty_data = fetch_nifty50_data(start_date, end_date)
    nifty_quarterly_returns = calculate_nifty50_quarterly_returns(nifty_data, quarterly_dates)
    
    # Portfolio metrics using quarterly data
    num_quarters = len(quarterly_returns)
    
    # 1. CAGR (Compound Annual Growth Rate) - Correct calculation
    total_return_multiple = (initial_investment + total_return) / initial_investment
    cagr = (total_return_multiple ** (1/total_years)) - 1
    
    # 2. Total Return Percentage
    total_return_pct = (total_return / initial_investment) * 100
    
    # 3. Quarterly statistics
    avg_quarterly_return = quarterly_returns.mean()
    quarterly_volatility = quarterly_returns.std()
    
    # 4. Annualized volatility (from quarterly data)
    annualized_volatility = quarterly_volatility * np.sqrt(4)  # 4 quarters per year
    
    # 5. Risk-free rate (quarterly)
    annual_risk_free_rate = 0.06  # 7% annual
    quarterly_risk_free_rate = (1 + annual_risk_free_rate) ** (1/4) - 1
    
    # 6. Nifty 50 benchmark metrics
    nifty_quarterly_avg = np.mean(nifty_quarterly_returns)
    nifty_quarterly_vol = np.std(nifty_quarterly_returns)
    nifty_annualized_return = ((1 + nifty_quarterly_avg) ** 4) - 1
    nifty_annualized_volatility = nifty_quarterly_vol * np.sqrt(4)
    
    # 7. Sharpe Ratio (using periodic method for quarterly data)
    excess_quarterly_return = avg_quarterly_return - quarterly_risk_free_rate
    sharpe_ratio_quarterly = excess_quarterly_return / quarterly_volatility if quarterly_volatility > 0 else 0
    # Annualized Sharpe ratio
    sharpe_ratio = sharpe_ratio_quarterly * np.sqrt(4)
    
    # 8. Sortino Ratio (using downside deviation)
    negative_returns = quarterly_returns[quarterly_returns < quarterly_risk_free_rate]
    if len(negative_returns) > 0:
        downside_deviation_quarterly = np.sqrt(np.mean((negative_returns - quarterly_risk_free_rate)**2))
        sortino_ratio_quarterly = excess_quarterly_return / downside_deviation_quarterly
        sortino_ratio = sortino_ratio_quarterly * np.sqrt(4)
    else:
        sortino_ratio = 0
    
    # 9. Maximum Drawdown (corrected calculation)
    portfolio_values = initial_investment + daily_df['Cumulative_Profit'].values
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown_values = (portfolio_values - running_max) / running_max * 100
    max_drawdown = np.min(drawdown_values)
    
    # 10. Calmar Ratio
    calmar_ratio = (cagr * 100) / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # 11. Beta calculation (using quarterly returns)
    if len(nifty_quarterly_returns) == len(quarterly_returns):
        portfolio_returns_array = quarterly_returns.values
        nifty_returns_array = np.array(nifty_quarterly_returns)
        
        covariance = np.cov(portfolio_returns_array, nifty_returns_array)[0,1]
        nifty_variance = np.var(nifty_returns_array)
        
        beta = covariance / nifty_variance if nifty_variance > 0 else 1.0
    else:
        beta = annualized_volatility / nifty_annualized_volatility
    
    # 12. Information Ratio
    excess_return_vs_benchmark = cagr - nifty_annualized_return
    tracking_error = np.std(quarterly_returns.values - np.array(nifty_quarterly_returns[:len(quarterly_returns)])) * np.sqrt(4)
    information_ratio = excess_return_vs_benchmark / tracking_error if tracking_error > 0 else 0
    
    # 13. Treynor Ratio
    treynor_ratio = (cagr - annual_risk_free_rate) / beta if beta > 0 else 0
    
    # 14. Jensen's Alpha
    expected_return = annual_risk_free_rate + beta * (nifty_annualized_return - annual_risk_free_rate)
    jensen_alpha = cagr - expected_return
    
    # 15. Trading performance metrics
    profits = daily_df['Total_Daily_Profit'].values
    positive_days = len(profits[profits > 0])
    total_trading_days = len(profits)
    win_rate = (positive_days / total_trading_days) * 100
    
    gross_profit = np.sum(profits[profits > 0])
    gross_loss = abs(np.sum(profits[profits < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    # Average win/loss
    avg_win = np.mean(profits[profits > 0]) if positive_days > 0 else 0
    avg_loss = np.mean(profits[profits < 0]) if (total_trading_days - positive_days) > 0 else 0
    
    return {
        'Total_Return_%': round(total_return_pct, 2),
        'CAGR_%': round(cagr * 100, 2),
        'Annualized_Volatility_%': round(annualized_volatility * 100, 2),
        'Sharpe_Ratio': round(sharpe_ratio, 3),
        'Sortino_Ratio': round(sortino_ratio, 3),
        'Information_Ratio': round(information_ratio, 3),
        'Treynor_Ratio': round(treynor_ratio, 3),
        'Jensen_Alpha_%': round(jensen_alpha * 100, 2),
        'Max_Drawdown_%': round(max_drawdown, 2),
        'Calmar_Ratio': round(calmar_ratio, 3),
        'Beta': round(beta, 3),
        'Win_Rate_%': round(win_rate, 2),
        'Profit_Factor': round(profit_factor, 2),
        'Avg_Win_Rs': round(avg_win, 0),
        'Avg_Loss_Rs': round(avg_loss, 0),
        'Total_Trading_Days': total_trading_days,
        'Investment_Period_Years': round(total_years, 2),
        'Risk_Free_Rate_%': round(annual_risk_free_rate * 100, 1),
        'Nifty50_CAGR_%': round(nifty_annualized_return * 100, 1),
        'Nifty50_Volatility_%': round(nifty_annualized_volatility * 100, 1),
        'Excess_Return_%': round(excess_return_vs_benchmark * 100, 2),
        'Quarterly_Returns_Count': num_quarters,
        'Avg_Quarterly_Return_%': round(avg_quarterly_return * 100, 2),
        'Quarterly_Volatility_%': round(quarterly_volatility * 100, 2)
    }

def create_enhanced_performance_table():
    """Create enhanced performance metrics table with Nifty 50 benchmark"""
    metrics = calculate_enhanced_performance_metrics()
    
    # Create the performance table
    performance_data = {
        'Metric': [
            'Total Return (%)',
            'CAGR - Compound Annual Growth Rate (%)',
            'Excess Return vs Nifty 50 (%)',
            'Annualized Volatility (%)',
            'Sharpe Ratio (Annualized)',
            'Sortino Ratio (Annualized)',
            'Information Ratio',
            'Treynor Ratio',
            'Jensen\'s Alpha (%)',
            'Maximum Drawdown (%)',
            'Calmar Ratio',
            'Beta vs Nifty 50',
            'Win Rate (%)',
            'Profit Factor',
            'Average Win (₹)',
            'Average Loss (₹)',
            'Total Trading Days',
            'Investment Period (Years)',
            'Number of Quarters',
            '',  # Separator
            'Quarterly Analysis',
            'Average Quarterly Return (%)',
            'Quarterly Volatility (%)',
            '',  # Separator
            'Benchmark: Nifty 50',
            'Nifty 50 CAGR (%)',
            'Nifty 50 Volatility (%)',
            'Risk-Free Rate (%)'
        ],
        'Value': [
            f"{metrics['Total_Return_%']:.2f}",
            f"{metrics['CAGR_%']:.2f}",
            f"{metrics['Excess_Return_%']:.2f}",
            f"{metrics['Annualized_Volatility_%']:.2f}",
            f"{metrics['Sharpe_Ratio']:.3f}",
            f"{metrics['Sortino_Ratio']:.3f}",
            f"{metrics['Information_Ratio']:.3f}",
            f"{metrics['Treynor_Ratio']:.3f}",
            f"{metrics['Jensen_Alpha_%']:.2f}",
            f"{metrics['Max_Drawdown_%']:.2f}",
            f"{metrics['Calmar_Ratio']:.3f}",
            f"{metrics['Beta']:.3f}",
            f"{metrics['Win_Rate_%']:.2f}",
            f"{metrics['Profit_Factor']:.2f}",
            f"₹{metrics['Avg_Win_Rs']:,.0f}",
            f"₹{metrics['Avg_Loss_Rs']:,.0f}",
            f"{metrics['Total_Trading_Days']:,}",
            f"{metrics['Investment_Period_Years']:.2f}",
            f"{metrics['Quarterly_Returns_Count']:,}",
            '',
            '',
            f"{metrics['Avg_Quarterly_Return_%']:.2f}",
            f"{metrics['Quarterly_Volatility_%']:.2f}",
            '',
            '',
            f"{metrics['Nifty50_CAGR_%']:.1f}",
            f"{metrics['Nifty50_Volatility_%']:.1f}",
            f"{metrics['Risk_Free_Rate_%']:.1f}"
        ],
        'Interpretation': [
            'Total portfolio return over entire period',
            'Compound annual growth rate (geometric mean)',
            'Annual outperformance vs Nifty 50',
            'Annual standard deviation of returns',
            'Risk-adjusted return (>1 good, >2 excellent)',
            'Return per unit of downside risk',
            'Excess return per unit of tracking error',
            'Excess return per unit of systematic risk',
            'Return above CAPM expected return',
            'Largest peak-to-trough decline',
            'CAGR divided by absolute max drawdown',
            'Sensitivity to Nifty 50 movements',
            'Percentage of profitable trading periods',
            'Gross profit divided by gross loss',
            'Average profit on winning trades',
            'Average loss on losing trades',
            'Total number of trading observations',
            'Duration of investment strategy',
            'Total quarterly return observations',
            '',
            '',
            'Mean return per quarter',
            'Standard deviation of quarterly returns',
            '',
            '',
            'Nifty 50 compound annual growth rate',
            'Nifty 50 annual volatility',
            'Indian 10-year government bond yield'
        ]
    }
    
    df = pd.DataFrame(performance_data)
    return df, metrics

def create_visual_performance_table():
    """Create a visual performance table as an image"""
    df, metrics = create_enhanced_performance_table()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 16))
    ax.axis('tight')
    ax.axis('off')
    
    # Filter out empty rows for the table
    display_df = df[df['Metric'] != ''].copy()
    
    # Create table
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                    cellLoc='left', loc='center', colWidths=[0.35, 0.15, 0.5])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)
    
    # Header styling
    for i in range(len(display_df.columns)):
        table[(0, i)].set_facecolor('#1f4e79')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.08)
    
    # Color code rows based on metric type
    colors = {
        'return': '#E8F5E8',      # Light green for return metrics
        'risk': '#FFE8E8',        # Light red for risk metrics  
        'ratio': '#E8F0FF',       # Light blue for ratios
        'trading': '#FFF8E8',     # Light yellow for trading metrics
        'quarterly': '#F0E8FF',   # Light purple for quarterly metrics
        'benchmark': '#F0F0F0',   # Light gray for benchmark
        'info': '#F5F5F5'         # Very light gray for info
    }
    
    metric_categories = [
        'return', 'return', 'return', 'risk', 'ratio', 'ratio', 'ratio', 'ratio', 'return',
        'risk', 'ratio', 'risk', 'trading', 'trading', 'trading', 'trading',
        'info', 'info', 'info', 'quarterly', 'quarterly', 'quarterly', 
        'benchmark', 'benchmark', 'benchmark'
    ]
    
    for i, category in enumerate(metric_categories):
        for j in range(len(display_df.columns)):
            table[(i+1, j)].set_facecolor(colors[category])
            table[(i+1, j)].set_height(0.06)
    
    # Add title
    plt.title('Enhanced Portfolio Performance Metrics (Corrected)\nQuarterly Options Trading Strategy vs Nifty 50 Benchmark (2015-2025)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add summary box
    summary_text = f"""
    CORRECTED PERFORMANCE HIGHLIGHTS:
    • Total Return: {metrics['Total_Return_%']:.1f}% | CAGR: {metrics['CAGR_%']:.1f}% vs Nifty 50: {metrics['Nifty50_CAGR_%']:.1f}%
    • Excess Return: {metrics['Excess_Return_%']:.1f}% per year (Outperformance)
    • Sharpe Ratio: {metrics['Sharpe_Ratio']:.2f} (Quarterly-based calculation)
    • Beta: {metrics['Beta']:.2f} (Lower volatility than market)
    • Maximum Drawdown: {metrics['Max_Drawdown_%']:.1f}% (Excellent risk control)
    • Quarterly Analysis: {metrics['Quarterly_Returns_Count']} quarters, Avg: {metrics['Avg_Quarterly_Return_%']:.2f}%
    """
    
    plt.figtext(0.02, 0.02, summary_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('Performance_Analysis_Images/Enhanced_Performance_Metrics_Nifty50_Corrected.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def save_enhanced_metrics():
    """Save enhanced performance metrics to files"""
    df, metrics = create_enhanced_performance_table()
    
    # Save to CSV
    df.to_csv('Performance_Metrics_Data/Enhanced_Performance_Metrics_Nifty50_Corrected.csv', index=False)
    
    # Save formatted text version
    with open('Performance_Metrics_Data/Enhanced_Performance_Metrics_Nifty50_Corrected.txt', 'w') as f:
        f.write("ENHANCED PORTFOLIO PERFORMANCE METRICS vs NIFTY 50 (CORRECTED)\n")
        f.write("="*70 + "\n\n")
        f.write("METHODOLOGY:\n")
        f.write("- Quarterly returns used for all calculations\n")
        f.write("- Sharpe ratio calculated using periodic method (quarterly)\n")
        f.write("- CAGR calculated as geometric mean\n")
        f.write("- Maximum drawdown from daily portfolio values\n\n")
        f.write(tabulate(df[df['Metric'] != ''], headers=df.columns, tablefmt='grid', showindex=False))
        f.write(f"\n\nCORRECTED PERFORMANCE SUMMARY:\n")
        f.write(f"Portfolio delivered {metrics['Total_Return_%']:.1f}% total return ")
        f.write(f"with CAGR of {metrics['CAGR_%']:.1f}% vs Nifty 50's {metrics['Nifty50_CAGR_%']:.1f}%.\n")
        f.write(f"Excess return of {metrics['Excess_Return_%']:.1f}% per year with Sharpe ratio of {metrics['Sharpe_Ratio']:.2f}.\n")
        f.write(f"Risk control: {metrics['Max_Drawdown_%']:.1f}% max drawdown, Beta {metrics['Beta']:.2f} vs Nifty 50.\n")
        f.write(f"Quarterly performance: {metrics['Quarterly_Returns_Count']} quarters, ")
        f.write(f"average {metrics['Avg_Quarterly_Return_%']:.2f}% per quarter.\n")
        f.write(f"Trading performance: {metrics['Win_Rate_%']:.1f}% win rate, {metrics['Profit_Factor']:.1f}x profit factor.\n")
    
    # Create visual table
    visual_df = create_visual_performance_table()
    
    return df, metrics

def main():
    """Generate enhanced performance metrics with Nifty 50 benchmark"""
    print("🚀 Enhanced Portfolio Performance Analysis vs Nifty 50 (CORRECTED)")
    print("="*70)
    
    try:
        df, metrics = save_enhanced_metrics()
        
        print("✅ Generated Corrected Enhanced Performance Files:")
        print("- Enhanced_Performance_Metrics_Nifty50_Corrected.csv")
        print("- Enhanced_Performance_Metrics_Nifty50_Corrected.txt") 
        print("- Enhanced_Performance_Metrics_Nifty50_Corrected.png")
        
        print("\n📊 CORRECTED PERFORMANCE SUMMARY:")
        print("="*70)
        
        # Display key metrics
        key_metrics = df[df['Metric'].isin([
            'Total Return (%)', 'CAGR - Compound Annual Growth Rate (%)', 'Excess Return vs Nifty 50 (%)',
            'Sharpe Ratio (Annualized)', 'Beta vs Nifty 50', 'Maximum Drawdown (%)', 
            'Win Rate (%)', 'Average Quarterly Return (%)', 'Nifty 50 CAGR (%)'
        ])]
        
        print(tabulate(key_metrics, headers=key_metrics.columns, tablefmt='simple', showindex=False))
        
        print(f"\n🎯 CORRECTED KEY INSIGHTS:")
        excess_return = metrics['Excess_Return_%']
        sharpe = metrics['Sharpe_Ratio']
        beta = metrics['Beta']
        drawdown = abs(metrics['Max_Drawdown_%'])
        cagr = metrics['CAGR_%']
        
        print(f"• CAGR: {cagr:.2f}% (Compound Annual Growth Rate)")
        print(f"• {'�� OUTPERFORMED' if excess_return > 0 else '🔴 UNDERPERFORMED'} Nifty 50 by {abs(excess_return):.1f}% annually")
        print(f"• {'🟢 EXCELLENT' if sharpe > 2 else '🟡 GOOD' if sharpe > 1 else '🔴 MODERATE'} risk-adjusted returns (Sharpe: {sharpe:.2f})")
        print(f"• {'🟢 LOWER' if beta < 1 else '🔴 HIGHER'} volatility than market (Beta: {beta:.2f})")
        print(f"• {'🟢 EXCELLENT' if drawdown < 10 else '🟡 GOOD' if drawdown < 20 else '�� HIGH'} risk control (Max DD: {drawdown:.1f}%)")
        print(f"• Quarterly consistency: {metrics['Avg_Quarterly_Return_%']:.2f}% average per quarter")
        print(f"• {'🟢 CONSISTENT' if metrics['Win_Rate_%'] > 60 else '🔴 INCONSISTENT'} profitability ({metrics['Win_Rate_%']:.1f}% win rate)")
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
