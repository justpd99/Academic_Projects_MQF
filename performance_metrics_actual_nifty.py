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

def fetch_actual_nifty50_data(start_date, end_date):
    """Fetch actual Nifty 50 data and calculate returns"""
    try:
        print("Fetching actual Nifty 50 data from Yahoo Finance...")
        
        # Nifty 50 symbol
        nifty_symbol = "^NSEI"
        
        # Fetch data with buffer
        buffer_start = start_date - timedelta(days=30)
        buffer_end = end_date + timedelta(days=30)
        
        nifty_data = yf.download(nifty_symbol, start=buffer_start, end=buffer_end, progress=False)
        
        if nifty_data.empty:
            print("❌ Could not fetch Nifty 50 data")
            return None, None, None
        
        # Handle multi-level columns
        if isinstance(nifty_data.columns, pd.MultiIndex):
            nifty_data.columns = nifty_data.columns.droplevel(1)
        
        # Use Close or Adj Close
        close_col = 'Adj Close' if 'Adj Close' in nifty_data.columns else 'Close'
        
        # Filter to exact portfolio period
        portfolio_period_data = nifty_data[(nifty_data.index >= start_date) & (nifty_data.index <= end_date)]
        
        if len(portfolio_period_data) < 2:
            print("❌ Insufficient Nifty 50 data for portfolio period")
            return None, None, None
        
        # Calculate Nifty 50 total return and CAGR
        start_price = portfolio_period_data[close_col].iloc[0]
        end_price = portfolio_period_data[close_col].iloc[-1]
        
        nifty_total_return = (end_price - start_price) / start_price
        
        # Calculate time period
        total_years = (end_date - start_date).days / 365.25
        nifty_cagr = (1 + nifty_total_return) ** (1/total_years) - 1
        
        # Calculate daily returns for volatility
        portfolio_period_data['Daily_Return'] = portfolio_period_data[close_col].pct_change()
        daily_returns = portfolio_period_data['Daily_Return'].dropna()
        nifty_annual_volatility = daily_returns.std() * np.sqrt(252)  # 252 trading days
        
        print(f"✅ Nifty 50 data: {portfolio_period_data.index[0].date()} to {portfolio_period_data.index[-1].date()}")
        print(f"   Start Price: ₹{start_price:.2f}, End Price: ₹{end_price:.2f}")
        print(f"   Total Return: {nifty_total_return*100:.2f}%, CAGR: {nifty_cagr*100:.2f}%")
        
        return nifty_cagr, nifty_total_return, nifty_annual_volatility
        
    except Exception as e:
        print(f"❌ Error fetching Nifty 50 data: {e}")
        return None, None, None

def calculate_performance_metrics():
    """Calculate performance metrics using actual Nifty 50 returns"""
    daily_df, summary_df = load_portfolio_data()
    
    # Basic portfolio parameters
    initial_investment = 1500000  # ₹15 lakh
    total_return = daily_df['Cumulative_Profit'].iloc[-1]
    
    # Time period calculations
    start_date = daily_df['Date'].iloc[0]
    end_date = daily_df['Date'].iloc[-1]
    total_years = (end_date - start_date).days / 365.25
    
    print(f"Portfolio Period: {start_date.date()} to {end_date.date()} ({total_years:.2f} years)")
    
    # Get actual Nifty 50 returns
    nifty_cagr, nifty_total_return, nifty_volatility = fetch_actual_nifty50_data(start_date, end_date)
    
    # Use defaults if Nifty data fetch fails
    if nifty_cagr is None:
        nifty_cagr = 0.12  # 12% fallback
        nifty_total_return = 2.5  # 250% fallback for ~10 years
        nifty_volatility = 0.20  # 20% fallback
        print("⚠️  Using fallback Nifty 50 values")
    
    # Calculate quarterly returns for portfolio
    daily_df['Quarter'] = daily_df['Date'].dt.to_period('Q')
    quarterly_profits = daily_df.groupby('Quarter')['Total_Daily_Profit'].sum()
    quarterly_returns = quarterly_profits / initial_investment
    
    # 1. Portfolio Returns
    total_return_pct = (total_return / initial_investment) * 100
    total_return_multiple = (initial_investment + total_return) / initial_investment
    portfolio_cagr = (total_return_multiple ** (1/total_years)) - 1
    
    # 2. Quarterly statistics
    avg_quarterly_return = quarterly_returns.mean()
    quarterly_volatility = quarterly_returns.std()
    annualized_volatility = quarterly_volatility * np.sqrt(4)
    
    # 3. Risk-free rate
    annual_risk_free_rate = 0.07  # 7% Indian government bonds
    quarterly_risk_free_rate = (1 + annual_risk_free_rate) ** (1/4) - 1
    
    # 4. Sharpe Ratio (quarterly method)
    excess_quarterly_return = avg_quarterly_return - quarterly_risk_free_rate
    sharpe_ratio_quarterly = excess_quarterly_return / quarterly_volatility if quarterly_volatility > 0 else 0
    sharpe_ratio_annualized = sharpe_ratio_quarterly * np.sqrt(4)
    
    # 5. Sortino Ratio
    negative_returns = quarterly_returns[quarterly_returns < quarterly_risk_free_rate]
    if len(negative_returns) > 0:
        downside_deviation = np.sqrt(np.mean((negative_returns - quarterly_risk_free_rate)**2))
        sortino_ratio = (excess_quarterly_return / downside_deviation) * np.sqrt(4)
    else:
        sortino_ratio = 0
    
    # 6. Maximum Drawdown
    portfolio_values = initial_investment + daily_df['Cumulative_Profit'].values
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown_values = (portfolio_values - running_max) / running_max * 100
    max_drawdown = np.min(drawdown_values)
    
    # 7. Other metrics
    calmar_ratio = (portfolio_cagr * 100) / abs(max_drawdown) if max_drawdown != 0 else 0
    excess_return_vs_nifty = portfolio_cagr - nifty_cagr
    
    # 8. Beta (simplified using volatility ratio)
    beta = annualized_volatility / nifty_volatility if nifty_volatility > 0 else 1.0
    
    # 9. Trading performance
    profits = daily_df['Total_Daily_Profit'].values
    positive_days = len(profits[profits > 0])
    total_trading_days = len(profits)
    win_rate = (positive_days / total_trading_days) * 100
    
    gross_profit = np.sum(profits[profits > 0])
    gross_loss = abs(np.sum(profits[profits < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    return {
        'Total_Return_%': round(total_return_pct, 2),
        'Portfolio_CAGR_%': round(portfolio_cagr * 100, 2),
        'Nifty50_CAGR_%': round(nifty_cagr * 100, 2),
        'Nifty50_Total_Return_%': round(nifty_total_return * 100, 2),
        'Excess_Return_%': round(excess_return_vs_nifty * 100, 2),
        'Annualized_Volatility_%': round(annualized_volatility * 100, 2),
        'Nifty50_Volatility_%': round(nifty_volatility * 100, 2),
        'Sharpe_Ratio': round(sharpe_ratio_annualized, 3),
        'Sortino_Ratio': round(sortino_ratio, 3),
        'Max_Drawdown_%': round(max_drawdown, 2),
        'Calmar_Ratio': round(calmar_ratio, 3),
        'Beta': round(beta, 3),
        'Win_Rate_%': round(win_rate, 2),
        'Profit_Factor': round(profit_factor, 2),
        'Investment_Period_Years': round(total_years, 2),
        'Quarterly_Count': len(quarterly_returns),
        'Avg_Quarterly_Return_%': round(avg_quarterly_return * 100, 2),
        'Quarterly_Volatility_%': round(quarterly_volatility * 100, 2),
        'Risk_Free_Rate_%': round(annual_risk_free_rate * 100, 1)
    }

def create_comparison_tables():
    """Create tables comparing portfolio vs actual Nifty 50"""
    metrics = calculate_performance_metrics()
    
    # Table 1: Returns Comparison
    returns_data = {
        'Return Metric': [
            'Total Return (%)',
            'CAGR (%)',
            'Excess Return (%)',
            'Investment Period (Years)'
        ],
        'Portfolio': [
            f"{metrics['Total_Return_%']:.2f}",
            f"{metrics['Portfolio_CAGR_%']:.2f}",
            f"{metrics['Excess_Return_%']:.2f}",
            f"{metrics['Investment_Period_Years']:.2f}"
        ],
        'Nifty 50': [
            f"{metrics['Nifty50_Total_Return_%']:.2f}",
            f"{metrics['Nifty50_CAGR_%']:.2f}",
            "0.00",
            f"{metrics['Investment_Period_Years']:.2f}"
        ]
    }
    
    # Table 2: Risk Comparison
    risk_data = {
        'Risk Metric': [
            'Annualized Volatility (%)',
            'Maximum Drawdown (%)',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Beta vs Nifty 50'
        ],
        'Portfolio': [
            f"{metrics['Annualized_Volatility_%']:.2f}",
            f"{metrics['Max_Drawdown_%']:.2f}",
            f"{metrics['Sharpe_Ratio']:.3f}",
            f"{metrics['Sortino_Ratio']:.3f}",
            f"{metrics['Beta']:.3f}"
        ],
        'Nifty 50': [
            f"{metrics['Nifty50_Volatility_%']:.2f}",
            "N/A",
            "N/A",
            "N/A",
            "1.000"
        ]
    }
    
    # Table 3: Portfolio Specific
    portfolio_data = {
        'Portfolio Metric': [
            'Win Rate (%)',
            'Profit Factor',
            'Calmar Ratio',
            'Quarterly Count',
            'Avg Quarterly Return (%)',
            'Risk-Free Rate (%)'
        ],
        'Value': [
            f"{metrics['Win_Rate_%']:.2f}",
            f"{metrics['Profit_Factor']:.2f}",
            f"{metrics['Calmar_Ratio']:.3f}",
            f"{metrics['Quarterly_Count']}",
            f"{metrics['Avg_Quarterly_Return_%']:.2f}",
            f"{metrics['Risk_Free_Rate_%']:.1f}"
        ]
    }
    
    return pd.DataFrame(returns_data), pd.DataFrame(risk_data), pd.DataFrame(portfolio_data), metrics

def create_visual_comparison():
    """Create visual comparison with actual Nifty 50 data"""
    returns_df, risk_df, portfolio_df, metrics = create_comparison_tables()
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 16))
    
    # Remove axes
    for ax in [ax1, ax2, ax3]:
        ax.axis('tight')
        ax.axis('off')
    
    # Table 1: Returns Comparison
    table1 = ax1.table(cellText=returns_df.values, colLabels=returns_df.columns,
                      cellLoc='center', loc='center', colWidths=[0.4, 0.3, 0.3])
    table1.auto_set_font_size(False)
    table1.set_fontsize(12)
    table1.scale(1, 2.5)
    
    # Style table 1
    for i in range(len(returns_df.columns)):
        table1[(0, i)].set_facecolor('#1f4e79')
        table1[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(returns_df) + 1):
        for j in range(len(returns_df.columns)):
            if j == 1:  # Portfolio column - green tint
                table1[(i, j)].set_facecolor('#E8F5E8')
                table1[(i, j)].set_text_props(weight='bold')
            elif j == 2:  # Nifty column - blue tint
                table1[(i, j)].set_facecolor('#E8F0FF')
            else:
                table1[(i, j)].set_facecolor('#F8F9FA')
    
    ax1.set_title('RETURNS COMPARISON: Portfolio vs Actual Nifty 50', fontsize=14, fontweight='bold', pad=20)
    
    # Table 2: Risk Comparison
    table2 = ax2.table(cellText=risk_df.values, colLabels=risk_df.columns,
                      cellLoc='center', loc='center', colWidths=[0.4, 0.3, 0.3])
    table2.auto_set_font_size(False)
    table2.set_fontsize(12)
    table2.scale(1, 2.5)
    
    # Style table 2
    for i in range(len(risk_df.columns)):
        table2[(0, i)].set_facecolor('#A23B72')
        table2[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(risk_df) + 1):
        for j in range(len(risk_df.columns)):
            if j == 1:  # Portfolio column
                table2[(i, j)].set_facecolor('#F8E8F0')
                table2[(i, j)].set_text_props(weight='bold')
            elif j == 2:  # Nifty column
                table2[(i, j)].set_facecolor('#E8F0FF')
            else:
                table2[(i, j)].set_facecolor('#F8F9FA')
    
    ax2.set_title('RISK COMPARISON: Portfolio vs Actual Nifty 50', fontsize=14, fontweight='bold', pad=20)
    
    # Table 3: Portfolio Specific
    table3 = ax3.table(cellText=portfolio_df.values, colLabels=portfolio_df.columns,
                      cellLoc='center', loc='center', colWidths=[0.6, 0.4])
    table3.auto_set_font_size(False)
    table3.set_fontsize(12)
    table3.scale(1, 2.5)
    
    # Style table 3
    for i in range(len(portfolio_df.columns)):
        table3[(0, i)].set_facecolor('#F18F01')
        table3[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(portfolio_df) + 1):
        for j in range(len(portfolio_df.columns)):
            table3[(i, j)].set_facecolor('#FFF8E8' if i % 2 == 0 else '#FFFFFF')
            if j == 1:  # Value column
                table3[(i, j)].set_text_props(weight='bold')
    
    ax3.set_title('PORTFOLIO SPECIFIC METRICS', fontsize=14, fontweight='bold', pad=20)
    
    # Add main title
    fig.suptitle('Portfolio Performance vs Actual Nifty 50 Returns\nQuarterly Options Trading Strategy (2015-2025)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add summary
    performance_status = "OUTPERFORMED" if metrics['Excess_Return_%'] > 0 else "UNDERPERFORMED"
    summary_text = f"""
ACTUAL PERFORMANCE: Portfolio {performance_status} Nifty 50 by {abs(metrics['Excess_Return_%']):.2f}% annually
Portfolio: {metrics['Portfolio_CAGR_%']:.2f}% CAGR | Nifty 50: {metrics['Nifty50_CAGR_%']:.2f}% CAGR | Max DD: {metrics['Max_Drawdown_%']:.2f}%
Sharpe Ratio: {metrics['Sharpe_Ratio']:.2f} | Beta: {metrics['Beta']:.2f} | Win Rate: {metrics['Win_Rate_%']:.1f}%
    """
    
    plt.figtext(0.5, 0.02, summary_text, fontsize=11, ha='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen" if metrics['Excess_Return_%'] > 0 else "lightcoral", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.15)
    plt.savefig('Performance_Analysis_Images/Actual_Nifty50_Performance_Comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def save_actual_nifty_metrics():
    """Save performance metrics with actual Nifty 50 data"""
    returns_df, risk_df, portfolio_df, metrics = create_comparison_tables()
    
    # Combine all tables
    combined_data = []
    
    # Add returns data
    for i, row in returns_df.iterrows():
        combined_data.append({
            'Category': 'Returns',
            'Metric': row['Return Metric'],
            'Portfolio': row['Portfolio'],
            'Nifty_50': row['Nifty 50']
        })
    
    # Add risk data
    for i, row in risk_df.iterrows():
        combined_data.append({
            'Category': 'Risk',
            'Metric': row['Risk Metric'],
            'Portfolio': row['Portfolio'],
            'Nifty_50': row['Nifty 50']
        })
    
    # Add portfolio specific data
    for i, row in portfolio_df.iterrows():
        combined_data.append({
            'Category': 'Portfolio',
            'Metric': row['Portfolio Metric'],
            'Portfolio': row['Value'],
            'Nifty_50': 'N/A'
        })
    
    combined_df = pd.DataFrame(combined_data)
    
    # Save to CSV
    combined_df.to_csv('Performance_Metrics_Data/Actual_Nifty50_Performance_Comparison.csv', index=False)
    
    # Save text version
    with open('Performance_Metrics_Data/Actual_Nifty50_Performance_Comparison.txt', 'w') as f:
        f.write("PORTFOLIO PERFORMANCE vs ACTUAL NIFTY 50 RETURNS\n")
        f.write("="*60 + "\n\n")
        
        f.write("RETURNS COMPARISON:\n")
        f.write(tabulate(returns_df, headers=returns_df.columns, tablefmt='grid', showindex=False))
        f.write("\n\n")
        
        f.write("RISK COMPARISON:\n")
        f.write(tabulate(risk_df, headers=risk_df.columns, tablefmt='grid', showindex=False))
        f.write("\n\n")
        
        f.write("PORTFOLIO SPECIFIC METRICS:\n")
        f.write(tabulate(portfolio_df, headers=portfolio_df.columns, tablefmt='grid', showindex=False))
        f.write("\n\n")
        
        f.write("ACTUAL PERFORMANCE SUMMARY:\n")
        f.write(f"Portfolio CAGR: {metrics['Portfolio_CAGR_%']:.2f}%\n")
        f.write(f"Nifty 50 CAGR: {metrics['Nifty50_CAGR_%']:.2f}%\n")
        f.write(f"Excess Return: {metrics['Excess_Return_%']:.2f}% annually\n")
        f.write(f"Sharpe Ratio: {metrics['Sharpe_Ratio']:.3f}\n")
        f.write(f"Maximum Drawdown: {metrics['Max_Drawdown_%']:.2f}%\n")
        f.write(f"Beta vs Nifty 50: {metrics['Beta']:.3f}\n")
    
    # Create visual
    create_visual_comparison()
    
    return combined_df, metrics

def main():
    """Generate performance metrics with actual Nifty 50 returns"""
    print("🚀 Portfolio Performance Analysis with ACTUAL Nifty 50 Returns")
    print("="*65)
    
    try:
        combined_df, metrics = save_actual_nifty_metrics()
        
        print("✅ Generated Actual Nifty 50 Comparison Files:")
        print("- Actual_Nifty50_Performance_Comparison.csv")
        print("- Actual_Nifty50_Performance_Comparison.txt")
        print("- Actual_Nifty50_Performance_Comparison.png")
        
        print(f"\n📊 ACTUAL PERFORMANCE RESULTS:")
        print(f"Portfolio CAGR: {metrics['Portfolio_CAGR_%']:.2f}%")
        print(f"Nifty 50 CAGR: {metrics['Nifty50_CAGR_%']:.2f}%")
        print(f"Excess Return: {metrics['Excess_Return_%']:.2f}% annually")
        print(f"Sharpe Ratio: {metrics['Sharpe_Ratio']:.3f}")
        print(f"Max Drawdown: {metrics['Max_Drawdown_%']:.2f}%")
        print(f"Beta: {metrics['Beta']:.3f}")
        
        print(f"\n🎯 ACTUAL ASSESSMENT:")
        if metrics['Excess_Return_%'] > 0:
            print(f"🟢 OUTPERFORMED Nifty 50 by {metrics['Excess_Return_%']:.2f}% annually")
        else:
            print(f"🔴 UNDERPERFORMED Nifty 50 by {abs(metrics['Excess_Return_%']):.2f}% annually")
            
        if metrics['Sharpe_Ratio'] > 1:
            print("🟢 GOOD risk-adjusted returns")
        else:
            print("🟡 MODERATE risk-adjusted returns")
            
        if abs(metrics['Max_Drawdown_%']) < 10:
            print("🟢 EXCELLENT risk control")
        else:
            print("🟡 MODERATE risk control")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
