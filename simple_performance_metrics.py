import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
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

def calculate_performance_metrics():
    """Calculate performance metrics using correct quarterly methodology"""
    daily_df, summary_df = load_portfolio_data()
    
    # Basic portfolio parameters
    initial_investment = 1500000  # ₹15 lakh
    total_return = daily_df['Cumulative_Profit'].iloc[-1]
    
    # Time period calculations
    start_date = daily_df['Date'].iloc[0]
    end_date = daily_df['Date'].iloc[-1]
    total_years = (end_date - start_date).days / 365.25
    
    # Calculate quarterly returns for portfolio
    daily_df['Quarter'] = daily_df['Date'].dt.to_period('Q')
    quarterly_profits = daily_df.groupby('Quarter')['Total_Daily_Profit'].sum()
    quarterly_returns = quarterly_profits / initial_investment
    
    # 1. Total Return and CAGR
    total_return_pct = (total_return / initial_investment) * 100
    total_return_multiple = (initial_investment + total_return) / initial_investment
    cagr = (total_return_multiple ** (1/total_years)) - 1
    
    # 2. Quarterly statistics
    avg_quarterly_return = quarterly_returns.mean()
    quarterly_volatility = quarterly_returns.std()
    annualized_volatility = quarterly_volatility * np.sqrt(4)
    
    # 3. Risk-free rate and benchmark (using realistic Indian market data)
    annual_risk_free_rate = 0.07  # 7% Indian government bonds
    quarterly_risk_free_rate = (1 + annual_risk_free_rate) ** (1/4) - 1
    nifty_annual_return = 0.12  # 12% historical Nifty 50 CAGR
    
    # 4. Sharpe Ratio (quarterly method as per your image)
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
    
    # 6. Maximum Drawdown (corrected calculation)
    portfolio_values = initial_investment + daily_df['Cumulative_Profit'].values
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown_values = (portfolio_values - running_max) / running_max * 100
    max_drawdown = np.min(drawdown_values)
    
    # 7. Other metrics
    calmar_ratio = (cagr * 100) / abs(max_drawdown) if max_drawdown != 0 else 0
    excess_return_vs_nifty = cagr - nifty_annual_return
    
    # 8. Trading performance
    profits = daily_df['Total_Daily_Profit'].values
    positive_days = len(profits[profits > 0])
    total_trading_days = len(profits)
    win_rate = (positive_days / total_trading_days) * 100
    
    gross_profit = np.sum(profits[profits > 0])
    gross_loss = abs(np.sum(profits[profits < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    return {
        'Total_Return_%': round(total_return_pct, 2),
        'CAGR_%': round(cagr * 100, 2),
        'Excess_Return_%': round(excess_return_vs_nifty * 100, 2),
        'Annualized_Volatility_%': round(annualized_volatility * 100, 2),
        'Sharpe_Ratio': round(sharpe_ratio_annualized, 3),
        'Sortino_Ratio': round(sortino_ratio, 3),
        'Max_Drawdown_%': round(max_drawdown, 2),
        'Calmar_Ratio': round(calmar_ratio, 3),
        'Win_Rate_%': round(win_rate, 2),
        'Profit_Factor': round(profit_factor, 2),
        'Investment_Period_Years': round(total_years, 2),
        'Quarterly_Count': len(quarterly_returns),
        'Avg_Quarterly_Return_%': round(avg_quarterly_return * 100, 2),
        'Quarterly_Volatility_%': round(quarterly_volatility * 100, 2),
        'Nifty50_CAGR_%': round(nifty_annual_return * 100, 1),
        'Risk_Free_Rate_%': round(annual_risk_free_rate * 100, 1)
    }

def create_simple_table():
    """Create a simple, readable table"""
    metrics = calculate_performance_metrics()
    
    # Create two separate tables for better readability
    
    # Table 1: Returns and Performance
    returns_data = {
        'Performance Metric': [
            'Total Return (%)',
            'CAGR (%)',
            'Excess Return vs Nifty 50 (%)',
            'Investment Period (Years)',
            'Number of Quarters'
        ],
        'Value': [
            f"{metrics['Total_Return_%']:.2f}",
            f"{metrics['CAGR_%']:.2f}",
            f"{metrics['Excess_Return_%']:.2f}",
            f"{metrics['Investment_Period_Years']:.2f}",
            f"{metrics['Quarterly_Count']}"
        ]
    }
    
    # Table 2: Risk Metrics
    risk_data = {
        'Risk Metric': [
            'Annualized Volatility (%)',
            'Maximum Drawdown (%)',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Calmar Ratio'
        ],
        'Value': [
            f"{metrics['Annualized_Volatility_%']:.2f}",
            f"{metrics['Max_Drawdown_%']:.2f}",
            f"{metrics['Sharpe_Ratio']:.3f}",
            f"{metrics['Sortino_Ratio']:.3f}",
            f"{metrics['Calmar_Ratio']:.3f}"
        ]
    }
    
    # Table 3: Trading and Quarterly Analysis
    trading_data = {
        'Trading & Quarterly Metric': [
            'Win Rate (%)',
            'Profit Factor',
            'Average Quarterly Return (%)',
            'Quarterly Volatility (%)',
            'Nifty 50 CAGR (%)',
            'Risk-Free Rate (%)'
        ],
        'Value': [
            f"{metrics['Win_Rate_%']:.2f}",
            f"{metrics['Profit_Factor']:.2f}",
            f"{metrics['Avg_Quarterly_Return_%']:.2f}",
            f"{metrics['Quarterly_Volatility_%']:.2f}",
            f"{metrics['Nifty50_CAGR_%']:.1f}",
            f"{metrics['Risk_Free_Rate_%']:.1f}"
        ]
    }
    
    return pd.DataFrame(returns_data), pd.DataFrame(risk_data), pd.DataFrame(trading_data), metrics

def create_visual_tables():
    """Create clean visual tables"""
    returns_df, risk_df, trading_df, metrics = create_simple_table()
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
    
    # Remove axes
    for ax in [ax1, ax2, ax3]:
        ax.axis('tight')
        ax.axis('off')
    
    # Table 1: Returns
    table1 = ax1.table(cellText=returns_df.values, colLabels=returns_df.columns,
                      cellLoc='center', loc='center', colWidths=[0.6, 0.4])
    table1.auto_set_font_size(False)
    table1.set_fontsize(12)
    table1.scale(1, 2)
    
    # Style table 1
    for i in range(len(returns_df.columns)):
        table1[(0, i)].set_facecolor('#2E86AB')
        table1[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(returns_df) + 1):
        for j in range(len(returns_df.columns)):
            table1[(i, j)].set_facecolor('#E8F4F8' if i % 2 == 0 else '#FFFFFF')
            if j == 1:  # Value column
                table1[(i, j)].set_text_props(weight='bold')
    
    ax1.set_title('PORTFOLIO RETURNS & PERFORMANCE', fontsize=14, fontweight='bold', pad=20)
    
    # Table 2: Risk
    table2 = ax2.table(cellText=risk_df.values, colLabels=risk_df.columns,
                      cellLoc='center', loc='center', colWidths=[0.6, 0.4])
    table2.auto_set_font_size(False)
    table2.set_fontsize(12)
    table2.scale(1, 2)
    
    # Style table 2
    for i in range(len(risk_df.columns)):
        table2[(0, i)].set_facecolor('#A23B72')
        table2[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(risk_df) + 1):
        for j in range(len(risk_df.columns)):
            table2[(i, j)].set_facecolor('#F8E8F0' if i % 2 == 0 else '#FFFFFF')
            if j == 1:  # Value column
                table2[(i, j)].set_text_props(weight='bold')
    
    ax2.set_title('RISK METRICS', fontsize=14, fontweight='bold', pad=20)
    
    # Table 3: Trading
    table3 = ax3.table(cellText=trading_df.values, colLabels=trading_df.columns,
                      cellLoc='center', loc='center', colWidths=[0.6, 0.4])
    table3.auto_set_font_size(False)
    table3.set_fontsize(12)
    table3.scale(1, 2)
    
    # Style table 3
    for i in range(len(trading_df.columns)):
        table3[(0, i)].set_facecolor('#F18F01')
        table3[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(trading_df) + 1):
        for j in range(len(trading_df.columns)):
            table3[(i, j)].set_facecolor('#FFF8E8' if i % 2 == 0 else '#FFFFFF')
            if j == 1:  # Value column
                table3[(i, j)].set_text_props(weight='bold')
    
    ax3.set_title('TRADING & QUARTERLY ANALYSIS', fontsize=14, fontweight='bold', pad=20)
    
    # Add main title
    fig.suptitle('Portfolio Performance Metrics - Quarterly Options Trading Strategy\nvs Nifty 50 Benchmark (2015-2025)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add summary at bottom
    summary_text = f"""
SUMMARY: Portfolio delivered {metrics['Total_Return_%']:.1f}% total return ({metrics['CAGR_%']:.1f}% CAGR) vs Nifty 50's {metrics['Nifty50_CAGR_%']:.1f}% CAGR.
Excess return: {metrics['Excess_Return_%']:.1f}% annually | Max drawdown: {metrics['Max_Drawdown_%']:.1f}% | Sharpe ratio: {metrics['Sharpe_Ratio']:.2f}
Methodology: Quarterly returns used for Sharpe ratio calculation as per periodic method.
    """
    
    plt.figtext(0.5, 0.02, summary_text, fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.15)
    plt.savefig('Performance_Analysis_Images/Simple_Performance_Metrics.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def save_simple_metrics():
    """Save simple performance metrics"""
    returns_df, risk_df, trading_df, metrics = create_simple_table()
    
    # Combine all tables
    combined_df = pd.concat([
        returns_df.rename(columns={'Performance Metric': 'Metric'}),
        risk_df.rename(columns={'Risk Metric': 'Metric'}),
        trading_df.rename(columns={'Trading & Quarterly Metric': 'Metric'})
    ], ignore_index=True)
    
    # Save to CSV
    combined_df.to_csv('Performance_Metrics_Data/Simple_Performance_Metrics.csv', index=False)
    
    # Save text version
    with open('Performance_Metrics_Data/Simple_Performance_Metrics.txt', 'w') as f:
        f.write("PORTFOLIO PERFORMANCE METRICS - QUARTERLY OPTIONS STRATEGY\n")
        f.write("="*60 + "\n\n")
        
        f.write("RETURNS & PERFORMANCE:\n")
        f.write(tabulate(returns_df, headers=returns_df.columns, tablefmt='grid', showindex=False))
        f.write("\n\n")
        
        f.write("RISK METRICS:\n")
        f.write(tabulate(risk_df, headers=risk_df.columns, tablefmt='grid', showindex=False))
        f.write("\n\n")
        
        f.write("TRADING & QUARTERLY ANALYSIS:\n")
        f.write(tabulate(trading_df, headers=trading_df.columns, tablefmt='grid', showindex=False))
        f.write("\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("- Sharpe Ratio: (Avg Quarterly Return - Risk-Free Rate) / Quarterly Volatility × √4\n")
        f.write("- CAGR: (Final Value / Initial Value)^(1/Years) - 1\n")
        f.write("- Maximum Drawdown: From daily portfolio values\n")
        f.write("- All metrics calculated using quarterly returns as base\n")
    
    # Create visual
    create_visual_tables()
    
    return combined_df, metrics

def main():
    """Generate simple, clean performance metrics"""
    print("🚀 Simple Portfolio Performance Analysis")
    print("="*50)
    
    try:
        combined_df, metrics = save_simple_metrics()
        
        print("✅ Generated Simple Performance Files:")
        print("- Simple_Performance_Metrics.csv")
        print("- Simple_Performance_Metrics.txt")
        print("- Simple_Performance_Metrics.png")
        
        print(f"\n📊 KEY RESULTS:")
        print(f"Total Return: {metrics['Total_Return_%']:.2f}%")
        print(f"CAGR: {metrics['CAGR_%']:.2f}%")
        print(f"Excess Return vs Nifty 50: {metrics['Excess_Return_%']:.2f}%")
        print(f"Sharpe Ratio: {metrics['Sharpe_Ratio']:.3f}")
        print(f"Max Drawdown: {metrics['Max_Drawdown_%']:.2f}%")
        print(f"Win Rate: {metrics['Win_Rate_%']:.2f}%")
        
        print(f"\n🎯 ASSESSMENT:")
        if metrics['CAGR_%'] > metrics['Nifty50_CAGR_%']:
            print("🟢 OUTPERFORMED Nifty 50")
        else:
            print("🔴 UNDERPERFORMED Nifty 50")
            
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
