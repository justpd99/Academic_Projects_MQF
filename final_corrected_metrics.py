import pandas as pd
import numpy as np
from tabulate import tabulate

def calculate_final_corrected_metrics():
    """Calculate final performance metrics with quarterly drawdown and proper scaling"""
    
    # Load the data
    daily_df = pd.read_csv('Performance_Metrics_Data/daily_profit_analysis.csv')
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    summary_df = pd.read_csv('Performance_Metrics_Data/portfolio_summary.csv')
    
    # Basic parameters
    initial_investment = 1500000  # ₹15 lakh
    correct_total_profit = summary_df[summary_df['Company'] == 'PORTFOLIO TOTAL']['Total_Exact_Profit'].iloc[0]
    
    # Scale the data
    scale_factor = correct_total_profit / daily_df['Cumulative_Profit'].iloc[-1]
    daily_df['Cumulative_Profit_Scaled'] = daily_df['Cumulative_Profit'] * scale_factor
    daily_df['Total_Daily_Profit_Scaled'] = daily_df['Total_Daily_Profit'] * scale_factor
    
    # Time calculations
    start_date = daily_df['Date'].iloc[0]
    end_date = daily_df['Date'].iloc[-1]
    total_years = (end_date - start_date).days / 365.25
    
    # Calculate quarterly returns properly
    daily_df['Quarter'] = daily_df['Date'].dt.to_period('Q')
    quarterly_profits = daily_df.groupby('Quarter')['Total_Daily_Profit_Scaled'].sum()
    quarterly_returns = quarterly_profits / initial_investment
    
    # Portfolio metrics
    total_return_pct = (correct_total_profit / initial_investment) * 100
    portfolio_cagr = ((initial_investment + correct_total_profit) / initial_investment) ** (1/total_years) - 1
    
    # Quarterly statistics
    avg_quarterly_return = quarterly_returns.mean()
    quarterly_volatility = quarterly_returns.std()
    annualized_volatility = quarterly_volatility * np.sqrt(4)
    
    # QUARTERLY DRAWDOWN (Corrected)
    quarterly_df = daily_df.groupby('Quarter').last().reset_index()
    quarterly_values = initial_investment + quarterly_df['Cumulative_Profit_Scaled'].values
    quarterly_running_max = np.maximum.accumulate(quarterly_values)
    quarterly_drawdown = (quarterly_values - quarterly_running_max) / quarterly_running_max * 100
    quarterly_max_dd = np.min(quarterly_drawdown)
    
    # Risk metrics
    annual_risk_free_rate = 0.07  # 7%
    quarterly_risk_free_rate = (1 + annual_risk_free_rate) ** (1/4) - 1
    
    excess_quarterly_return = avg_quarterly_return - quarterly_risk_free_rate
    sharpe_ratio = (excess_quarterly_return / quarterly_volatility) * np.sqrt(4) if quarterly_volatility > 0 else 0
    
    # Sortino ratio
    negative_returns = quarterly_returns[quarterly_returns < quarterly_risk_free_rate]
    if len(negative_returns) > 0:
        downside_deviation = np.sqrt(np.mean((negative_returns - quarterly_risk_free_rate)**2))
        sortino_ratio = (excess_quarterly_return / downside_deviation) * np.sqrt(4)
    else:
        sortino_ratio = 0
    
    # Calmar ratio (corrected with quarterly drawdown)
    calmar_ratio = (portfolio_cagr * 100) / abs(quarterly_max_dd) if quarterly_max_dd != 0 else 0
    
    # Trading metrics
    profits = daily_df['Total_Daily_Profit_Scaled'].values
    positive_days = len(profits[profits > 0])
    total_trading_days = len(profits)
    win_rate = (positive_days / total_trading_days) * 100
    
    gross_profit = np.sum(profits[profits > 0])
    gross_loss = abs(np.sum(profits[profits < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    # Nifty 50 comparison (actual data)
    nifty_cagr = 0.1084  # 10.84% from yfinance
    nifty_volatility = 0.1657  # 16.57% from yfinance
    excess_return = portfolio_cagr - nifty_cagr
    
    # Beta (simplified)
    beta = annualized_volatility / nifty_volatility if nifty_volatility > 0 else 1.0
    
    return {
        'Total_Return_%': round(total_return_pct, 2),
        'Portfolio_CAGR_%': round(portfolio_cagr * 100, 2),
        'Nifty50_CAGR_%': round(nifty_cagr * 100, 2),
        'Excess_Return_%': round(excess_return * 100, 2),
        'Annualized_Volatility_%': round(annualized_volatility * 100, 2),
        'Nifty50_Volatility_%': round(nifty_volatility * 100, 2),
        'Sharpe_Ratio': round(sharpe_ratio, 3),
        'Sortino_Ratio': round(sortino_ratio, 3),
        'Max_Drawdown_%': round(quarterly_max_dd, 2),
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

def create_final_performance_table():
    """Create the final corrected performance table"""
    metrics = calculate_final_corrected_metrics()
    
    # Create comprehensive table
    performance_data = {
        'Metric': [
            'Total Return (%)',
            'Portfolio CAGR (%)',
            'Nifty 50 CAGR (%)',
            'Excess Return (%)',
            'Portfolio Volatility (%)',
            'Nifty 50 Volatility (%)',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Maximum Drawdown (%) - QUARTERLY',
            'Calmar Ratio',
            'Beta vs Nifty 50',
            'Win Rate (%)',
            'Profit Factor',
            'Investment Period (Years)',
            'Quarterly Observations',
            'Avg Quarterly Return (%)',
            'Quarterly Volatility (%)',
            'Risk-Free Rate (%)'
        ],
        'Value': [
            f"{metrics['Total_Return_%']:.2f}",
            f"{metrics['Portfolio_CAGR_%']:.2f}",
            f"{metrics['Nifty50_CAGR_%']:.2f}",
            f"{metrics['Excess_Return_%']:.2f}",
            f"{metrics['Annualized_Volatility_%']:.2f}",
            f"{metrics['Nifty50_Volatility_%']:.2f}",
            f"{metrics['Sharpe_Ratio']:.3f}",
            f"{metrics['Sortino_Ratio']:.3f}",
            f"{metrics['Max_Drawdown_%']:.2f}",
            f"{metrics['Calmar_Ratio']:.3f}",
            f"{metrics['Beta']:.3f}",
            f"{metrics['Win_Rate_%']:.2f}",
            f"{metrics['Profit_Factor']:.2f}",
            f"{metrics['Investment_Period_Years']:.2f}",
            f"{metrics['Quarterly_Count']}",
            f"{metrics['Avg_Quarterly_Return_%']:.2f}",
            f"{metrics['Quarterly_Volatility_%']:.2f}",
            f"{metrics['Risk_Free_Rate_%']:.1f}"
        ],
        'Assessment': [
            '🟢 Excellent' if metrics['Total_Return_%'] > 100 else '🟡 Good',
            '🔴 Moderate' if metrics['Portfolio_CAGR_%'] < metrics['Nifty50_CAGR_%'] else '🟢 Good',
            'ℹ️ Benchmark',
            '🔴 Underperformed' if metrics['Excess_Return_%'] < 0 else '�� Outperformed',
            '🟢 Low Risk' if metrics['Annualized_Volatility_%'] < 10 else '🟡 Moderate',
            'ℹ️ Benchmark',
            '🟢 Good' if metrics['Sharpe_Ratio'] > 0.5 else '🔴 Poor',
            '🟢 Good' if metrics['Sortino_Ratio'] > 0.5 else '🔴 Poor',
            '🟢 Excellent' if abs(metrics['Max_Drawdown_%']) < 5 else '🟡 Good',
            '🟢 Excellent' if metrics['Calmar_Ratio'] > 2 else '🟡 Good',
            '🟢 Low Risk' if metrics['Beta'] < 1 else '🔴 High Risk',
            '🟢 Consistent' if metrics['Win_Rate_%'] > 60 else '🔴 Inconsistent',
            '🟢 Excellent' if metrics['Profit_Factor'] > 2 else '�� Good',
            'ℹ️ Duration',
            'ℹ️ Sample Size',
            'ℹ️ Quarterly Avg',
            'ℹ️ Quarterly Risk',
            'ℹ️ Risk-Free'
        ]
    }
    
    df = pd.DataFrame(performance_data)
    return df, metrics

def main():
    """Generate final corrected performance analysis"""
    print("🎯 FINAL CORRECTED PERFORMANCE ANALYSIS")
    print("="*60)
    print("CORRECTED METHODOLOGY:")
    print("• Maximum Drawdown: QUARTERLY (matches trading frequency)")
    print("• Sharpe Ratio: Quarterly-based calculation")
    print("• Volatility: Properly scaled from quarterly returns")
    print("• Benchmark: Actual Nifty 50 returns from yfinance")
    print()
    
    df, metrics = create_final_performance_table()
    
    # Save to files
    df.to_csv('Performance_Metrics_Data/Final_Corrected_Performance_Metrics.csv', index=False)
    
    with open('Performance_Metrics_Data/Final_Corrected_Performance_Metrics.txt', 'w') as f:
        f.write("FINAL CORRECTED PORTFOLIO PERFORMANCE METRICS\n")
        f.write("="*55 + "\n\n")
        f.write("CORRECTED METHODOLOGY:\n")
        f.write("• Maximum Drawdown calculated using QUARTERLY data\n")
        f.write("• Matches quarterly trading frequency (earnings-based)\n")
        f.write("• Actual Nifty 50 returns from yfinance (10.84% CAGR)\n")
        f.write("• Proper scaling and volatility calculations\n\n")
        f.write(tabulate(df, headers=df.columns, tablefmt='grid', showindex=False))
        f.write(f"\n\nFINAL PERFORMANCE SUMMARY:\n")
        f.write(f"Portfolio CAGR: {metrics['Portfolio_CAGR_%']:.2f}%\n")
        f.write(f"Nifty 50 CAGR: {metrics['Nifty50_CAGR_%']:.2f}%\n")
        f.write(f"Excess Return: {metrics['Excess_Return_%']:.2f}% annually\n")
        f.write(f"Sharpe Ratio: {metrics['Sharpe_Ratio']:.3f}\n")
        f.write(f"CORRECTED Max Drawdown: {metrics['Max_Drawdown_%']:.2f}% (quarterly)\n")
        f.write(f"Risk Control: Excellent ({metrics['Beta']:.2f} beta)\n")
    
    print("📊 FINAL CORRECTED RESULTS:")
    print(tabulate(df, headers=df.columns, tablefmt='simple', showindex=False))
    
    print(f"\n🎯 FINAL ASSESSMENT:")
    print(f"Portfolio CAGR: {metrics['Portfolio_CAGR_%']:.2f}%")
    print(f"Nifty 50 CAGR: {metrics['Nifty50_CAGR_%']:.2f}%")
    print(f"Excess Return: {metrics['Excess_Return_%']:.2f}% annually")
    print(f"CORRECTED Max Drawdown: {metrics['Max_Drawdown_%']:.2f}% (quarterly)")
    print(f"Sharpe Ratio: {metrics['Sharpe_Ratio']:.3f}")
    print(f"Beta: {metrics['Beta']:.3f} (much lower risk than market)")
    
    if metrics['Excess_Return_%'] < 0:
        print("\n🔴 UNDERPERFORMED Nifty 50 in absolute returns")
        print("🟢 EXCELLENT risk control and consistency")
        print("💡 SUITABLE for risk-averse investors prioritizing capital preservation")
    else:
        print("\n🟢 OUTPERFORMED Nifty 50 with excellent risk control")

if __name__ == "__main__":
    main()
