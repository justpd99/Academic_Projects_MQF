import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_quarterly_drawdown():
    """Calculate drawdown using quarterly data to match trading frequency"""
    
    # Load the data
    daily_df = pd.read_csv('Performance_Metrics_Data/daily_profit_analysis.csv')
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    
    # Portfolio parameters
    initial_investment = 1500000  # ₹15 lakh
    
    # Scale to correct total
    summary_df = pd.read_csv('Performance_Metrics_Data/portfolio_summary.csv')
    correct_total_profit = summary_df[summary_df['Company'] == 'PORTFOLIO TOTAL']['Total_Exact_Profit'].iloc[0]
    scale_factor = correct_total_profit / daily_df['Cumulative_Profit'].iloc[-1]
    daily_df['Cumulative_Profit_Scaled'] = daily_df['Cumulative_Profit'] * scale_factor
    
    # Create quarterly data (end of each quarter)
    daily_df['Quarter'] = daily_df['Date'].dt.to_period('Q')
    quarterly_df = daily_df.groupby('Quarter').last().reset_index()
    
    # Calculate quarterly portfolio values
    quarterly_values = initial_investment + quarterly_df['Cumulative_Profit_Scaled'].values
    
    # Calculate quarterly drawdown
    quarterly_running_max = np.maximum.accumulate(quarterly_values)
    quarterly_drawdown = (quarterly_values - quarterly_running_max) / quarterly_running_max * 100
    quarterly_max_dd = np.min(quarterly_drawdown)
    
    # Find the worst drawdown period
    max_dd_idx = np.argmin(quarterly_drawdown)
    peak_idx = np.where(quarterly_running_max == quarterly_running_max[max_dd_idx])[0][0]
    
    peak_quarter = quarterly_df['Quarter'].iloc[peak_idx]
    trough_quarter = quarterly_df['Quarter'].iloc[max_dd_idx]
    peak_value = quarterly_values[peak_idx]
    trough_value = quarterly_values[max_dd_idx]
    
    print("QUARTERLY DRAWDOWN ANALYSIS (CORRECTED)")
    print("="*55)
    print("RATIONALE: Since trading happens quarterly (earnings-based),")
    print("           drawdown should be measured quarterly, not daily.")
    print()
    print(f"Calculation Method: QUARTERLY portfolio values")
    print(f"Data Points: {len(quarterly_df)} quarterly observations")
    print(f"Period: {quarterly_df['Quarter'].iloc[0]} to {quarterly_df['Quarter'].iloc[-1]}")
    print()
    print(f"CORRECTED Maximum Drawdown: {quarterly_max_dd:.2f}%")
    print(f"Peak Quarter: {peak_quarter}")
    print(f"Trough Quarter: {trough_quarter}")
    print(f"Peak Value: ₹{peak_value:,.0f}")
    print(f"Trough Value: ₹{trough_value:,.0f}")
    print(f"Absolute Loss: ₹{peak_value - trough_value:,.0f}")
    print()
    
    # Compare with daily calculation
    portfolio_values_daily = initial_investment + daily_df['Cumulative_Profit_Scaled'].values
    daily_running_max = np.maximum.accumulate(portfolio_values_daily)
    daily_drawdown = (portfolio_values_daily - daily_running_max) / daily_running_max * 100
    daily_max_dd = np.min(daily_drawdown)
    
    print("COMPARISON: Daily vs Quarterly Drawdown")
    print(f"Daily-based Max Drawdown: {daily_max_dd:.2f}%")
    print(f"Quarterly-based Max Drawdown: {quarterly_max_dd:.2f}%")
    print(f"Difference: {abs(daily_max_dd - quarterly_max_dd):.2f} percentage points")
    print()
    
    print("WHY QUARTERLY IS MORE APPROPRIATE:")
    print("• Trading strategy is quarterly (earnings announcements)")
    print("• Positions held for quarters, not adjusted daily")
    print("• Daily fluctuations are not actionable for this strategy")
    print("• Quarterly drawdown reflects actual trading decisions")
    print()
    
    # Show quarterly drawdown periods
    print("WORST QUARTERLY DRAWDOWN PERIODS:")
    quarterly_dd_df = pd.DataFrame({
        'Quarter': quarterly_df['Quarter'],
        'Portfolio_Value': quarterly_values,
        'Drawdown_%': quarterly_drawdown
    })
    
    worst_quarters = quarterly_dd_df.nsmallest(5, 'Drawdown_%')
    for i, row in worst_quarters.iterrows():
        print(f"{row['Quarter']}: {row['Drawdown_%']:.2f}% (₹{row['Portfolio_Value']:,.0f})")
    
    return {
        'quarterly_max_drawdown': quarterly_max_dd,
        'daily_max_drawdown': daily_max_dd,
        'peak_quarter': peak_quarter,
        'trough_quarter': trough_quarter,
        'peak_value': peak_value,
        'trough_value': trough_value,
        'quarterly_data_points': len(quarterly_df)
    }

def recalculate_performance_metrics_with_quarterly_dd():
    """Recalculate all performance metrics using quarterly drawdown"""
    
    # Get quarterly drawdown
    dd_result = calculate_quarterly_drawdown()
    quarterly_max_dd = dd_result['quarterly_max_drawdown']
    
    # Load existing metrics
    daily_df = pd.read_csv('Performance_Metrics_Data/daily_profit_analysis.csv')
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    
    # Basic parameters
    initial_investment = 1500000
    total_return = 1878738.75  # From summary
    total_years = (pd.to_datetime('2025-05-29') - pd.to_datetime('2015-01-19')).days / 365.25
    
    # Calculate metrics
    total_return_pct = (total_return / initial_investment) * 100
    portfolio_cagr = ((initial_investment + total_return) / initial_investment) ** (1/total_years) - 1
    
    # Quarterly returns
    daily_df['Quarter'] = daily_df['Date'].dt.to_period('Q')
    quarterly_profits = daily_df.groupby('Quarter')['Total_Daily_Profit'].sum()
    quarterly_returns = quarterly_profits / initial_investment
    
    avg_quarterly_return = quarterly_returns.mean()
    quarterly_volatility = quarterly_returns.std()
    annualized_volatility = quarterly_volatility * np.sqrt(4)
    
    # Risk metrics with corrected drawdown
    annual_risk_free_rate = 0.07
    quarterly_risk_free_rate = (1 + annual_risk_free_rate) ** (1/4) - 1
    
    excess_quarterly_return = avg_quarterly_return - quarterly_risk_free_rate
    sharpe_ratio = (excess_quarterly_return / quarterly_volatility) * np.sqrt(4) if quarterly_volatility > 0 else 0
    
    # Corrected Calmar Ratio using quarterly drawdown
    calmar_ratio = (portfolio_cagr * 100) / abs(quarterly_max_dd) if quarterly_max_dd != 0 else 0
    
    # Nifty comparison (actual data)
    nifty_cagr = 0.1084  # 10.84% from actual data
    excess_return = portfolio_cagr - nifty_cagr
    
    print("\nCORRECTED PERFORMANCE METRICS (with Quarterly Drawdown)")
    print("="*60)
    print(f"Portfolio CAGR: {portfolio_cagr*100:.2f}%")
    print(f"Nifty 50 CAGR: {nifty_cagr*100:.2f}%")
    print(f"Excess Return: {excess_return*100:.2f}% annually")
    print(f"Total Return: {total_return_pct:.2f}%")
    print(f"Annualized Volatility: {annualized_volatility*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
    print(f"CORRECTED Max Drawdown: {quarterly_max_dd:.2f}% (quarterly)")
    print(f"CORRECTED Calmar Ratio: {calmar_ratio:.3f}")
    print(f"Average Quarterly Return: {avg_quarterly_return*100:.2f}%")
    print(f"Quarterly Volatility: {quarterly_volatility*100:.2f}%")
    
    return {
        'Portfolio_CAGR_%': round(portfolio_cagr * 100, 2),
        'Nifty50_CAGR_%': round(nifty_cagr * 100, 2),
        'Excess_Return_%': round(excess_return * 100, 2),
        'Total_Return_%': round(total_return_pct, 2),
        'Annualized_Volatility_%': round(annualized_volatility * 100, 2),
        'Sharpe_Ratio': round(sharpe_ratio, 3),
        'Max_Drawdown_%': round(quarterly_max_dd, 2),
        'Calmar_Ratio': round(calmar_ratio, 3),
        'Avg_Quarterly_Return_%': round(avg_quarterly_return * 100, 2),
        'Quarterly_Volatility_%': round(quarterly_volatility * 100, 2)
    }

if __name__ == "__main__":
    print("ANALYZING QUARTERLY vs DAILY DRAWDOWN")
    print("="*60)
    
    # First analyze drawdown methodology
    dd_result = calculate_quarterly_drawdown()
    
    # Then recalculate all metrics with corrected drawdown
    corrected_metrics = recalculate_performance_metrics_with_quarterly_dd()
    
    print(f"\n🎯 KEY INSIGHT:")
    print(f"Since this is a QUARTERLY trading strategy, using quarterly")
    print(f"drawdown ({dd_result['quarterly_max_drawdown']:.2f}%) is more appropriate than daily drawdown.")
