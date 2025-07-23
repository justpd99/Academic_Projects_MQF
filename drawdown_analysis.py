import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_drawdown_calculation():
    """Analyze how maximum drawdown is calculated and verify the method"""
    
    # Load the data
    daily_df = pd.read_csv('Performance_Metrics_Data/daily_profit_analysis.csv')
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    
    # Portfolio parameters
    initial_investment = 1500000  # ₹15 lakh
    
    # Scale to correct total (from portfolio summary)
    summary_df = pd.read_csv('Performance_Metrics_Data/portfolio_summary.csv')
    correct_total_profit = summary_df[summary_df['Company'] == 'PORTFOLIO TOTAL']['Total_Exact_Profit'].iloc[0]
    scale_factor = correct_total_profit / daily_df['Cumulative_Profit'].iloc[-1]
    daily_df['Cumulative_Profit_Scaled'] = daily_df['Cumulative_Profit'] * scale_factor
    
    # Calculate portfolio values (this is what's used for drawdown)
    portfolio_values = initial_investment + daily_df['Cumulative_Profit_Scaled'].values
    
    # Calculate running maximum (peak values)
    running_max = np.maximum.accumulate(portfolio_values)
    
    # Calculate drawdown at each point
    drawdown_values = (portfolio_values - running_max) / running_max * 100
    
    # Find maximum drawdown
    max_drawdown = np.min(drawdown_values)
    max_drawdown_idx = np.argmin(drawdown_values)
    max_drawdown_date = daily_df['Date'].iloc[max_drawdown_idx]
    
    # Find the peak before this drawdown
    peak_idx = np.where(running_max == running_max[max_drawdown_idx])[0][0]
    peak_date = daily_df['Date'].iloc[peak_idx]
    peak_value = portfolio_values[peak_idx]
    trough_value = portfolio_values[max_drawdown_idx]
    
    print("MAXIMUM DRAWDOWN ANALYSIS")
    print("="*50)
    print(f"Calculation Method: DAILY portfolio values")
    print(f"Data Points: {len(daily_df)} daily observations")
    print(f"Period: {daily_df['Date'].iloc[0].date()} to {daily_df['Date'].iloc[-1].date()}")
    print()
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"Peak Date: {peak_date.date()}")
    print(f"Trough Date: {max_drawdown_date.date()}")
    print(f"Peak Value: ₹{peak_value:,.0f}")
    print(f"Trough Value: ₹{trough_value:,.0f}")
    print(f"Absolute Loss: ₹{peak_value - trough_value:,.0f}")
    print()
    
    # Check if this is based on daily data vs quarterly data
    print("DATA FREQUENCY ANALYSIS:")
    print(f"• Using DAILY cumulative profit data ({len(daily_df)} points)")
    print(f"• NOT using quarterly returns (which would be {len(daily_df.groupby(daily_df['Date'].dt.to_period('Q')))} points)")
    print(f"• This gives a more accurate and granular drawdown measurement")
    print()
    
    # Compare with quarterly-based calculation
    quarterly_df = daily_df.groupby(daily_df['Date'].dt.to_period('Q')).last().reset_index()
    quarterly_values = initial_investment + quarterly_df['Cumulative_Profit_Scaled'].values
    quarterly_running_max = np.maximum.accumulate(quarterly_values)
    quarterly_drawdown = (quarterly_values - quarterly_running_max) / quarterly_running_max * 100
    quarterly_max_dd = np.min(quarterly_drawdown)
    
    print("COMPARISON: Daily vs Quarterly Drawdown Calculation")
    print(f"Daily-based Max Drawdown: {max_drawdown:.2f}%")
    print(f"Quarterly-based Max Drawdown: {quarterly_max_dd:.2f}%")
    print(f"Difference: {abs(max_drawdown - quarterly_max_dd):.2f} percentage points")
    print()
    
    # Show some examples of drawdown periods
    print("DRAWDOWN PERIODS (Top 5 worst):")
    drawdown_df = pd.DataFrame({
        'Date': daily_df['Date'],
        'Portfolio_Value': portfolio_values,
        'Running_Max': running_max,
        'Drawdown_%': drawdown_values
    })
    
    worst_drawdowns = drawdown_df.nsmallest(5, 'Drawdown_%')
    for i, row in worst_drawdowns.iterrows():
        print(f"{row['Date'].date()}: {row['Drawdown_%']:.2f}% (₹{row['Portfolio_Value']:,.0f})")
    
    return {
        'method': 'Daily portfolio values',
        'data_points': len(daily_df),
        'max_drawdown': max_drawdown,
        'peak_date': peak_date,
        'trough_date': max_drawdown_date,
        'peak_value': peak_value,
        'trough_value': trough_value
    }

if __name__ == "__main__":
    result = analyze_drawdown_calculation()
