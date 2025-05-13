import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
from datetime import datetime
from skforecast.recursive._forecaster_recursive import ForecasterRecursive

# Define helper functions
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error given the true and
    predicted values
    
    Args:
        - y_true: true values
        - y_pred: predicted values
    
    Returns:
        - mape: MAPE value for the given predicted values
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def create_features(df, target_col, lag_periods=48, forecast_horizon=48):
    """
    Create lag features for time series forecasting
    
    Args:
        df: Dataframe containing time series data
        target_col: Target column to create lags for
        lag_periods: Number of lag periods to create
        forecast_horizon: Number of periods ahead to forecast
        
    Returns:
        X: Feature dataframe
        y: Target series
    """
    data = df.copy()
    
    # Create lag features
    for lag in range(1, lag_periods + 1):
        data[f'lag_{lag}'] = data[target_col].shift(lag)
    
    # Create the target variable (forecast_horizon steps ahead)
    data['target'] = data[target_col].shift(-forecast_horizon)
    
    # Drop NaN values that result from shifting
    data = data.dropna()
    
    # Separate features and target
    y = data['target']
    X = data.drop('target', axis=1)
    
    return X, y

def train_test_split_ts(X, y, test_size=0.2):
    """
    Split time series data chronologically into train and test sets
    
    Args:
        X: Feature dataframe
        y: Target series
        test_size: Fraction of data to use for testing
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test

def add_datepart(df):
    # Convert 'settlement_date' to datetime
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])

    # Calculate the period offset in minutes
    df["period_offset"] = pd.to_timedelta((df["settlement_period"] - 1) * 30, unit="m")

    # Add the period offset to the settlement_date to get the timestamp
    df["timestamp"] = df["settlement_date"] + df["period_offset"]

    # Create time-related features from timestamp
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["timestamp"].dt.month
    df["quarter"] = df["timestamp"].dt.quarter
    df["is_month_end"] = df["timestamp"].dt.is_month_end.astype(int)
    df["is_month_start"] = df["timestamp"].dt.is_month_start.astype(int)
    df["is_quarter_end"] = df["timestamp"].dt.is_quarter_end.astype(int)
    df["is_quarter_start"] = df["timestamp"].dt.is_quarter_start.astype(int)
    df["is_year_end"] = df["timestamp"].dt.is_year_end.astype(int)
    df["is_year_start"] = df["timestamp"].dt.is_year_start.astype(int)
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    return df

def recursive_forecast(model, X_test, steps=48):
    """
    Make recursive forecasts using an XGBoost model
    
    Args:
        model: Trained XGBoost model
        X_test: Initial test data with lag features
        steps: Number of steps to forecast
        
    Returns:
        Forecasted values
    """
    data = X_test.iloc[0:1].copy()
    forecasts = []
    
    # Get the initial lag feature names
    lag_cols = [col for col in data.columns if 'lag_' in col]
    lag_cols.sort(key=lambda x: int(x.split('_')[1]))
    
    # Get the target column (will be used to update lag features)
    target_col = 'tsd'
    
    # Make recursive predictions
    for i in range(steps):
        if i % 100 == 0 or i == steps - 1:
            print(f"Forecasting step {i+1}/{steps}...")
            
        # Make prediction for the current step
        pred = model.predict(data.iloc[-1:])
        forecasts.append(pred[0])
        
        # If we've reached the desired forecast horizon, stop
        if i == steps - 1:
            break
        
        # Prepare data for the next prediction by shifting lag features
        last_row = data.iloc[-1].copy()
        
        # Shift lag values
        for j in range(1, len(lag_cols)):
            last_row[lag_cols[j-1]] = last_row[lag_cols[j]]
        
        # The most recent lag gets the predicted value
        last_row[lag_cols[-1]] = last_row[target_col]
        
        # The target value becomes our prediction
        last_row[target_col] = pred[0]
        
        # Add the updated row to our data
        data = pd.concat([data, pd.DataFrame([last_row])], ignore_index=True)
    
    return forecasts

def direct_forecast(model, X_test, steps=48, lag_periods=48):
    """
    Make direct forecasts for multiple horizons using an XGBoost model
    
    Args:
        model: Dictionary of trained XGBoost models for each horizon
        X_test: Test data with lag features
        steps: Number of steps to forecast
        lag_periods: Number of lag periods used
        
    Returns:
        Forecasted values
    """
    forecasts = []
    
    # Get the first row of test data
    initial_data = X_test.iloc[0:1].copy()
    
    # For each horizon, use the appropriate model to make a prediction
    for i in range(steps):
        if i % 100 == 0 or i == steps - 1:
            print(f"Direct forecasting step {i+1}/{steps}...")
            
        # Load the appropriate model if available, otherwise use the last available model
        if i < len(model):
            horizon_model = model[i]
        else:
            # If we don't have a model for this horizon, use the model for the last horizon
            horizon_model = model[len(model)-1]
        
        # Make prediction for this horizon
        pred = horizon_model.predict(initial_data)
        forecasts.append(pred[0])
    
    return forecasts

def create_skforecast_forecaster(xgb_params, lags=48):
    """
    Create a ForecasterRecursive with XGBoost
    
    Args:
        xgb_params: XGBoost parameters
        lags: Number of lags to use
        
    Returns:
        Configured forecaster
    """
    # Create XGBoost regressor with provided parameters
    regressor = xgb.XGBRegressor(**xgb_params)
    
    # Create lag features
    lags_grid = list(range(1, lags+1))
    
    # Initialize the forecaster
    forecaster = ForecasterRecursive(
        regressor=regressor,
        lags=lags_grid
    )
    
    return forecaster

def compare_forecast_methods(df, lag_periods=48, forecast_horizon=48, test_size=0.2):
    """
    Compare different forecasting methods:
    1. Regular XGBoost (direct prediction)
    2. Recursive XGBoost
    3. SKForecast ForecasterAutoReg
    
    Args:
        df: DataFrame with time series data
        lag_periods: Number of lag periods to use
        forecast_horizon: Forecasting horizon
        test_size: Proportion of data for testing
        
    Returns:
        DataFrame with comparison results
    """
    print("\n===== COMPARING FORECAST METHODS =====")
    
    # Create features for xgboost models
    TARGET_COL = 'tsd'
    
    # For regular XGBoost (direct prediction)
    print("\n1. Setting up Regular XGBoost (direct prediction)...")
    X, y = create_features(df, TARGET_COL, lag_periods, forecast_horizon=1)  # 1-step ahead for direct prediction
    X_train, X_test, y_train, y_test = train_test_split_ts(X, y, test_size)
    
    # For recursive forecasting
    print("\n2. Setting up Recursive XGBoost...")
    X_recursive, y_recursive = create_features(df, TARGET_COL, lag_periods, forecast_horizon=1)
    X_train_rec, X_test_rec, y_train_rec, y_test_rec = train_test_split_ts(X_recursive, y_recursive, test_size)
    
    # For SKForecast
    print("\n3. Setting up SKForecast...")
    series = df[TARGET_COL].copy()
    split_idx = int(len(series) * (1 - test_size))
    train_series = series.iloc[:split_idx]
    test_series = series.iloc[split_idx:]
    
    # XGBoost parameters
    xgb_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'random_state': 0
    }
    
    # Train Regular XGBoost model
    print("\nTraining Regular XGBoost model...")
    regular_start_time = time.time()
    regular_model = xgb.XGBRegressor(**xgb_params)
    regular_model.fit(X_train, y_train)
    regular_train_time = time.time() - regular_start_time
    print(f"Regular XGBoost training completed in {regular_train_time:.2f} seconds")
    
    # Train Recursive XGBoost model
    print("\nTraining Recursive XGBoost model...")
    recursive_start_time = time.time()
    recursive_model = xgb.XGBRegressor(**xgb_params)
    recursive_model.fit(X_train_rec, y_train_rec)
    recursive_train_time = time.time() - recursive_start_time
    print(f"Recursive XGBoost training completed in {recursive_train_time:.2f} seconds")
    
    # Create and train SKForecast model
    print("\nTraining SKForecast model...")
    skf_start_time = time.time()
    forecaster = create_skforecast_forecaster(xgb_params, lags=lag_periods)
    forecaster.fit(y=train_series)
    skf_train_time = time.time() - skf_start_time
    print(f"SKForecast training completed in {skf_train_time:.2f} seconds")
    
    # Make predictions with each method
    # 1. Regular XGBoost direct prediction
    print("\nMaking Regular XGBoost predictions...")
    direct_start_time = time.time()
    direct_preds = direct_forecast(regular_model, X_test.iloc[:forecast_horizon])
    direct_pred_time = time.time() - direct_start_time
    
    # 2. Recursive XGBoost prediction
    print("\nMaking Recursive XGBoost predictions...")
    recursive_start_time = time.time()
    recursive_preds = recursive_forecast(recursive_model, X_test_rec.iloc[:1], steps=forecast_horizon)
    recursive_pred_time = time.time() - recursive_start_time
    
    # 3. SKForecast prediction
    print("\nMaking SKForecast predictions...")
    skf_start_time = time.time()
    skf_preds = forecaster.predict(steps=forecast_horizon)
    skf_pred_time = time.time() - skf_start_time
    
    # Get actual values
    actuals = test_series.iloc[:forecast_horizon].values
    
    # Calculate metrics for each method
    # Regular XGBoost
    direct_mae = mean_absolute_error(actuals, direct_preds)
    direct_rmse = np.sqrt(mean_squared_error(actuals, direct_preds))
    direct_mape = mean_absolute_percentage_error(actuals, direct_preds)
    
    # Recursive XGBoost
    recursive_mae = mean_absolute_error(actuals, recursive_preds)
    recursive_rmse = np.sqrt(mean_squared_error(actuals, recursive_preds))
    recursive_mape = mean_absolute_percentage_error(actuals, recursive_preds)
    
    # SKForecast
    skf_mae = mean_absolute_error(actuals, skf_preds.values)
    skf_rmse = np.sqrt(mean_squared_error(actuals, skf_preds.values))
    skf_mape = mean_absolute_percentage_error(actuals, skf_preds.values)
    
    # Calculate average MAPE across horizons for each method
    all_horizon_mapes = []
    
    # For regular direct prediction
    direct_horizon_mapes = []
    for horizon in range(1, forecast_horizon+1):
        if horizon % 10 == 0 or horizon == 1 or horizon == forecast_horizon:
            print(f"Evaluating horizon {horizon} for direct prediction...")
        horizon_preds = direct_preds[:horizon]
        horizon_actuals = actuals[:horizon]
        mape = mean_absolute_percentage_error(horizon_actuals, horizon_preds)
        direct_horizon_mapes.append(mape)
    avg_direct_mape = np.mean(direct_horizon_mapes)
    
    # For recursive prediction
    recursive_horizon_mapes = []
    for horizon in range(1, forecast_horizon+1):
        if horizon % 10 == 0 or horizon == 1 or horizon == forecast_horizon:
            print(f"Evaluating horizon {horizon} for recursive prediction...")
        horizon_preds = recursive_preds[:horizon]
        horizon_actuals = actuals[:horizon]
        mape = mean_absolute_percentage_error(horizon_actuals, horizon_preds)
        recursive_horizon_mapes.append(mape)
    avg_recursive_mape = np.mean(recursive_horizon_mapes)
    
    # For SKForecast
    skf_horizon_mapes = []
    for horizon in range(1, forecast_horizon+1):
        if horizon % 10 == 0 or horizon == 1 or horizon == forecast_horizon:
            print(f"Evaluating horizon {horizon} for SKForecast...")
        horizon_preds = skf_preds.values[:horizon]
        horizon_actuals = actuals[:horizon]
        mape = mean_absolute_percentage_error(horizon_actuals, horizon_preds)
        skf_horizon_mapes.append(mape)
    avg_skf_mape = np.mean(skf_horizon_mapes)
    
    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'Metric': ['Training Time (s)', 'Prediction Time (s)', 'MAE', 'RMSE', 'MAPE (%)', 'Avg Horizon MAPE (%)'],
        'Regular XGBoost': [regular_train_time, direct_pred_time, direct_mae, direct_rmse, direct_mape, avg_direct_mape],
        'Recursive XGBoost': [recursive_train_time, recursive_pred_time, recursive_mae, recursive_rmse, recursive_mape, avg_recursive_mape],
        'SKForecast': [skf_train_time, skf_pred_time, skf_mae, skf_rmse, skf_mape, avg_skf_mape]
    })
    
    # Plot comparison
    plt.figure(figsize=(15, 7))
    plt.plot(range(forecast_horizon), actuals, label='Actual', marker='o', linestyle='-', alpha=0.7)
    plt.plot(range(forecast_horizon), direct_preds, label=f'Regular XGBoost (Avg MAPE: {avg_direct_mape:.2f}%)', marker='^', linestyle='--', alpha=0.7)
    plt.plot(range(forecast_horizon), recursive_preds, label=f'Recursive XGBoost (Avg MAPE: {avg_recursive_mape:.2f}%)', marker='x', linestyle='--', alpha=0.7)
    plt.plot(range(forecast_horizon), skf_preds.values, label=f'SKForecast (Avg MAPE: {avg_skf_mape:.2f}%)', marker='s', linestyle='--', alpha=0.7)
    plt.title('Comparison of Forecasting Methods')
    plt.xlabel('Forecast Horizon')
    plt.ylabel('Demand (tsd)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('forecast_methods_comparison.png')
    plt.show()
    
    # Plot MAPE by horizon for each method
    plt.figure(figsize=(15, 7))
    plt.plot(range(1, forecast_horizon+1), direct_horizon_mapes, label='Regular XGBoost', marker='^', alpha=0.7)
    plt.plot(range(1, forecast_horizon+1), recursive_horizon_mapes, label='Recursive XGBoost', marker='x', alpha=0.7)
    plt.plot(range(1, forecast_horizon+1), skf_horizon_mapes, label='SKForecast', marker='s', alpha=0.7)
    plt.title('MAPE by Horizon for Different Forecast Methods')
    plt.xlabel('Forecast Horizon')
    plt.ylabel('MAPE (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('forecast_methods_mape_by_horizon.png')
    plt.show()
    
    # Save results to CSV
    comparison.to_csv('forecast_methods_comparison.csv', index=False)
    
    print("\n===== FORECAST METHODS COMPARISON RESULTS =====")
    print(comparison)
    print("\nResults saved to forecast_methods_comparison.csv")
    print("Plots saved as forecast_methods_comparison.png and forecast_methods_mape_by_horizon.png")
    
    return comparison

# Main execution
if __name__ == "__main__":
    print("Loading and preprocessing data...")
    df = pd.read_csv("historic_demand_2009_2024_noNaN.csv", index_col=0)
    
    # Filter out unneeded columns and prepare data
    df = df[['settlement_date', 'settlement_period', 'tsd', 'is_holiday']]
    
    # Add date-based features
    add_datepart(df)
    
    # Clean up and set the timestamp as index
    df.drop(columns=["period_offset", "settlement_date"], inplace=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    
    # Compare the forecast methods
    comparison_results = compare_forecast_methods(
        df, 
        lag_periods=48, 
        forecast_horizon=48, 
        test_size=0.2
    )