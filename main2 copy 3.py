# Cell 1: Introduction and Imports
# # Time Series Forecasting with XGBoost
# ## Predicting Energy Demand with Lag Features and Recursive Forecasting

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import pmdarima as pm  # Commented out as not currently used
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import time
from datetime import datetime

# This notebook implements energy demand forecasting using XGBoost with:
# - Time series feature engineering
# - Direct and recursive forecasting methods
# - Multi-horizon forecast evaluation
# - Performance visualization and comparison

# Cell 2: Data Loading and Preprocessing
# ## Loading and Cleaning the Dataset

# Load the historical demand dataset
df = pd.read_csv("historic_demand_2009_2024.csv", index_col=0)

# Print basic information about the loaded dataset
print("Original dataset shape:", df.shape)
print("Column names:", df.columns.tolist())

# Remove columns with null values
# Note: These columns contain values that started appearing after a specific year
# and might be useful in future analyses
df.drop(columns=["nsl_flow", "eleclink_flow", "scottish_transfer", "viking_flow", "greenlink_flow"], axis=1, inplace=True)

# Drop rows where settlement_period value is greater than 48 (data quality issue)
df.drop(index=df[df["settlement_period"] > 48].index, inplace=True)

df.reset_index(drop=True, inplace=True)

# Select only the columns we need for this analysis
df = df[['settlement_date', 'settlement_period', 'tsd', 'is_holiday']]

# Remove days with zero demand (likely data quality issues)
null_days = df.loc[df["tsd"] == 0.0, "settlement_date"].unique().tolist()
null_days_index = []

for day in null_days:
    null_days_index.append(df[df["settlement_date"] == day].index.tolist())

null_days_index = [item for sublist in null_days_index for item in sublist]

df.drop(index=null_days_index, inplace=True)
df.reset_index(drop=True, inplace=True)

# Display the cleaned dataset
print("\nCleaned dataset shape:", df.shape)
print("First 5 rows of cleaned data:")
print(df.head())
print("\nData summary statistics:")
print(df.describe())

# Cell 3: Feature Engineering
# ## Creating Datetime Features

def add_datepart(df):
    """
    Create comprehensive datetime features from the settlement date and period.
    
    Args:
        df (DataFrame): DataFrame containing settlement_date and settlement_period columns
        
    Returns:
        DataFrame: DataFrame with added datetime features
    """
    # Convert 'settlement_date' to datetime (ensure it's in the correct format)
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])

    # Ensure that 'settlement_period' is an integer and calculate the period offset in minutes
    df["period_offset"] = pd.to_timedelta((df["settlement_period"] - 1) * 30, unit="m")

    # Add the period offset (Timedelta) to the settlement_date (Datetime) to get the timestamp
    df["timestamp"] = df["settlement_date"] + df["period_offset"]

    # Ensure 'timestamp' is in datetime format (in case it's not already)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Create time-related features from timestamp
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # Monday=0, Sunday=6
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

# Apply the datetime feature engineering function
print("Adding datetime features...")
df = add_datepart(df)

# Clean up and set the timestamp as index
df.drop(columns=["period_offset", "settlement_date"], inplace=True)
df.set_index("timestamp", inplace=True)
df.sort_index(inplace=True)

# Show the dataframe with new features
print("\nDataset with datetime features:")
print(df.head())
print(f"Number of features: {df.shape[1]}")

# Cell 4: Data Exploration
# ## Exploring Data Patterns and Correlations

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error given the true and predicted values.

    Args:
        y_true (array-like): Ground truth values
        y_pred (array-like): Predicted values

    Returns:
        float: MAPE value for the given predicted values (as percentage)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

# Plot the distribution of the target variable (tsd)
plt.figure(figsize=(12, 6))
plt.hist(df['tsd'], bins=50, alpha=0.7)
plt.title('Distribution of Energy Demand (tsd)')
plt.xlabel('Demand Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

# Display correlation matrix heatmap
print("Generating correlation matrix heatmap...")
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()

# Plot time series of the target variable
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['tsd'])
plt.title('Time Series of Energy Demand')
plt.xlabel('Date')
plt.ylabel('Demand (tsd)')
plt.grid(True, alpha=0.3)
plt.show()

# Examine patterns by hour of day
hourly_avg = df.groupby(df.index.hour)['tsd'].mean()
plt.figure(figsize=(12, 6))
plt.bar(hourly_avg.index, hourly_avg.values)
plt.title('Average Energy Demand by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Average Demand')
plt.grid(True, alpha=0.3)
plt.show()

# Examine patterns by day of week
daily_avg = df.groupby('day_of_week')['tsd'].mean()
plt.figure(figsize=(12, 6))
plt.bar(daily_avg.index, daily_avg.values)
plt.title('Average Energy Demand by Day of Week')
plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
plt.ylabel('Average Demand')
plt.grid(True, alpha=0.3)
plt.show()

# Cell 5: Time Series Feature Creation
# ## Creating Lag Features for Forecasting

def create_features(df, target_col, lag_periods=48, forecast_horizon=48):
    """
    Create lag features for time series forecasting.
    
    Args:
        df (DataFrame): Dataframe containing time series data
        target_col (str): Target column to create lags for
        lag_periods (int): Number of lag periods to create
        forecast_horizon (int): Number of periods ahead to forecast
        
    Returns:
        tuple: (X, y) Feature dataframe and target series
    """
    # Create a copy of the dataframe to avoid modifying the original
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

# Cell 6: Train-Test Split for Time Series
# ## Preparing Data for Model Training

def train_test_split_ts(X, y, test_size=0.2):
    """
    Split time series data chronologically into train and test sets.
    
    Args:
        X (DataFrame): Feature dataframe
        y (Series): Target series
        test_size (float): Fraction of data to use for testing
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test

# Cell 7: Training Progress Callback
# ## Custom Callback for XGBoost Training

class TrainingProgressCallback:
    """
    Custom callback to track and display XGBoost training progress.
    
    This callback provides real-time monitoring of training progress,
    displays metrics at regular intervals, and supports early stopping.
    """
    def __init__(self, start_time=None, early_stopping_rounds=None, print_interval=10):
        self.start_time = start_time or time.time()
        self.early_stopping_rounds = early_stopping_rounds
        self.print_interval = print_interval
        self.best_score = float('inf')
        self.best_iteration = 0
        self.current_rounds_no_improve = 0
        
    def __call__(self, env):
        """Called after each iteration"""
        # Extract evaluation results
        score = env.evaluation_result_list[1][1]  # Validation score
        iteration = env.iteration
        
        # Track early stopping
        if score < self.best_score:
            self.best_score = score
            self.best_iteration = iteration
            self.current_rounds_no_improve = 0
        else:
            self.current_rounds_no_improve += 1
        
        # Print status on selected intervals or if it's the last iteration
        if iteration % self.print_interval == 0 or (self.early_stopping_rounds and 
                                                  self.current_rounds_no_improve >= self.early_stopping_rounds):
            elapsed_time = time.time() - self.start_time
            remaining = "Unknown"
            
            if iteration > 0:
                time_per_iter = elapsed_time / (iteration + 1)
                if self.early_stopping_rounds and self.current_rounds_no_improve < self.early_stopping_rounds:
                    estimated_remaining = (env.end_iteration - iteration) * time_per_iter
                    remaining = f"{estimated_remaining:.1f} sec"
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Iteration {iteration:4d} | "
                  f"Train RMSE: {env.evaluation_result_list[0][1]:.4f} | "
                  f"Valid RMSE: {score:.4f} | "
                  f"Best: {self.best_score:.4f} @ {self.best_iteration} | "
                  f"No improve: {self.current_rounds_no_improve}/{self.early_stopping_rounds or 'None'} | "
                  f"Elapsed: {elapsed_time:.1f} sec | "
                  f"Remaining: {remaining}")
        
        # Stop if we've reached the stopping rounds
        if self.early_stopping_rounds and self.current_rounds_no_improve >= self.early_stopping_rounds:
            print(f"\nEarly stopping at iteration {iteration}. Best score: {self.best_score:.4f} at iteration {self.best_iteration}")
            return True
            
        return False

# Cell 8: Recursive Forecasting Implementation
# ## Function for Multi-Step Ahead Predictions

def recursive_forecast(model, X_test, steps=48):
    """
    Make recursive forecasts using an XGBoost model.
    
    This function makes multi-step forecasts by using each prediction as
    an input feature for the next prediction (recursive approach).
    
    Args:
        model (XGBRegressor): Trained XGBoost model
        X_test (DataFrame): Initial test data with lag features
        steps (int): Number of steps to forecast
        
    Returns:
        list: Forecasted values
    """
    # Make a copy of the test data to avoid modifying the original
    data = X_test.iloc[0:1].copy()
    
    # Array to store forecasts
    forecasts = []
    
    # Get the initial lag feature names
    lag_cols = [col for col in data.columns if 'lag_' in col]
    lag_cols.sort(key=lambda x: int(x.split('_')[1]))
    
    # Get the target column (will be used to update lag features)
    target_col = 'tsd'
    
    # Show progress bar for recursive forecasting
    print("\nMaking recursive forecasts...")
    
    # Make recursive predictions
    for i in range(steps):
        if i % 10 == 0 or i == steps - 1:
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
    
    print("Recursive forecasting complete.")
    return forecasts

# Cell 9: Model Training Setup
# ## Setting Parameters and Creating Features

# Setting parameters for forecasting
TARGET_COL = 'tsd'
LAG_PERIODS = 48  # One day of lags
FORECAST_HORIZON = 48  # One day ahead forecast
TEST_SIZE = 0.2

# Create features with lag values
print("Creating lag features...")
X, y = create_features(df, TARGET_COL, LAG_PERIODS, FORECAST_HORIZON)
print(f"Feature shape: {X.shape}, Target shape: {y.shape}")

# Split data chronologically into train and test sets
X_train, X_test, y_train, y_test = train_test_split_ts(X, y, TEST_SIZE)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
print(f"Training period: {X_train.index.min()} to {X_train.index.max()}")
print(f"Testing period: {X_test.index.min()} to {X_test.index.max()}")

# Cell 10: Model Training
# ## Training XGBoost Model

# Initialize and train XGBoost model
print("\nInitializing XGBoost model with the following parameters:")
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

for param, value in xgb_params.items():
    print(f"  {param}: {value}")

print("\nTraining XGBoost model...")
xgb_model = xgb.XGBRegressor(**xgb_params)

# Start timing the training
training_start_time = time.time()

# Create callback for early stopping
progress_callback = TrainingProgressCallback(start_time=training_start_time, early_stopping_rounds=50, print_interval=10)

# Train the model
xgb_model.fit(
    X_train, 
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=True
)

training_time = time.time() - training_start_time
print(f"\nTraining completed in {training_time:.2f} seconds")
# print(f"Best iteration: {xgb_model.best_iteration}, Best score: {xgb_model.best_score:.4f}")

# Cell 11: Direct Forecasting Evaluation
# ## Evaluate One-Step-Ahead Predictions

# Make predictions on test set
print("\nMaking direct predictions on test set...")
prediction_start_time = time.time()
y_pred = xgb_model.predict(X_test)
prediction_time = time.time() - prediction_start_time
print(f"Predictions completed in {prediction_time:.2f} seconds")

# Calculate and display performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"\nDirect Prediction Results:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"Average Direct Prediction MAPE: {mape:.2f}%")  # Added line showing direct prediction MAPE

# Cell 12: Feature Importance Analysis
# ## Examining Important Features

# Plot feature importance
plt.figure(figsize=(12, 6))
xgb.plot_importance(xgb_model, max_num_features=15)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

# Create a DataFrame with feature importance values
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance_df.head(10))

# Cell 13: Recursive Forecasting
# ## Multi-Step Ahead Forecasting

# Now perform recursive forecasting
print("\nPerforming recursive forecasting...")
# Get a starting point from the test set
start_idx = 0
forecast_length = min(48, len(y_test))  # Forecast for up to 48 steps or the length of test set

# Get actual values for the forecast period
actual_values = y_test.iloc[start_idx:start_idx+forecast_length].values

# Get initial data for recursive forecasting
initial_data = X_test.iloc[start_idx:start_idx+1]

# Start timing recursive forecasting
recursive_start_time = time.time()

# Make recursive forecasts
recursive_preds = recursive_forecast(xgb_model, initial_data, steps=forecast_length)

recursive_time = time.time() - recursive_start_time
print(f"Recursive forecasting completed in {recursive_time:.2f} seconds")

# Calculate metrics for recursive forecasting
rec_mae = mean_absolute_error(actual_values, recursive_preds)
rec_mse = mean_squared_error(actual_values, recursive_preds)
rec_rmse = np.sqrt(rec_mse)
rec_mape = mean_absolute_percentage_error(actual_values, recursive_preds)

print(f"\nRecursive Prediction Results:")
print(f"MAE: {rec_mae:.2f}")
print(f"RMSE: {rec_rmse:.2f}")
print(f"MAPE: {rec_mape:.2f}%")
print(f"Average Recursive Prediction MAPE: {rec_mape:.2f}%")  # Added line showing recursive prediction MAPE

# Cell 14: Visualizing Recursive Forecast Results
# ## Comparing Predictions to Actual Values

# Plot recursive forecasting results
plt.figure(figsize=(15, 7))
plt.plot(actual_values, label='Actual', marker='o', linestyle='-', alpha=0.7)
plt.plot(recursive_preds, label='Recursive Forecast', marker='x', linestyle='--', alpha=0.7)
plt.title('Recursive Forecasting vs Actual Values')
plt.xlabel('Time Steps')
plt.ylabel('Demand (tsd)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Visualize predictions vs actual over time
plt.figure(figsize=(15, 7))
time_idx = y_test.index[start_idx:start_idx+forecast_length]
plt.plot(time_idx, actual_values, label='Actual', marker='o', linestyle='-', alpha=0.7)
plt.plot(time_idx, recursive_preds, label='Recursive Forecast', marker='x', linestyle='--', alpha=0.7)
plt.title('Recursive Forecasting vs Actual Values Over Time')
plt.xlabel('Date/Time')
plt.ylabel('Demand (tsd)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cell 15: Direct vs Recursive Forecasting Comparison
# ## Comparing Different Forecasting Approaches

# Compare direct vs recursive forecasting
plt.figure(figsize=(15, 7))
direct_preds = y_pred[start_idx:start_idx+forecast_length]
plt.plot(time_idx, actual_values, label='Actual', marker='o', linestyle='-', alpha=0.7)
plt.plot(time_idx, direct_preds, label='Direct Forecast', marker='^', linestyle='--', alpha=0.7)
plt.plot(time_idx, recursive_preds, label='Recursive Forecast', marker='x', linestyle='--', alpha=0.7)
plt.title('Direct vs Recursive Forecasting')
plt.xlabel('Date/Time')
plt.ylabel('Demand (tsd)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create a DataFrame to compare direct and recursive forecasts
comparison_df = pd.DataFrame({
    'Actual': actual_values,
    'Direct': direct_preds,
    'Recursive': recursive_preds
})
print("\nComparison of Direct vs Recursive Forecasts (First 10 rows):")
print(comparison_df.head(10))

# Cell 16: Multi-horizon Forecast Evaluation
# ## Evaluating Performance Across Different Horizons

def evaluate_multi_horizon(model, X_test, y_test, max_horizon=48):
    """
    Evaluate model performance across multiple forecast horizons.
    
    Args:
        model (XGBRegressor): Trained model
        X_test (DataFrame): Test features
        y_test (Series): Test targets
        max_horizon (int): Maximum forecast horizon to evaluate
        
    Returns:
        DataFrame: Performance metrics for each horizon
    """
    results = []
    
    print(f"\nEvaluating performance across forecast horizons (1 to {max_horizon})...")
    
    for horizon in range(1, max_horizon+1):
        if horizon % 10 == 0 or horizon == 1 or horizon == max_horizon:
            print(f"Evaluating horizon {horizon}/{max_horizon}...")
            
        # Make recursive forecasts for this horizon
        forecasts = []
        actuals = []
        
        # We need enough data in the test set to make a forecast for this horizon
        for i in range(len(X_test) - horizon):
            initial_data = X_test.iloc[i:i+1]
            forecast = recursive_forecast(model, initial_data, steps=horizon)
            forecasts.append(forecast[-1])  # We only care about the forecast at the target horizon
            actuals.append(y_test.iloc[i+horizon-1])
        
        # Calculate metrics
        if len(forecasts) > 0:
            mae = mean_absolute_error(actuals, forecasts)
            mse = mean_squared_error(actuals, forecasts)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(actuals, forecasts)
            
            results.append({
                'Horizon': horizon,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'Samples': len(forecasts)
            })
    
    return pd.DataFrame(results)

# Evaluate performance across multiple horizons (limited to a few samples for speed)
print("\nEvaluating performance across forecast horizons...")
max_samples = min(48, len(X_test) - 48)  # Use at most 48 samples
horizon_results = evaluate_multi_horizon(xgb_model, X_test[:max_samples], y_test[:max_samples], max_horizon=48)

# Save the horizon results to a CSV file
horizon_results.to_csv('forecast_horizon_metrics.csv', index=False)
print("Saved horizon metrics to forecast_horizon_metrics.csv")

# Display summary of horizon results
print("\nSummary of forecast performance by horizon:")
print(f"Average MAPE across all horizons: {horizon_results['MAPE'].mean():.2f}%")
print(f"Best MAPE: {horizon_results['MAPE'].min():.2f}% at horizon {horizon_results.loc[horizon_results['MAPE'].idxmin(), 'Horizon']}")
print(f"Worst MAPE: {horizon_results['MAPE'].max():.2f}% at horizon {horizon_results.loc[horizon_results['MAPE'].idxmax(), 'Horizon']}")

# Cell 17: Visualizing Performance by Horizon
# ## Plotting Error Metrics Across Forecast Horizons

# Plot results by horizon
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.plot(horizon_results['Horizon'], horizon_results['RMSE'], marker='o')
plt.title('RMSE by Forecast Horizon')
plt.xlabel('Forecast Horizon (periods)')
plt.ylabel('RMSE')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(horizon_results['Horizon'], horizon_results['MAPE'], marker='o')
plt.title('MAPE by Forecast Horizon')
plt.xlabel('Forecast Horizon (periods)')
plt.ylabel('MAPE (%)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot a more detailed view of horizon performance
plt.figure(figsize=(12, 6))
plt.plot(horizon_results['Horizon'], horizon_results['MAE'], marker='o', label='MAE')
plt.plot(horizon_results['Horizon'], horizon_results['RMSE'], marker='s', label='RMSE')
plt.title('Error Metrics by Forecast Horizon')
plt.xlabel('Forecast Horizon (periods)')
plt.ylabel('Error')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Cell 18: Saving Results
# ## Export Model and Forecasts

# Save model for later use
model_filename = 'xgboost_recursive_forecast_model.json'
xgb_model.save_model(model_filename)
print(f"Model saved to {model_filename}")

# Save predictions for further analysis
results_df = pd.DataFrame({
    'timestamp': time_idx,
    'actual': actual_values,
    'direct_forecast': direct_preds,
    'recursive_forecast': recursive_preds
})
results_df.to_csv('forecast_results.csv')
print("Saved forecast results to forecast_results.csv")

# Cell 19: Final Summary
# ## Performance Comparison and Conclusions

# Final summary of performance metrics
print("\n=== FINAL PERFORMANCE SUMMARY ===")
print(f"Direct Prediction Average MAPE: {mape:.2f}%")
print(f"Recursive Prediction Average MAPE: {rec_mape:.2f}%")
print(f"Difference: {abs(mape - rec_mape):.2f}%")
print(f"Percentage Improvement: {(abs(mape - rec_mape) / mape * 100):.2f}%")
print("==================================")

# Create a summary DataFrame for easy comparison
summary_df = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'MAPE (%)'],
    'Direct Forecasting': [mae, rmse, mape],
    'Recursive Forecasting': [rec_mae, rec_rmse, rec_mape],
    'Difference': [abs(mae - rec_mae), abs(rmse - rec_rmse), abs(mape - rec_mape)]
})
print("\nFull Performance Comparison:")
print(summary_df)

print("\nXGBoost Recursive Forecasting Model Complete.")
print("This notebook demonstrates both direct and recursive forecasting approaches for time series data.")
print("The results highlight the trade-offs between these approaches and provide insights for model selection.")

