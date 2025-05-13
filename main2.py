import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import pmdarima as pm
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import time
from datetime import datetime
from skforecast.recursive._forecaster_recursive import ForecasterRecursive

df = pd.read_csv("historic_demand_2009_2024.csv", index_col=0)

# removed columns with null values
# the values started appearing after a specific year, maybe we can use them in the recursive prediction

df.drop(columns=["nsl_flow", "eleclink_flow", "scottish_transfer", "viking_flow", "greenlink_flow"], axis=1, inplace=True)

# Drop rows where settlement_period value is greater than 48
df.drop(index=df[df["settlement_period"] > 48].index, inplace=True)

df.reset_index(drop=True, inplace=True)

df = df[['settlement_date', 'settlement_period', 'tsd', 'is_holiday']]

null_days = df.loc[df["tsd"] == 0.0, "settlement_date"].unique().tolist()

null_days_index = []

for day in null_days:
    null_days_index.append(df[df["settlement_date"] == day].index.tolist())

null_days_index = [item for sublist in null_days_index for item in sublist]

df.drop(index=null_days_index, inplace=True)
df.reset_index(drop=True, inplace=True)

def add_datepart(df):
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


add_datepart(df)

df.drop(columns=["period_offset", "settlement_date"], inplace=True)

df.set_index("timestamp", inplace=True)
df.sort_index(inplace=True)


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Pertange Error given the true and
    predicted values

    Args:
        - y_true: true values
        - y_pred: predicted values

    Returns:
        - mape: MAPE value for the given predicted values
    """

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

# Display correlation matrix heatmap
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()

# Create lag features for time series forecasting
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

# Split data into train and test sets chronologically
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

# Custom callback to track and display training progress
class TrainingProgressCallback:
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

# Recursive forecasting function
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

# Plot feature importance
plt.figure(figsize=(12, 6))
xgb.plot_importance(xgb_model, max_num_features=15)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

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

# Function to perform multi-step forecast evaluation
def evaluate_multi_horizon(model, X_test, y_test, max_horizon=48):
    """
    Evaluate model performance across multiple forecast horizons
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        max_horizon: Maximum forecast horizon to evaluate
        
    Returns:
        DataFrame with performance metrics for each horizon
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

# Now let's add and train a third model using SKForecast's ForecasterRecursive
print("\nTraining SKForecast ForecasterRecursive model...")
skf_start_time = time.time()

# Create the forecaster with the same XGBoost parameters for fair comparison
regressor = xgb.XGBRegressor(**xgb_params)
lags_grid = list(range(1, LAG_PERIODS + 1))

# Initialize the forecaster
forecaster = ForecasterRecursive(
    regressor=regressor,
    lags=lags_grid
)

# Extract the training series and align it with the exogenous features
# We need to make sure the indices align perfectly between train_series and X_train
train_series = df[TARGET_COL].loc[X_train.index[0]:X_train.index[-1]]

# # Make sure the series is the same length as X_train
# if len(train_series) != len(X_train):
#     print(f"Adjusting train_series length to match X_train: {len(train_series)} → {len(X_train)}")
#     train_series = train_series.loc[X_train.index]

# print(f"Train series: {train_series.index.min()} to {train_series.index.max()} (n={len(train_series)})")
# print(f"X_train: {X_train.index.min()} to {X_train.index.max()} (n={len(X_train)})")

# Fit the forecaster
forecaster.fit(y=train_series, exog=X_train)
skf_train_time = time.time() - skf_start_time
print(f"SKForecast training completed in {skf_train_time:.2f} seconds")

# Make predictions with SKForecast
print("\nMaking SKForecast predictions...")
skf_start_pred_time = time.time()
# Use the exact test indices for the forecast period
forecast_index = time_idx
# For SKForecast, we need the exogenous variables for the forecast period
exog_forecast = X_test.loc[forecast_index]
skf_preds = forecaster.predict(steps=forecast_length, exog=exog_forecast)
skf_pred_time = time.time() - skf_start_pred_time
print(f"SKForecast prediction completed in {skf_pred_time:.2f} seconds")

# Calculate metrics for SKForecast
skf_mae = mean_absolute_error(actual_values, skf_preds.values)
skf_mse = mean_squared_error(actual_values, skf_preds.values)
skf_rmse = np.sqrt(skf_mse)
skf_mape = mean_absolute_percentage_error(actual_values, skf_preds.values)

print(f"\nSKForecast Prediction Results:")
print(f"MAE: {skf_mae:.2f}")
print(f"RMSE: {skf_rmse:.2f}")
print(f"MAPE: {skf_mape:.2f}%")

# Update the results dataframe to include SKForecast predictions
results_df['skforecast'] = skf_preds.values
results_df.to_csv('forecast_results.csv')
print("Saved forecast results to forecast_results.csv")

# Compare all three methods in a plot
plt.figure(figsize=(15, 7))
plt.plot(time_idx, actual_values, label='Actual', marker='o', linestyle='-', alpha=0.7)
plt.plot(time_idx, direct_preds, label=f'Direct XGBoost (MAPE: {mape:.2f}%)', marker='^', linestyle='--', alpha=0.7)
plt.plot(time_idx, recursive_preds, label=f'Recursive XGBoost (MAPE: {rec_mape:.2f}%)', marker='x', linestyle='--', alpha=0.7)
plt.plot(time_idx, skf_preds.values, label=f'SKForecast (MAPE: {skf_mape:.2f}%)', marker='s', linestyle='--', alpha=0.7)
plt.title('Comparison of Forecasting Methods')
plt.xlabel('Date/Time')
plt.ylabel('Demand (tsd)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('forecast_methods_comparison.png')
plt.show()

# Create a comparison table
comparison = pd.DataFrame({
    'Metric': ['Training Time (s)', 'Prediction Time (s)', 'MAE', 'RMSE', 'MAPE (%)'],
    'Direct XGBoost': [training_time, prediction_time, mae, rmse, mape],
    'Recursive XGBoost': [training_time, recursive_time, rec_mae, rec_rmse, rec_mape],
    'SKForecast': [skf_train_time, skf_pred_time, skf_mae, skf_rmse, skf_mape]
})

# Display and save comparison table
print("\n=== MODEL COMPARISON ===")
print(comparison)
comparison.to_csv('model_comparison.csv', index=False)
print("Saved model comparison to model_comparison.csv")

print("XGBoost Recursive Forecasting Model Complete.")

# Final summary of performance metrics
print("\n=== FINAL PERFORMANCE SUMMARY ===")
print(f"Direct Prediction Average MAPE: {mape:.2f}%")
print(f"Recursive Prediction Average MAPE: {rec_mape:.2f}%")
print(f"SKForecast Prediction Average MAPE: {skf_mape:.2f}%")
print("==================================")

