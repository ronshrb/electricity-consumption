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
# Import skforecast for autoregressive forecasting
from skforecast.recursive._forecaster_recursive import ForecasterRecursive

# This notebook implements energy demand forecasting using XGBoost with:
# - Time series feature engineering
# - Direct and recursive forecasting methods
# - Multi-horizon forecast evaluation
# - Performance visualization and comparison
# - Methods comparison (regular XGBoost, recursive, and SKForecast)

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
    X_train, X_test, y_train, y_test = X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]
    
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

# Cell 8: SKForecast Implementation
# ## Using ForecasterRecursive for Multi-Step Forecasting

def create_skforecast_forecaster(xgb_params, lags=48):
    """
    Create a ForecasterRecursive with XGBoost.
    
    Args:
        xgb_params (dict): XGBoost parameters
        lags (int): Number of lags to use as predictors
        
    Returns:
        ForecasterRecursive: Configured forecaster
    """
    # Create XGBoost regressor with provided parameters
    regressor = xgb.XGBRegressor(**xgb_params)
    
    # Create lag features (equivalent to lags 1,2,3,...,lags)
    lags_grid = list(range(1, lags+1))
    
    # Initialize the forecaster
    forecaster = ForecasterRecursive(
        regressor=regressor,
        lags=lags_grid,
        transformer_y=None  # You can add scalers here if needed
    )
    
    return forecaster

# Cell 9: Training and Prediction with SKForecast
# ## Using ForecasterAutoReg with XGBoost

# Define constants for forecasting
TEST_SIZE = 0.2  # Portion of data to use for testing
LAG_PERIODS = 48  # Number of lag periods to use (one day)
FORECAST_HORIZON = 48  # Forecasting horizon (one day ahead)

# Initialize XGBoost parameters
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

# Create the forecaster
print("\nCreating SKForecast forecaster with XGBoost...")
forecaster = create_skforecast_forecaster(xgb_params, lags=LAG_PERIODS)

# Prepare data for SKForecast
# In skforecast, we work with the time series directly
series = df['tsd'].copy()

# Split data chronologically into train and test sets
split_idx = int(len(series) * (1 - TEST_SIZE))
train_series = series.iloc[:split_idx]
test_series = series.iloc[split_idx:]

print(f"Training series size: {len(train_series)}, Test series size: {len(test_series)}")
print(f"Training period: {train_series.index.min()} to {train_series.index.max()}")
print(f"Testing period: {test_series.index.min()} to {test_series.index.max()}")

# Start timing the training
training_start_time = time.time()

# Train the forecaster
print("\nTraining SKForecast model...")
forecaster.fit(y=train_series)

# Calculate training time
training_time = time.time() - training_start_time
print(f"\nTraining completed in {training_time:.2f} seconds")

# Cell 10: Making Predictions with SKForecast
# ## Multi-Step Forecasting with SKForecast

# Start timing the forecast
predict_start_time = time.time()

# Make multi-step forecasts with skforecast
print("\nMaking multi-step forecasts with SKForecast...")
predictions = forecaster.predict(steps=FORECAST_HORIZON)

# Calculate prediction time
predict_time = time.time() - predict_start_time
print(f"Multi-step forecasting completed in {predict_time:.2f} seconds")

# Get actuals for the forecast period
actuals = test_series.iloc[:FORECAST_HORIZON]

# Calculate metrics for the forecasts
mae = mean_absolute_error(actuals, predictions)
mse = mean_squared_error(actuals, predictions)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(actuals, predictions)

print(f"\nMulti-step Prediction Results:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# Cell 11: Visualizing Forecast Results
# ## Comparing Predictions to Actual Values

# Plot forecasting results
plt.figure(figsize=(15, 7))
plt.plot(actuals.values, label='Actual', marker='o', linestyle='-', alpha=0.7)
plt.plot(predictions.values, label='SKForecast Predictions', marker='x', linestyle='--', alpha=0.7)
plt.title('SKForecast Multi-step Forecasting vs Actual Values')
plt.xlabel('Time Steps')
plt.ylabel('Demand (tsd)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Visualize predictions vs actual over time
plt.figure(figsize=(15, 7))
plt.plot(actuals.index, actuals.values, label='Actual', marker='o', linestyle='-', alpha=0.7)
plt.plot(predictions.index, predictions.values, label='SKForecast Predictions', marker='x', linestyle='--', alpha=0.7)
plt.title('Multi-step Forecasting vs Actual Values Over Time')
plt.xlabel('Date/Time')
plt.ylabel('Demand (tsd)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cell 12: Multi-horizon Evaluation
# ## Evaluating Performance Across Different Horizons

def evaluate_multi_horizon_skforecast(forecaster, test_series, max_horizon=48):
    """
    Evaluate forecaster performance across multiple forecast horizons.
    
    Args:
        forecaster (ForecasterAutoReg): Trained forecaster
        test_series (Series): Test data
        max_horizon (int): Maximum forecast horizon to evaluate
        
    Returns:
        DataFrame: Performance metrics for each horizon
    """
    results = []
    
    print(f"\nEvaluating performance across forecast horizons (1 to {max_horizon})...")
    
    # Make predictions for max_horizon steps
    predictions = forecaster.predict(steps=max_horizon)
    
    # Calculate metrics for each horizon
    for horizon in range(1, max_horizon+1):
        if horizon % 10 == 0 or horizon == 1 or horizon == max_horizon:
            print(f"Evaluating horizon {horizon}/{max_horizon}...")
        
        # Get predictions and actuals for this horizon
        horizon_preds = predictions.iloc[:horizon]
        horizon_actuals = test_series.iloc[:horizon]
        
        # Calculate metrics
        mae = mean_absolute_error(horizon_actuals, horizon_preds)
        mse = mean_squared_error(horizon_actuals, horizon_preds)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(horizon_actuals, horizon_preds)
        
        results.append({
            'Horizon': horizon,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Samples': len(horizon_preds)
        })
    
    return pd.DataFrame(results)

# Evaluate performance across multiple horizons
print("\nEvaluating performance across forecast horizons...")
horizon_results = evaluate_multi_horizon_skforecast(forecaster, test_series, max_horizon=48)

# Save the horizon results to a CSV file
horizon_results.to_csv('skforecast_horizon_metrics.csv', index=False)
print("Saved horizon metrics to skforecast_horizon_metrics.csv")

# Plot multi-horizon performance
plt.figure(figsize=(14, 7))
plt.plot(horizon_results['Horizon'], horizon_results['MAPE'], 'o-', label='MAPE')
plt.xlabel('Forecast Horizon')
plt.ylabel('MAPE (%)')
plt.title('Forecast Performance by Horizon')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Cell 13: Feature Importance
# ## Analyzing Feature Importance from the Model

# Get feature importance from the model
try:
    # For XGBoost models inside SKForecast
    feature_importance = pd.DataFrame({
        'Feature': forecaster.regressor.feature_names_in_,
        'Importance': forecaster.regressor.feature_importances_
    })
    
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance in SKForecast XGBoost Model')
    plt.tight_layout()
    plt.show()
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
except Exception as e:
    print(f"\nCouldn't extract feature importance: {e}")

# Cell 14: Save the Model
# ## Saving the Trained Model for Future Use

# Save the trained forecaster
print("\nSaving the SKForecast model...")
try:
    forecaster.save('skforecast_xgboost_model.pkl')
    print("Model saved as 'skforecast_xgboost_model.pkl'")
except Exception as e:
    print(f"Error saving model: {e}")

# Cell 15: Conclusion
# ## Summary of Results and Findings

print("\n===== Forecasting Results Summary =====")
print(f"Model: SKForecast ForecasterAutoReg with XGBoost")
print(f"Training time: {training_time:.2f} seconds")
print(f"Prediction time: {predict_time:.2f} seconds")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print("\nAdvantages of using SKForecast:")
print("1. Simplified workflow with automatic lag creation")
print("2. More efficient implementation for recursive forecasting")
print("3. Built-in support for various model types and time series operations")
print("4. Standardized interface consistent with scikit-learn APIs")
print("=====================================\n")

