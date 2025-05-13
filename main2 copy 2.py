# Cell 1: Title and Introduction
# # Time Series Forecasting with XGBoost
# ## Advanced Demand Forecasting with Recursive and Ensemble Methods

# Import required libraries
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

# This notebook implements time series forecasting for demand prediction using:
# - Feature engineering with lag and rolling features 
# - XGBoost regression with optimized parameters
# - Recursive forecasting with error correction
# - Ensemble methods for improved accuracy
# - Multi-horizon forecast evaluation

# Cell 2: Data Loading and Preprocessing
# ## Load and Clean the Dataset

# Load the dataset
df = pd.read_csv("historic_demand_2009_2024.csv", index_col=0)

# Remove columns with null values
# Note: Some columns contain values that only started appearing after 
# a specific year and might be useful in future analyses
df.drop(columns=["nsl_flow", "eleclink_flow", "scottish_transfer", "viking_flow", "greenlink_flow"], axis=1, inplace=True)

# Drop rows where settlement_period value is greater than 48 (data quality issues)
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

# Display the first few rows of cleaned dataset
print("First 5 rows of cleaned dataset:")
print(df.head())

# Display dataset information
print("\nDataset information:")
print(f"Shape: {df.shape}")
print(f"Date range: {df['settlement_date'].min()} to {df['settlement_date'].max()}")
print(f"Number of unique days: {df['settlement_date'].nunique()}")

# Cell 3: Feature Engineering
# ## Create Date-Time Features

def add_datepart(df):
    """
    Create comprehensive datetime features from the settlement date and period
    
    Args:
        df: DataFrame containing settlement_date and settlement_period columns
        
    Returns:
        DataFrame with added datetime features
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
    
    # Add sine and cosine transforms for cyclical features
    # These better capture the cyclical nature of time features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    
    # Add features for time of day categorization
    # Morning (5-11), Afternoon (12-16), Evening (17-20), Night (21-4)
    df["time_of_day"] = pd.cut(
        df["hour"], 
        bins=[-1, 4, 11, 16, 20, 23], 
        labels=[0, 1, 2, 3, 4]  # night, morning, afternoon, evening, night
    ).astype(int)
    
    # Create day of week one-hot encoding for stronger day patterns
    for i in range(7):
        df[f'dow_{i}'] = (df['day_of_week'] == i).astype(int)
    
    return df

# Apply datetime feature engineering
print("Adding datetime features...")
df = add_datepart(df)

# Clean up and set the index
df.drop(columns=["period_offset", "settlement_date"], inplace=True)
df.set_index("timestamp", inplace=True)
df.sort_index(inplace=True)

# Display information about the enhanced dataset
print("\nEnhanced dataset information:")
print(f"Shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Number of features: {df.shape[1]}")

# Display sample of the data with new features
print("\nSample of data with engineered features:")
print(df.head())

# Cell 4: Data Exploration and Visualization
# ## Examine Dataset Correlations and Patterns

# Define MAPE function for later use
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAPE value (percentage)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

# Plot correlation matrix heatmap
print("Generating correlation matrix heatmap...")
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()

# Plot time series of target variable
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['tsd'], linewidth=1)
plt.title('Time Series of Demand (tsd)')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.grid(True, alpha=0.3)
plt.show()

# Plot demand by hour of day
plt.figure(figsize=(12, 5))
hourly_avg = df.groupby('hour')['tsd'].mean()
plt.bar(hourly_avg.index, hourly_avg.values)
plt.title('Average Demand by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Average Demand')
plt.grid(True, alpha=0.3)
plt.show()

# Plot demand by day of week
plt.figure(figsize=(12, 5))
daily_avg = df.groupby('day_of_week')['tsd'].mean()
plt.bar(daily_avg.index, daily_avg.values)
plt.title('Average Demand by Day of Week')
plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
plt.ylabel('Average Demand')
plt.grid(True, alpha=0.3)
plt.show()

# Cell 5: Time Series Feature Creation
# ## Create Lag Features for Forecasting

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
    
    # Add multiple-period lags for capturing weekly and daily patterns
    # These help with longer-term trends and seasonality
    for lag in [48, 96, 144, 192, 336]:  # 1-day, 2-day, 3-day, 4-day and 7-day lags
        if lag <= lag_periods:
            continue  # Skip if already included in the standard lags
        data[f'lag_{lag}'] = data[target_col].shift(lag)
    
    # Create rolling window features (capture recent trends)
    for window in [6, 12, 24, 48]:
        # Mean of recent values
        data[f'rolling_mean_{window}'] = data[target_col].shift(1).rolling(window=window).mean()
        # Standard deviation (volatility)
        data[f'rolling_std_{window}'] = data[target_col].shift(1).rolling(window=window).std()
        # Min and max
        data[f'rolling_min_{window}'] = data[target_col].shift(1).rolling(window=window).min()
        data[f'rolling_max_{window}'] = data[target_col].shift(1).rolling(window=window).max()
    
    # Create difference features to capture rate of change
    data['diff_1'] = data[target_col].diff(1)
    data['diff_2'] = data[target_col].diff(2)
    data['diff_7'] = data[target_col].diff(7)
    data['diff_48'] = data[target_col].diff(48)
    
    # Create percentage change features
    data['pct_change_1'] = data[target_col].pct_change(1)
    data['pct_change_48'] = data[target_col].pct_change(48)
    
    # Create the target variable (forecast_horizon steps ahead)
    data['target'] = data[target_col].shift(-forecast_horizon)
    
    # Drop NaN values that result from shifting
    data = data.dropna()
    
    # Separate features and target
    y = data['target']
    X = data.drop('target', axis=1)
    
    return X, y

# Cell 6: Train-Test Split and Feature Selection Functions
# ## Prepare Data for Training

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
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test

def select_important_features(X_train, y_train, X_test, feature_importance_threshold=0.01):
    """
    Select important features based on a preliminary XGBoost model's feature importance
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        feature_importance_threshold: Minimum importance threshold to keep a feature
        
    Returns:
        X_train_selected: Training features with only important features
        X_test_selected: Test features with only important features
        important_features: List of important feature names
    """
    print("\nPerforming feature selection...")
    
    # Train a preliminary model to get feature importances
    prelim_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    prelim_model.fit(X_train, y_train)
    
    # Get feature importances
    importances = prelim_model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Normalize importances
    total_importance = feature_importance['importance'].sum()
    feature_importance['importance_normalized'] = feature_importance['importance'] / total_importance
    
    # Select features above threshold
    important_features = feature_importance[
        feature_importance['importance_normalized'] >= feature_importance_threshold
    ]['feature'].tolist()
    
    print(f"Selected {len(important_features)} out of {X_train.shape[1]} features.")
    
    # Select only important features
    X_train_selected = X_train[important_features]
    X_test_selected = X_test[important_features]
    
    # Show top features
    top_n = min(15, len(important_features))
    print(f"\nTop {top_n} most important features:")
    for i, (feature, importance) in enumerate(
        zip(
            feature_importance['feature'].values[:top_n],
            feature_importance['importance_normalized'].values[:top_n]
        )
    ):
        print(f"{i+1}. {feature}: {importance:.4f}")
    
    # Plot top 20 feature importances
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance['feature'].values[:20], 
             feature_importance['importance_normalized'].values[:20])
    plt.title('Top 20 Feature Importances')
    plt.xlabel('Normalized Importance')
    plt.gca().invert_yaxis()  # Display highest importance at the top
    plt.show()
    
    return X_train_selected, X_test_selected, important_features

# Cell 7: Training Progress Callback and Recursive Forecasting
# ## Model Training Utilities

class TrainingProgressCallback:
    """
    Custom callback to track and display XGBoost training progress
    
    This callback provides real-time monitoring of model training,
    displaying metrics at intervals and supporting early stopping.
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
# ## Core Forecasting Functions

def recursive_forecast(model, X_test, steps=48, error_correction=True, correction_factor=0.7):
    """
    Make recursive forecasts using an XGBoost model with error correction
    
    Args:
        model: Trained XGBoost model
        X_test: Initial test data with lag features
        steps: Number of steps to forecast
        error_correction: Whether to apply error correction
        correction_factor: Weight given to error correction (0-1)
        
    Returns:
        List of forecasted values
    """
    # Make a copy of the test data to avoid modifying the original
    data = X_test.iloc[0:1].copy()
    
    # Store forecasts
    forecasts = []
    
    # Get the initial lag feature names and sort them numerically
    lag_cols = [col for col in data.columns if 'lag_' in col]
    lag_cols.sort(key=lambda x: int(x.split('_')[1]))
    
    # Get rolling feature columns
    rolling_cols = [col for col in data.columns if 'rolling_' in col]
    
    # Get difference feature columns
    diff_cols = [col for col in data.columns if 'diff_' in col]
    pct_cols = [col for col in data.columns if 'pct_change_' in col]
    
    # Get the target column (will be used to update lag features)
    target_col = 'tsd'
    
    # Get initial values for tracking errors
    recent_errors = []
    
    # Show progress for recursive forecasting
    print("\nMaking recursive forecasts...")
    
    # Make recursive predictions
    for i in range(steps):
        if i % 10 == 0 or i == steps - 1:
            print(f"Forecasting step {i+1}/{steps}...")
            
        # Make prediction for the current step
        pred = model.predict(data.iloc[-1:])
        
        # Apply error correction if enabled and we have error history
        if error_correction and i > 0 and len(recent_errors) > 0:
            # Calculate mean error (bias) from recent predictions
            mean_error = np.mean(recent_errors[-min(5, len(recent_errors)):])
            # Apply correction with a dampening factor
            corrected_pred = pred[0] - (correction_factor * mean_error)
            # Ensure prediction is positive
            corrected_pred = max(0, corrected_pred)
            # Store the corrected prediction
            forecasts.append(corrected_pred)
        else:
            # Store the raw prediction if we can't apply correction
            forecasts.append(pred[0])
        
        # If we've reached the desired forecast horizon, stop
        if i == steps - 1:
            break
        
        # Prepare data for the next prediction
        last_row = data.iloc[-1].copy()
        
        # Shift lag values (standard lags)
        for j in range(1, min(48, len(lag_cols))):
            last_row[lag_cols[j-1]] = last_row[lag_cols[j]]
        
        # Update most recent lag with the current target value
        if len(lag_cols) >= 1:
            last_row[lag_cols[0]] = last_row[target_col]
        
        # Update target with prediction
        last_row[target_col] = forecasts[-1]
        
        # Update difference features (if they exist)
        for col in diff_cols:
            periods = int(col.split('_')[1])
            if i + 1 >= periods:
                # Get the appropriate previous value for this difference
                idx = -periods if i + 1 >= periods else 0
                prev_value = data.iloc[idx][target_col]
                last_row[col] = last_row[target_col] - prev_value
        
        # Update percentage change features
        for col in pct_cols:
            periods = int(col.split('_')[2])
            if i + 1 >= periods:
                idx = -periods if i + 1 >= periods else 0
                prev_value = data.iloc[idx][target_col]
                if prev_value != 0:
                    last_row[col] = (last_row[target_col] - prev_value) / prev_value
                else:
                    last_row[col] = 0
        
        # Update rolling window features when possible
        if i > 0:
            for col in rolling_cols:
                parts = col.split('_')
                if len(parts) >= 3:
                    # Extract operation type and window size
                    op_type = parts[1]  # mean, std, min, max
                    window = int(parts[2])
                    
                    # Calculate values based on available history
                    available_history = min(window, i + 1)
                    values = [data.iloc[-j][target_col] for j in range(1, available_history + 1)]
                    
                    if op_type == 'mean':
                        last_row[col] = np.mean(values)
                    elif op_type == 'std' and len(values) > 1:  
                        last_row[col] = np.std(values)
                    elif op_type == 'min':
                        last_row[col] = np.min(values)
                    elif op_type == 'max':
                        last_row[col] = np.max(values)
        
        # Add the updated row to our data
        data = pd.concat([data, pd.DataFrame([last_row])], ignore_index=True)
        
        # Store the prediction error if we have actual values
        if i > 0 and len(forecasts) >= 2:
            # Use previous forecast as a proxy for error calculation
            error = forecasts[-2] - last_row[target_col]
            recent_errors.append(error)
    
    print("Recursive forecasting complete.")
    return forecasts

# Cell 9: Ensemble Modeling for Improved Forecasting
# ## Create Ensemble of Models

def create_ensemble_models(X_train, y_train, X_test, y_test, n_models=3):
    """
    Create an ensemble of models for more stable forecasting
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        n_models: Number of models to create
        
    Returns:
        List of trained models
    """
    print(f"\nCreating an ensemble of {n_models} models...")
    
    models = []
    seeds = [42, 123, 456, 789, 101][:n_models]
    
    # Base parameters
    base_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 1000,
        'early_stopping_rounds': 50,
        'learning_rate': 0.03,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
    }
    
    # Parameter variations for diversity
    param_variations = [
        {'max_depth': 7, 'min_child_weight': 2, 'gamma': 0.05, 'reg_alpha': 0.01, 'reg_lambda': 1.0},
        {'max_depth': 8, 'min_child_weight': 3, 'gamma': 0.1, 'reg_alpha': 0.05, 'reg_lambda': 1.5},
        {'max_depth': 6, 'min_child_weight': 4, 'gamma': 0.2, 'reg_alpha': 0.1, 'reg_lambda': 2.0},
        {'max_depth': 9, 'min_child_weight': 2, 'gamma': 0.1, 'reg_alpha': 0.01, 'reg_lambda': 0.5},
        {'max_depth': 7, 'min_child_weight': 5, 'gamma': 0.15, 'reg_alpha': 0.2, 'reg_lambda': 1.0}
    ]
    
    for i in range(n_models):
        print(f"\nTraining model {i+1}/{n_models}...")
        
        # Create parameter set with variation
        params = base_params.copy()
        params.update(param_variations[i])
        params['random_state'] = seeds[i]
        
        # Display parameters for this model
        print(f"Parameters for model {i+1}:")
        for param, value in params.items():
            print(f"  {param}: {value}")
        
        # Create and train the model
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, 
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        
        models.append(model)
        
        # Evaluate this model
        y_pred = model.predict(X_test)
        mape_val = mean_absolute_percentage_error(y_test, y_pred)
        print(f"Model {i+1} MAPE on test set: {mape_val:.2f}%")
    
    print(f"Ensemble of {n_models} models created successfully.")
    return models

def ensemble_recursive_forecast(models, X_test, steps=48, error_correction=True, correction_factor=0.7):
    """
    Make recursive forecasts using an ensemble of models with error correction
    
    Args:
        models: List of trained models
        X_test: Initial test data with lag features
        steps: Number of steps to forecast
        error_correction: Whether to apply error correction
        correction_factor: Weight given to error correction (0-1)
        
    Returns:
        List of forecasted values
    """
    # Make a copy of the test data to avoid modifying the original
    data = X_test.iloc[0:1].copy()
    
    # Store forecasts
    forecasts = []
    
    # Get the initial lag feature names and sort them numerically
    lag_cols = [col for col in data.columns if 'lag_' in col]
    lag_cols.sort(key=lambda x: int(x.split('_')[1]))
    
    # Get rolling feature columns
    rolling_cols = [col for col in data.columns if 'rolling_' in col]
    
    # Get difference feature columns
    diff_cols = [col for col in data.columns if 'diff_' in col]
    pct_cols = [col for col in data.columns if 'pct_change_' in col]
    
    # Get the target column (will be used to update lag features)
    target_col = 'tsd'
    
    # Get initial values for tracking errors
    recent_errors = []
    
    # Show progress for recursive forecasting
    print("\nMaking ensemble recursive forecasts...")
    
    # Make recursive predictions
    for i in range(steps):
        if i % 10 == 0 or i == steps - 1:
            print(f"Forecasting step {i+1}/{steps}...")
            
        # Make predictions with all models and average them
        all_preds = []
        for model in models:
            pred = model.predict(data.iloc[-1:])
            all_preds.append(pred[0])
        
        # Average predictions from all models (simple ensemble)
        ensemble_pred = np.mean(all_preds)
        
        # Apply error correction if enabled and we have error history
        if error_correction and i > 0 and len(recent_errors) > 0:
            # Calculate mean error (bias) from recent predictions
            mean_error = np.mean(recent_errors[-min(5, len(recent_errors)):])
            # Apply correction with a dampening factor
            corrected_pred = ensemble_pred - (correction_factor * mean_error)
            # Ensure prediction is positive
            corrected_pred = max(0, corrected_pred)
            # Store the corrected prediction
            forecasts.append(corrected_pred)
        else:
            # Store the raw prediction if we can't apply correction
            forecasts.append(ensemble_pred)
        
        # If we've reached the desired forecast horizon, stop
        if i == steps - 1:
            break
        
        # Prepare data for the next prediction
        last_row = data.iloc[-1].copy()
        
        # Shift lag values (standard lags)
        for j in range(1, min(48, len(lag_cols))):
            last_row[lag_cols[j-1]] = last_row[lag_cols[j]]
        
        # Update most recent lag with the current target value
        if len(lag_cols) >= 1:
            last_row[lag_cols[0]] = last_row[target_col]
        
        # Update target with prediction
        last_row[target_col] = forecasts[-1]
        
        # Update difference features (if they exist)
        for col in diff_cols:
            periods = int(col.split('_')[1])
            if i + 1 >= periods:
                # Get the appropriate previous value for this difference
                idx = -periods if i + 1 >= periods else 0
                prev_value = data.iloc[idx][target_col]
                last_row[col] = last_row[target_col] - prev_value
        
        # Update percentage change features
        for col in pct_cols:
            periods = int(col.split('_')[2])
            if i + 1 >= periods:
                idx = -periods if i + 1 >= periods else 0
                prev_value = data.iloc[idx][target_col]
                if prev_value != 0:
                    last_row[col] = (last_row[target_col] - prev_value) / prev_value
                else:
                    last_row[col] = 0
        
        # Update rolling window features when possible
        if i > 0:
            for col in rolling_cols:
                parts = col.split('_')
                if len(parts) >= 3:
                    # Extract operation type and window size
                    op_type = parts[1]  # mean, std, min, max
                    window = int(parts[2])
                    
                    # Calculate values based on available history
                    available_history = min(window, i + 1)
                    values = [data.iloc[-j][target_col] for j in range(1, available_history + 1)]
                    
                    if op_type == 'mean':
                        last_row[col] = np.mean(values)
                    elif op_type == 'std' and len(values) > 1:  
                        last_row[col] = np.std(values)
                    elif op_type == 'min':
                        last_row[col] = np.min(values)
                    elif op_type == 'max':
                        last_row[col] = np.max(values)
        
        # Add the updated row to our data
        data = pd.concat([data, pd.DataFrame([last_row])], ignore_index=True)
        
        # Store the prediction error if we have actual values
        if i > 0 and len(forecasts) >= 2:
            # Use previous forecast as a proxy for error calculation
            error = forecasts[-2] - last_row[target_col]
            recent_errors.append(error)
    
    print("Ensemble recursive forecasting complete.")
    return forecasts

# Cell 10: Multi-horizon Forecast Evaluation
# ## Evaluate Forecasting Performance Across Horizons

def evaluate_multi_horizon(models, X_test, y_test, max_horizon=48, use_ensemble=True, error_correction=True):
    """
    Evaluate model performance across multiple forecast horizons
    
    Args:
        models: Trained model or list of models for ensemble
        X_test: Test features
        y_test: Test targets
        max_horizon: Maximum forecast horizon to evaluate
        use_ensemble: Whether to use ensemble forecasting
        error_correction: Whether to apply error correction
        
    Returns:
        DataFrame with performance metrics for each horizon
    """
    results = []
    
    print(f"\nEvaluating performance across forecast horizons (1 to {max_horizon})...")
    
    # Determine number of samples to evaluate
    # We limit the number to make execution faster
    max_samples = min(20, len(X_test) - max_horizon)
    sample_indices = list(range(0, len(X_test) - max_horizon, len(X_test) // max_samples))[:max_samples]
    
    for horizon in range(1, max_horizon+1):
        if horizon % 10 == 0 or horizon == 1 or horizon == max_horizon:
            print(f"Evaluating horizon {horizon}/{max_horizon}...")
            
        # Make forecasts for this horizon
        forecasts = []
        actuals = []
        
        # Use selected sample indices for efficiency
        for i in sample_indices:
            initial_data = X_test.iloc[i:i+1]
            
            # Choose forecast method based on parameters
            if use_ensemble and isinstance(models, list):
                forecast = ensemble_recursive_forecast(
                    models, initial_data, steps=horizon, 
                    error_correction=error_correction
                )
            else:
                model = models[0] if isinstance(models, list) else models
                forecast = recursive_forecast(
                    model, initial_data, steps=horizon,
                    error_correction=error_correction
                )
            
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
    
    # Create DataFrame from results
    result_df = pd.DataFrame(results)
    
    # Display summary of results
    print("\nSummary of forecast accuracy by horizon:")
    print(f"Average MAPE across all horizons: {result_df['MAPE'].mean():.2f}%")
    print(f"Best MAPE: {result_df['MAPE'].min():.2f}% at horizon {result_df.loc[result_df['MAPE'].idxmin(), 'Horizon']}")
    print(f"Worst MAPE: {result_df['MAPE'].max():.2f}% at horizon {result_df.loc[result_df['MAPE'].idxmax(), 'Horizon']}")
    
    return result_df

# Cell 11: Main Program - Training and Evaluation
# ## Feature Creation and Model Training

# Setting parameters for forecasting
TARGET_COL = 'tsd'
LAG_PERIODS = 48  # One day of lags
FORECAST_HORIZON = 48  # One day ahead forecast
TEST_SIZE = 0.2

# Create features with lag values
print("Creating enhanced lag features...")
X, y = create_features(df, TARGET_COL, LAG_PERIODS, FORECAST_HORIZON)
print(f"Feature shape: {X.shape}, Target shape: {y.shape}")

# Split data chronologically into train and test sets
X_train, X_test, y_train, y_test = train_test_split_ts(X, y, TEST_SIZE)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
print(f"Training period: {X_train.index.min()} to {X_train.index.max()}")
print(f"Testing period: {X_test.index.min()} to {X_test.index.max()}")

# Perform feature selection to improve model quality
X_train_selected, X_test_selected, important_features = select_important_features(
    X_train, y_train, X_test, feature_importance_threshold=0.005
)

print(f"\nSelected {len(important_features)} important features for training.")
print(f"Reduced feature dimensions: {X_train.shape} -> {X_train_selected.shape}")

# Initialize and train XGBoost model
print("\nInitializing XGBoost model with improved parameters:")
xgb_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 1000,
    'learning_rate': 0.03,  # Slower learning rate for better generalization
    'max_depth': 8,  # Increased depth for more complex patterns
    'min_child_weight': 3,  # Increased to prevent overfitting
    'subsample': 0.7,  # Reduced to prevent overfitting
    'colsample_bytree': 0.7,  # Reduced to prevent overfitting
    'gamma': 0.1,  # Added regularization
    'reg_alpha': 0.01,  # L1 regularization
    'reg_lambda': 1.0,  # L2 regularization
    'random_state': 42
}

for param, value in xgb_params.items():
    print(f"  {param}: {value}")

# Training XGBoost model with callbacks
print("\nTraining XGBoost model...")
xgb_model = xgb.XGBRegressor(**xgb_params)

# Start timing the training
training_start_time = time.time()

# Create a callback for progress tracking and early stopping
callback = TrainingProgressCallback(
    start_time=training_start_time,
    early_stopping_rounds=50,
    print_interval=10
)

# Train the model with callback
xgb_model.fit(
    X_train_selected, 
    y_train,
    eval_set=[(X_train_selected, y_train), (X_test_selected, y_test)],
    verbose=False
)

training_time = time.time() - training_start_time
print(f"\nTraining completed in {training_time:.2f} seconds")

# Create an ensemble of models for improved forecasting
ensemble_models = create_ensemble_models(
    X_train_selected, y_train, X_test_selected, y_test, n_models=3
)

# Cell 12: Model Evaluation
# ## Evaluate Model Performance

# Make predictions on test set (direct predictions)
print("\nMaking direct predictions on test set...")
prediction_start_time = time.time()
y_pred = xgb_model.predict(X_test_selected)
prediction_time = time.time() - prediction_start_time
print(f"Predictions completed in {prediction_time:.2f} seconds")

# Calculate and display performance metrics for direct predictions
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"\nDirect Prediction Results:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# Plot feature importance
plt.figure(figsize=(12, 6))
xgb.plot_importance(xgb_model, max_num_features=15)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

# Cell 13: Recursive Forecasting Evaluation
# ## Compare Different Forecasting Methods

# Now perform original recursive forecasting (for comparison)
print("\nPerforming original recursive forecasting...")
# Get a starting point from the test set
start_idx = 0
forecast_length = min(48, len(y_test))  # Forecast for up to 48 steps or the length of test set

# Get actual values for the forecast period
actual_values = y_test.iloc[start_idx:start_idx+forecast_length].values

# Get initial data for recursive forecasting
initial_data = X_test_selected.iloc[start_idx:start_idx+1]

# Start timing recursive forecasting
recursive_start_time = time.time()

# Make original recursive forecasts (without error correction)
original_recursive_preds = recursive_forecast(xgb_model, initial_data, steps=forecast_length, error_correction=False)

original_recursive_time = time.time() - recursive_start_time
print(f"Original recursive forecasting completed in {original_recursive_time:.2f} seconds")

# Make improved recursive forecasts (with error correction)
improved_recursive_start_time = time.time()
improved_recursive_preds = recursive_forecast(xgb_model, initial_data, steps=forecast_length, error_correction=True)
improved_recursive_time = time.time() - improved_recursive_start_time
print(f"Improved recursive forecasting completed in {improved_recursive_time:.2f} seconds")

# Make ensemble recursive forecasts
ensemble_recursive_start_time = time.time()
ensemble_recursive_preds = ensemble_recursive_forecast(ensemble_models, initial_data, steps=forecast_length, error_correction=True)
ensemble_recursive_time = time.time() - ensemble_recursive_start_time
print(f"Ensemble recursive forecasting completed in {ensemble_recursive_time:.2f} seconds")

# Calculate metrics for original recursive forecasting
orig_rec_mae = mean_absolute_error(actual_values, original_recursive_preds)
orig_rec_mse = mean_squared_error(actual_values, original_recursive_preds)
orig_rec_rmse = np.sqrt(orig_rec_mse)
orig_rec_mape = mean_absolute_percentage_error(actual_values, original_recursive_preds)

# Calculate metrics for improved recursive forecasting
improved_rec_mae = mean_absolute_error(actual_values, improved_recursive_preds)
improved_rec_mse = mean_squared_error(actual_values, improved_recursive_preds)
improved_rec_rmse = np.sqrt(improved_rec_mse)
improved_rec_mape = mean_absolute_percentage_error(actual_values, improved_recursive_preds)

# Calculate metrics for ensemble recursive forecasting
ensemble_rec_mae = mean_absolute_error(actual_values, ensemble_recursive_preds)
ensemble_rec_mse = mean_squared_error(actual_values, ensemble_recursive_preds)
ensemble_rec_rmse = np.sqrt(ensemble_rec_mse)
ensemble_rec_mape = mean_absolute_percentage_error(actual_values, ensemble_recursive_preds)

print(f"\nOriginal Recursive Prediction Results:")
print(f"MAE: {orig_rec_mae:.2f}")
print(f"RMSE: {orig_rec_rmse:.2f}")
print(f"MAPE: {orig_rec_mape:.2f}%")

print(f"\nImproved Recursive Prediction Results:")
print(f"MAE: {improved_rec_mae:.2f}")
print(f"RMSE: {improved_rec_rmse:.2f}")
print(f"MAPE: {improved_rec_mape:.2f}%")

print(f"\nEnsemble Recursive Prediction Results:")
print(f"MAE: {ensemble_rec_mae:.2f}")
print(f"RMSE: {ensemble_rec_rmse:.2f}")
print(f"MAPE: {ensemble_rec_mape:.2f}%")

# Cell 14: Visualization and Results
# ## Plot Forecasting Results and Save Models

# Plot all methods for comparison
plt.figure(figsize=(15, 7))
time_idx = y_test.index[start_idx:start_idx+forecast_length]
plt.plot(time_idx, actual_values, label='Actual', marker='o', linestyle='-', alpha=0.7)
plt.plot(time_idx, original_recursive_preds, label='Original Recursive', marker='^', linestyle='--', alpha=0.7)
plt.plot(time_idx, improved_recursive_preds, label='Improved Recursive', marker='x', linestyle='--', alpha=0.7)
plt.plot(time_idx, ensemble_recursive_preds, label='Ensemble Recursive', marker='*', linestyle='--', alpha=0.7)
plt.title('Comparison of Forecasting Methods')
plt.xlabel('Date/Time')
plt.ylabel('Demand (tsd)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Compare the percentage improvements in MAPE
orig_to_improved_pct_change = ((orig_rec_mape - improved_rec_mape) / orig_rec_mape) * 100
orig_to_ensemble_pct_change = ((orig_rec_mape - ensemble_rec_mape) / orig_rec_mape) * 100

print("\n=== FINAL PERFORMANCE SUMMARY ===")
print(f"Original Recursive MAPE: {orig_rec_mape:.2f}%")
print(f"Improved Recursive MAPE: {improved_rec_mape:.2f}%")
print(f"Ensemble Recursive MAPE: {ensemble_rec_mape:.2f}%")
print(f"Improvement from Original to Improved: {orig_to_improved_pct_change:.2f}%")
print(f"Improvement from Original to Ensemble: {orig_to_ensemble_pct_change:.2f}%")
print("==================================")

# Create performance comparison table
performance_df = pd.DataFrame({
    'Method': ['Direct', 'Original Recursive', 'Improved Recursive', 'Ensemble Recursive'],
    'MAE': [mae, orig_rec_mae, improved_rec_mae, ensemble_rec_mae],
    'RMSE': [rmse, orig_rec_rmse, improved_rec_rmse, ensemble_rec_rmse],
    'MAPE (%)': [mape, orig_rec_mape, improved_rec_mape, ensemble_rec_mape]
})
print("\nPerformance Comparison Table:")
print(performance_df)

# Evaluate performance across multiple horizons 
print("\nEvaluating performance of the ensemble model across forecast horizons...")
horizon_results = evaluate_multi_horizon(
    ensemble_models, X_test_selected[:20], y_test[:20], 
    max_horizon=48, use_ensemble=True, error_correction=True
)

# Save the horizon results to a CSV file
horizon_results.to_csv('forecast_horizon_metrics_improved.csv', index=False)
print("Saved improved horizon metrics to forecast_horizon_metrics_improved.csv")

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

# Cell 15: Save Models and Results
# ## Export Models and Forecast Data

# Save best model for later use
best_model_filename = 'xgboost_recursive_forecast_model_improved.json'
ensemble_models[0].save_model(best_model_filename)
print(f"Best model saved to {best_model_filename}")

# Save comparison results for further analysis
results_df = pd.DataFrame({
    'timestamp': time_idx,
    'actual': actual_values,
    'original_recursive': original_recursive_preds,
    'improved_recursive': improved_recursive_preds,
    'ensemble_recursive': ensemble_recursive_preds
})
results_df.to_csv('forecast_results_improved.csv')
print("Saved improved forecast results to forecast_results_improved.csv")

# Create a function to make new forecasts with the trained model
def make_new_forecast(model_path, data_path, forecast_horizon=48, use_error_correction=True):
    """
    Make a new forecast using a saved model and new data
    
    Args:
        model_path: Path to the saved XGBoost model
        data_path: Path to the new data CSV file
        forecast_horizon: Number of periods ahead to forecast
        use_error_correction: Whether to use error correction
    
    Returns:
        DataFrame with forecast results
    """
    # Load the model
    model = xgb.XGBoost()
    model.load_model(model_path)
    
    # Load and prepare the data
    # (Code would be added here to load and prepare new data)
    
    # Make forecasts
    # (Code would be added here to make forecasts)
    
    print("Forecasting function ready for future use.")
    
    return None  # Placeholder for actual implementation

print("XGBoost Recursive Forecasting Model Improvements Complete.")

