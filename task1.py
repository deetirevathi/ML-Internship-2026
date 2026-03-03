import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# LOAD & CLEAN DATA
# ==============================

df = pd.read_csv("Sample - Superstore.csv", encoding='latin1')
df = df.dropna()
df = df.drop_duplicates()

df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

df = df.sort_values(['Order Date', 'Ship Date'])

# ==============================
# TIME FEATURES
# ==============================

df['Order_year'] = df['Order Date'].dt.year
df['Order_month'] = df['Order Date'].dt.month
df['Order_day'] = df['Order Date'].dt.day
df['Order_dayOfWeek'] = df['Order Date'].dt.dayofweek
df['Order_quarter'] = df['Order Date'].dt.quarter
df['Order_week'] = df['Order Date'].dt.isocalendar().week

# ==============================
# DAILY & MONTHLY SALES
# ==============================

daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()

df['Order_month'] = df['Order Date'].dt.to_period('M')
monthly_sales = df.groupby('Order_month')['Sales'].sum().reset_index()

# ==============================
# DAILY SALES PLOT
# ==============================

plt.figure(figsize=(12,6))
plt.plot(daily_sales['Order Date'], daily_sales['Sales'])
plt.title('Daily Sales Over Time')
plt.xlabel('Order Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.grid(True)

# IMPORTANT: Do not block execution
plt.show(block=False)
plt.pause(3)
plt.close()

# ==============================
# STEP 9: MODEL EVALUATION
# ==============================

from statsmodels.tsa.arima.model import ARIMA


print("Starting Model Evaluation...")

# Prepare monthly data for ARIMA
model_df = monthly_sales.copy()
model_df['Order_month'] = model_df['Order_month'].dt.to_timestamp()
model_df = model_df.set_index('Order_month')

# Ensure proper monthly frequency
model_df = model_df.asfreq('MS')

# Split data (last 6 months for testing)
train = model_df[:-6]
test = model_df[-6:]

# Build ARIMA model
model = ARIMA(train['Sales'], order=(1,1,1))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test))

# Align forecast index with test
forecast.index = test.index

# Evaluate
mae = np.mean(np.abs(test['Sales'] - forecast))
rmse = np.sqrt(np.mean((test['Sales'] - forecast) ** 2))

print("MAE:", mae)
print("RMSE:", rmse)

# ==============================
# Plot Forecast vs Actual
# ==============================

plt.figure(figsize=(12,6))
plt.plot(train.index, train['Sales'], label='Train')
plt.plot(test.index, test['Sales'], label='Actual')
plt.plot(test.index, forecast, label='Forecast', linestyle='--')

plt.title("ARIMA Model Evaluation (Monthly Sales)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)

plt.show()