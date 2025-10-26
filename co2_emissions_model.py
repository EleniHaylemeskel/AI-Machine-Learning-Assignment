# ---------------------------------------------------------
# AI for Sustainable Development ; SDG 13: Climate Action
# Predicting CO₂ Emissions per Capita using ML
# ---------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/co2_emissions.csv")

# Basic preprocessing
df = df.dropna()
X = df[['GDP_per_capita', 'Energy_use', 'Renewable_energy_percent']]
y = df['CO2_emissions_per_capita']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.3f}")
print(f"R² Score: {r2:.3f}")

# Save metrics
with open("results/evaluation_metrics.txt", "w") as f:
    f.write(f"MAE: {mae:.3f}\nR²: {r2:.3f}\n")

# Save predictions
output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
output.to_csv("results/co2_predictions.csv", index=False)

# Visualization
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual CO₂ Emissions")
plt.ylabel("Predicted CO₂ Emissions")
plt.title("Actual vs Predicted CO₂ Emissions")
plt.grid(True)
plt.savefig("screenshots/model_results.png")
plt.show()
