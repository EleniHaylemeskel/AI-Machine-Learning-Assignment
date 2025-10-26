#  AI for Sustainable Development : *Predicting CO₂ Emissions (SDG 13: Climate Action)*

### “Machine Learning Meets the UN Sustainable Development Goals (SDGs)” 🤖

---

##  Project Overview

Climate change is one of humanity’s most urgent challenges. CO₂ emissions are the leading cause of global warming, yet many developing nations lack predictive insights to plan sustainable policies.

This project uses **machine learning** to predict a country’s **CO₂ emissions per capita** based on factors such as GDP, energy use, and renewable energy consumption. The goal is to demonstrate how **AI can support SDG 13: Climate Action** by providing data-driven tools for sustainability and environmental policy.

---

##  Objectives

- Use supervised learning to model and predict CO₂ emissions.  
- Identify key factors influencing emission levels.  
- Support sustainable decision-making through data insights.

---

##  SDG Alignment

**United Nations SDG 13 : Climate Action**  
> *“Take urgent action to combat climate change and its impacts.”*

**How this project contributes:**
- Provides early warnings for rising emissions.  
- Helps policymakers design sustainable energy strategies.  
- Promotes environmental accountability through open data and transparency.

---

##  Machine Learning Approach

**Approach:** Supervised Learning : Regression  
**Model Used:** Random Forest Regressor  
**Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

---

###  Workflow

1. **Data Collection:**  
   - CO₂ emissions, GDP per capita, energy use, and renewable energy data sourced from the **World Bank Open Data** and **Kaggle**.

2. **Data Preprocessing:**  
   - Handle missing values, normalize numerical features, and prepare training/test sets.

3. **Model Training:**  
   - Train a Random Forest Regressor to predict CO₂ emissions.

4. **Evaluation:**  
   - Metrics: Mean Absolute Error (MAE), R² Score.  
   - Visualization: Actual vs Predicted CO₂ Emissions Scatter Plot.

---


# 📊 Dataset Information

**Source:**  
- [World Bank Open Data](https://data.worldbank.org/)
- [Kaggle CO2 Emissions Dataset](https://www.kaggle.com/datasets)

**Columns Used:**
- `Country` — Country name
- `GDP_per_capita` — Economic indicator
- `Energy_use` — Energy consumption per capita
- `Renewable_energy_percent` — % of renewable energy in total consumption
- `CO2_emissions_per_capita` — Target variable (metric tons per person)

**Note:**  
Data cleaned and preprocessed for training the CO₂ emission prediction model.


##  Example Code Snippet

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
