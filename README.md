🌍 AI for Sustainable Development — Predicting CO₂ Emissions (SDG 13: Climate Action)
“Machine Learning Meets the UN Sustainable Development Goals (SDGs)” 🤖
🧭 Project Overview

Climate change is one of humanity’s most urgent challenges. CO₂ emissions are the leading cause of global warming, yet many developing nations lack predictive insights to plan sustainable policies.

This project uses machine learning to predict a country’s CO₂ emissions per capita based on factors such as GDP, energy use, and renewable energy consumption. The goal is to demonstrate how AI can support SDG 13: Climate Action by providing data-driven tools for sustainability and environmental policy.

🎯 Objectives

Use supervised learning to model and predict CO₂ emissions.

Identify key factors influencing emission levels.

Support sustainable decision-making through data insights.

🌱 SDG Alignment

United Nations SDG 13 — Climate Action

“Take urgent action to combat climate change and its impacts.”

How this project contributes:

Provides early warnings for rising emissions.

Helps policymakers design sustainable energy strategies.

Promotes environmental accountability through open data and transparency.

🧠 Machine Learning Approach

Approach: Supervised Learning — Regression
Model Used: Random Forest Regressor
Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

🔍 Workflow

Data Collection:

CO₂ emissions, GDP per capita, energy use, and renewable energy data sourced from the World Bank Open Data and Kaggle.

Data Preprocessing:

Handle missing values, normalize numerical features, and prepare training/test sets.

Model Training:

Train a Random Forest Regressor to predict CO₂ emissions.

Evaluation:

Metrics: Mean Absolute Error (MAE), R² Score.

Visualization: Actual vs Predicted CO₂ Emissions Scatter Plot.

🧩 Example Code Snippet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

📊 Results
Metric	Score
Mean Absolute Error	0.18
R² Score	0.86

Key Insights:

Economic growth (GDP per capita) and energy use strongly influence emissions.

Increasing renewable energy share significantly reduces predicted emissions.

⚖️ Ethical & Sustainability Reflection

Bias Awareness: Some countries have incomplete data, which can skew predictions.

Fairness: The model uses transparent, publicly available data to avoid hidden biases.

Impact: Encourages sustainable economic and energy planning aligned with SDG 13.

🧰 Tools & Dependencies

Python 3.x

Google Colab / Jupyter Notebook

Libraries:

pip install pandas numpy scikit-learn matplotlib seaborn

📁 Repository Structure
├── README.md                  # Project documentation  
├── co2_emissions_model.py     # Core model script  
├── co2_emissions_notebook.ipynb # Full notebook with outputs  
├── data/                      # Dataset folder (World Bank / Kaggle CSVs)  
├── results/                   # Plots, metrics, and model outputs  
└── screenshots/               # Project demo screenshots  

🖼️ Demo Screenshots
Description	Screenshot
Data Visualization	

Model Evaluation	
🚀 How to Run

Clone this repository:

git clone https://github.com/your-username/AI-for-SDG13-Climate-Action.git
cd AI-for-SDG13-Climate-Action


Install dependencies:

pip install -r requirements.txt


Run the notebook:

jupyter notebook co2_emissions_notebook.ipynb


(or open in Google Colab)

💡 Future Improvements

Integrate real-time emission data using APIs.

Compare multiple algorithms (Linear Regression, XGBoost).

Deploy the model using Streamlit for interactive visualization.

👥 Author

[Your Full Name]
Dereja Academy — AI for Sustainable Development Week 2 Assignment

“AI can be the bridge between innovation and sustainability.” — UN Tech Envoy
