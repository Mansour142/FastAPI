import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("project/data.csv")

numerical_features = joblib.load("project/numerical_features.joblib")
categorical_features = joblib.load("project/categorical_features.joblib")
target_columns = [
    'Potential_Savings_Groceries', 'Potential_Savings_Transport',
    'Potential_Savings_Eating_Out', 'Potential_Savings_Entertainment',
    'Potential_Savings_Utilities', 'Potential_Savings_Healthcare',
    'Potential_Savings_Education', 'Potential_Savings_Miscellaneous'
]

X = df[numerical_features + categorical_features]
y = df[target_columns]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

encoder = joblib.load("project/encoder.joblib")
cat_encoded = encoder.transform(X_train[categorical_features]).toarray()
scaler = joblib.load("project/scaler.joblib")
num_scaled = scaler.transform(X_train[numerical_features])

feature_order = joblib.load("project/feature_order.joblib")
X_train_processed = pd.DataFrame(
    np.concatenate([num_scaled, cat_encoded], axis=1),
    columns=feature_order
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train_processed, y_train)

joblib.dump(model, "project/savings_predictor_forest.joblib")
