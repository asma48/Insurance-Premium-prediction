import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression



data = pd.read_csv("app\data\insurance.csv")
data.head()


num_features= ["age", "bmi", "children"]
cat_features = ["smoker"]


X = data[num_features + cat_features].copy()
Y = data["charges"]

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size= 0.8, random_state= 42)

cat_encoder = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_encoder, cat_features)  
    ],
    remainder="passthrough"
)

rf = RandomForestRegressor()


model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("randomforest", rf)
    ]
)

model.fit(X_train, Y_train)
y_predict = model.predict(X_valid)

rmse = np.sqrt(mean_squared_error(Y_valid, y_predict))
print("RMSE:", rmse)


joblib.dump(model, "app\model\model_rf.pkl")