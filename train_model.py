import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA_FILE = "Nassau Candy Distributor (1).csv"
MODEL_FILE = "leadtime_model.pkl"

def load_data(path=DATA_FILE):
    df = pd.read_csv(path)
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")
    df["Lead_Time"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Month"] = df["Order Date"].dt.month
    df["Weekday"] = df["Order Date"].dt.dayofweek
    return df.dropna(subset=["Lead_Time", "Order Date", "Ship Date"]).copy()

def main():
    df = load_data()
    features = ["Division", "Region", "Ship Mode", "Product Name", "Sales", "Units", "Gross Profit", "Cost", "Month", "Weekday"]
    X = df[features]
    y = df["Lead_Time"]

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), ["Division", "Region", "Ship Mode", "Product Name"])],
        remainder="passthrough",
    )

    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("rf", RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(f"MAE: {mean_absolute_error(y_test, pred):.3f}")
    print(f"R2 : {r2_score(y_test, pred):.3f}")

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved model to {MODEL_FILE}")

if __name__ == "__main__":
    main()
