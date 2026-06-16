import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

st.title("Nassau Candy Decision Intelligence Dashboard")

df = pd.read_csv("Nassau Candy Distributor (1).csv")
df["Order Date"]=pd.to_datetime(df["Order Date"],dayfirst=True)
df["Ship Date"]=pd.to_datetime(df["Ship Date"],dayfirst=True)
df["Lead_Time"]=(df["Ship Date"]-df["Order Date"]).dt.days
df["Month"]=df["Order Date"].dt.month
df["Weekday"]=df["Order Date"].dt.dayofweek

X=df[["Division","Region","Ship Mode","Product Name","Sales","Units","Gross Profit","Cost","Month","Weekday"]]
y=df["Lead_Time"]

prep=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),
["Division","Region","Ship Mode","Product Name"])],remainder="passthrough")

model=Pipeline([("prep",prep),("rf",RandomForestRegressor(n_estimators=50,random_state=42))])

model.fit(X,y)

st.success("Model trained successfully without pkl file.")
