import pickle
import pandas as pd
import plotly.express as px
import streamlit as st

from simulation_engine import add_default_factory, simulate_factory_scenarios
from recommendation_engine import risk_label

st.set_page_config(page_title="Nassau Candy Decision Intelligence Dashboard", layout="wide")
st.title("🍭 Nassau Candy Decision Intelligence Dashboard")

DATA_FILE = "Nassau Candy Distributor (1).csv"
MODEL_FILE = "leadtime_model.pkl"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")
    df["Lead_Time"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Month"] = df["Order Date"].dt.month
    df["Weekday"] = df["Order Date"].dt.dayofweek
    df = df.dropna(subset=["Lead_Time", "Order Date", "Ship Date"]).copy()
    return add_default_factory(df)

@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

df = load_data()
model = load_model()

with st.sidebar:
    st.header("Filters")
    product = st.selectbox("Select Product", sorted(df["Product Name"].dropna().unique()))
    region = st.selectbox("Select Region", ["All"] + sorted(df["Region"].dropna().unique()))
    ship_mode = st.selectbox("Select Ship Mode", ["All"] + sorted(df["Ship Mode"].dropna().unique()))
    priority = st.slider("Optimization Priority (Speed)", 0, 100, 70)

filtered = df[df["Product Name"] == product].copy()
if region != "All":
    filtered = filtered[filtered["Region"] == region]
if ship_mode != "All":
    filtered = filtered[filtered["Ship Mode"] == ship_mode]
if filtered.empty:
    st.warning("No records found for this combination. Showing all records for selected product.")
    filtered = df[df["Product Name"] == product].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Orders", len(filtered))
c2.metric("Average Lead Time", round(filtered["Lead_Time"].mean(), 2))
c3.metric("Total Sales", round(filtered["Sales"].sum(), 2))
c4.metric("Total Profit", round(filtered["Gross Profit"].sum(), 2))

st.subheader("Predicted Lead Time")
sample = filtered.iloc[0]
x = pd.DataFrame([{
    "Division": sample["Division"],
    "Region": sample["Region"],
    "Ship Mode": sample["Ship Mode"],
    "Product Name": sample["Product Name"],
    "Sales": sample["Sales"],
    "Units": sample["Units"],
    "Gross Profit": sample["Gross Profit"],
    "Cost": sample["Cost"],
    "Month": int(sample["Month"]),
    "Weekday": int(sample["Weekday"]),
}])
prediction = float(model.predict(x)[0])
st.success(f"{prediction:.2f} Days")
st.caption(f"Risk label: {risk_label(prediction)}")

st.subheader("Factory Scenario Simulation")
scenario = simulate_factory_scenarios(sample, model)
st.dataframe(scenario, use_container_width=True)
st.plotly_chart(px.bar(scenario, x="Factory", y="Predicted Lead Time", title="Predicted Lead Time by Factory"), use_container_width=True)

st.subheader("Average Lead Time by Region")
region_df = df.groupby("Region", as_index=False)["Lead_Time"].mean()
st.plotly_chart(px.bar(region_df, x="Region", y="Lead_Time", color="Lead_Time"), use_container_width=True)

st.subheader("Lead Time by Ship Mode")
ship_df = df.groupby("Ship Mode", as_index=False)["Lead_Time"].mean()
st.plotly_chart(px.pie(ship_df, names="Ship Mode", values="Lead_Time"), use_container_width=True)

st.subheader("Top Products by Sales")
top_products = df.groupby("Product Name", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False).head(10)
st.plotly_chart(px.bar(top_products, x="Sales", y="Product Name", orientation="h"), use_container_width=True)

st.subheader("Lead Time Distribution")
st.plotly_chart(px.histogram(df, x="Lead_Time", nbins=20), use_container_width=True)

st.subheader("Filtered Data")
st.dataframe(filtered, use_container_width=True)

st.download_button(
    "Download Filtered Data",
    filtered.to_csv(index=False),
    file_name="filtered_data.csv",
    mime="text/csv"
)
