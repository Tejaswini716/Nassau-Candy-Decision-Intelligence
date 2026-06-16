import pandas as pd
from factory_mapping import FACTORIES

def rank_factory_recommendations(row, model):
    records = []
    for factory in FACTORIES:
        x = pd.DataFrame([{
            "Division": row["Division"],
            "Region": row["Region"],
            "Ship Mode": row["Ship Mode"],
            "Product Name": row["Product Name"],
            "Sales": row["Sales"],
            "Units": row["Units"],
            "Gross Profit": row["Gross Profit"],
            "Cost": row["Cost"],
            "Month": int(row["Month"]),
            "Weekday": int(row["Weekday"]),
        }])
        lead_time = float(model.predict(x)[0])
        records.append({
            "Product Name": row["Product Name"],
            "Current Factory": row.get("Factory", None),
            "Recommended Factory": factory,
            "Predicted Lead Time": lead_time,
        })

    return pd.DataFrame(records).sort_values("Predicted Lead Time").reset_index(drop=True)

def risk_label(predicted_lead_time):
    if predicted_lead_time <= 3:
        return "Low"
    if predicted_lead_time <= 6:
        return "Medium"
    return "High"
