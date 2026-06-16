import pandas as pd
from factory_mapping import FACTORIES

def simulate_factory_scenarios(row, model, factories=FACTORIES):
    records = []
    for factory in factories:
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
        records.append({"Factory": factory, "Predicted Lead Time": lead_time})

    out = pd.DataFrame(records).sort_values("Predicted Lead Time", ascending=True).reset_index(drop=True)
    out["Improvement_vs_Worst"] = out["Predicted Lead Time"].max() - out["Predicted Lead Time"]
    return out

def add_default_factory(df):
    from factory_mapping import PRODUCT_FACTORY
    out = df.copy()
    out["Factory"] = out["Product Name"].map(PRODUCT_FACTORY)
    return out
