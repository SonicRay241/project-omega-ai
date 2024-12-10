from flask import Flask, request, jsonify
from dotenv import dotenv_values

# base libraries
import numpy as np
import pandas as pd

#For exporting model
import pickle
import os

# Init
env = dotenv_values(".env") # Load environment
export_path = 'model'

#scalers
with open(os.path.join(export_path,'modal_scaler.pkl'), 'rb') as f:
    loaded_modal_scaler = pickle.load(f)

with open(os.path.join(export_path,'stock_scaler.pkl'), 'rb') as f:
    loaded_stock_scaler = pickle.load(f)

with open(os.path.join(export_path,'competitor_scaler.pkl'), 'rb') as f:
    loaded_competitor_scaler = pickle.load(f)

with open(os.path.join(export_path,'demand_scaler.pkl'), 'rb') as f:
    loaded_demand_scaler = pickle.load(f)

with open(os.path.join(export_path,'final_scaler.pkl'), 'rb') as f:
    loaded_final_scaler = pickle.load(f)
    
app = Flask(__name__)

#model
with open(os.path.join(export_path,'SVR_Model.pkl'), 'rb') as f:
    loaded_model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        demand_rate = request.json["demandRate"]
        key = request.json["secretKey"]

        if not demand_rate or not key or key != env["SECRET_KEY"]:
            return jsonify(
                isError = True,
                message = "Incomplete JSON",
                statusCode = 400,
                data = data
            ), 400

        competitor_price = 40_000 # Average from db

        scaled_modal = loaded_modal_scaler.transform([[20_000]]) # Fetch db
        scaled_stock = loaded_stock_scaler.transform([[40]]) # Fetch db + demand
        scaled_demand_rate = loaded_demand_scaler.transform([[float(demand_rate)]]) # Demand
        scaled_competitor_price = loaded_competitor_scaler.transform([[competitor_price]]) # Average All Competitor
        profit_margin = [0.05] # 0-1, fetch db

        data = pd.DataFrame({
            "Modal": scaled_modal[0],
            "Stock": scaled_stock[0],
            "DemandRate": scaled_demand_rate[0],
            "CompetitorPrice": scaled_competitor_price[0],
            "ProfitMargin": profit_margin
        })
        
        prediction = loaded_model.predict(data).reshape(-1,1)

        final_price = (np.round(loaded_final_scaler.inverse_transform(prediction)[0])).item()
    
        return {
            "finalPrice": final_price
        }
    else:
        return jsonify(
            isError = False,
            message = "Success",
            statusCode = 405,
            data = data
        ), 405