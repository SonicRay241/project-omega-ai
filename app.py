from flask import Flask, request, jsonify
from dotenv import dotenv_values

# base libraries
import numpy as np
import pandas as pd
import math

#For exporting model
import pickle
import os

# Init
env = dotenv_values(".env") # Load environment
export_path = 'model'
web_url = env["WEB_URL"]

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
        competitor_price = request.json["competitorPrice"] # Average from db
        modal = request.json["basePrice"]
        
        profit_margin = [0.05] # 0-1, fetch db
        stock = 50
        # stock = request.json["stock"]

        key = request.json["secretKey"]

        print({
            "demand": demand_rate,
            "comp_price": competitor_price,
            "modal": modal
        })

        if not demand_rate or not modal or not stock or not key or key != env["SECRET_KEY"]:
            return jsonify(
                isError = True,
                message = "Incomplete JSON",
                statusCode = 400,
                data = data
            ), 400

        
        scaled_modal = loaded_modal_scaler.transform([[float(modal)]]) # Fetch db
        scaled_stock = loaded_stock_scaler.transform([[float(stock)]]) # Fetch db + demand
        scaled_demand_rate = loaded_demand_scaler.transform([[float(demand_rate)]]) # Demand
        scaled_competitor_price = loaded_competitor_scaler.transform([[competitor_price]]) # Average All Competitor

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
            "finalPrice": max(math.ceil(final_price/100) * 100, modal)
        }
    
    else: # Method other than POST
        return jsonify(
            isError = False,
            message = "E",
            statusCode = 405,
            data = data
        ), 405