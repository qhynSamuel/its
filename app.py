import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pulp

app = Flask(__name__)
CORS(app)

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# Recipe data
recipes = {
    'Kung Pao Chicken': {'Chicken': 200, 'Peanuts': 50, 'Bell Pepper': 30, 'Chili': 10},
    'Mapo Tofu': {'Tofu': 250, 'Ground Beef': 100, 'Sichuan Peppercorn': 5},
    'Braised Pork': {'Pork': 300, 'Soy Sauce': 10, 'Sugar': 20},
    'Hot and Sour Soup': {'Egg': 2, 'Wood Ear Mushroom': 20, 'Vinegar': 15},
    'Steamed Fish': {'Fish': 400, 'Ginger': 10, 'Cooking Wine': 10}
}

# Initialize model
model = None

def simulate_sales_data(weeks=10):
    np.random.seed(42)
    sales_data = []

    for week in range(weeks):
        for dish in recipes.keys():
            sales_count = np.random.randint(50, 200)  # Sales quantity per dish per week
            sales_data.append([week + 1, dish, sales_count])

    return pd.DataFrame(sales_data, columns=['Week', 'Dish', 'Sales'])

def prepare_training_data(sales_df):
    recipe_df = pd.DataFrame(recipes).T.reset_index().rename(columns={'index': 'Dish'})
    merged_data = pd.merge(sales_df, recipe_df, on='Dish')

    for ingredient in recipe_df.columns[1:]:
        merged_data[ingredient] = merged_data[ingredient] * merged_data['Sales']

    merged_data = merged_data.fillna(0)
    weekly_usage = merged_data.groupby('Week').sum(numeric_only=True).reset_index()

    return weekly_usage

def initialize_inventory(X, model):
    next_week_inventory = model.predict(pd.DataFrame([X.iloc[-1].values], columns=X.columns))[0]
    ingredients = list(X.columns)
    inventory = {ingredients[i]: next_week_inventory[i] for i in range(len(ingredients))}
    days_in_inventory = {ingredient: 0 for ingredient in inventory.keys()}

    return inventory, days_in_inventory

def train_model():
    global model, inventory, days_in_inventory
    
    try:
        sales_df = simulate_sales_data()
        logging.info("Simulated sales data generated.")
        
        weekly_usage = prepare_training_data(sales_df)
        logging.info("Training data prepared.")
        
        X = weekly_usage.drop(columns=['Week'])
        y = weekly_usage.drop(columns=['Week', 'Sales'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logging.info("Model training completed successfully.")

        inventory, days_in_inventory = initialize_inventory(X, model)
        logging.info("Inventory initialized.")
        
        return True
    except Exception as e:
        logging.error("Error during model training: %s", str(e))
        return False

@app.route('/')
def home():
    return "Welcome to the Inventory Management System!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            raise ValueError("Model is not initialized.")
        
        data = request.json
        X = pd.DataFrame([data['features']])
        prediction = model.predict(X)[0]
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        logging.error("Error during prediction: %s", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/promote', methods=['POST'])
def promote():
    try:
        data = request.json
        inventory = data['inventory']
        days_in_inventory = data['days_in_inventory']
        profits = {
            'Kung Pao Chicken': 20,
            'Mapo Tofu': 15,
            'Braised Pork': 25,
            'Hot and Sour Soup': 10,
            'Steamed Fish': 30
        }
        shelf_life = {
            'Chicken': 5, 'Peanuts': 90, 'Bell Pepper': 7, 'Chili': 180,
            'Tofu': 3, 'Ground Beef': 5, 'Sichuan Peppercorn': 365,
            'Pork': 7, 'Soy Sauce': 365, 'Sugar': 365,
            'Egg': 21, 'Wood Ear Mushroom': 180, 'Vinegar': 365,
            'Fish': 5, 'Ginger': 30, 'Cooking Wine': 365
        }

        shelf_life_factor = {
            ingredient: max(0, 1 - days_in_inventory[ingredient] / shelf_life[ingredient])
            for ingredient in shelf_life.keys()
        }

        prob = pulp.LpProblem("PromotionPlan", pulp.LpMaximize)

        promotion_vars = {dish: pulp.LpVariable(dish, lowBound=0, cat='Integer') for dish in recipes.keys()}

        profit_weight = 0.7
        shelf_life_weight = 0.3

        prob += pulp.lpSum(profit_weight * profits[dish] * promotion_vars[dish] +
                           shelf_life_weight * pulp.lpSum(shelf_life_factor[ingredient] * promotion_vars[dish] * recipes[dish].get(ingredient, 0)
                                                          for ingredient in recipes[dish].keys())
                           for dish in recipes.keys())

        for ingredient in inventory.keys():
            prob += pulp.lpSum(promotion_vars[dish] * recipes[dish].get(ingredient, 0) for dish in recipes.keys()) <= inventory[ingredient]

        prob.solve()

        promotion_plan = {var.name: var.varValue for var in promotion_vars.values() if var.varValue > 0}

        return jsonify({'promotion_plan': promotion_plan})
    except Exception as e:
        logging.error("Error during promotion planning: %s", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    model_initialized = train_model()
    if model_initialized:
        app.run(debug=True)
    else:
        logging.error("Failed to initialize the model.")
