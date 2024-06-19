import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pulp
import os
import joblib
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

class InventoryModel:
    def __init__(self):
        self.recipes = {
            'Kung Pao Chicken': {'Chicken': 200, 'Peanuts': 50, 'Bell Pepper': 30, 'Chili': 10},
            'Mapo Tofu': {'Tofu': 250, 'Ground Beef': 100, 'Sichuan Peppercorn': 5},
            'Braised Pork': {'Pork': 300, 'Soy Sauce': 10, 'Sugar': 20},
            'Hot and Sour Soup': {'Egg': 2, 'Wood Ear Mushroom': 20, 'Vinegar': 15},
            'Steamed Fish': {'Fish': 400, 'Ginger': 10, 'Cooking Wine': 10}
        }
        self.recipe_df = pd.DataFrame(self.recipes).T.reset_index()
        self.recipe_df = self.recipe_df.rename(columns={'index': 'Dish'})
        self.sales_data = self.generate_sales_data(10)
        self.model, self.X = self.update_and_train_model()

    def generate_sales_data(self, weeks):
        sales_data = []
        np.random.seed(42)
        for week in range(weeks):
            for dish in self.recipes.keys():
                sales_count = np.random.randint(50, 200)  # Sales quantity per dish per week
                sales_data.append([week + 1, dish, sales_count])
        return pd.DataFrame(sales_data, columns=['Week', 'Dish', 'Sales'])

    def update_and_train_model(self):
        sales_df = self.sales_data

        # Merge recipe data and sales data
        merged_data = pd.merge(sales_df, self.recipe_df, on='Dish')

        # Multiply ingredient amounts by sales quantity to get weekly ingredient usage
        for ingredient in self.recipe_df.columns[1:]:
            merged_data[ingredient] = merged_data[ingredient] * merged_data['Sales']

        # Ensure there are no missing values and the types are correct
        merged_data = merged_data.fillna(0)

        # Calculate total weekly ingredient usage
        weekly_usage = merged_data.groupby('Week').sum(numeric_only=True).reset_index()

        # Select features and target variables
        X = weekly_usage.drop(columns=['Week', 'Sales'])
        y = weekly_usage.drop(columns=['Week'])

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train random forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save the trained model
        joblib.dump(model, 'inventory_model.pkl')

        return model, X

# Initialize the inventory model
inventory_model = InventoryModel()

@app.route('/')
def index():
    return "Hello, World! The Flask app is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        app.logger.debug("Received data for prediction: %s", data)
        # Process input data
        X_input = pd.DataFrame([data])
        prediction = inventory_model.model.predict(X_input)
        app.logger.debug("Prediction result: %s", prediction.tolist())
        return jsonify(prediction.tolist())
    except Exception as e:
        app.logger.error("Error in /predict: %s", e)
        return "Internal Server Error", 500

@app.route('/manage_inventory', methods=['POST'])
def manage_inventory():
    try:
        data = request.json
        app.logger.debug("Received data for manage_inventory: %s", data)
        inventory = data['inventory']
        days_in_inventory = data['days_in_inventory']
        
        # Manage inventory logic
        updated_inventory = manage_inventory_logic(inventory, days_in_inventory, inventory_model.recipes)
        
        return jsonify(updated_inventory)
    except Exception as e:
        app.logger.error("Error in /manage_inventory: %s", e)
        return "Internal Server Error", 500

def manage_inventory_logic(inventory, days_in_inventory, recipes):
    days = 70  # Simulate 70 days
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
    
    for day in range(1, days + 1):
        # Update days in inventory
        days_in_inventory = {ingredient: days_in_inventory[ingredient] + 1 for ingredient in days_in_inventory.keys()}
        
        # Calculate shelf life factor
        shelf_life_factor = {
            ingredient: max(0, 1 - days_in_inventory[ingredient] / shelf_life[ingredient])
            for ingredient in shelf_life.keys()
        }
        
        # Build linear programming problem
        prob = pulp.LpProblem("PromotionPlan", pulp.LpMaximize)
        
        # Decision variables: promotion quantity for each dish
        promotion_vars = {dish: pulp.LpVariable(dish, lowBound=0, cat='Integer') for dish in recipes.keys()}
        
        # Objective function: maximize weighted profit
        profit_weight = 0.7
        shelf_life_weight = 0.3
        
        prob += pulp.lpSum(profit_weight * profits[dish] * promotion_vars[dish] +
                           shelf_life_weight * pulp.lpSum(shelf_life_factor[ingredient] * promotion_vars[dish] * recipes[dish].get(ingredient, 0)
                                                          for ingredient in recipes[dish].keys())
                           for dish in recipes.keys())
        
        # Constraints: promotion quantity for each ingredient cannot exceed its inventory
        for ingredient in inventory.keys():
            prob += pulp.lpSum(promotion_vars[dish] * recipes[dish].get(ingredient, 0) for dish in recipes.keys()) <= inventory[ingredient]
        
        # Solve problem
        prob.solve()
        
        # Output suggested promotion quantity
        promotion_plan = {var.name: var.varValue for var in prob.variables() if var.varValue > 0}
        
        # Simulate actual sales and update inventory
        for dish, promotions in promotion_plan.items():
            for ingredient, amount in recipes[dish].items():
                inventory[ingredient] -= amount * promotions
        
        # Remove expired ingredients
        for ingredient, days in days_in_inventory.items():
            if days > shelf_life[ingredient]:
                inventory[ingredient] = 0
    
    return inventory

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
