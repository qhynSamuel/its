from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pulp

app = Flask(__name__)

# Recipe data
recipes = {
    'Kung Pao Chicken': {'Chicken': 200, 'Peanuts': 50, 'Bell Pepper': 30, 'Chili': 10},
    'Mapo Tofu': {'Tofu': 250, 'Ground Beef': 100, 'Sichuan Peppercorn': 5},
    'Braised Pork': {'Pork': 300, 'Soy Sauce': 10, 'Sugar': 20},
    'Hot and Sour Soup': {'Egg': 2, 'Wood Ear Mushroom': 20, 'Vinegar': 15},
    'Steamed Fish': {'Fish': 400, 'Ginger': 10, 'Cooking Wine': 10}
}

# Initialize global variables
model = None
next_week_inventory = []

def train_model():
    global model, next_week_inventory
    try:
        # Create recipe DataFrame
        recipe_df = pd.DataFrame(recipes).T.reset_index()
        recipe_df = recipe_df.rename(columns={'index': 'Dish'})
        
        # Simulate weekly sales data
        np.random.seed(42)
        weeks = 10  # Assume 10 weeks of data
        sales_data = []

        for week in range(weeks):
            for dish in recipes.keys():
                sales_count = np.random.randint(50, 200)  # Sales quantity per dish per week
                sales_data.append([week + 1, dish, sales_count])

        sales_df = pd.DataFrame(sales_data, columns=['Week', 'Dish', 'Sales'])

        # Merge recipe data and sales data
        merged_data = pd.merge(sales_df, recipe_df, on='Dish')

        # Multiply ingredient amounts by sales quantity to get weekly ingredient usage
        for ingredient in recipe_df.columns[1:]:
            merged_data[ingredient] = merged_data[ingredient] * merged_data['Sales']

        # Ensure there are no missing values and the types are correct
        merged_data = merged_data.fillna(0)

        # Calculate total weekly ingredient usage
        weekly_usage = merged_data.groupby('Week').sum(numeric_only=True).reset_index()

        # Select features and target variables
        X = weekly_usage.drop(columns=['Week'])
        y = weekly_usage.drop(columns=['Week', 'Sales'])

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train random forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

        # Predict next week's inventory demand
        next_week_inventory = model.predict(pd.DataFrame([X.iloc[-1].values], columns=X.columns))[0]
        print("Predicted Inventory for Next Week:", next_week_inventory)

        # Debugging output
        print(f"next_week_inventory: {next_week_inventory}")
    except Exception as e:
        print(f"Error during model training: {e}")

@app.route('/')
def home():
    return "Welcome to the Inventory Management System!"

@app.route('/train', methods=['GET'])
def train():
    train_model()
    return jsonify({"message": "Model trained successfully"})

@app.route('/predict', methods=['GET'])
def predict():
    global next_week_inventory
    try:
        if not next_week_inventory:
            return jsonify({"error": "Predicted inventory is empty. Please train the model first."}), 400
        return jsonify({"Predicted Inventory for Next Week": next_week_inventory.tolist()})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An error occurred during prediction."}), 500

@app.route('/simulate', methods=['POST'])
def simulate():
    global next_week_inventory
    if not next_week_inventory:
        return jsonify({"error": "Predicted inventory is empty. Please train the model first."}), 400

    data = request.json
    days = data.get('days', 70)
    
    inventory = {
        'Chicken': next_week_inventory[0],
        'Peanuts': next_week_inventory[1],
        'Bell Pepper': next_week_inventory[2],
        'Chili': next_week_inventory[3],
        'Tofu': next_week_inventory[4],
        'Ground Beef': next_week_inventory[5],
        'Sichuan Peppercorn': next_week_inventory[6],
        'Pork': next_week_inventory[7],
        'Soy Sauce': next_week_inventory[8],
        'Sugar': next_week_inventory[9],
        'Egg': next_week_inventory[10],
        'Wood Ear Mushroom': next_week_inventory[11],
        'Vinegar': next_week_inventory[12],
        'Fish': next_week_inventory[13],
        'Ginger': next_week_inventory[14],
        'Cooking Wine': next_week_inventory[15]
    }

    days_in_inventory = {ingredient: 0 for ingredient in inventory.keys()}

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
        print(f"Day {day}")

        days_in_inventory = {ingredient: days_in_inventory[ingredient] + 1 for ingredient in days_in_inventory.keys()}

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

        print("Suggested Promotion Plan:")
        for var in promotion_vars.values():
            if var.varValue > 0:
                print(f"{var.name}: {var.varValue}")

        actual_sales = data.get('actual_sales', {})

        for dish, counts in actual_sales.items():
            if counts['promotions'] > 0:
                for ingredient, amount in recipes[dish].items():
                    inventory[ingredient] -= amount * counts['promotions']

        for ingredient, days in days_in_inventory.items():
            if days > shelf_life[ingredient]:
                inventory[ingredient] = 0

        print("Updated Inventory:")
        print(inventory)
        print("\n")

    return jsonify({"message": "Simulation complete", "Updated Inventory": inventory})

if __name__ == '__main__':
    app.run(debug=True)
