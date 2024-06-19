from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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

# Initial inventory and days in inventory
inventory = {}
days_in_inventory = {}
model = None

def train_model():
    global model, inventory, days_in_inventory
    
    # Simulate weekly sales data
    np.random.seed(42)
    weeks = 10  # Assume 10 weeks of data
    sales_data = []

    for week in range(weeks):
        for dish in recipes.keys():
            sales_count = np.random.randint(50, 200)  # Sales quantity per dish per week
            sales_data.append([week + 1, dish, sales_count])

    sales_df = pd.DataFrame(sales_data, columns=['Week', 'Dish', 'Sales'])

    # Create recipe DataFrame
    recipe_df = pd.DataFrame(recipes).T.reset_index()
    recipe_df = recipe_df.rename(columns={'index': 'Dish'})

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

    # Predict next week's inventory demand
    next_week_inventory = model.predict(pd.DataFrame([X.iloc[-1].values], columns=X.columns))[0]

    # Initialize inventory based on the prediction
    ingredients = list(X.columns)
    inventory = {ingredients[i]: next_week_inventory[i] for i in range(len(ingredients))}
    days_in_inventory = {ingredient: 0 for ingredient in inventory.keys()}
    
@app.route('/')
def home():
    return "Welcome to the Inventory Management System!"

@app.route('/predict', methods=['POST'])
def predict():
    # Predict next week's inventory demand based on current data
    data = request.json
    X = pd.DataFrame([data['features']])
    prediction = model.predict(X)[0]
    return jsonify({'prediction': prediction.tolist()})

@app.route('/promote', methods=['POST'])
def promote():
    # Generate promotion strategy
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

    # Generate promotion plan
    promotion_plan = {var.name: var.varValue for var in promotion_vars.values() if var.varValue > 0}

    return jsonify({'promotion_plan': promotion_plan})

if __name__ == '__main__':
    train_model()
    app.run(debug=True)
