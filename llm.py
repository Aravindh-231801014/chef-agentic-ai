import ollama
import json
import os

# ---- CORE AGENTS ----

def goal_agent(user_input):
    """Identifies user goal."""
    prompt = f"""
    You are the Goal Agent for ChefMind-AI. 
    Analyze the user input and identify the primary intent.
    
    Possible Intents:
    - generate_recipe: User wants a full recipe based on ingredients or dish name.
    - leftover_mode: User has specific ingredients and wants to make something with ONLY those.
    - meal_planner: User wants a multi-day (usually 7 days) meal plan.
    - ask_chef: General cooking questions, tips, or troubleshooting.
    - nutrition_analysis: User wants nutritional info for a dish or ingredient list.
    
    User Input: "{user_input}"
    
    Return ONLY the intent name (one of the five above).
    """
    try:
        response = ollama.chat(
            model="llama3", 
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2, "num_predict": 350}
        )
        return response["message"]["content"].strip().lower()
    except Exception as e:
        # Fallback based on keywords
        ui = user_input.lower()
        if "leftover" in ui: return "leftover_mode"
        if "plan" in ui: return "meal_planner"
        if "nutrition" in ui or "calorie" in ui: return "nutrition_analysis"
        if "recipe" in ui or "cook" in ui: return "generate_recipe"
        return "ask_chef"

def planner_agent(goal):
    """Creates execution steps for a goal."""
    plans = {
        "generate_recipe": ["retrieve", "generate", "scale", "evaluate"],
        "leftover_mode": ["retrieve", "generate", "evaluate"],
        "meal_planner": ["generate_plan"],
        "ask_chef": ["chat"],
        "nutrition_analysis": ["analyze"]
    }
    return plans.get(goal, ["chat"])

# ---- FEATURE FUNCTIONS ----

def generate_recipe_ai(name, ingredients, cuisine, servings, context=""):
    """Generates a detailed recipe using RAG context."""
    prompt = f"""
    You are a professional Master Chef. 
    Generate a detailed recipe based on the following:
    - Dish Name/Input: {name}
    - User Ingredients: {ingredients}
    - Preferred Cuisine: {cuisine}
    - Target Servings: {servings}
    - Expert Reference Knowledge: {context}
    
    STRICT FORMAT:
    Title: [Dish Name]
    Description: [Brief description]
    Ingredients:
    - [Item] ([Quantity])
    Steps:
    1. [Instruction]
    Nutrition:
    - Calories: [Value]
    - Protein: [Value]
    - Carbs: [Value]
    - Fats: [Value]
    """
    try:
        response = ollama.chat(
            model="llama3", 
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2, "num_predict": 450}
        )
        return response["message"]["content"]
    except Exception:
        return f"### Mock Recipe: {name}\nThis is a simulated response because Ollama is still downloading the models.\n\n**Ingredients:**\n- {ingredients}\n- 1 cup Water\n- Salt to taste\n\n**Steps:**\n1. Combine ingredients.\n2. Cook for 20 minutes.\n3. Serve hot."

def generate_leftover_recipe(ingredients, context=""):
    """Generates a recipe using ONLY the provided leftovers."""
    prompt = f"""
    You are an expert at Zero-Waste Cooking.
    Create a delicious recipe using ONLY these ingredients: {ingredients}.
    You can assume basic pantry staples like oil, salt, and pepper are available.
    
    SPECIAL INSTRUCTION: If ingredients like "day-old rice" or "stale bread" are provided, prioritize classic transformation techniques (e.g., Fried Rice, Panzanella, French Toast, or Arancini).
    
    Context: {context}
    
    Format:
    Title: ...
    Ingredients: ...
    Steps: ...
    """
    try:
        response = ollama.chat(
            model="llama3", 
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2, "num_predict": 400}
        )
        return response["message"]["content"]
    except Exception:
        return f"### Mock Leftover Recipe\nUsing: {ingredients}\n\n**Steps:**\n1. Sauté everything in a pan.\n2. Add seasoning.\n3. Enjoy your quick zero-waste meal!"

def generate_meal_plan(days, goal, profile=""):
    """Generates a multi-day meal plan."""
    prompt = f"""
    Generate a {days}-day meal plan.
    User Profile: {profile}
    Health Goal: {goal}
    
    Provide Breakfast, Lunch, and Dinner for each day.
    Keep it varied and healthy.
    """
    try:
        response = ollama.chat(
            model="llama3", 
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 500}
        )
        return response["message"]["content"]
    except Exception:
        return f"### Mock {days}-Day Meal Plan\nGoal: {goal}\n\n**Day 1:**\n- Breakfast: Oats\n- Lunch: Salad\n- Dinner: Grilled Veggies"

def analyze_nutrition(input_text):
    """Analyzes nutritional value."""
    prompt = f"""
    Provide detailed nutritional information for: {input_text}.
    Include Calories, Protein, Carbs, and Fats.
    Also give a health rating (1-10) and why.
    """
    try:
        response = ollama.chat(
            model="llama3", 
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 300}
        )
        return response["message"]["content"]
    except Exception:
        return f"### Nutrition Analysis: {input_text}\n- Calories: ~250 kcal\n- Protein: 10g\n- Carbs: 30g\n- Fats: 5g\n\n*This is a mock result while models download.*"

def chat_with_chef(user_input, history=[]):
    """General chat with the AI chef."""
    messages = history + [{"role": "user", "content": user_input}]
    try:
        response = ollama.chat(
            model="llama3", 
            messages=messages,
            options={"temperature": 0.4, "num_predict": 300}
        )
        return response["message"]["content"]
    except Exception:
        return "I'm currently in 'Offline Mode' while my knowledge base downloads. How else can I help with basic tips?"