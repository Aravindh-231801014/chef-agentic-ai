import ollama
import json
import os
import streamlit as st
from groq import Groq

# ---- LLM CLIENT SELECTION ----
# Detect if we should use Groq (Cloud) or Ollama (Local)
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

def get_llm_response(prompt, messages=None, temperature=0.2, max_tokens=500):
    """Orchestrates between Groq and Ollama."""
    if GROQ_API_KEY:
        try:
            client = Groq(api_key=GROQ_API_KEY)
            if messages:
                # Chat mode
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model="llama3-8b-8192",
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                # Single prompt mode
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-8b-8192",
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Groq Error: {e}")
            # Fallback to local if possible or mock
    
    # Local Ollama Implementation
    try:
        if messages:
            response = ollama.chat(
                model="llama3", 
                messages=messages,
                options={"temperature": temperature, "num_predict": max_tokens}
            )
        else:
            response = ollama.chat(
                model="llama3", 
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature, "num_predict": max_tokens}
            )
        return response["message"]["content"]
    except Exception as e:
        print(f"Ollama/Local Error: {e}")
        return None

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
    res = get_llm_response(prompt, max_tokens=100)
    if res:
        return res.strip().lower()
    
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
    res = get_llm_response(prompt, max_tokens=500)
    if res: return res
    return f"### Mock Recipe: {name}\n**Ingredients:**\n- {ingredients}\n- 1 cup Water\n- Salt to taste\n\n**Steps:**\n1. Combine ingredients.\n2. Cook for 20 minutes."

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
    res = get_llm_response(prompt, max_tokens=400)
    if res: return res
    return f"### Mock Leftover Recipe\nUsing: {ingredients}\n\n**Steps:**\n1. Sauté everything in a pan.\n2. Add seasoning."

def generate_meal_plan(days, goal, profile=""):
    """Generates a multi-day meal plan."""
    prompt = f"""
    Generate a {days}-day meal plan.
    User Profile: {profile}
    Health Goal: {goal}
    
    Provide Breakfast, Lunch, and Dinner for each day.
    Keep it varied and healthy.
    """
    res = get_llm_response(prompt, max_tokens=600)
    if res: return res
    return f"### Mock {days}-Day Meal Plan\nGoal: {goal}\n- Day 1: Oats, Salad, Grilled Veggies"

def analyze_nutrition(input_text):
    """Analyzes nutritional value."""
    prompt = f"""
    Provide detailed nutritional information for: {input_text}.
    Include Calories, Protein, Carbs, and Fats.
    Also give a health rating (1-10) and why.
    """
    res = get_llm_response(prompt, temperature=0.1, max_tokens=300)
    if res: return res
    return f"### Nutrition Analysis: {input_text}\n- Calories: ~250 kcal\n- Protein: 10g\n- Carbs: 30g"

def chat_with_chef(user_input, history=[]):
    """General chat with the AI chef."""
    messages = history + [{"role": "user", "content": user_input}]
    res = get_llm_response("", messages=messages, temperature=0.4, max_tokens=300)
    if res: return res
    return "I'm currently in 'Offline Mode'. How else can I help with basic tips?"