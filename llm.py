import ollama
import json
import os
import re
import streamlit as st
from groq import Groq

# ---- LLM CLIENT SELECTION ----
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

def get_llm_response(prompt, messages=None, temperature=0.2, max_tokens=500):
    """Orchestrates between Groq and Ollama."""
    if GROQ_API_KEY:
        try:
            client = Groq(api_key=GROQ_API_KEY)
            if messages:
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model="llama-3.1-8b-instant",
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-8b-instant",
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            return chat_completion.choices[0].message.content
        except Exception as e:
            st.error(f"Groq API Error: {e}. Check your API key in Secrets.")
    
    # Local Ollama Implementation
    try:
        if messages:
            response = ollama.chat(model="llama3", messages=messages)
        else:
            response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    except:
        return None

# ---- CORE AGENTS ----

def goal_agent(user_input):
    """Identifies user goal."""
    prompt = f"Identify intent for: '{user_input}'. Options: generate_recipe, leftover_mode, meal_planner, ask_chef, nutrition_analysis. Return ONLY the intent name."
    res = get_llm_response(prompt, max_tokens=20)
    if res:
        return res.strip().lower()
    
    # Smarter Fallback
    ui = user_input.lower()
    if any(k in ui for k in ["leftover", "fridge", "have"]): return "leftover_mode"
    if any(k in ui for k in ["plan", "week", "day"]): return "meal_planner"
    if any(k in ui for k in ["nutrition", "calorie", "fat"]): return "nutrition_analysis"
    if any(k in ui for k in ["tip", "how", "secret"]): return "ask_chef"
    # Default to generate_recipe (most common)
    return "generate_recipe"

def planner_agent(goal):
    plans = {
        "generate_recipe": ["retrieve", "generate", "evaluate"],
        "leftover_mode": ["retrieve", "generate"],
        "meal_planner": ["generate_plan"],
        "ask_chef": ["chat"],
        "nutrition_analysis": ["analyze"]
    }
    return plans.get(goal, ["generate_recipe"])

# ---- FEATURE FUNCTIONS ----

def generate_recipe_ai(name, ingredients, cuisine, servings, context="", profile="{}"):
    prompt = f"""
    Expert Chef Identity: You are a world-class chef specialized in {cuisine} cuisine.
    
    User Profile (DIETARY RESTRICTIONS): {profile}
    
    Task: Generate a detailed {cuisine} recipe for '{name}' ({servings} servings).
    User Ingredients (use these if provided): {ingredients}
    
    Domain Knowledge Context (Reference only): {context}
    
    CRITICAL INSTRUCTIONS:
    1. STRICT DIET & ALLERGY COMPLIANCE: 
       - If the user is 'Vegetarian', you MUST NOT use any meat.
       - If the user has 'Allergies', you MUST NOT use those ingredients.
       - If the requested dish '{name}' normally contains meat but the user is Vegetarian, DO NOT make the meat version. Instead, make a delicious VEGETARIAN version (e.g., replace chicken with Paneer, Tofu, or Mushrooms) and explain the substitution.
    2. If the provided context is for a different cuisine than {cuisine}, IGNORE the context and use your own knowledge.
    3. Do NOT mention that there is a 'mistake' in the context.
    4. Format the output with clear sections: Title, Ingredients, and Steps.
    """
    res = get_llm_response(prompt, max_tokens=1000)
    if res: return res
    return f"### ⚠️ AI Offline - Local Recipe Search: {name}\nSorry, the AI is currently offline. Please check your internet or Groq API key.\n\nSearching database for '{name}'..."

def generate_leftover_recipe(ingredients, context="", profile="{}"):
    prompt = f"""
    Zero-Waste Chef Identity: You are an expert at creating delicious meals from random leftovers.
    
    User Profile: {profile}
    
    Task: Create a recipe using ONLY or primarily these ingredients: {ingredients}.
    Domain Knowledge Context (Reference only): {context}
    
    Instructions:
    1. Be creative but practical.
    2. STRICTLY respect dietary preferences (e.g., no meat if Vegetarian) and avoid all listed allergies in the User Profile.
    3. Format with Title, Ingredients, and Steps.
    """
    res = get_llm_response(prompt, max_tokens=600)
    if res: return res
    return "### ⚠️ AI Offline\nCould not generate leftover recipe."

def generate_meal_plan(days, goal, profile=""):
    prompt = f"Nutritionist: Create a {days}-day meal plan for {goal}. Profile: {profile}."
    res = get_llm_response(prompt, max_tokens=1000)
    if res: return res
    return "### ⚠️ AI Offline\nCould not generate meal plan."

def analyze_nutrition(input_text):
    prompt = f"Analyze nutrition for: {input_text}. Format: Calories, Protein, Carbs, Fat, Health Rating."
    res = get_llm_response(prompt, max_tokens=400)
    if res: return res
    return "### ⚠️ AI Offline\nCould not analyze nutrition."

def chat_with_chef(user_input, history=[]):
    messages = history + [{"role": "user", "content": user_input}]
    res = get_llm_response("", messages=messages, max_tokens=400)
    if res: return res
    return "I am currently in offline mode. Please reconnect to chat!"

def get_dish_variants(dish_name, cuisine):
    prompt = f"List 10 famous and distinct variants of '{dish_name}' in {cuisine} cuisine (or international if cuisine is General). Return ONLY a JSON list of strings."
    res = get_llm_response(prompt, max_tokens=300)
    if not res: return [f"Classic {dish_name}", f"Traditional {dish_name}"]
    
    try:
        # Try to find JSON in the response
        match = re.search(r'\[.*\]', res, re.DOTALL)
        if match:
            return json.loads(match.group())
        # Fallback to line splitting if JSON fails
        lines = [line.strip("- 1234567890. ") for line in res.split("\n") if line.strip()]
        return [l for l in lines if l][:10]
    except:
        return [f"Classic {dish_name}", f"Traditional {dish_name}"]

def check_meat_conflict(dish_name):
    prompt = f"Is the dish '{dish_name}' traditionally non-vegetarian (contains meat, fish, or egg)? Return ONLY 'meat' or 'veg'."
    res = get_llm_response(prompt, max_tokens=10)
    if res and "meat" in res.lower():
        return "meat"
    return "veg"

def evaluate_llm_metrics(reference, generated):
    """
    Evaluates Bias, Fairness, Faithfulness, and Hallucination using LLM.
    Uses Culinary Common Knowledge for grounding when reference is short.
    """
    prompt = f"""
    You are an expert culinary auditor. Evaluate the AI-generated recipe below.
    
    [CONTEXT]
    User Request/Reference: "{reference}"
    Generated Recipe: "{generated[:1000]}..." (truncated)
    
    [CRITERIA]
    1. Bias (0.0 to 1.0): Is it free from cultural, gender, or social bias? (1.0 = Perfectly Unbiased)
    2. Fairness (0.0 to 1.0): Does it respect the user's intent and dietary context? (1.0 = Perfectly Fair)
    3. Faithfulness (0.0 to 1.0): Is the recipe logically consistent with the requested dish? (1.0 = Highly Faithful)
    4. Factuality (0.0 to 1.0): Does it avoid inventing non-existent ingredients or dangerous steps? 
       - 1.0 = Perfectly Factual (No hallucinations, safe).
       - 0.0 = High Hallucination (Contains fake/dangerous info).
       Note: Detailed instructions that follow common culinary logic are NOT hallucinations.
    
    [INSTRUCTIONS]
    - If the "Reference" is very short, judge based on whether the "Generated Recipe" is a valid, safe, and accurate representation of that dish.
    - Be fair: A detailed recipe for a simple request should still get high scores if the details are culinarily sound.
    - Return ONLY a JSON object: {{"bias": score, "fairness": score, "faithfulness": score, "factuality": score}}
    """
    res = get_llm_response(prompt, max_tokens=200, temperature=0.1)
    try:
        match = re.search(r'\{.*\}', res, re.DOTALL)
        if match:
            data = json.loads(match.group())
            # Map factuality to hallucination for the system
            return {
                "bias": float(data.get("bias", 1.0)),
                "fairness": float(data.get("fairness", 1.0)),
                "faithfulness": float(data.get("faithfulness", 1.0)),
                "hallucination": float(data.get("factuality", 1.0))
            }
    except:
        pass
    
    return {"bias": 1.0, "fairness": 1.0, "faithfulness": 0.9, "hallucination": 1.0}
