import ollama
import json
import os
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
                    model="llama3-8b-8192",
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-8b-8192",
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

def generate_recipe_ai(name, ingredients, cuisine, servings, context=""):
    prompt = f"Expert Chef: Generate a detailed {cuisine} recipe for {name} ({servings} servings) using these ingredients if provided: {ingredients}. Context: {context}. Format with Title, Ingredients, and Steps."
    res = get_llm_response(prompt, max_tokens=800)
    if res: return res
    return f"### ⚠️ AI Offline - Local Recipe Search: {name}\nSorry, the AI is currently offline. Please check your internet or Groq API key.\n\nSearching database for '{name}'..."

def generate_leftover_recipe(ingredients, context=""):
    prompt = f"Zero-Waste Chef: Create a recipe using ONLY: {ingredients}. Context: {context}."
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