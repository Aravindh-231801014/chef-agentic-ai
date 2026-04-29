import streamlit as st
st.set_page_config(page_title="ChefMind-AI", page_icon="🍳", layout="wide")
import json
import os
import re
from llm import (
    goal_agent, planner_agent, generate_recipe_ai, 
    generate_leftover_recipe, generate_meal_plan, 
    analyze_nutrition, chat_with_chef
)
from retriever import search
from evaluation.metrics import evaluate

import base64
from pathlib import Path

# ---- CONFIG ----

# ---- HELPERS ----
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return ""

# ---- SESSION STATE ----
if "user" not in st.session_state: st.session_state.user = None
if "page" not in st.session_state: st.session_state.page = "login"
if "profile" not in st.session_state: st.session_state.profile = {}
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# ---- STYLING ----
bg_img_file = "data/img/chef_bg_login.png" if st.session_state.get("page", "login") == "login" else "data/img/chef_bg_inside.png"
bg_base64 = get_base64_image(bg_img_file)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

/* Main Container Styling */
.stApp {{
    background: linear-gradient(rgba(11, 12, 16, 0.85), rgba(11, 12, 16, 0.85)), 
                url("data:image/png;base64,{bg_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: 'Outfit', sans-serif;
    color: #C5C6C7;
}}

/* Responsive Feature Cards */
.feature-card {{
    background: rgba(31, 40, 51, 0.4);
    backdrop-filter: blur(8px);
    border-radius: 20px;
    padding: 2rem;
    border: none;
    transition: all 0.3s ease-in-out;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}}

/* Mobile Adjustments */
@media (max-width: 768px) {{
    .stApp {{
        background-attachment: scroll; /* Better performance on mobile */
    }}
    .feature-card {{
        padding: 1.2rem;
        margin-bottom: 15px;
    }}
    h1 {{
        font-size: 2.2rem !important;
    }}
    h2 {{
        font-size: 1.8rem !important;
    }}
    h3 {{
        font-size: 1.4rem !important;
    }}
}}

.feature-card:hover {{
    transform: translateY(-5px);
    background: rgba(31, 40, 51, 0.6);
    box-shadow: 0 12px 40px rgba(102, 252, 241, 0.15);
}}

/* Subtle Buttons */
.stButton>button {{
    width: 100%;
    border-radius: 12px;
    padding: 10px 20px;
    background: rgba(102, 252, 241, 0.1);
    color: #66FCF1;
    font-weight: 600;
    border: 1px solid rgba(102, 252, 241, 0.3);
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.3s ease;
}}
.stButton>button:hover {{
    background: #45A29E;
    color: #0B0C10;
    border-color: #66FCF1;
    transform: translateY(-2px);
}}

/* Typography */
h1, h2, h3 {{
    font-weight: 800 !important;
    color: #66FCF1 !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
}}

/* Chat & Inputs */
.stChatMessage {{
    background: rgba(31, 40, 51, 0.4) !important;
    backdrop-filter: blur(5px);
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}}

div[data-baseweb="input"] {{
    background-color: rgba(31, 40, 51, 0.4) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(102, 252, 241, 0.2) !important;
    color: #C5C6C7 !important;
}}
</style>
""", unsafe_allow_html=True)

# ---- HELPERS ----
def save_profile(profile):
    os.makedirs("user", exist_ok=True)
    with open("user/profile.json", "w") as f:
        json.dump(profile, f)

def load_profile():
    if os.path.exists("user/profile.json"):
        try:
            with open("user/profile.json", "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

# ---- LOGIN PAGE ----
def login_page():
    st.markdown("<h1 style='text-align:center; font-size: 3rem;'>ChefMind-AI</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if email and password:
                    st.session_state.user = email
                    st.session_state.profile = load_profile()
                    if not st.session_state.profile:
                        st.session_state.page = "setup"
                    else:
                        st.session_state.page = "dashboard"
                    st.rerun()
                else:
                    st.error("Please enter email and password")
        
        if st.button("Continue as Guest"):
            st.session_state.user = "Guest"
            st.session_state.page = "dashboard"
            st.rerun()

# ---- PROFILE SETUP ----
def setup_page():
    st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
    st.markdown("## 👤 First Time Setup")
    st.markdown("<p>Help us understand your cooking preferences for better recommendations.</p>", unsafe_allow_html=True)
    
    with st.form("profile_wizard"):
        diet = st.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian", "Vegan", "Keto", "Paleo"])
        allergies = st.text_input("Allergies (comma separated, leave blank if none)")
        cuisine = st.multiselect("Favorite Cuisines", ["Indian", "Italian", "Chinese", "Mexican", "Japanese", "French", "Thai"])
        goal = st.selectbox("Health Goal", ["Weight Loss", "Muscle Gain", "Healthy Living", "Balanced Diet"])
        skill = st.select_slider("Cooking Skill Level", options=["Beginner", "Intermediate", "Expert"])
        
        if st.form_submit_button("Save Profile"):
            profile = {
                "diet": diet,
                "allergies": allergies,
                "cuisine": cuisine,
                "goal": goal,
                "skill": skill
            }
            save_profile(profile)
            st.session_state.profile = profile
            st.session_state.page = "dashboard"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ---- DASHBOARD ----
def dashboard():
    st.sidebar.markdown(f"### Hello, {st.session_state.user}!")
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.session_state.page = "login"
        st.rerun()
    
    st.markdown("# 🏠 Home Dashboard")
    
    cols = st.columns(3)
    
    features = [
        ("Generate Recipe", "🍳", "recipe"),
        ("Leftover Mode", "🥕", "leftover"),
        ("Meal Planner", "📅", "planner"),
        ("Ask Chef AI", "🤖", "chat"),
        ("Nutrition Analyzer", "🧮", "nutrition"),
        ("Profile Settings", "👤", "profile")
    ]
    
    for i, (name, icon, key) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class='feature-card'>
                <h2 style='text-align:center;'>{icon}</h2>
                <h3 style='text-align:center; margin-bottom: 20px;'>{name}</h3>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Go to {name}", key=f"nav_{key}"):
                st.session_state.page = key
                st.rerun()

# ---- FEATURE PAGES ----

def recipe_gen():
    st.markdown("# 🍳 Generate Smart Recipe")
    if st.button("⬅ Back to Dashboard"): st.session_state.page = "dashboard"; st.rerun()
    
    col1, col2 = st.columns(2)
    with col1:
        dish_name = st.text_input("Name a dish (e.g. Paneer Tikka)")
    with col2:
        ingredients = st.text_input("Or list ingredients (e.g. paneer, onion, capsicum)")
        
    servings = st.slider("Servings", 1, 10, 2)
    
    user_cuisines = st.session_state.profile.get("cuisine", [])
    cuisine = st.selectbox("Cuisine Style", ["General"] + user_cuisines)
    
    if st.button("✨ Generate Recipe"):
        if not (dish_name or ingredients):
            st.warning("Please provide a dish name or ingredients.")
        else:
            with st.spinner("Chef AI is orchestrating agents..."):
                # Goal & Planner Agents
                goal = goal_agent(dish_name or ingredients)
                plan = planner_agent(goal)
                
                # RAG Retrieval
                st.info(f"Agents identified goal: {goal}. Retrieving domain knowledge...")
                context_data = search(dish_name or ingredients)
                context_str = json.dumps(context_data)
                
                # Generation
                recipe = generate_recipe_ai(dish_name or ingredients, ingredients, cuisine, servings, context_str)
                
                st.markdown("---")
                st.markdown(recipe)
                
                # Evaluation
                if context_data:
                    ref_recipe = context_data[0]
                    ref_text = f"{ref_recipe.get('title', '')} Ingredients: {', '.join(ref_recipe.get('ingredients', []))}"
                    scores = evaluate(ref_text, recipe)
                    
                    st.markdown("---")
                    st.markdown("### 📊 AI Quality Metrics")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("BLEU Score", scores['BLEU'])
                    m2.metric("ROUGE-1", scores['ROUGE-1'])
                    m3.metric("ROUGE-L", scores['ROUGE-L'])
                    st.caption("Metrics compared against retrieved reference recipe from FAISS database.")

def leftover_mode():
    st.markdown("# 🥕 Leftover Mode")
    if st.button("⬅ Back"): st.session_state.page = "dashboard"; st.rerun()
    
    st.markdown("### What's in your fridge?")
    items = st.text_area("List your leftovers (e.g. leftover rice, 1 egg, half an onion)")
    
    if st.button("Generate Zero-Waste Meal"):
        if not items:
            st.warning("Please list some ingredients.")
        else:
            with st.spinner("Creating magic with leftovers..."):
                recipe = generate_leftover_recipe(items)
                st.markdown("---")
                st.markdown(recipe)

def meal_planner():
    st.markdown("# 📅 Meal Planner")
    if st.button("⬅ Back"): st.session_state.page = "dashboard"; st.rerun()
    
    col1, col2 = st.columns(2)
    with col1:
        days = st.selectbox("Plan Duration", [3, 7], index=1)
    with col2:
        goal = st.selectbox("Nutrition Goal", ["Balanced", "High Protein", "Low Carb", "Weight Loss"])
    
    if st.button("Generate Plan"):
        with st.spinner("Generating your personalized plan..."):
            plan = generate_meal_plan(days, goal, json.dumps(st.session_state.profile))
            st.markdown("---")
            st.markdown(plan)

def ask_chef():
    st.markdown("# 🤖 Ask Chef AI")
    if st.button("⬅ Back"): st.session_state.page = "dashboard"; st.rerun()
    
    # Initialize chat history if empty
    if not st.session_state.chat_history:
        st.session_state.chat_history = [{"role": "assistant", "content": "Hello! I'm Chef AI. How can I help you in the kitchen today?"}]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    user_input = st.chat_input("Ask about techniques, substitutions, or troubleshooting...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)
        
        with st.spinner("Chef is typing..."):
            reply = chat_with_chef(user_input, st.session_state.chat_history[:-1])
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

def nutrition_analyzer():
    st.markdown("# 🧮 Nutrition Analyzer")
    if st.button("⬅ Back"): st.session_state.page = "dashboard"; st.rerun()
    
    dish = st.text_area("Enter a dish name or a list of ingredients to analyze")
    if st.button("Analyze Nutrition"):
        if not dish:
            st.warning("Please enter something to analyze.")
        else:
            with st.spinner("Analyzing nutritional value..."):
                result = analyze_nutrition(dish)
                st.markdown("---")
                st.markdown(result)

def profile_page():
    st.markdown("# 👤 Profile Settings")
    if st.button("⬅ Back"): st.session_state.page = "dashboard"; st.rerun()
    setup_page()

# ---- ROUTER ----
if st.session_state.page == "login": login_page()
elif st.session_state.page == "setup": setup_page()
elif st.session_state.page == "dashboard": dashboard()
elif st.session_state.page == "recipe": recipe_gen()
elif st.session_state.page == "leftover": leftover_mode()
elif st.session_state.page == "planner": meal_planner()
elif st.session_state.page == "chat": ask_chef()
elif st.session_state.page == "nutrition": nutrition_analyzer()
elif st.session_state.page == "profile": profile_page()