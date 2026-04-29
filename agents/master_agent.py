from agents.goal_agent import get_goal
from agents.planner_agent import create_plan
from rag.retriever import retrieve

def run_agent(user_input):
    goal = get_goal(user_input)
    plan = create_plan(goal)

    results = None

    for step in plan:
        if step == "retrieve_recipes":
            results = retrieve(user_input)

    return results