# Install dependecies: !pip install langgraph langchain langchain-groq streamlit
# Run: streamlit run ui_planningpattern.py

import streamlit as st
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
import random
import time

import os
from pathlib import Path
import sys

# add parent folder to sys.path so we can import get_api from parent directory
sys.path.insert(0, str(Path.cwd().parent))
import get_api

api_keys = get_api.get_api_keys()
os.environ["GROQ_API_KEY"] = api_keys["GROQ_API_KEY"]
# --------------------------------------------------
# UI CONFIG
# --------------------------------------------------

st.set_page_config(page_title="Planning Agent Demo", layout="wide")

st.title("🧠 Planning Pattern Agent")
st.caption("Visualizing Plan → Execute → Replan loop")

# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatGroq(model="mixtral-8x7b-32768")

# --------------------------------------------------
# State
# --------------------------------------------------

class AgentState(TypedDict, total=False):
    input: str
    tasks: List[str]
    completed: List[str]
    credit_score: int
    products: List[str]
    documents: str
    status: str
    plan_history: List[List[str]]

# --------------------------------------------------
# Tools
# --------------------------------------------------

def fetch_credit_score(state):
    score = random.randint(600, 800)
    return {"credit_score": score}


def fetch_prime_products(state):
    return {"products": ["Prime Loan A", "Prime Loan B"]}


def fetch_subprime_products(state):
    return {"products": ["Subprime Loan X", "Subprime Loan Y"]}


def gather_documents(state):
    return {"documents": "Collected"}


def submit_application(state):
    return {"status": "Submitted"}


def request_guarantor(state):
    return {"documents": "Guarantor Added"}

TOOLS = {
    "fetch_credit_score": fetch_credit_score,
    "fetch_prime_products": fetch_prime_products,
    "fetch_subprime_products": fetch_subprime_products,
    "gather_documents": gather_documents,
    "submit_application": submit_application,
    "request_guarantor": request_guarantor
}

# --------------------------------------------------
# Planner
# --------------------------------------------------

def planner_node(state: AgentState):
    plan_history = state.get("plan_history", [])

    if "credit_score" not in state:
        tasks = ["fetch_credit_score"]
        reason = "Need credit score"
        stage = "Initial Plan"

    elif "products" not in state:
        if state["credit_score"] < 650:
            tasks = ["request_guarantor"]
            reason = "Low score → guarantor"
        elif state["credit_score"] >= 700:
            tasks = ["fetch_prime_products"]
            reason = "High score → prime"
        else:
            tasks = ["fetch_subprime_products"]
            reason = "Medium score → subprime"
        stage = "Replan"

    elif "documents" not in state:
        tasks = ["gather_documents"]
        reason = "Collect documents"
        stage = "Replan"

    elif "status" not in state:
        tasks = ["submit_application"]
        reason = "Submit application"
        stage = "Replan"

    else:
        tasks = []
        reason = "Done"
        stage = "Completed"

    plan_history.append(tasks)

    return {
        "tasks": tasks,
        "plan_history": plan_history,
        "reason": reason,
        "stage": stage
    }

# --------------------------------------------------
# Executor
# --------------------------------------------------

def executor_node(state: AgentState):
    task = state.get("tasks", [None])[0]
    if not task:
        return state

    result = TOOLS[task](state)

    completed = state.get("completed", []) + [task]

    return {
        **state,
        **result,
        "completed": completed,
        "tasks": []
    }

# --------------------------------------------------
# UI Helpers
# --------------------------------------------------

def render_step(step, plan, reason, state):
    with st.container():
        st.markdown(f"### 🔹 Step {step}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.info(f"**Stage**\n{plan['stage']}")

        with col2:
            st.success(f"**Task**\n{plan['tasks']}")

        with col3:
            st.warning(f"**Reason**\n{reason}")

        st.progress(min(step * 20, 100))

        st.markdown("**State Snapshot**")
        st.json({k: v for k, v in state.items() if k != "plan_history"})

        st.markdown("---")

# --------------------------------------------------
# Run
# --------------------------------------------------

if st.button("🚀 Run Agent"):

    state = {
        "input": "Apply for home loan",
        "tasks": [],
        "completed": [],
        "plan_history": []
    }

    timeline = []

    for step in range(1, 10):

        plan = planner_node(state)
        state.update(plan)

        render_step(step, plan, plan["reason"], state)

        state = executor_node(state)

        timeline.append(plan["tasks"])

        time.sleep(0.8)  # animation feel

        if state.get("status") == "Submitted":
            st.success("🎉 Loan Application Submitted")
            break

    # --------------------------------------------------
    # Timeline View
    # --------------------------------------------------

    st.subheader("📈 Plan Timeline")

    for i, t in enumerate(timeline, 1):
        st.write(f"Step {i} → {t}")

    # --------------------------------------------------
    # Final Summary
    # --------------------------------------------------

    st.subheader("📊 Final Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Credit Score", state.get("credit_score", "-"))
    col2.metric("Products", len(state.get("products", [])))
    col3.metric("Status", state.get("status", "-"))

    st.json(state)