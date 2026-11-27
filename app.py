from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from openai import OpenAI  # Or use your preferred LLM provider API

# Page Configuration
st.set_page_config(
    page_title="AI Personal Finance Manager & Assistant",
    page_icon="🤖",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "model_outputs"

# Load Models & Data
@st.cache_resource
def load_resources():
    classifier = joblib.load(MODELS_DIR / "category_classifier.joblib") if (MODELS_DIR / "category_classifier.joblib").exists() else None
    summary_df = pd.read_csv(OUTPUTS_DIR / "spending_by_category.csv") if (OUTPUTS_DIR / "spending_by_category.csv").exists() else None
    return classifier, summary_df

classifier, summary_df = load_resources()

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a Feature:", [
    "📊 Spending Analysis & Leaks",
    "🔍 Live Categorizer",
    "💬 AI Financial Assistant",
    "🔮 What-If & Goal Simulator"
])

# --- 1. SPENDING ANALYSIS & MONEY LEAK ALERTS ---
if app_mode == "📊 Spending Analysis & Leaks":
    st.header("📊 Spending Analysis & Money Leak Detection")
    
    if summary_df is not None:
        total_spend = summary_df["amount"].sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Tracked Spending", f"${total_spend:,.2f}")
        col2.metric("Top Spending Category", summary_df.iloc[0]["category"].title())
        col3.metric("Categories Tracked", len(summary_df))

        st.subheader("Category Breakdown")
        st.dataframe(summary_df, use_container_width=True)
        st.bar_chart(summary_df.set_index("category")["amount"])

        # Money Leak Detection Rule
        st.subheader("🚨 Potential Money Leaks")
        leaks = summary_df[summary_df["category"].isin(["television", "coffee shops", "fast food", "alcohol & bars"])]
        if not leaks.empty:
            st.warning("We detected high discretionary spending in these categories. Consider reviewing subscriptions or dining habits to optimize savings!")
            st.dataframe(leaks, use_container_width=True)
        else:
            st.success("No major money leaks detected!")
    else:
        st.warning("Please run `python main.py` first to generate your analytics data.")

# --- 2. LIVE CATEGORIZER ---
elif app_mode == "🔍 Live Categorizer":
    st.header("🔍 Automatic Expense Categorization")
    desc = st.text_input("Enter Transaction Description:", "starbucks coffee morning")
    amt = st.number_input("Amount ($):", min_value=0.0, value=6.50)

    if st.button("Categorize"):
        if classifier:
            pred = classifier.predict(pd.DataFrame([{"description": desc, "amount": amt}]))[0]
            st.success(f"Predicted Category: **{pred.upper()}**")
        else:
            st.error("Model not found.")

# --- 3. AI FINANCIAL ASSISTANT (LLM + NLP) ---
elif app_mode == "💬 AI Financial Assistant":
    st.header("💬 AI Financial Assistant & Smart Insights")
    st.markdown("Ask questions about your spending habits, how to save more, or get personalized financial advice based on your data.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_prompt := st.chat_input("Ex: 'Where am I spending the most money and how can I save $500 next month?'"):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            # Prepare context from your summary data
            context_str = summary_df.to_string() if summary_df is not None else "No data available."
            
            # Placeholder for LLM response generation (Configure your API key / client here)
            # Example using OpenAI / Groq client:
            # client = OpenAI(api_key="your-api-key")
            # response = client.chat.completions.create(
            #     model="gpt-4o-mini",
            #     messages=[
            #         {"role": "system", "content": f"You are an expert AI financial advisor. Here is the user's spending summary:\n{context_str}"},
            #         {"role": "user", "content": user_prompt}
            #     ]
            # )
            # bot_reply = response.choices[0].message.content
            
            bot_reply = f"Based on your financial data, your highest spending is in categories like paycheck allocations and major fixed expenses. To save more, focus on trimming discretionary categories shown in your analytics. (Connect your LLM API key in `app.py` to get fully dynamic AI responses!)"
            
            st.markdown(bot_reply)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})

# --- 4. WHAT-IF & GOAL SIMULATOR ---
elif app_mode == "🔮 What-If & Goal Simulator":
    st.header("🔮 Financial What-If Simulator & Reverse Goal Planner")
    
    st.markdown("Simulate how changing your monthly habits impacts your future savings over 1 year.")
    
    reduction_pct = st.slider("Reduce discretionary spending by (%):", 0, 50, 20)
    current_monthly_spend = summary_df["amount"].sum() / 12 if summary_df is not None else 1000
    
    saved_amount = current_monthly_spend * (reduction_pct / 100) * 12
    
    st.info(f"💡 By cutting discretionary spending by **{reduction_pct}%**, you could save approximately **${saved_amount:,.2f}** over the next year!")