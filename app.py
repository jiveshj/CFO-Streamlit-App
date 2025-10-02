import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys
from pathlib import Path
import json

# Add the agent directory to the Python path
sys.path.append(str(Path(__file__).parent / "agent"))

from agent.planner import CFOPlanner
from agent.tools import FinancialTools

# Page configuration
st.set_page_config(
    page_title="CFO Copilot",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    .response-container {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_financial_data():
    """Load and cache financial data"""
    try:
        fixtures_path = Path(__file__).parent / "fixtures"
        
        # Load CSV files
        actuals = pd.read_csv(fixtures_path / "actuals.csv")
        budget = pd.read_csv(fixtures_path / "budget.csv") 
        fx = pd.read_csv(fixtures_path / "fx.csv")
        cash = pd.read_csv(fixtures_path / "cash.csv")
        
        return actuals, budget, fx, cash
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "planner" not in st.session_state:
        # Load data
        actuals, budget, fx, cash = load_financial_data()
        if actuals is not None:
            tools = FinancialTools(actuals, budget, fx, cash)
            st.session_state.planner = CFOPlanner(tools)
            st.session_state.tools = tools

def display_key_metrics():
    """Display key financial metrics in the sidebar"""
    if "tools" in st.session_state:
        tools = st.session_state.tools
        
        # Get latest month data from available months
        try:
            latest_month = max(tools.month_columns)
        except:
            latest_month = "2025-06"
        
        try:
            # Revenue
            revenue_actual = tools.get_revenue_vs_budget(latest_month, latest_month)
            revenue_val = revenue_actual['actual_usd'].iloc[0] if not revenue_actual.empty else 0
            
            # Gross Margin
            margin_data = tools.get_gross_margin_trend(latest_month, latest_month)
            margin_val = margin_data['gross_margin_pct'].iloc[0] if not margin_data.empty else 0
            
            # Cash Runway
            runway = tools.get_cash_runway()
            
            month_display = datetime.strptime(latest_month, '%Y-%m').strftime('%B %Y')
            st.sidebar.markdown(f"### Key Metrics ({month_display})")
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("Revenue", f"${revenue_val/1000000:.1f}M")
                st.metric("Gross Margin", f"{margin_val:.1f}%")
            with col2:
                st.metric("Cash Runway", f"{runway:.1f} months" if runway else "N/A")
                
        except Exception as e:
            st.sidebar.error(f"Error calculating metrics: {str(e)}")

def display_chat_interface():
    """Display the main chat interface"""
    st.markdown('<h1 class="main-header"> CFO Copilot</h1>', unsafe_allow_html=True)
    st.markdown("### Ask me anything about your financial performance!")
    
    # Sample questions
    st.markdown("**Try asking:**")
    sample_questions = [
        "What was June 2025 revenue vs budget in USD?",
        "Show Gross Margin % trend for the last 3 months",
        "Break down Opex by category for June",
        "What is our cash runway right now?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(sample_questions):
        with cols[i % 2]:
            if st.button(question, key=f"sample_{i}", help="Click to use this question"):
                st.session_state.messages.append({"role": "user", "content": question})
                handle_query(question)

def handle_query(query: str):
    """Handle user query and generate response"""
    if "planner" not in st.session_state:
        st.error("Financial data not loaded. Please refresh the page.")
        return
    
    with st.spinner("Analyzing your query..."):
        try:
            # Get response from planner
            response = st.session_state.planner.process_query(query)
            
            # Add assistant response to messages
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

def display_conversation():
    """Display the conversation history"""
    if st.session_state.messages:
        st.markdown("### 💬 Conversation")
        
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                with st.container():
                    st.markdown('<div class="response-container">', unsafe_allow_html=True)
                    
                    # Parse response for charts
                    content = message["content"]
                    if "CHART_DATA:" in content:
                        parts = content.split("CHART_DATA:")
                        text_part = parts[0]
                        chart_json = parts[1] if len(parts) > 1 else ""
                        try:
                            chart_data = pd.DataFrame(json.loads(chart_json))
                            # Example: line chart of revenue
                            if "revenue_usd" in chart_data.columns:
                                fig = px.line(chart_data, x="month", y="revenue_usd",
                                              title="Revenue Trend", markers=True)
                                fig.update_layout(yaxis_title="Revenue (USD)")
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error rendering chart: {e}")
                    else:
                        st.markdown(f"**CFO Copilot:** {content}")
        
                    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application function"""
    initialize_session_state()
    
    # Sidebar
    st.sidebar.title(" Dashboard")
    display_key_metrics()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("###  Tools")
    if st.sidebar.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
    
    if st.sidebar.button("Export PDF Report"):
        st.sidebar.info("PDF export functionality would be implemented here")
    
    # Main content
    display_chat_interface()
    
    # Chat input
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Use columns for better layout
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_query = st.text_input(
            "Ask a question about your financials:",
            placeholder="e.g., What was our revenue last month?",
            key="user_input"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        submit_button = st.button("Ask", type="primary")
    
    if submit_button and user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        handle_query(user_query)
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display conversation
    display_conversation()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "CFO Copilot - AI-Powered Financial Assistant | "
        "Built with Streamlit & OpenAI"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
