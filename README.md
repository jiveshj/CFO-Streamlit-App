Step 1: Create Project Structure 
bash# Create main directory
mkdir cfo-copilot
cd cfo-copilot

# Create subdirectories
mkdir agent fixtures tests

# Create empty __init__.py files
touch agent/__init__.py
touch fixtures/__init__.py
touch tests/__init__.py

Step 2: Copy All Files 
Copy the following files from the artifacts into your project:
Root Directory:

app.py - Main Streamlit application
requirements.txt - Dependencies
README.md - Documentation
convert_data.py - Data conversion utility

agent/ Directory:

agent/planner.py - Intent classification
agent/tools.py - Financial calculations

fixtures/ Directory:

fixtures/actuals.csv - Actual financial data
fixtures/budget.csv - Budget data
fixtures/fx.csv - Exchange rates
fixtures/cash.csv - Cash balances

tests/ Directory:

tests/test_agent.py - Agent tests
tests/test_tools.py - Tools tests

Step 3: Install Dependencies 
bash# Optional: Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
# Install packages
pip install -r requirements.txt

Step 4: Verify Installation 
bash# Run tests to verify everything works
pytest -v
# Should see: All tests passed ✓

Step 5: Launch App 
bashstreamlit run app.py
Browser opens automatically at http://localhost:8501
