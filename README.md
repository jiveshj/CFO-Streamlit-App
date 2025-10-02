# Step 1: Create Project Structure 
bash# Create main directory
mkdir cfo-copilot
cd cfo-copilot

#Create subdirectories
mkdir agent fixtures tests

Alternatively you can Git clone the repository using https on your local machine. 

# Step 2: Copy All Files 
Copy the following files from the artifacts into your project:
Root Directory:

app.py - Main Streamlit application
requirements.txt - Dependencies
README.md - Documentation

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

# Step 3: Install Dependencies 
#Optional: Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
#Install packages
pip install -r requirements.txt

# Step 4: Verify Installation 
#Run tests to verify everything works
pytest -v
#Should see: All tests passed ✓
#you can also run individual pytest to test tools.py and planner.py. For checking tools.py, run pytest teests/test_tools.py -v and for checking planner.py, run pytest tests/test_agent.py

# Step 5: Launch App 
streamlit run app.py
Browser opens automatically at http://localhost:8501
