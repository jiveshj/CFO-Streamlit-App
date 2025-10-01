import pytest
import pandas as pd
import sys
from pathlib import Path

# Add the agent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "agent"))

from agent.tools import FinancialTools

@pytest.fixture
def sample_data():
    """Create sample financial data for testing"""
    # Sample actuals data
    actuals_data = {
        'entity': ['US', 'US', 'UK', 'UK'],
        'account': ['Revenue - Subscription', 'COGS - Hosting', 'Revenue - Subscription', 'COGS - Hosting'],
        'currency': ['USD', 'USD', 'GBP', 'GBP'],
        '2025-01': [1000000, 100000, 500000, 50000],
        '2025-02': [1100000, 110000, 550000, 55000],
        '2025-03': [1200000, 120000, 600000, 60000]
    }
    
    # Sample budget data
    budget_data = {
        'entity': ['US', 'US', 'UK', 'UK'],
        'account': ['Revenue - Subscription', 'COGS - Hosting', 'Revenue - Subscription', 'COGS - Hosting'],
        'currency': ['USD', 'USD', 'GBP', 'GBP'],
        '2025-01': [950000, 95000, 480000, 48000],
        '2025-02': [1050000, 105000, 520000, 52000],
        '2025-03': [1150000, 115000, 580000, 58000]
    }
    
    # Sample FX data
    fx_data = {
        'currency': ['USD', 'GBP'],
        '2025-01': [1.0, 1.25],
        '2025-02': [1.0, 1.26],
        '2025-03': [1.0, 1.27]
    }
    
    # Sample cash data
    cash_data = {
        'entity': ['US', 'UK'],
        'currency': ['USD', 'GBP'],
        '2025-01': [5000000, 1000000],
        '2025-02': [4800000, 950000],
        '2025-03': [4600000, 900000]
    }
    
    return (
        pd.DataFrame(actuals_data),
        pd.DataFrame(budget_data),
        pd.DataFrame(fx_data),
        pd.DataFrame(cash_data)
    )

@pytest.fixture
def financial_tools(sample_data):
    """Create FinancialTools instance with sample data"""
    actuals, budget, fx, cash = sample_data
    return FinancialTools(actuals, budget, fx, cash)

def test_revenue_vs_budget(financial_tools):
    """Test revenue vs budget calculation"""
    result = financial_tools.get_revenue_vs_budget('2025-01', '2025-01')
    
    assert not result.empty
    assert 'actual_usd' in result.columns
    assert 'budget_usd' in result.columns
    assert 'variance_usd' in result.columns
    
    # Check that USD conversion works
    row = result.iloc[0]
    assert row['actual_usd'] > 0
    assert row['budget_usd'] > 0

def test_gross_margin_calculation(financial_tools):
    """Test gross margin calculation"""
    result = financial_tools.get_gross_margin_trend('2025-01', '2025-02')
    
    assert not result.empty
    assert 'gross_margin_pct' in result.columns
    assert 'revenue_usd' in result.columns
    assert 'cogs_usd' in result.columns
    
    # Check margin is calculated correctly
    for _, row in result.iterrows():
        expected_margin = ((row['revenue_usd'] - row['cogs_usd']) / row['revenue_usd']) * 100
        assert abs(row['gross_margin_pct'] - expected_margin) < 0.01

def test_cash_runway_calculation(financial_tools):
    """Test cash runway calculation"""
    runway = financial_tools.get_cash_runway()
    
    # Should return a positive number (months)
    if runway is not None:
        assert runway > 0
        assert isinstance(runway, float)

def test_current_cash_balance(financial_tools):
    """Test current cash balance retrieval"""
    balance = financial_tools.get_current_cash_balance()
    
    if balance is not None:
        assert balance > 0
        assert isinstance(balance, (int, float))

def test_expense_categorization(financial_tools):
    """Test expense categorization logic"""
    # Test different account types
    assert financial_tools._categorize_expense('R&D Salaries') == 'R&D'
    assert financial_tools._categorize_expense('Sales Team Wages') == 'Sales & Marketing'
    assert financial_tools._categorize_expense('Office Rent') == 'Facilities'
    assert financial_tools._categorize_expense('Legal Fees') == 'General & Admin'
    assert financial_tools._categorize_expense('Random Expense') == 'Other'

def test_currency_conversion(financial_tools):
    """Test currency conversion functionality"""
    # Create a small DataFrame to test conversion
    test_df = pd.DataFrame({
        'entity': ['UK'],
        'account': ['Test Account'],
        'currency': ['GBP'],
        '2025-01': [1000]
    })
    
    converted = financial_tools._convert_to_usd(test_df, '2025-01')
    
    # Should have USD column
    assert '2025-01_usd' in converted.columns
    # Should be converted (GBP rate is 1.25 in sample data)
    assert converted.iloc[0]['2025-01_usd'] == 1000 * 1.25

def test_ebitda_calculation(financial_tools):
    """Test EBITDA calculation"""
    ebitda_data = financial_tools.get_ebitda('2025-01')
    
    if ebitda_data is not None:
        assert 'revenue' in ebitda_data
        assert 'cogs' in ebitda_data
        assert 'opex' in ebitda_data
        assert 'ebitda' in ebitda_data
        
        # EBITDA should equal revenue - cogs - opex
        expected_ebitda = ebitda_data['revenue'] - ebitda_data['cogs'] - ebitda_data['opex']
        assert abs(ebitda_data['ebitda'] - expected_ebitda) < 0.01

if __name__ == "__main__":
    pytest.main([__file__])
