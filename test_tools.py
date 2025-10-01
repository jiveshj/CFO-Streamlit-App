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
    # Sample actuals data - new format
    actuals_data = {
        'month': ['2025-01', '2025-01', '2025-01', '2025-01', 
                  '2025-02', '2025-02', '2025-02', '2025-02'],
        'entity': ['ParentCo', 'ParentCo', 'EMEA', 'EMEA',
                   'ParentCo', 'ParentCo', 'EMEA', 'EMEA'],
        'account_category': ['Revenue', 'COGS', 'Revenue', 'COGS',
                            'Revenue', 'COGS', 'Revenue', 'COGS'],
        'amount': [1000000, 150000, 500000, 75000,
                   1100000, 165000, 550000, 82500],
        'currency': ['USD', 'USD', 'EUR', 'EUR',
                    'USD', 'USD', 'EUR', 'EUR']
    }
    
    # Sample budget data
    budget_data = {
        'month': ['2025-01', '2025-01', '2025-01', '2025-01',
                  '2025-02', '2025-02', '2025-02', '2025-02'],
        'entity': ['ParentCo', 'ParentCo', 'EMEA', 'EMEA',
                   'ParentCo', 'ParentCo', 'EMEA', 'EMEA'],
        'account_category': ['Revenue', 'COGS', 'Revenue', 'COGS',
                            'Revenue', 'COGS', 'Revenue', 'COGS'],
        'amount': [950000, 142500, 480000, 72000,
                   1050000, 157500, 520000, 78000],
        'currency': ['USD', 'USD', 'EUR', 'EUR',
                    'USD', 'USD', 'EUR', 'EUR']
    }
    
    # Sample FX data
    fx_data = {
        'month': ['2025-01', '2025-01', '2025-02', '2025-02'],
        'currency': ['USD', 'EUR', 'USD', 'EUR'],
        'rate_to_usd': [1.0, 1.08, 1.0, 1.09]
    }
    
    # Sample cash data
    cash_data = {
        'month': ['2025-01', '2025-02', '2025-03'],
        'entity': ['Consolidated', 'Consolidated', 'Consolidated'],
        'cash_usd': [5000000, 4800000, 4600000]
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
    
    # ParentCo: 1M USD + EMEA: 500K EUR * 1.08 = 1M + 540K = 1.54M
    assert row['actual_usd'] > 1500000

def test_gross_margin_calculation(financial_tools):
    """Test gross margin calculation"""
    result = financial_tools.get_gross_margin_trend('2025-01', '2025-02')
    
    assert not result.empty
    assert 'gross_margin_pct' in result.columns
    assert 'revenue_usd' in result.columns
    assert 'cogs_usd' in result.columns
    
    # Check margin is between 0 and 100
    for _, row in result.iterrows():
        assert 0 <= row['gross_margin_pct'] <= 100

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
        # Should be the latest month (2025-03 = 4.6M)
        assert balance == 4600000

def test_currency_conversion(financial_tools):
    """Test currency conversion functionality"""
    # Create a small DataFrame to test conversion
    test_df = pd.DataFrame({
        'month': ['2025-01'],
        'entity': ['EMEA'],
        'account_category': ['Revenue'],
        'amount': [1000],
        'currency': ['EUR']
    })
    
    converted = financial_tools._convert_to_usd(test_df)
    
    # Should have amount_usd column
    assert 'amount_usd' in converted.columns
    # Should be converted (EUR rate is 1.08 in sample data)
    assert converted.iloc[0]['amount_usd'] == 1000 * 1.08

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

def test_opex_breakdown(financial_tools):
    """Test OpEx breakdown with real Opex data"""
    # Add some Opex data to test
    opex_data = pd.DataFrame({
        'month': ['2025-01', '2025-01', '2025-01'],
        'entity': ['ParentCo', 'ParentCo', 'ParentCo'],
        'account_category': ['Opex:Marketing', 'Opex:Sales', 'Opex:R&D'],
        'amount': [100000, 80000, 120000],
        'currency': ['USD', 'USD', 'USD']
    })
    
    # Add to actuals
    financial_tools.actuals = pd.concat([financial_tools.actuals, opex_data], ignore_index=True)
    
    result = financial_tools.get_opex_breakdown('2025-01')
    
    assert not result.empty
    assert 'category' in result.columns
    assert 'amount_usd' in result.columns
    
    # Should have 3 categories
    assert len(result) == 3

def test_revenue_trend(financial_tools):
    """Test revenue trend over multiple months"""
    result = financial_tools.get_revenue_trend('2025-01', '2025-02')
    
    assert not result.empty
    assert len(result) == 2
    assert 'month' in result.columns
    assert 'revenue_usd' in result.columns
    
    # Revenue should be positive
    for _, row in result.iterrows():
        assert row['revenue_usd'] > 0

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
