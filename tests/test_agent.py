import pytest
import pandas as pd
import sys
from pathlib import Path

# Add the agent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "agent"))

from agent.planner import CFOPlanner
from agent.tools import FinancialTools

@pytest.fixture
def sample_data():
    """Create sample financial data for testing"""
    # Sample actuals data - using your format
    actuals_data = {
        'month': ['2025-01', '2025-01', '2025-01', '2025-01', '2025-01', '2025-01',
                  '2025-02', '2025-02', '2025-02', '2025-02', '2025-02', '2025-02',
                  '2025-03', '2025-03', '2025-03', '2025-03', '2025-03', '2025-03',
                  '2025-04', '2025-04', '2025-04', '2025-04', '2025-04', '2025-04',
                  '2025-05', '2025-05', '2025-05', '2025-05', '2025-05', '2025-05',
                  '2025-06', '2025-06', '2025-06', '2025-06', '2025-06', '2025-06'],
        'entity': ['ParentCo', 'ParentCo', 'ParentCo', 'ParentCo', 'ParentCo', 'ParentCo',
                   'ParentCo', 'ParentCo', 'ParentCo', 'ParentCo', 'ParentCo', 'ParentCo',
                   'ParentCo', 'ParentCo', 'ParentCo', 'ParentCo', 'ParentCo', 'ParentCo',
                   'ParentCo', 'ParentCo', 'ParentCo', 'ParentCo', 'ParentCo', 'ParentCo',
                   'ParentCo', 'ParentCo', 'ParentCo', 'ParentCo', 'ParentCo', 'ParentCo',
                   'ParentCo', 'ParentCo', 'ParentCo', 'ParentCo', 'ParentCo', 'ParentCo'],
        'account_category': ['Revenue', 'COGS', 'Opex:Marketing', 'Opex:Sales', 'Opex:R&D', 'Opex:Admin',
                            'Revenue', 'COGS', 'Opex:Marketing', 'Opex:Sales', 'Opex:R&D', 'Opex:Admin',
                            'Revenue', 'COGS', 'Opex:Marketing', 'Opex:Sales', 'Opex:R&D', 'Opex:Admin',
                            'Revenue', 'COGS', 'Opex:Marketing', 'Opex:Sales', 'Opex:R&D', 'Opex:Admin',
                            'Revenue', 'COGS', 'Opex:Marketing', 'Opex:Sales', 'Opex:R&D', 'Opex:Admin',
                            'Revenue', 'COGS', 'Opex:Marketing', 'Opex:Sales', 'Opex:R&D', 'Opex:Admin'],
        'amount': [695000, 104250, 139000, 83400, 55600, 41700,
                   725000, 108750, 145000, 87000, 58000, 43500,
                   745000, 111750, 149000, 89400, 59600, 44700,
                   730000, 109500, 146000, 87600, 58400, 43800,
                   765000, 114750, 153000, 91800, 61200, 45900,
                   780000, 117000, 156000, 93600, 62400, 46800],
        'currency': ['USD'] * 36
    }
    
    # Sample budget data
    budget_data = {
        'month': ['2025-01', '2025-01', '2025-01', '2025-01', '2025-01', '2025-01',
                  '2025-02', '2025-02', '2025-02', '2025-02', '2025-02', '2025-02',
                  '2025-03', '2025-03', '2025-03', '2025-03', '2025-03', '2025-03',
                  '2025-04', '2025-04', '2025-04', '2025-04', '2025-04', '2025-04',
                  '2025-05', '2025-05', '2025-05', '2025-05', '2025-05', '2025-05',
                  '2025-06', '2025-06', '2025-06', '2025-06', '2025-06', '2025-06'],
        'entity': ['ParentCo'] * 36,
        'account_category': ['Revenue', 'COGS', 'Opex:Marketing', 'Opex:Sales', 'Opex:R&D', 'Opex:Admin'] * 6,
        'amount': [710000, 99400, 127800, 78100, 56800, 42600,
                   740000, 103600, 133200, 81400, 59200, 44400,
                   760000, 106400, 136800, 83600, 60800, 45600,
                   750000, 108000, 142000, 86000, 59500, 44500,
                   780000, 112000, 148000, 89000, 62000, 46500,
                   800000, 112000, 144000, 88000, 64000, 48000],
        'currency': ['USD'] * 36
    }
    
    # Sample FX data
    fx_data = {
        'month': ['2025-01', '2025-02', '2025-03', '2025-04', '2025-05', '2025-06'],
        'currency': ['USD', 'USD', 'USD', 'USD', 'USD', 'USD'],
        'rate_to_usd': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    }
    
    # Sample cash data
    cash_data = {
        'month': ['2025-01', '2025-02', '2025-03', '2025-04', '2025-05', '2025-06'],
        'entity': ['Consolidated'] * 6,
        'cash_usd': [5000000, 4800000, 4600000, 4400000, 4200000, 4000000]
    }
    
    return (
        pd.DataFrame(actuals_data),
        pd.DataFrame(budget_data),
        pd.DataFrame(fx_data),
        pd.DataFrame(cash_data)
    )

@pytest.fixture
def cfo_planner(sample_data):
    """Create CFOPlanner instance with sample data"""
    actuals, budget, fx, cash = sample_data
    tools = FinancialTools(actuals, budget, fx, cash)
    return CFOPlanner(tools)

def test_intent_classification_revenue_vs_budget(cfo_planner):
    """Test intent classification for revenue vs budget queries"""
    queries = [
        "What was June 2025 revenue vs budget?",
        "Show me actual revenue compared to budget for June",
        "How did our revenue perform against budget in June?",
        "Revenue vs budget for June"
    ]
    
    for query in queries:
        intent, params = cfo_planner.classify_intent(query)
        assert intent == 'revenue_vs_budget', f"Failed for query: {query}"
        assert '2025-06' in params.get('specific_month', ''), f"Failed to extract June 2025 from: {query}"

def test_intent_classification_gross_margin(cfo_planner):
    """Test intent classification for gross margin queries"""
    queries = [
        "Show gross margin trend for the last 3 months",
        "What's our margin percentage?",
        "How is our profitability looking?",
        "Gross margin for June"
    ]
    
    for query in queries:
        intent, params = cfo_planner.classify_intent(query)
        assert intent == 'gross_margin', f"Failed for query: {query}"

def test_intent_classification_cash_runway(cfo_planner):
    """Test intent classification for cash runway queries"""
    queries = [
        "What is our cash runway?",
        "How long will our cash last?",
        "What's our runway looking like?",
        "Cash runway analysis"
    ]
    
    for query in queries:
        intent, params = cfo_planner.classify_intent(query)
        assert intent == 'cash_runway', f"Failed for query: {query}"

def test_intent_classification_opex_breakdown(cfo_planner):
    """Test intent classification for opex breakdown queries"""
    queries = [
        "Break down opex by category for June",
        "Show me operating expenses breakdown",
        "OpEx breakdown"
    ]
    
    for query in queries:
        intent, params = cfo_planner.classify_intent(query)
        assert intent == 'opex_breakdown', f"Failed for query: {query}"

def test_intent_classification_revenue_trend(cfo_planner):
    """Test intent classification for revenue trend queries"""
    queries = [
        "Show revenue trend for the last 3 months",
        "Revenue over time",
        "Show me revenue trends"
    ]
    
    for query in queries:
        intent, params = cfo_planner.classify_intent(query)
        assert intent == 'revenue_trend', f"Failed for query: {query}"

def test_time_parameter_extraction_specific_month(cfo_planner):
    """Test extraction of specific month from queries"""
    test_cases = [
        ("What was June 2025 revenue?", '2025-06'),
        ("Show me January performance", '2025-01'),
        ("March 2025 numbers", '2025-03'),
        ("Feb revenue", '2025-02')
    ]
    
    for query, expected_month in test_cases:
        intent, params = cfo_planner.classify_intent(query)
        assert params.get('specific_month') == expected_month, \
            f"Failed to extract {expected_month} from: {query}"

def test_time_parameter_extraction_relative_period(cfo_planner):
    """Test extraction of relative time periods"""
    test_cases = [
        ("Show revenue trend for last 3 months", 'last_3_months'),
        ("Cash trend over the last 6 months", 'last_6_months'),
        ("Last month's performance", 'last_month')
    ]
    
    for query, expected_period in test_cases:
        intent, params = cfo_planner.classify_intent(query)
        assert params.get('period') == expected_period, \
            f"Failed to extract {expected_period} from: {query}"

def test_process_query_revenue_vs_budget(cfo_planner):
    """Test processing of revenue vs budget query"""
    query = "What was June 2025 revenue vs budget?"
    response = cfo_planner.process_query(query)
    
    assert isinstance(response, str)
    assert len(response) > 0
    assert "revenue" in response.lower() or "Revenue" in response
    assert "June" in response or "2025-06" in response

def test_process_query_gross_margin(cfo_planner):
    """Test processing of gross margin query"""
    query = "What's our gross margin for June?"
    response = cfo_planner.process_query(query)
    
    assert isinstance(response, str)
    assert len(response) > 0
    assert "margin" in response.lower()

def test_process_query_cash_runway(cfo_planner):
    """Test processing of cash runway query"""
    query = "What is our cash runway?"
    response = cfo_planner.process_query(query)
    
    assert isinstance(response, str)
    assert len(response) > 0
    assert "runway" in response.lower() or "cash" in response.lower()

def test_process_query_opex_breakdown(cfo_planner):
    """Test processing of opex breakdown query"""
    query = "Break down opex by category for June"
    response = cfo_planner.process_query(query)
    
    assert isinstance(response, str)
    assert len(response) > 0
    assert ("opex" in response.lower() or "expense" in response.lower() or 
            "operating" in response.lower() or "Marketing" in response)

def test_process_query_revenue_trend(cfo_planner):
    """Test processing of revenue trend query"""
    query = "Show revenue trend for the last 3 months"
    response = cfo_planner.process_query(query)
    
    assert isinstance(response, str)
    assert len(response) > 0
    assert "revenue" in response.lower() or "Revenue" in response

def test_process_query_general(cfo_planner):
    """Test processing of general/unrecognized query"""
    query = "Hello, what can you help me with?"
    response = cfo_planner.process_query(query)
    
    assert isinstance(response, str)
    assert len(response) > 0
    # Should provide helpful guidance
    assert ("help" in response.lower() or "ask" in response.lower() or 
            "revenue" in response.lower())

def test_error_handling_empty_query(cfo_planner):
    """Test error handling with empty query"""
    response = cfo_planner.process_query("")
    assert isinstance(response, str)
    assert len(response) > 0

def test_error_handling_nonsense_query(cfo_planner):
    """Test error handling with nonsensical query"""
    response = cfo_planner.process_query("asdfghjkl qwerty")
    assert isinstance(response, str)
    assert len(response) > 0

def test_multiple_queries_session(cfo_planner):
    """Test processing multiple queries in a session"""
    queries = [
        "What was June 2025 revenue vs budget?",
        "Show gross margin for June",
        "What is our cash runway?"
    ]
    
    for query in queries:
        response = cfo_planner.process_query(query)
        assert isinstance(response, str)
        assert len(response) > 0

def test_case_insensitivity(cfo_planner):
    """Test that intent classification is case-insensitive"""
    queries = [
        "what was june 2025 revenue vs budget?",
        "WHAT WAS JUNE 2025 REVENUE VS BUDGET?",
        "What Was June 2025 Revenue Vs Budget?"
    ]
    
    intents = []
    for query in queries:
        intent, params = cfo_planner.classify_intent(query)
        intents.append(intent)
    
    # All should produce the same intent
    assert len(set(intents)) == 1
    assert intents[0] == 'revenue_vs_budget'

def test_response_formatting(cfo_planner):
    """Test that responses are properly formatted"""
    query = "What was June 2025 revenue vs budget?"
    response = cfo_planner.process_query(query)
    
    # Response should be multi-line and well-formatted
    assert '\n' in response or len(response) > 50
    # Should contain some formatting characters or structure
    assert '•' in response or '-' in response or ':' in response or 'M' in response

if __name__ == "__main__":
    pytest.main([__file__, '-v'])