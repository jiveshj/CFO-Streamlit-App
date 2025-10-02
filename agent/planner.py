import re
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
import plotly.express as px

class CFOPlanner:
    """
    Main planning agent that interprets CFO queries and routes to appropriate tools
    """
    
    def __init__(self, financial_tools):
        self.tools = financial_tools
        self.intent_patterns = self._build_intent_patterns()
    
    def _build_intent_patterns(self) -> Dict[str, List[str]]:
        """Build regex patterns for intent classification"""
        return {
            'revenue_vs_budget': [
                r'revenue.*vs.*budget',
                r'revenue.*compared.*budget',
                r'actual.*revenue.*budget',
                r'budget.*vs.*revenue',
                r'revenue.*against.*budget'
            ],
            'revenue_trend': [
                r'revenue.*trend',
                r'revenue.*over.*time',
                r'revenue.*last.*months?',
                r'show.*revenue'
            ],
            'gross_margin': [
                r'gross.*margin',
                r'margin.*trend',
                r'margin.*percent',
                r'profitability'
            ],
            'opex_breakdown': [
                r'opex.*breakdown',
                r'operating.*expense',
                r'break.*down.*expense',
                r'expense.*categor',
                r'spending.*breakdown',
                r'break.*down.*opex',
                r'spending.*category'
            ],
            'cash_runway': [
                r'cash.*runway',
                r'cash.*burn',
                r'how.*long.*cash',
                r'runway'
            ],
            'cash_trend': [
                r'cash.*trend',
                r'cash.*over.*time',
                r'cash.*balance'
            ],
            'ebitda': [
                r'ebitda',
                r'operating.*profit',
                r'earnings'
            ]
        }
    
    def _get_latest_month(self) -> str:
        """Get the latest month from available data"""
        if self.tools.month_columns:
            return max(self.tools.month_columns)
        return '2025-12'  # Fallback
    
    def _calculate_month_range(self, months_back: int, end_month: str = None) -> Tuple[str, str]:
        """
        Calculate start and end months for a relative period
        
        Args:
            months_back: Number of months to go back
            end_month: End month (defaults to latest)
            
        Returns:
            Tuple of (start_month, end_month)
        """
        if end_month is None:
            end_month = self._get_latest_month()
        
        end_dt = datetime.strptime(end_month, '%Y-%m')
        start_dt = end_dt - relativedelta(months=months_back - 1)
        start_month = start_dt.strftime('%Y-%m')
        
        return start_month, end_month
    
    def classify_intent(self, query: str) -> Tuple[str, Dict]:
        """
        Classify user intent and extract parameters
        
        Args:
            query: User's natural language query
            
        Returns:
            Tuple of (intent, parameters)
        """
        query_lower = query.lower()
        
        # Extract month/time parameters
        params = self._extract_time_params(query_lower)
        
        # Check each intent pattern
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent, params
        
        return 'general', params
    
    def _extract_time_params(self, query: str) -> Dict:
        """Extract time-related parameters from query"""
        params = {}
        
        # Extract specific months
        months = {
            'january': '01', 'jan': '01',
            'february': '02', 'feb': '02', 
            'march': '03', 'mar': '03',
            'april': '04', 'apr': '04',
            'may': '05',
            'june': '06', 'jun': '06',
            'july': '07', 'jul': '07',
            'august': '08', 'aug': '08',
            'september': '09', 'sep': '09',
            'october': '10', 'oct': '10',
            'november': '11', 'nov': '11',
            'december': '12', 'dec': '12'
        }
        
        for month_name, month_num in months.items():
            if month_name in query:
                # Assume 2025 if no year specified
                year = '2025'
                if re.search(r'202[3-6]', query):
                    year = re.search(r'202[3-6]', query).group()
                params['specific_month'] = f"{year}-{month_num}"
                break
        
        # Extract relative time periods
        if re.search(r'last.*3.*months?', query):
            params['period'] = 'last_3_months'
        elif re.search(r'last.*6.*months?', query):
            params['period'] = 'last_6_months'
        elif re.search(r'last.*month', query):
            params['period'] = 'last_month'
        elif re.search(r'this.*month', query):
            params['period'] = 'current_month'
        elif re.search(r'ytd|year.*to.*date', query):
            params['period'] = 'ytd'
        elif re.search(r'q[1-4]|quarter', query):
            quarter_match = re.search(r'q([1-4])', query)
            if quarter_match:
                params['period'] = f'q{quarter_match.group(1)}'
        
        # Default to latest available month if no time specified
        if 'specific_month' not in params and 'period' not in params:
            params['specific_month'] = self._get_latest_month()
        
        return params
    
    def process_query(self, query: str) -> str:
        """
        Process a user query and return formatted response with chart data
        
        Args:
            query: User's natural language query
            
        Returns:
            Formatted response string
        """
        try:
            intent, params = self.classify_intent(query)
            
            # Route to appropriate handler
            if intent == 'revenue_vs_budget':
                return self._handle_revenue_vs_budget(params)
            elif intent == 'revenue_trend':
                return self._handle_revenue_trend(params)
            elif intent == 'gross_margin':
                return self._handle_gross_margin(params)
            elif intent == 'opex_breakdown':
                return self._handle_opex_breakdown(params)
            elif intent == 'cash_runway':
                return self._handle_cash_runway(params)
            elif intent == 'cash_trend':
                return self._handle_cash_trend(params)
            elif intent == 'ebitda':
                return self._handle_ebitda(params)
            else:
                return self._handle_general_query(query)
                
        except Exception as e:
            return f"I apologize, but I encountered an error processing your query: {str(e)}. Please try rephrasing your question."
    
    def _handle_revenue_vs_budget(self, params: Dict) -> str:
        """Handle revenue vs budget queries"""
        month = params.get('specific_month', self._get_latest_month())
        
        try:
            data = self.tools.get_revenue_vs_budget(month, month)
            
            if data.empty:
                return f"No revenue data found for {month}."
            
            row = data.iloc[0]
            actual = row['actual_usd'] / 1_000_000
            budget = row['budget_usd'] / 1_000_000
            variance = ((actual - budget) / budget * 100) if budget > 0 else 0
            
            month_name = datetime.strptime(month, '%Y-%m').strftime('%B %Y')
            
            response = f"**Revenue Performance for {month_name}:**\n\n"
            response += f"• **Actual Revenue:** ${actual:.1f}M\n"
            response += f"• **Budgeted Revenue:** ${budget:.1f}M\n"
            response += f"• **Variance:** {variance:+.1f}% "
            
            if variance > 0:
                response += "🟢 (Above budget)"
            else:
                response += "🔴 (Below budget)"
            
            return response
            
        except Exception as e:
            return f"Error retrieving revenue data: {str(e)}"
    
    def _handle_revenue_trend(self, params: Dict) -> str:
        """Handle revenue trend queries"""
        try:
            latest_month = self._get_latest_month()
            
            if params.get('period') == 'last_3_months':
                start_month, end_month = self._calculate_month_range(3, latest_month)
            elif params.get('period') == 'last_6_months':
                start_month, end_month = self._calculate_month_range(6, latest_month)
            else:
                # Default to last 6 months
                start_month, end_month = self._calculate_month_range(6, latest_month)
            
            data = self.tools.get_revenue_trend(start_month, end_month)
            
            if data.empty:
                return "No revenue trend data available."
            
            response = f"**Revenue Trend ({start_month} to {end_month}):**\n\n"
            
            for _, row in data.iterrows():
                month_name = datetime.strptime(row['month'], '%Y-%m').strftime('%b %Y')
                revenue = row['revenue_usd'] / 1_000_000
                response += f"• {month_name}: ${revenue:.1f}M\n"
            
           

            
            # Calculate growth
            if len(data) > 1:
                first_month = data.iloc[0]['revenue_usd']
                last_month = data.iloc[-1]['revenue_usd']
                growth = ((last_month - first_month) / first_month * 100) if first_month > 0 else 0
                response += f"\n**Total Growth:** {growth:+.1f}%"
            # Attach chart data
            response += "\nCHART_DATA:" + data.to_json(orient="records")
            return response
            
        except Exception as e:
            return f"Error retrieving revenue trend: {str(e)}"
    
    def _handle_gross_margin(self, params: Dict) -> str:
        """Handle gross margin queries"""
        try:
            latest_month = self._get_latest_month()
            
            if params.get('period') == 'last_3_months':
                start_month, end_month = self._calculate_month_range(3, latest_month)
            else:
                month = params.get('specific_month', latest_month)
                start_month = end_month = month
            
            data = self.tools.get_gross_margin_trend(start_month, end_month)
            
            if data.empty:
                return "No gross margin data available."
            
            if start_month == end_month:
                # Single month
                row = data.iloc[0]
                month_name = datetime.strptime(row['month'], '%Y-%m').strftime('%B %Y')
                margin = row['gross_margin_pct']
                
                response = f"**Gross Margin for {month_name}:** {margin:.1f}%\n\n"
                response += f"• **Revenue:** ${row['revenue_usd']/1_000_000:.1f}M\n"
                response += f"• **COGS:** ${row['cogs_usd']/1_000_000:.1f}M"
            else:
                # Trend
                response = f"**Gross Margin Trend ({start_month} to {end_month}):**\n\n"
                
                for _, row in data.iterrows():
                    month_name = datetime.strptime(row['month'], '%Y-%m').strftime('%b %Y')
                    margin = row['gross_margin_pct']
                    response += f"• {month_name}: {margin:.1f}%\n"
                
                # Average margin
                avg_margin = data['gross_margin_pct'].mean()
                response += f"\n**Average Margin:** {avg_margin:.1f}%"
            
            return response
            
        except Exception as e:
            return f"Error retrieving gross margin data: {str(e)}"
    
    def _handle_opex_breakdown(self, params: Dict) -> str:
        """Handle operating expense breakdown queries"""
        month = params.get('specific_month', self._get_latest_month())
        
        try:
            data = self.tools.get_opex_breakdown(month)
            
            if data.empty:
                return f"No operating expense data found for {month}."
            
            month_name = datetime.strptime(month, '%Y-%m').strftime('%B %Y')
            total_opex = data['amount_usd'].sum() / 1_000_000
            
            response = f"**Operating Expenses for {month_name}:**\n\n"
            response += f"**Total OpEx:** ${total_opex:.1f}M\n\n"
            response += "**Breakdown by Category:**\n"
            
            # Sort by amount descending
            data_sorted = data.sort_values('amount_usd', ascending=False)
            
            for _, row in data_sorted.iterrows():
                category = row['category']
                amount = row['amount_usd'] / 1_000_000
                percentage = (row['amount_usd'] / data['amount_usd'].sum()) * 100
                response += f"• **{category}:** ${amount:.1f}M ({percentage:.1f}%)\n"
            
            return response
            
        except Exception as e:
            return f"Error retrieving OpEx breakdown: {str(e)}"
    
    def _handle_cash_runway(self, params: Dict) -> str:
        """Handle cash runway queries"""
        try:
            runway_months = self.tools.get_cash_runway()
            current_cash = self.tools.get_current_cash_balance()
            burn_rate = self.tools.get_average_burn_rate()
            
            response = "**Cash Runway Analysis:**\n\n"
            
            if runway_months:
                response += f"• **Current Runway:** {runway_months:.1f} months\n"
                
                if runway_months < 6:
                    response += "• **Status:** 🔴 Critical - Consider fundraising\n"
                elif runway_months < 12:
                    response += "• **Status:** 🟡 Caution - Monitor closely\n"
                else:
                    response += "• **Status:** 🟢 Healthy\n"
            
            if current_cash:
                response += f"• **Current Cash:** ${current_cash/1_000_000:.1f}M\n"
            
            if burn_rate:
                response += f"• **Avg Monthly Burn:** ${abs(burn_rate)/1_000_000:.1f}M\n"
            
            return response
            
        except Exception as e:
            return f"Error calculating cash runway: {str(e)}"
    
    def _handle_cash_trend(self, params: Dict) -> str:
        """Handle cash trend queries"""
        try:
            latest_month = self._get_latest_month()
            
            if params.get('period') == 'last_6_months':
                start_month, end_month = self._calculate_month_range(6, latest_month)
            else:
                # Default to last 6 months
                start_month, end_month = self._calculate_month_range(6, latest_month)
            
            data = self.tools.get_cash_trend(start_month, end_month)
            
            if data.empty:
                return "No cash trend data available."
            
            response = f"**Cash Balance Trend ({start_month} to {end_month}):**\n\n"
            
            for _, row in data.iterrows():
                month_name = datetime.strptime(row['month'], '%Y-%m').strftime('%b %Y')
                balance = row['cash_balance_usd'] / 1_000_000
                response += f"• {month_name}: ${balance:.1f}M\n"
            
            # Calculate net change
            if len(data) > 1:
                first_balance = data.iloc[0]['cash_balance_usd']
                last_balance = data.iloc[-1]['cash_balance_usd']
                net_change = (last_balance - first_balance) / 1_000_000
                response += f"\n**Net Change:** ${net_change:+.1f}M"
            
            return response
            
        except Exception as e:
            return f"Error retrieving cash trend: {str(e)}"
    
    def _handle_ebitda(self, params: Dict) -> str:
        """Handle EBITDA queries"""
        month = params.get('specific_month', self._get_latest_month())
        
        try:
            ebitda_data = self.tools.get_ebitda(month)
            
            if not ebitda_data:
                return f"No EBITDA data available for {month}."
            
            month_name = datetime.strptime(month, '%Y-%m').strftime('%B %Y')
            
            response = f"**EBITDA for {month_name}:**\n\n"
            response += f"• **Revenue:** ${ebitda_data['revenue']/1_000_000:.1f}M\n"
            response += f"• **COGS:** ${ebitda_data['cogs']/1_000_000:.1f}M\n"
            response += f"• **OpEx:** ${ebitda_data['opex']/1_000_000:.1f}M\n"
            response += f"• **EBITDA:** ${ebitda_data['ebitda']/1_000_000:.1f}M\n\n"
            
            if ebitda_data['revenue'] > 0:
                ebitda_margin = (ebitda_data['ebitda'] / ebitda_data['revenue']) * 100
                response += f"**EBITDA Margin:** {ebitda_margin:.1f}%"
            
            return response
            
        except Exception as e:
            return f"Error calculating EBITDA: {str(e)}"
    
    def _handle_general_query(self, query: str) -> str:
        """Handle general queries that don't match specific intents"""
        response = "I can help you with various financial analyses. Here are some things you can ask:\n\n"
        response += "📊 **Revenue Analysis:**\n"
        response += "• 'What was June 2025 revenue vs budget?'\n"
        response += "• 'Show revenue trend for last 3 months'\n\n"
        response += "💰 **Profitability:**\n"
        response += "• 'What's our gross margin for June?'\n"
        response += "• 'Show EBITDA for this month'\n\n"
        response += "💸 **Expenses:**\n"
        response += "• 'Break down OpEx by category'\n"
        response += "• 'How much did we spend on R&D?'\n\n"
        response += "🏦 **Cash Management:**\n"
        response += "• 'What is our cash runway?'\n"
        response += "• 'Show cash trend over 6 months'\n\n"
        response += "Try asking one of these questions!"
        
        return response
