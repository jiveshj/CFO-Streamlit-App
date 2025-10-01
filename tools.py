import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class FinancialTools:
    """
    Financial calculation tools for CFO Copilot
    """
    
    def __init__(self, actuals_df: pd.DataFrame, budget_df: pd.DataFrame, 
                 fx_df: pd.DataFrame, cash_df: pd.DataFrame):
        """
        Initialize with financial data
        
        Args:
            actuals_df: Monthly actual financial data
            budget_df: Monthly budget data
            fx_df: Foreign exchange rates
            cash_df: Cash balance data
        """
        self.actuals = actuals_df.copy()
        self.budget = budget_df.copy()
        self.fx = fx_df.copy()
        self.cash = cash_df.copy()
        
        # Get month columns (assuming format like 2025-01, 2025-02, etc.)
        self.month_columns = [col for col in self.actuals.columns if col.startswith('2025-')]
        
        # Preprocess data
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess and clean financial data"""
        # Fill NaN values with 0
        for df in [self.actuals, self.budget, self.cash]:
            df[self.month_columns] = df[self.month_columns].fillna(0)
        
        # Ensure FX rates exist for all currencies
        self.fx = self.fx.fillna(1.0)  # Default to 1.0 USD rate
    
    def _convert_to_usd(self, df: pd.DataFrame, month: str) -> pd.DataFrame:
        """Convert amounts to USD using FX rates"""
        result_df = df.copy()
        
        if month not in self.month_columns:
            return result_df
        
        for idx, row in result_df.iterrows():
            currency = row.get('currency', 'USD')
            
            # Get FX rate for this currency and month
            fx_rate = 1.0  # Default USD rate
            if currency != 'USD':
                fx_row = self.fx[self.fx['currency'] == currency]
                if not fx_row.empty and month in fx_row.columns:
                    fx_rate = fx_row[month].iloc[0]
                    if pd.isna(fx_rate) or fx_rate == 0:
                        fx_rate = 1.0
            
            # Convert amount to USD
            if month in result_df.columns:
                result_df.loc[idx, f'{month}_usd'] = result_df.loc[idx, month] * fx_rate
        
        return result_df
    
    def get_revenue_vs_budget(self, start_month: str, end_month: str) -> pd.DataFrame:
        """
        Get revenue actual vs budget comparison
        
        Args:
            start_month: Start month in YYYY-MM format
            end_month: End month in YYYY-MM format
            
        Returns:
            DataFrame with actual and budget revenue in USD
        """
        # Filter for revenue accounts
        revenue_accounts = self.actuals[self.actuals['account'].str.contains('Revenue|Sales', case=False, na=False)]
        budget_revenue = self.budget[self.budget['account'].str.contains('Revenue|Sales', case=False, na=False)]
        
        results = []
        
        # Get months between start and end
        months = [col for col in self.month_columns if start_month <= col <= end_month]
        
        for month in months:
            if month not in self.month_columns:
                continue
                
            # Calculate actual revenue
            actual_with_usd = self._convert_to_usd(revenue_accounts, month)
            actual_total = actual_with_usd[f'{month}_usd'].sum() if f'{month}_usd' in actual_with_usd.columns else 0
            
            # Calculate budget revenue
            budget_with_usd = self._convert_to_usd(budget_revenue, month)
            budget_total = budget_with_usd[f'{month}_usd'].sum() if f'{month}_usd' in budget_with_usd.columns else 0
            
            results.append({
                'month': month,
                'actual_usd': actual_total,
                'budget_usd': budget_total,
                'variance_usd': actual_total - budget_total,
                'variance_pct': ((actual_total - budget_total) / budget_total * 100) if budget_total > 0 else 0
            })
        
        return pd.DataFrame(results)
    
    def get_revenue_trend(self, start_month: str, end_month: str) -> pd.DataFrame:
        """Get revenue trend over time"""
        revenue_accounts = self.actuals[self.actuals['account'].str.contains('Revenue|Sales', case=False, na=False)]
        
        results = []
        months = [col for col in self.month_columns if start_month <= col <= end_month]
        
        for month in months:
            if month not in self.month_columns:
                continue
                
            actual_with_usd = self._convert_to_usd(revenue_accounts, month)
            revenue_total = actual_with_usd[f'{month}_usd'].sum() if f'{month}_usd' in actual_with_usd.columns else 0
            
            results.append({
                'month': month,
                'revenue_usd': revenue_total
            })
        
        return pd.DataFrame(results)
    
    def get_gross_margin_trend(self, start_month: str, end_month: str) -> pd.DataFrame:
        """Calculate gross margin trend"""
        revenue_accounts = self.actuals[self.actuals['account'].str.contains('Revenue|Sales', case=False, na=False)]
        cogs_accounts = self.actuals[self.actuals['account'].str.contains('COGS|Cost of Goods|Cost of Sales', case=False, na=False)]
        
        results = []
        months = [col for col in self.month_columns if start_month <= col <= end_month]
        
        for month in months:
            if month not in self.month_columns:
                continue
            
            # Calculate revenue
            revenue_with_usd = self._convert_to_usd(revenue_accounts, month)
            revenue_total = revenue_with_usd[f'{month}_usd'].sum() if f'{month}_usd' in revenue_with_usd.columns else 0
            
            # Calculate COGS
            cogs_with_usd = self._convert_to_usd(cogs_accounts, month)
            cogs_total = cogs_with_usd[f'{month}_usd'].sum() if f'{month}_usd' in cogs_with_usd.columns else 0
            
            # Calculate gross margin
            gross_profit = revenue_total - cogs_total
            gross_margin_pct = (gross_profit / revenue_total * 100) if revenue_total > 0 else 0
            
            results.append({
                'month': month,
                'revenue_usd': revenue_total,
                'cogs_usd': cogs_total,
                'gross_profit_usd': gross_profit,
                'gross_margin_pct': gross_margin_pct
            })
        
        return pd.DataFrame(results)
    
    def get_opex_breakdown(self, month: str) -> pd.DataFrame:
        """Get operating expense breakdown by category"""
        # Filter for OpEx accounts (exclude Revenue and COGS)
        opex_accounts = self.actuals[
            ~self.actuals['account'].str.contains('Revenue|Sales|COGS|Cost of Goods|Cost of Sales', case=False, na=False)
        ]
        
        if month not in self.month_columns:
            return pd.DataFrame()
        
        # Convert to USD
        opex_with_usd = self._convert_to_usd(opex_accounts, month)
        
        # Group by account category (simplified categorization)
        results = []
        
        for _, row in opex_with_usd.iterrows():
            account = row['account']
            amount_usd = row.get(f'{month}_usd', 0)
            
            if amount_usd == 0:
                continue
            
            # Categorize expenses (simplified logic)
            category = self._categorize_expense(account)
            
            results.append({
                'account': account,
                'category': category,
                'amount_usd': amount_usd,
                'month': month
            })
        
        df = pd.DataFrame(results)
        
        # Group by category
        if not df.empty:
            grouped = df.groupby('category')['amount_usd'].sum().reset_index()
            grouped['month'] = month
            return grouped
        
        return df
    
    def _categorize_expense(self, account: str) -> str:
        """Categorize expense accounts into broad categories"""
        account_lower = account.lower()
        
        if any(term in account_lower for term in ['r&d', 'research', 'development', 'engineering']):
            return 'R&D'
        elif any(term in account_lower for term in ['sales', 'marketing', 'advertising']):
            return 'Sales & Marketing'
        elif any(term in account_lower for term in ['admin', 'general', 'legal', 'finance', 'hr']):
            return 'General & Admin'
        elif any(term in account_lower for term in ['rent', 'office', 'utilities', 'facilities']):
            return 'Facilities'
        elif any(term in account_lower for term in ['salary', 'wages', 'payroll', 'compensation']):
            return 'Personnel'
        else:
            return 'Other'
    
    def get_cash_runway(self) -> Optional[float]:
        """Calculate cash runway in months"""
        try:
            current_cash = self.get_current_cash_balance()
            avg_burn = self.get_average_burn_rate(months=3)
            
            if current_cash and avg_burn and avg_burn < 0:
                runway_months = current_cash / abs(avg_burn)
                return runway_months
            
            return None
            
        except Exception as e:
            print(f"Error calculating cash runway: {e}")
            return None
    
    def get_current_cash_balance(self) -> Optional[float]:
        """Get current cash balance in USD"""
        try:
            # Get the latest month with data
            latest_month = max(self.month_columns)
            
            cash_with_usd = self._convert_to_usd(self.cash, latest_month)
            
            if f'{latest_month}_usd' in cash_with_usd.columns:
                return cash_with_usd[f'{latest_month}_usd'].sum()
            
            return None
            
        except Exception as e:
            print(f"Error getting current cash balance: {e}")
            return None
    
    def get_average_burn_rate(self, months: int = 3) -> Optional[float]:
        """Calculate average monthly burn rate over specified months"""
        try:
            # Get cash trend for the last N months
            latest_months = self.month_columns[-months:]
            
            cash_balances = []
            for month in latest_months:
                cash_with_usd = self._convert_to_usd(self.cash, month)
                if f'{month}_usd' in cash_with_usd.columns:
                    balance = cash_with_usd[f'{month}_usd'].sum()
                    cash_balances.append(balance)
            
            if len(cash_balances) < 2:
                return None
            
            # Calculate monthly changes
            monthly_changes = []
            for i in range(1, len(cash_balances)):
                change = cash_balances[i] - cash_balances[i-1]
                monthly_changes.append(change)
            
            # Return average monthly change (negative = burn)
            return np.mean(monthly_changes)
            
        except Exception as e:
            print(f"Error calculating burn rate: {e}")
            return None
    
    def get_cash_trend(self, start_month: str, end_month: str) -> pd.DataFrame:
        """Get cash balance trend"""
        results = []
        months = [col for col in self.month_columns if start_month <= col <= end_month]
        
        for month in months:
            if month not in self.month_columns:
                continue
                
            cash_with_usd = self._convert_to_usd(self.cash, month)
            balance = cash_with_usd[f'{month}_usd'].sum() if f'{month}_usd' in cash_with_usd.columns else 0
            
            results.append({
                'month': month,
                'cash_balance_usd': balance
            })
        
        return pd.DataFrame(results)
    
    def get_ebitda(self, month: str) -> Optional[Dict[str, float]]:
        """Calculate EBITDA for a given month"""
        try:
            if month not in self.month_columns:
                return None
            
            # Revenue
            revenue_accounts = self.actuals[self.actuals['account'].str.contains('Revenue|Sales', case=False, na=False)]
            revenue_with_usd = self._convert_to_usd(revenue_accounts, month)
            revenue = revenue_with_usd[f'{month}_usd'].sum() if f'{month}_usd' in revenue_with_usd.columns else 0
            
            # COGS
            cogs_accounts = self.actuals[self.actuals['account'].str.contains('COGS|Cost of Goods|Cost of Sales', case=False, na=False)]
            cogs_with_usd = self._convert_to_usd(cogs_accounts, month)
            cogs = cogs_with_usd[f'{month}_usd'].sum() if f'{month}_usd' in cogs_with_usd.columns else 0
            
            # OpEx
            opex_accounts = self.actuals[
                ~self.actuals['account'].str.contains('Revenue|Sales|COGS|Cost of Goods|Cost of Sales', case=False, na=False)
            ]
            opex_with_usd = self._convert_to_usd(opex_accounts, month)
            opex = opex_with_usd[f'{month}_usd'].sum() if f'{month}_usd' in opex_with_usd.columns else 0
            
            # EBITDA = Revenue - COGS - OpEx (simplified, ignoring D&A)
            ebitda = revenue - cogs - opex
            
            return {
                'revenue': revenue,
                'cogs': cogs,
                'opex': opex,
                'ebitda': ebitda,
                'month': month
            }
            
        except Exception as e:
            print(f"Error calculating EBITDA: {e}")
            return None
