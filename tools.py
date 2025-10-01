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
            actuals_df: Monthly actual financial data (month, entity, account_category, amount, currency)
            budget_df: Monthly budget data (month, entity, account_category, amount, currency)
            fx_df: Foreign exchange rates (month, currency, rate_to_usd)
            cash_df: Cash balance data (month, entity, cash_usd)
        """
        self.actuals = actuals_df.copy()
        self.budget = budget_df.copy()
        self.fx = fx_df.copy()
        self.cash = cash_df.copy()
        
        # Ensure month columns are strings in YYYY-MM format
        self.actuals['month'] = pd.to_datetime(self.actuals['month']).dt.strftime('%Y-%m')
        self.budget['month'] = pd.to_datetime(self.budget['month']).dt.strftime('%Y-%m')
        self.fx['month'] = pd.to_datetime(self.fx['month']).dt.strftime('%Y-%m')
        self.cash['month'] = pd.to_datetime(self.cash['month']).dt.strftime('%Y-%m')
        
        # Get available months
        self.month_columns = sorted(self.actuals['month'].unique())
        
        # Create FX lookup dictionary for faster access
        self._create_fx_lookup()
    
    def _create_fx_lookup(self):
        """Create a dictionary for fast FX rate lookup"""
        self.fx_lookup = {}
        for _, row in self.fx.iterrows():
            key = (row['month'], row['currency'])
            self.fx_lookup[key] = row['rate_to_usd']
    
    def _convert_to_usd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert amounts to USD using FX rates
        
        Args:
            df: DataFrame with columns: month, currency, amount
            
        Returns:
            DataFrame with additional amount_usd column
        """
        result_df = df.copy()
        
        # Initialize amount_usd column
        result_df['amount_usd'] = 0.0
        
        for idx, row in result_df.iterrows():
            month = row['month']
            currency = row['currency']
            amount = row['amount']
            
            # Get FX rate
            fx_rate = self.fx_lookup.get((month, currency), 1.0)
            
            # Convert to USD
            result_df.loc[idx, 'amount_usd'] = amount * fx_rate
        
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
        revenue_actuals = self.actuals[
            (self.actuals['account_category'] == 'Revenue') &
            (self.actuals['month'] >= start_month) &
            (self.actuals['month'] <= end_month)
        ].copy()
        
        revenue_budget = self.budget[
            (self.budget['account_category'] == 'Revenue') &
            (self.budget['month'] >= start_month) &
            (self.budget['month'] <= end_month)
        ].copy()
        
        # Convert to USD
        revenue_actuals_usd = self._convert_to_usd(revenue_actuals)
        revenue_budget_usd = self._convert_to_usd(revenue_budget)
        
        # Group by month
        actuals_by_month = revenue_actuals_usd.groupby('month')['amount_usd'].sum().reset_index()
        actuals_by_month.columns = ['month', 'actual_usd']
        
        budget_by_month = revenue_budget_usd.groupby('month')['amount_usd'].sum().reset_index()
        budget_by_month.columns = ['month', 'budget_usd']
        
        # Merge
        result = pd.merge(actuals_by_month, budget_by_month, on='month', how='outer')
        result = result.fillna(0)
        
        # Calculate variance
        result['variance_usd'] = result['actual_usd'] - result['budget_usd']
        result['variance_pct'] = (result['variance_usd'] / result['budget_usd'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
        
        return result.sort_values('month')
    
    def get_revenue_trend(self, start_month: str, end_month: str) -> pd.DataFrame:
        """Get revenue trend over time"""
        revenue_actuals = self.actuals[
            (self.actuals['account_category'] == 'Revenue') &
            (self.actuals['month'] >= start_month) &
            (self.actuals['month'] <= end_month)
        ].copy()
        
        # Convert to USD
        revenue_actuals_usd = self._convert_to_usd(revenue_actuals)
        
        # Group by month
        result = revenue_actuals_usd.groupby('month')['amount_usd'].sum().reset_index()
        result.columns = ['month', 'revenue_usd']
        
        return result.sort_values('month')
    
    def get_gross_margin_trend(self, start_month: str, end_month: str) -> pd.DataFrame:
        """Calculate gross margin trend"""
        # Get revenue
        revenue_actuals = self.actuals[
            (self.actuals['account_category'] == 'Revenue') &
            (self.actuals['month'] >= start_month) &
            (self.actuals['month'] <= end_month)
        ].copy()
        
        # Get COGS
        cogs_actuals = self.actuals[
            (self.actuals['account_category'] == 'COGS') &
            (self.actuals['month'] >= start_month) &
            (self.actuals['month'] <= end_month)
        ].copy()
        
        # Convert to USD
        revenue_usd = self._convert_to_usd(revenue_actuals)
        cogs_usd = self._convert_to_usd(cogs_actuals)
        
        # Group by month
        revenue_by_month = revenue_usd.groupby('month')['amount_usd'].sum().reset_index()
        revenue_by_month.columns = ['month', 'revenue_usd']
        
        cogs_by_month = cogs_usd.groupby('month')['amount_usd'].sum().reset_index()
        cogs_by_month.columns = ['month', 'cogs_usd']
        
        # Merge and calculate margin
        result = pd.merge(revenue_by_month, cogs_by_month, on='month', how='outer')
        result = result.fillna(0)
        
        result['gross_profit_usd'] = result['revenue_usd'] - result['cogs_usd']
        result['gross_margin_pct'] = (result['gross_profit_usd'] / result['revenue_usd'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
        
        return result.sort_values('month')
    
    def get_opex_breakdown(self, month: str) -> pd.DataFrame:
        """Get operating expense breakdown by category"""
        # Filter for OpEx accounts (those starting with 'Opex:')
        opex_actuals = self.actuals[
            (self.actuals['account_category'].str.startswith('Opex:', na=False)) &
            (self.actuals['month'] == month)
        ].copy()
        
        if opex_actuals.empty:
            return pd.DataFrame()
        
        # Convert to USD
        opex_usd = self._convert_to_usd(opex_actuals)
        
        # Extract category (everything after 'Opex:')
        opex_usd['category'] = opex_usd['account_category'].str.replace('Opex:', '')
        
        # Group by category
        result = opex_usd.groupby('category')['amount_usd'].sum().reset_index()
        result.columns = ['category', 'amount_usd']
        result['month'] = month
        
        return result.sort_values('amount_usd', ascending=False)
    
    def get_opex_trend(self, start_month: str, end_month: str) -> pd.DataFrame:
        """Get total OpEx trend over time"""
        opex_actuals = self.actuals[
            (self.actuals['account_category'].str.startswith('Opex:', na=False)) &
            (self.actuals['month'] >= start_month) &
            (self.actuals['month'] <= end_month)
        ].copy()
        
        # Convert to USD
        opex_usd = self._convert_to_usd(opex_actuals)
        
        # Group by month
        result = opex_usd.groupby('month')['amount_usd'].sum().reset_index()
        result.columns = ['month', 'opex_usd']
        
        return result.sort_values('month')
    
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
            if self.cash.empty:
                return None
            
            # Get the latest month
            latest_month = self.cash['month'].max()
            
            # Sum cash across all entities for the latest month
            current_cash = self.cash[self.cash['month'] == latest_month]['cash_usd'].sum()
            
            return current_cash
            
        except Exception as e:
            print(f"Error getting current cash balance: {e}")
            return None
    
    def get_average_burn_rate(self, months: int = 3) -> Optional[float]:
        """Calculate average monthly burn rate over specified months"""
        try:
            if self.cash.empty:
                return None
            
            # Get unique months sorted
            unique_months = sorted(self.cash['month'].unique())
            
            if len(unique_months) < 2:
                return None
            
            # Get last N months
            last_n_months = unique_months[-months:]
            
            # Calculate cash balances for these months
            cash_balances = []
            for month in last_n_months:
                balance = self.cash[self.cash['month'] == month]['cash_usd'].sum()
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
        cash_filtered = self.cash[
            (self.cash['month'] >= start_month) &
            (self.cash['month'] <= end_month)
        ].copy()
        
        # Group by month (in case there are multiple entities)
        result = cash_filtered.groupby('month')['cash_usd'].sum().reset_index()
        result.columns = ['month', 'cash_balance_usd']
        
        return result.sort_values('month')
    
    def get_ebitda(self, month: str) -> Optional[Dict[str, float]]:
        """Calculate EBITDA for a given month"""
        try:
            # Revenue
            revenue_actuals = self.actuals[
                (self.actuals['account_category'] == 'Revenue') &
                (self.actuals['month'] == month)
            ].copy()
            revenue_usd = self._convert_to_usd(revenue_actuals)
            revenue = revenue_usd['amount_usd'].sum()
            
            # COGS
            cogs_actuals = self.actuals[
                (self.actuals['account_category'] == 'COGS') &
                (self.actuals['month'] == month)
            ].copy()
            cogs_usd = self._convert_to_usd(cogs_actuals)
            cogs = cogs_usd['amount_usd'].sum()
            
            # OpEx
            opex_actuals = self.actuals[
                (self.actuals['account_category'].str.startswith('Opex:', na=False)) &
                (self.actuals['month'] == month)
            ].copy()
            opex_usd = self._convert_to_usd(opex_actuals)
            opex = opex_usd['amount_usd'].sum()
            
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
    
    def get_ebitda_trend(self, start_month: str, end_month: str) -> pd.DataFrame:
        """Get EBITDA trend over time"""
        months = [m for m in self.month_columns if start_month <= m <= end_month]
        
        results = []
        for month in months:
            ebitda_data = self.get_ebitda(month)
            if ebitda_data:
                results.append(ebitda_data)
        
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame()
