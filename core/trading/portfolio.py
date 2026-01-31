#!/usr/bin/env python3
"""
MARK3 Portfolio Optimizer v3.0
Modern Portfolio Theory and optimization algorithms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """Optimize portfolio allocation using Modern Portfolio Theory"""
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize portfolio optimizer
        
        Args:
            risk_free_rate: Annual risk-free rate (5% default)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_portfolio_stats(self, weights: np.ndarray, 
                                  returns: pd.DataFrame) -> Dict:
        """
        Calculate portfolio statistics
        
        Args:
            weights: Asset weights
            returns: Returns DataFrame (columns = assets)
        
        Returns:
            Portfolio statistics
        """
        # Annualized portfolio return
        portfolio_return = np.sum(returns.mean() * weights) * 252
        
        # Annualized portfolio volatility
        portfolio_std = np.sqrt(
            np.dot(weights.T, np.dot(returns.cov() * 252, weights))
        )
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_std,
            'sharpe_ratio': sharpe_ratio
        }
    
    def optimize_sharpe(self, returns: pd.DataFrame, 
                       constraints: Dict = None) -> Dict:
        """
        Optimize portfolio to maximize Sharpe ratio
        
        Args:
            returns: Historical returns DataFrame
            constraints: Optional constraints (min_weight, max_weight, etc.)
        
        Returns:
            Optimal weights and statistics
        """
        n_assets = len(returns.columns)
        
        # Objective function (negative Sharpe to minimize)
        def negative_sharpe(weights):
            stats = self.calculate_portfolio_stats(weights, returns)
            return -stats['sharpe_ratio']
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Bounds (default: no shorting)
        min_weight = constraints.get('min_weight', 0.0) if constraints else 0.0
        max_weight = constraints.get('max_weight', 1.0) if constraints else 1.0
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        init_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            negative_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if not result.success:
            return {'success': False, 'message': result.message}
        
        optimal_weights = result.x
        stats = self.calculate_portfolio_stats(optimal_weights, returns)
        
        return {
            'success': True,
            'weights': dict(zip(returns.columns, optimal_weights)),
            'stats': stats,
            'allocation': {
                ticker: round(weight * 100, 2) 
                for ticker, weight in zip(returns.columns, optimal_weights)
                if weight > 0.01  # Show only significant allocations
            }
        }
    
    def optimize_min_variance(self, returns: pd.DataFrame, 
                             target_return: float = None) -> Dict:
        """
        Optimize portfolio to minimize variance
        
        Args:
            returns: Historical returns DataFrame
            target_return: Optional target return constraint
        
        Returns:
            Optimal weights and statistics
        """
        n_assets = len(returns.columns)
        
        # Objective function (portfolio variance)
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(returns.cov() * 252, weights))
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # Add return constraint if specified
        if target_return is not None:
            constraints_list.append({
                'type': 'eq',
                'fun': lambda x: np.sum(returns.mean() * x) * 252 - target_return
            })
        
        bounds = tuple((0, 1) for _ in range(n_assets))
        init_weights = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            portfolio_variance,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if not result.success:
            return {'success': False, 'message': result.message}
        
        optimal_weights = result.x
        stats = self.calculate_portfolio_stats(optimal_weights, returns)
        
        return {
            'success': True,
            'weights': dict(zip(returns.columns, optimal_weights)),
            'stats': stats,
            'allocation': {
                ticker: round(weight * 100, 2)
                for ticker, weight in zip(returns.columns, optimal_weights)
                if weight > 0.01
            }
        }
    
    def efficient_frontier(self, returns: pd.DataFrame, 
                          n_points: int = 50) -> pd.DataFrame:
        """
        Generate efficient frontier
        
        Args:
            returns: Historical returns DataFrame
            n_points: Number of points on frontier
        
        Returns:
            DataFrame with frontier portfolios
        """
        # Calculate range of returns
        min_return = returns.mean().min() * 252
        max_return = returns.mean().max() * 252
        
        target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier_portfolios = []
        
        for target in target_returns:
            result = self.optimize_min_variance(returns, target_return=target)
            
            if result['success']:
                frontier_portfolios.append({
                    'return': result['stats']['return'],
                    'volatility': result['stats']['volatility'],
                    'sharpe': result['stats']['sharpe_ratio']
                })
        
        return pd.DataFrame(frontier_portfolios)
    
    def black_litterman(self, market_caps: Dict[str, float],
                       returns: pd.DataFrame,
                       views: Dict[str, float] = None,
                       tau: float = 0.05) -> Dict:
        """
        Black-Litterman model for incorporating market views
        
        Args:
            market_caps: Market capitalizations
            returns: Historical returns
            views: Optional views on expected returns
            tau: Uncertainty in prior (0.05 default)
        
        Returns:
            Adjusted expected returns and optimal weights
        """
        # Market weights (from market caps)
        total_cap = sum(market_caps.values())
        market_weights = np.array([
            market_caps[ticker] / total_cap 
            for ticker in returns.columns
        ])
        
        # Covariance matrix
        cov_matrix = returns.cov() * 252
        
        # Implied equilibrium returns (reverse optimization)
        risk_aversion = 2.5  # Typical value
        pi = risk_aversion * np.dot(cov_matrix, market_weights)
        
        if views:
            # Incorporate views (simplified version)
            # In full implementation, would use view matrix P and omega
            adjusted_returns = pi.copy()
            for i, ticker in enumerate(returns.columns):
                if ticker in views:
                    # Blend market implied with views
                    adjusted_returns[i] = 0.7 * pi[i] + 0.3 * views[ticker]
        else:
            adjusted_returns = pi
        
        # Optimize with adjusted returns
        def negative_utility(weights):
            portfolio_return = np.dot(weights, adjusted_returns)
            portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
            return -(portfolio_return - risk_aversion * portfolio_var / 2)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(len(returns.columns)))
        init_weights = market_weights
        
        result = minimize(
            negative_utility,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            return {'success': False}
        
        optimal_weights = result.x
        
        return {
            'success': True,
            'weights': dict(zip(returns.columns, optimal_weights)),
            'expected_returns': dict(zip(returns.columns, adjusted_returns)),
            'allocation': {
                ticker: round(weight * 100, 2)
                for ticker, weight in zip(returns.columns, optimal_weights)
                if weight > 0.01
            }
        }
    
    def risk_parity(self, returns: pd.DataFrame) -> Dict:
        """
        Risk parity portfolio - equal risk contribution
        
        Args:
            returns: Historical returns DataFrame
        
        Returns:
            Risk parity weights
        """
        n_assets = len(returns.columns)
        cov_matrix = returns.cov() * 252
        
        # Objective: minimize difference in risk contributions
        def risk_contribution_diff(weights):
            portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib
            
            # Target: equal risk contribution
            target_contrib = portfolio_var / n_assets
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0.01, 1) for _ in range(n_assets))
        init_weights = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            risk_contribution_diff,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            return {'success': False}
        
        optimal_weights = result.x
        stats = self.calculate_portfolio_stats(optimal_weights, returns)
        
        return {
            'success': True,
            'weights': dict(zip(returns.columns, optimal_weights)),
            'stats': stats,
            'allocation': {
                ticker: round(weight * 100, 2)
                for ticker, weight in zip(returns.columns, optimal_weights)
            }
        }


if __name__ == '__main__':
    print("Testing Portfolio Optimizer...")
    
    # Generate sample returns data
    np.random.seed(42)
    tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'WIPRO.NS']
    
    returns_data = {}
    for ticker in tickers:
        returns_data[ticker] = np.random.randn(252) * 0.02
    
    returns = pd.DataFrame(returns_data)
    
    optimizer = PortfolioOptimizer(risk_free_rate=0.05)
    
    print("\n1. Maximum Sharpe Ratio Portfolio...")
    max_sharpe = optimizer.optimize_sharpe(returns)
    
    if max_sharpe['success']:
        print(f"   Expected Return: {max_sharpe['stats']['return']*100:.2f}%")
        print(f"   Volatility: {max_sharpe['stats']['volatility']*100:.2f}%")
        print(f"   Sharpe Ratio: {max_sharpe['stats']['sharpe_ratio']:.2f}")
        print(f"   Allocation:")
        for ticker, weight in max_sharpe['allocation'].items():
            print(f"     {ticker}: {weight:.1f}%")
    
    print("\n2. Minimum Variance Portfolio...")
    min_var = optimizer.optimize_min_variance(returns)
    
    if min_var['success']:
        print(f"   Expected Return: {min_var['stats']['return']*100:.2f}%")
        print(f"   Volatility: {min_var['stats']['volatility']*100:.2f}%")
        print(f"   Allocation:")
        for ticker, weight in min_var['allocation'].items():
            print(f"     {ticker}: {weight:.1f}%")
    
    print("\n3. Risk Parity Portfolio...")
    risk_parity = optimizer.risk_parity(returns)
    
    if risk_parity['success']:
        print(f"   Allocation:")
        for ticker, weight in risk_parity['allocation'].items():
            print(f"     {ticker}: {weight:.1f}%")
    
    print("\n✓ Portfolio optimizer test complete")
