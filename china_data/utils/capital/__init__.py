"""
Capital stock calculation and projection utilities.

This package provides functions for calculating, projecting, and validating
capital stock data for economic analysis.
"""

from china_data.utils.capital.calculation import calculate_capital_stock
from china_data.utils.capital.projection import project_capital_stock
from china_data.utils.capital.validation import validate_capital_data
from china_data.utils.capital.investment import calculate_investment

__all__ = [
    'calculate_capital_stock',
    'project_capital_stock',
    'validate_capital_data',
    'calculate_investment',
]
