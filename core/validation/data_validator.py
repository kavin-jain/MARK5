"""
MARK5 DATA VALIDATOR v6.1 (STRICT MODE)
---------------------------------------
Enforces Data Integrity.
Philosophy: "Better to have no data than fake data."
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from datetime import datetime, timedelta

class DataValidator:
    """
    Enforces Data Integrity. 
    Philosophy: "Better to have no data than fake data."
    """
    
    REQUIRED_COLUMNS = {'open', 'high', 'low', 'close', 'volume'}
    
    @staticmethod
    def validate_strict(data: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Validates data WITHOUT modifying it. 
        Rejects the dataset if critical flaws exist.
        """
        issues = []
        warnings = []
        
        if data.empty:
            return data, {'valid': False, 'issues': ['Empty dataset']}
            
        # 1. Check for Missing Columns
        if not DataValidator.REQUIRED_COLUMNS.issubset(data.columns):
            return data, {'valid': False, 'issues': [f"Missing columns: {DataValidator.REQUIRED_COLUMNS - set(data.columns)}"]}

        # 2. Check for NaN (NO FILLING ALLOWED)
        if data[list(DataValidator.REQUIRED_COLUMNS)].isna().any().any():
            nan_counts = data[list(DataValidator.REQUIRED_COLUMNS)].isna().sum()
            issues.append(f"NaN values detected. We do not trade on guessed data. Counts: {nan_counts.to_dict()}")
        
        # 3. Logical Integrity (Low > High?)
        bad_rows = data[data['low'] > data['high']]
        if not bad_rows.empty:
            issues.append(f"Logical Error: Low > High in {len(bad_rows)} rows.")

        # 4. Zero Price Check
        if (data[['open', 'high', 'low', 'close']] <= 0).any().any():
            issues.append("Zero or Negative prices detected.")

        # 5. Volume Spike Check (Warning only)
        # 100x volume spike is suspicious but possible
        if 'volume' in data.columns:
            mean_vol = data['volume'].mean()
            spikes = data[data['volume'] > mean_vol * 100]
            if not spikes.empty:
                warnings.append(f"{len(spikes)} extreme volume spikes detected.")

        valid = len(issues) == 0
        
        report = {
            'valid': valid,
            'issues': issues,
            'warnings': warnings,
            'row_count': len(data)
        }
        
        return data, report

    @staticmethod
    def check_market_hours_integrity(data: pd.DataFrame) -> List[str]:
        """
        Ensures no data exists outside 09:15 - 15:30 IST.
        """
        issues = []
        if data.empty: return issues
        
        # Assuming index is DatetimeIndex
        times = data.index.time
        start = datetime.strptime("09:15", "%H:%M").time()
        end = datetime.strptime("15:30", "%H:%M").time()
        
        # Check for pre-market/post-market ghost data
        # Note: Depending on feed, 09:07 (pre-open) might exist, strategy must handle it.
        # This validator assumes we only want trading hours.
        mask = (times < start) | (times > end)
        ghost_data = data[mask]
        
        if not ghost_data.empty:
            issues.append(f"Found {len(ghost_data)} rows outside NSE trading hours.")
            
        return issues
