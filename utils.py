"""
Utility functions for Marine Life Research Dashboard
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class DataLoader:
    """Handle data loading and preprocessing"""
    
    def __init__(self):
        self.df = None
        self.numeric_columns = []
        
    def load_data(self, filepath: str) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """
        Load CSV/TSV file and preprocess
        
        Args:
            filepath: Path to the data file
            
        Returns:
            Tuple of (success, message, dataframe)
        """
        try:
            # Try reading as CSV first
            if filepath.endswith('.tsv') or filepath.endswith('.txt'):
                df = pd.read_csv(filepath, sep='\t', low_memory=False)
            else:
                df = pd.read_csv(filepath, low_memory=False)
            
            # Preprocess datetime columns
            if 'eventDate' in df.columns:
                df['eventDate'] = pd.to_datetime(df['eventDate'], errors='coerce')
                df['year'] = df['eventDate'].dt.year
                df['month'] = df['eventDate'].dt.month
                df['day'] = df['eventDate'].dt.day
            
            # Convert numeric columns
            numeric_cols = ['individualCount', 'depth', 'temperature', 'salinity',
                           'bathymetry', 'decimalLatitude', 'decimalLongitude',
                           'minimumDepthInMeters', 'maximumDepthInMeters']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            self.df = df
            self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            message = f"✅ Successfully loaded {len(df):,} rows × {len(df.columns)} columns"
            return True, message, df
            
        except Exception as e:
            return False, f"❌ Error loading file: {str(e)}", None
    
    def get_numeric_columns(self):
        """Return list of numeric column names"""
        return self.numeric_columns
    
    def get_dataframe(self):
        """Return the loaded dataframe"""
        return self.df


def format_number(num: float) -> str:
    """Format large numbers with commas"""
    if pd.isna(num):
        return "N/A"
    return f"{num:,.0f}"


def detect_indian_shores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter data for Indian shores (Bay of Bengal & Arabian Sea)
    
    Coordinates:
    - Bay of Bengal: Lat 5-22°N, Lon 80-95°E
    - Arabian Sea: Lat 8-24°N, Lon 65-77°E
    """
    if 'decimalLatitude' not in df.columns or 'decimalLongitude' not in df.columns:
        return df[df.index < 0]  # Return empty DataFrame
    
    # Bay of Bengal
    bay_bengal = (
        (df['decimalLatitude'].between(5, 22)) & 
        (df['decimalLongitude'].between(80, 95))
    )
    
    # Arabian Sea
    arabian_sea = (
        (df['decimalLatitude'].between(8, 24)) & 
        (df['decimalLongitude'].between(65, 77))
    )
    
    return df[bay_bengal | arabian_sea].copy()


def get_season(month: int) -> str:
    """Convert month to season"""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Unknown'
