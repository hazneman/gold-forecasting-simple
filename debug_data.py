#!/usr/bin/env python3
"""
Debug script to identify the broadcasting issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def debug_data_collection():
    """Debug the data collection process step by step"""
    
    print("🔍 Starting debug process...")
    
    try:
        # Step 1: Get basic data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 years
        
        print(f"📅 Getting data from {start_date} to {end_date}")
        gold = yf.download('GC=F', start=start_date, end=end_date, progress=False)
        
        if gold.empty:
            print("❌ No data for GC=F")
            return
            
        print(f"✅ Downloaded {len(gold)} rows")
        print(f"📊 Columns: {list(gold.columns)}")
        print(f"📈 Data shape: {gold.shape}")
        print(f"🗓️  Index type: {type(gold.index)}")
        
        # Handle multi-level columns from yfinance
        if gold.columns.nlevels > 1:
            print("🔧 Flattening multi-level columns...")
            gold.columns = [col[0] for col in gold.columns]
            print(f"📊 New columns: {list(gold.columns)}")
        
        # Step 2: Basic cleaning
        df = gold[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df = df.ffill()
        
        print(f"✅ After cleaning: {df.shape}")
        print(f"📊 Sample data:\n{df.head()}")
        
        # Step 3: Try simple indicators one by one
        print("\n🔧 Testing simple moving average...")
        df['sma_20'] = df['Close'].rolling(20).mean()
        print(f"✅ SMA_20 shape: {df['sma_20'].shape}")
        
        print("\n🔧 Testing price ratio...")
        # This is where the error likely occurs
        try:
            df['price_to_sma_20'] = df['Close'] / df['sma_20']
            print(f"✅ Price ratio shape: {df['price_to_sma_20'].shape}")
        except Exception as e:
            print(f"❌ Error in price ratio: {e}")
            print(f"Close shape: {df['Close'].shape}")
            print(f"SMA_20 shape: {df['sma_20'].shape}")
            print(f"Close type: {type(df['Close'])}")
            print(f"SMA_20 type: {type(df['sma_20'])}")
            return
        
        print("\n🔧 Testing RSI calculation...")
        try:
            # Simple RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            df['rsi'] = rsi.fillna(50)
            print(f"✅ RSI shape: {df['rsi'].shape}")
        except Exception as e:
            print(f"❌ Error in RSI: {e}")
            return
            
        print("\n🔧 Testing final cleanup...")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        print(f"✅ Final data shape: {df.shape}")
        print(f"📊 Final columns: {list(df.columns)}")
        
        print("\n🎉 Data collection debug completed successfully!")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_data_collection()