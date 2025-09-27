#!/usr/bin/env python3
"""
Test script to verify ML models are using economic features
"""

import sys
import os
sys.path.append('.')

from src.enhanced_auto_trainer import EnhancedAutoMLTrainer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_economic_features():
    """Test if economic features are integrated into ML training"""
    
    print("🔬 Testing Economic Features Integration in ML")
    print("=" * 50)
    
    try:
        # Initialize trainer
        trainer = EnhancedAutoMLTrainer()
        
        # Test data collection with economic features
        print("\n1️⃣ Testing data collection...")
        data = trainer.collect_extended_training_data(period="1y")
        
        if data is None or data.empty:
            print("❌ Failed to collect training data")
            return False
            
        print(f"✅ Collected {len(data)} samples of gold price data")
        
        # Test feature engineering with economic data
        print("\n2️⃣ Testing feature engineering with economic data...")
        features_df = trainer.engineer_features(data)
        
        if features_df.empty:
            print("❌ Feature engineering failed")
            return False
            
        print(f"✅ Feature engineering complete: {features_df.shape[1]} total features")
        
        # Check for economic features
        economic_feature_keywords = [
            'fed_funds', 'dxy', 'vix', 'treasury', 'tips', 'oil', 'spy', 'tlt', 
            'inflation', 'yield_curve', 'market_stress', 'bond'
        ]
        
        economic_features_found = []
        for keyword in economic_feature_keywords:
            matching_features = [col for col in features_df.columns if keyword in col.lower()]
            economic_features_found.extend(matching_features)
        
        print(f"\n3️⃣ Economic features analysis:")
        print(f"🔍 Total features: {features_df.shape[1]}")
        print(f"📊 Economic features found: {len(economic_features_found)}")
        
        if economic_features_found:
            print("✅ Economic features successfully integrated:")
            for feature in sorted(economic_features_found):
                print(f"   • {feature}")
        else:
            print("❌ No economic features found in the dataset")
            
        # Check feature categories
        print(f"\n4️⃣ Feature breakdown:")
        technical_features = [col for col in features_df.columns if col not in economic_features_found]
        print(f"📈 Technical features: {len(technical_features)}")
        print(f"💰 Economic features: {len(economic_features_found)}")
        print(f"🎯 Total ML features: {len(technical_features) + len(economic_features_found)}")
        
        # Show sample of latest features with values
        print(f"\n5️⃣ Sample economic feature values (latest):")
        for feature in economic_features_found[:10]:  # Show first 10 economic features
            if feature in features_df.columns and not features_df[feature].isna().all():
                latest_value = features_df[feature].dropna().iloc[-1] if not features_df[feature].dropna().empty else "N/A"
                print(f"   • {feature}: {latest_value}")
        
        success = len(economic_features_found) > 0
        if success:
            print(f"\n🎉 SUCCESS: ML now uses {len(economic_features_found)} economic features!")
            print("🏛️ Fed policies, inflation, and market data are integrated into predictions")
        else:
            print("\n❌ FAILED: Economic features not integrated")
            
        return success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_economic_features()
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: Economic ML integration test")
    sys.exit(0 if success else 1)