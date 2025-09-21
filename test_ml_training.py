#!/usr/bin/env python3
"""
Test script for ML training functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.auto_trainer import auto_trainer

def test_training_status():
    """Test getting training status"""
    print("🔍 Getting training status...")
    try:
        status = auto_trainer.get_training_status()
        print(f"✅ Status: {status}")
        return True
    except Exception as e:
        print(f"❌ Error getting status: {e}")
        return False

def test_data_collection():
    """Test collecting training data"""
    print("\n📊 Testing data collection...")
    try:
        data = auto_trainer.collect_training_data()
        print(f"✅ Collected {len(data)} days of data")
        print(f"📈 Data columns: {list(data.columns)}")
        print(f"📅 Date range: {data.index.min()} to {data.index.max()}")
        return True
    except Exception as e:
        print(f"❌ Error collecting data: {e}")
        return False

def test_model_training():
    """Test training models"""
    print("\n🚀 Testing model training...")
    try:
        model_paths = auto_trainer.train_prediction_models()
        print(f"✅ Training completed!")
        print(f"📁 Model paths: {model_paths}")
        return True
    except Exception as e:
        print(f"❌ Error training models: {e}")
        return False

def test_predictions():
    """Test making predictions"""
    print("\n🔮 Testing predictions...")
    try:
        for days in [1, 3, 7]:
            prediction = auto_trainer.predict_with_trained_model(horizon_days=days)
            print(f"✅ {days}-day prediction: ${prediction:.2f}")
        return True
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        return False

if __name__ == "__main__":
    print("🤖 ML Training System Test")
    print("=" * 50)
    
    success_count = 0
    total_tests = 4
    
    if test_training_status():
        success_count += 1
    
    if test_data_collection():
        success_count += 1
    
    if test_model_training():
        success_count += 1
    
    if test_predictions():
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {success_count}/{total_tests} passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! ML training system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")