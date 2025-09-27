#!/usr/bin/env python3

import requests
import json
import time

def test_extended_training():
    base_url = "http://localhost:8000"
    
    try:
        # Test cache status first
        print("🔍 Testing cache status...")
        response = requests.get(f"{base_url}/cache-status")
        if response.status_code == 200:
            cache_data = response.json()
            print(f"✅ Cache Status: {json.dumps(cache_data, indent=2)}")
        else:
            print(f"❌ Cache Status failed: {response.status_code} - {response.text}")
        
        print("\n" + "="*50)
        
        # Test extended training
        print("🚀 Starting extended training...")
        response = requests.post(f"{base_url}/start-extended-training-async", 
                               params={"force_retrain": True, "period": "2y"})
        
        if response.status_code == 200:
            training_data = response.json()
            print(f"✅ Training Started: {json.dumps(training_data, indent=2)}")
            
            # Monitor progress for a few seconds
            print("\n📊 Monitoring training progress...")
            for i in range(10):
                time.sleep(2)
                progress_response = requests.get(f"{base_url}/training-progress")
                if progress_response.status_code == 200:
                    progress = progress_response.json()
                    print(f"Progress: {progress.get('progress_percent', 0):.1f}% - {progress.get('current_step', 'Unknown')}")
                    
                    if progress.get('status') == 'completed':
                        print("🎉 Training completed!")
                        break
                    elif progress.get('status') == 'failed':
                        print("❌ Training failed!")
                        break
                else:
                    print(f"❌ Progress check failed: {progress_response.text}")
                    break
            
        else:
            print(f"❌ Training failed to start: {response.status_code} - {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure it's running on localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_extended_training()