import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'app'))

from nutrition_database import NutritionDatabase
from food_image_recognizer import FoodImageRecognizer

def test_food_model():
    """Test your food recognition model"""
    print("🧪 Testing Food Recognition Model")
    print("=" * 40)
    
    # Initialize components
    nutrition_db = NutritionDatabase('backend/data/indian_food_nutrition.csv')
    recognizer = FoodImageRecognizer(nutrition_db)
    
    # Test 1: Model info
    print("\n📊 MODEL INFORMATION:")
    model_info = recognizer.get_model_info()
    print(f"   Input Shape: {model_info['model_input_shape']}")
    print(f"   Output Shape: {model_info['model_output_shape']}")
    print(f"   Number of Classes: {model_info['num_classes']}")
    
    # Test 2: Database alignment
    print("\n🔗 DATABASE ALIGNMENT:")
    alignment = recognizer.validate_food_database_alignment()
    print(f"   Model Classes: {alignment['total_model_classes']}")
    print(f"   Database Foods: {alignment['total_db_foods']}")
    print(f"   Alignment: {alignment['alignment_percentage']:.1f}%")
    
    if alignment['model_only_foods']:
        print(f"   Foods in model but not in database: {len(alignment['model_only_foods'])}")
        for food in alignment['model_only_foods'][:5]:
            print(f"      - {food}")
    
    # Test 3: Sample prediction
    print("\n🎯 SAMPLE PREDICTION TEST:")
    test_result = recognizer.test_model_with_sample()
    print(f"   Test Status: {test_result['test_status']}")
    if test_result['test_status'] == 'success':
        print(f"   Predictions Generated: {test_result['predictions_count']}")
    else:
        print(f"   Error: {test_result['error']}")
    
    # Test 4: List supported foods
    print("\n📋 SUPPORTED FOODS (First 20):")
    supported_foods = model_info['supported_foods']
    for i, food in enumerate(supported_foods[:20], 1):
        display_name = food.replace('_', ' ').title()
        print(f"   {i:2d}. {display_name}")
    
    if len(supported_foods) > 20:
        print(f"   ... and {len(supported_foods) - 20} more")
    
    print(f"\n✅ Model testing completed!")
    print(f"🎯 Your model supports {len(supported_foods)} Indian food classes")

if __name__ == "__main__":
    test_food_model()