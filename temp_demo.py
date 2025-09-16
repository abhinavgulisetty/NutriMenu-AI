import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'app'))

from nutrition_database import NutritionDatabase
from nutrition_engine import NutritionEngine
import random

class SimplifiedNutriMenuDemo:
    def __init__(self):
        """Initialize demo with basic components only"""
        print("🍽️ Initializing Simplified NutriMenu AI Demo...")
        
        # Initialize basic components
        self.nutrition_db = NutritionDatabase('backend/data/indian_food_nutrition.csv')
        self.nutrition_engine = NutritionEngine(self.nutrition_db)
        
        print("✅ Basic components loaded successfully!\n")
    
    def display_menu(self):
        """Display simplified menu options"""
        print("=" * 50)
        print("🍽️  SIMPLIFIED NUTRIMENU AI DEMO")
        print("=" * 50)
        print("1. 🔍 Search Foods by Name")
        print("2. 📊 Get Nutrition Analysis")
        print("3. 📋 Browse Food Categories")
        print("4. ⚖️  Compare Foods")
        print("5. 🎲 Random Food Suggestions")
        print("6. ❌ Exit")
        print("=" * 50)
    
    def search_foods(self):
        """Search foods by name"""
        print("\n🔍 FOOD SEARCH")
        query = input("Enter food name to search: ").strip()
        
        if not query:
            print("❌ Please enter a food name!")
            return
        
        results = self.nutrition_engine.search_foods_by_name(query, limit=5)
        
        if results:
            print(f"\n📋 Found {len(results)} foods matching '{query}':")
            print("-" * 40)
            
            for i, food in enumerate(results, 1):
                food_name = food['food_name'].replace('_', ' ').title()
                print(f"{i}. {food_name}")
                print(f"   📊 {food['calories']} calories | {food['protein_g']}g protein")
                print(f"   🥗 Category: {food['category']} | Meal: {food['meal_type']}")
                print(f"   ✅ Veg: {food['vegetarian']} | Vegan: {food['vegan']}")
                print()
        else:
            print("❌ No foods found matching your search!")
    
    def nutrition_analysis(self):
        """Get detailed nutrition analysis for a food"""
        print("\n📊 NUTRITION ANALYSIS")
        food_name = input("Enter exact food name (use underscores for spaces): ").strip()
        
        if not food_name:
            print("❌ Please enter a food name!")
            return
        
        analysis = self.nutrition_engine.get_nutrition_analysis(food_name)
        
        if analysis:
            print(f"\n📊 Nutrition Analysis for {analysis['food_name'].replace('_', ' ').title()}")
            print("=" * 40)
            
            # Calorie breakdown
            cal_breakdown = analysis['calorie_breakdown']
            print(f"🔥 Total Calories: {cal_breakdown['total_calories']}")
            print(f"   - Protein: {cal_breakdown['protein_calories']} cal")
            print(f"   - Carbs: {cal_breakdown['carb_calories']} cal")
            print(f"   - Fat: {cal_breakdown['fat_calories']} cal")
            
            # Macronutrients
            print(f"\n🥩 MACRONUTRIENTS:")
            macros = analysis['macronutrients']
            print(f"   Protein: {macros['protein_g']}g")
            print(f"   Carbs: {macros['carbs_g']}g")
            print(f"   Fat: {macros['fat_g']}g")
            print(f"   Fiber: {macros['fiber_g']}g")
            
            # Micronutrients
            print(f"\n💊 MICRONUTRIENTS:")
            micros = analysis['micronutrients']
            for nutrient, value in micros.items():
                if value and value != 0:
                    print(f"   {nutrient.replace('_', ' ').title()}: {value}")
            
            # Dietary info
            print(f"\n🏷️  DIETARY INFO:")
            dietary = analysis['dietary_info']
            print(f"   Category: {dietary['category']}")
            print(f"   Meal Type: {dietary['meal_type']}")
            print(f"   Vegetarian: {'✅' if dietary['vegetarian'] else '❌'}")
            print(f"   Vegan: {'✅' if dietary['vegan'] else '❌'}")
            print(f"   Gluten-Free: {'✅' if dietary['gluten_free'] else '❌'}")
            
            # Nutrition score
            score = analysis['nutrition_score']
            print(f"\n⭐ NUTRITION SCORE: {score}/100")
            
        else:
            print("❌ Food not found! Try searching first to see available foods.")
    
    def browse_categories(self):
        """Browse foods by categories"""
        print("\n📋 FOOD CATEGORIES")
        
        all_foods = self.nutrition_db.get_all_foods()
        
        # Category breakdown
        print("\n🥘 FOOD CATEGORIES:")
        categories = all_foods['category'].value_counts()
        for category, count in categories.items():
            print(f"   {category}: {count} foods")
        
        # Meal type breakdown
        print(f"\n🍽️  MEAL TYPES:")
        meal_types = all_foods['meal_type'].value_counts()
        for meal_type, count in meal_types.items():
            print(f"   {meal_type}: {count} foods")
        
        # Dietary options
        print(f"\n🏷️  DIETARY OPTIONS:")
        vegetarian_count = all_foods['vegetarian'].sum()
        vegan_count = all_foods['vegan'].sum()
        gluten_free_count = all_foods['gluten_free'].sum()
        
        print(f"   Vegetarian: {vegetarian_count} foods")
        print(f"   Vegan: {vegan_count} foods")
        print(f"   Gluten-Free: {gluten_free_count} foods")
        
        # Browse specific category
        print(f"\nEnter category name to browse foods (or press Enter to skip):")
        category = input().strip()
        
        if category:
            category_foods = all_foods[all_foods['category'].str.lower() == category.lower()]
            if not category_foods.empty:
                print(f"\n🥘 Foods in '{category}' category:")
                for _, food in category_foods.iterrows():
                    food_name = food['food_name'].replace('_', ' ').title()
                    print(f"   • {food_name} ({food['calories']} cal)")
            else:
                print(f"❌ No foods found in category '{category}'")
    
    def compare_foods(self):
        """Compare multiple foods"""
        print("\n⚖️  FOOD COMPARISON")
        
        foods_to_compare = []
        
        # Get foods to compare
        while len(foods_to_compare) < 3:
            food_name = input(f"Enter food {len(foods_to_compare) + 1} name (or 'done' to compare): ").strip()
            
            if food_name.lower() == 'done' and len(foods_to_compare) >= 2:
                break
            elif food_name.lower() == 'done':
                print("❌ Please enter at least 2 foods to compare!")
                continue
            
            if not food_name:
                continue
            
            nutrition_data = self.nutrition_db.get_nutrition(food_name)
            if nutrition_data:
                foods_to_compare.append(nutrition_data)
                print(f"✅ Added {food_name.replace('_', ' ').title()}")
            else:
                print(f"❌ Food '{food_name}' not found!")
        
        if len(foods_to_compare) >= 2:
            # Display comparison
            print(f"\n📊 COMPARISON OF {len(foods_to_compare)} FOODS")
            print("=" * 60)
            
            # Headers
            print(f"{'Nutrient':<15}", end="")
            for food in foods_to_compare:
                food_name = food['food_name'].replace('_', ' ')[:12]
                print(f"{food_name:>12}", end="")
            print()
            print("-" * 60)
            
            # Comparison data
            nutrients = ['calories', 'protein_g', 'carbs_g', 'fat_g', 'fiber_g']
            for nutrient in nutrients:
                print(f"{nutrient.replace('_', ' ').title():<15}", end="")
                for food in foods_to_compare:
                    value = food.get(nutrient, 0)
                    print(f"{value:>12}", end="")
                print()
            
            # Find best in each category
            print(f"\n🏆 WINNERS:")
            print(f"   Lowest Calories: {min(foods_to_compare, key=lambda x: x['calories'])['food_name'].replace('_', ' ').title()}")
            print(f"   Highest Protein: {max(foods_to_compare, key=lambda x: x['protein_g'])['food_name'].replace('_', ' ').title()}")
            print(f"   Highest Fiber: {max(foods_to_compare, key=lambda x: x['fiber_g'])['food_name'].replace('_', ' ').title()}")
    
    def random_suggestions(self):
        """Show random food suggestions"""
        print("\n🎲 RANDOM FOOD SUGGESTIONS")
        
        all_foods = self.nutrition_db.get_all_foods()
        
        # Random suggestions by category
        categories = ['breakfast', 'lunch', 'dinner', 'snack', 'dessert']
        
        for category in categories:
            category_foods = all_foods[all_foods['meal_type'] == category]
            if not category_foods.empty:
                random_food = category_foods.sample(1).iloc[0]
                food_name = random_food['food_name'].replace('_', ' ').title()
                
                print(f"\n🍽️  {category.upper()} Suggestion:")
                print(f"   🥘 {food_name}")
                print(f"   📊 {random_food['calories']} calories | {random_food['protein_g']}g protein")
                print(f"   🏷️  {random_food['category']} | Veg: {random_food['vegetarian']}")
        
        # Bonus: Random healthy option
        healthy_foods = all_foods[all_foods['calories'] < 200]
        if not healthy_foods.empty:
            healthy_food = healthy_foods.sample(1).iloc[0]
            food_name = healthy_food['food_name'].replace('_', ' ').title()
            
            print(f"\n💚 BONUS - Healthy Option:")
            print(f"   🥗 {food_name}")
            print(f"   📊 {healthy_food['calories']} calories | {healthy_food['protein_g']}g protein")
    
    def run(self):
        """Main demo loop"""
        print("🚀 Welcome to NutriMenu AI Demo!")
        
        while True:
            try:
                self.display_menu()
                choice = input("\nEnter your choice (1-6): ").strip()
                
                if choice == '1':
                    self.search_foods()
                elif choice == '2':
                    self.nutrition_analysis()
                elif choice == '3':
                    self.browse_categories()
                elif choice == '4':
                    self.compare_foods()
                elif choice == '5':
                    self.random_suggestions()
                elif choice == '6':
                    print("\n👋 Thank you for using NutriMenu AI Demo!")
                    break
                else:
                    print("❌ Invalid choice! Please enter 1-6.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Demo interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                input("Press Enter to continue...")

if __name__ == "__main__":
    demo = SimplifiedNutriMenuDemo()
    demo.run()