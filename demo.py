import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'app'))

from nutrition_database import NutritionDatabase
from nutrition_engine import NutritionEngine
from menu_generator import MenuGenerator
from food_image_recognizer import FoodImageRecognizer
import json

# Simplified NLP processor to avoid heavy model loading
class SimplifiedNLPProcessor:
    def __init__(self):
        """Initialize simplified NLP processor with keyword matching"""
        print("🤖 Loading simplified NLP processor...")
        
        # Define keyword mappings
        self.dietary_keywords = {
            'vegetarian': ['vegetarian', 'veg', 'veggie', 'no meat', 'plant based'],
            'vegan': ['vegan', 'plant-based', 'no dairy', 'no animal products'],
            'gluten_free': ['gluten free', 'gluten-free', 'no gluten', 'celiac'],
            'healthy': ['healthy', 'light', 'low calorie', 'diet', 'nutritious'],
            'high_protein': ['protein', 'high protein', 'muscle', 'workout'],
            'low_carb': ['low carb', 'keto', 'ketogenic', 'no carbs']
        }
        
        self.meal_keywords = {
            'breakfast': ['breakfast', 'morning', 'first meal'],
            'lunch': ['lunch', 'midday', 'afternoon meal'],
            'dinner': ['dinner', 'evening', 'night meal', 'supper'],
            'snack': ['snack', 'light bite', 'quick eat'],
            'dessert': ['dessert', 'sweet', 'treat', 'after meal']
        }
        
        print("✅ Simplified NLP processor ready!")
    
    def extract_food_preferences(self, query: str) -> dict:
        """Extract dietary preferences using keyword matching"""
        query_lower = query.lower()
        preferences = {}
        
        for pref, keywords in self.dietary_keywords.items():
            preferences[pref] = any(keyword in query_lower for keyword in keywords)
        
        return preferences
    
    def extract_meal_type(self, query: str) -> str:
        """Extract meal type using keyword matching"""
        query_lower = query.lower()
        
        for meal, keywords in self.meal_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return meal
        
        return 'any'
    
    def parse_nutrition_query(self, query: str) -> dict:
        """Parse nutrition query using simplified methods"""
        return {
            'food_preferences': self.extract_food_preferences(query),
            'meal_type': self.extract_meal_type(query),
            'nutrition_focus': [],
            'original_query': query,
            'processed_query': query.lower().strip()
        }
    
    def generate_response_text(self, food_recommendations: list, query_analysis: dict) -> str:
        """Generate simple response text"""
        if not food_recommendations:
            return "I couldn't find any foods matching your criteria. Please try a different search."
        
        meal_type = query_analysis.get('meal_type', 'any')
        preferences = query_analysis.get('food_preferences', {})
        
        # Generate response
        if meal_type != 'any':
            response = f"Here are some great {meal_type} recommendations:\n"
        else:
            response = "Here are some food suggestions for you:\n"
        
        # Add top 3 recommendations
        for i, food in enumerate(food_recommendations[:3], 1):
            food_name = food['food_name'].replace('_', ' ').title()
            response += f"{i}. {food_name} - {food['calories']} calories, {food['protein_g']}g protein\n"
        
        # Add dietary confirmation
        active_prefs = [pref.replace('_', ' ') for pref, active in preferences.items() if active]
        if active_prefs:
            response += f"\nAll options are {', '.join(active_prefs)}."
        
        return response

class EnhancedNutriMenuDemo:
    def __init__(self):
        """Initialize the enhanced demo with all components including image recognition"""
        print("🍽️ Initializing Enhanced NutriMenu AI Demo...")
        
        # Initialize existing components
        self.nutrition_db = NutritionDatabase('backend/data/indian_food_nutrition.csv')
        self.nutrition_engine = NutritionEngine(self.nutrition_db)
        self.menu_generator = MenuGenerator(self.nutrition_db, self.nutrition_engine)
        
        # Use simplified NLP processor
        self.nlp_processor = SimplifiedNLPProcessor()
        
        # Initialize image recognition with your trained model
        try:
            self.image_recognizer = FoodImageRecognizer(
                self.nutrition_db,
                model_path='backend/ml_models/final_restarted_model.keras',
                label_encoder_path='backend/ml_models/label_encoder.pkl'
            )
            
            # Test model alignment
            alignment = self.image_recognizer.validate_food_database_alignment()
            print(f"📊 Model-Database Alignment: {alignment['alignment_percentage']:.1f}%")
            self.image_recognition_available = True
            
        except Exception as e:
            print(f"⚠️ Image recognition not available: {e}")
            print("📸 Demo will run without image recognition features")
            self.image_recognition_available = False
        
        print("✅ All components loaded successfully!\n")
    
    def display_menu(self):
        """Display the enhanced menu options"""
        print("=" * 60)
        print("🍽️  ENHANCED NUTRIMENU AI DEMO")
        print("=" * 60)
        print("1. 🔍 Search Foods by Name")
        print("2. 💬 Natural Language Food Query")
        print("3. 📊 Get Nutrition Analysis")
        print("4. 🍱 Generate Daily Menu")
        print("5. 📅 Generate Weekly Menu")
        print("6. ⚖️  Compare Foods")
        print("7. 🎲 Random Food Suggestions")
        print("8. 📋 Browse Food Categories")
        
        if self.image_recognition_available:
            print("9. 📸 Recognize Food from Image")
            print("10. 📷 Capture Food with Camera")
            print("11. 🔗 Analyze Food from URL")
            print("12. 📁 Batch Analyze Images")
            print("13. ❌ Exit")
        else:
            print("9. ❌ Exit")
        print("=" * 60)
    
    def search_foods(self):
        """Search foods by name"""
        print("\n🔍 FOOD SEARCH")
        query = input("Enter food name to search: ").strip()
        
        if not query:
            print("❌ Please enter a food name!")
            return
        
        results = self.nutrition_engine.search_foods_by_name(query, limit=5)
        
        if results:
            print(f"\n✅ Found {len(results)} matching foods:")
            for i, food in enumerate(results, 1):
                print(f"{i}. {food['food_name'].replace('_', ' ').title()}")
                print(f"   📊 {food['calories']} cal | {food['protein_g']}g protein | {food['carbs_g']}g carbs")
                print(f"   🏆 Nutrition Score: {food['nutrition_score']}/100")
                if 'similarity_score' in food:
                    print(f"   🎯 Similarity: {food['similarity_score']:.2%}")
                print()
        else:
            print("❌ No foods found matching your search!")
    
    def natural_language_query(self):
        """Process natural language food queries with examples"""
        print("\n💬 NATURAL LANGUAGE QUERY")
        print("\n📝 Try these example queries:")
        print("1. 'I want high protein vegetarian breakfast'")
        print("2. 'Show me healthy low calorie dinner options'")
        print("3. 'Find vegan gluten-free snacks'")
        print("4. 'Suggest protein rich lunch foods'")
        print("5. 'I need healthy breakfast options'")
        
        # Provide quick selection option
        example_choice = input("\nSelect example (1-5) or type custom query: ").strip()
        
        example_queries = {
            '1': 'I want high protein vegetarian breakfast',
            '2': 'Show me healthy low calorie dinner options', 
            '3': 'Find vegan gluten-free snacks',
            '4': 'Suggest protein rich lunch foods',
            '5': 'I need healthy breakfast options'
        }
        
        if example_choice in example_queries:
            query = example_queries[example_choice]
            print(f"📝 Using example: '{query}'")
        else:
            query = example_choice if example_choice else input("Enter your food request: ").strip()
        
        if not query:
            print("❌ Please enter a query!")
            return
        
        recommendations, analysis = self.nutrition_engine.get_recommendations(query, limit=5)
        
        if recommendations:
            print(f"\n✅ Query Analysis:")
            print(f"   🎯 Meal Type: {analysis['meal_type']}")
            preferences = analysis['food_preferences']
            active_prefs = [pref for pref, active in preferences.items() if active]
            print(f"   🥗 Preferences: {', '.join(active_prefs) if active_prefs else 'None'}")
            
            response = self.nlp_processor.generate_response_text(recommendations, analysis)
            print(f"\n🤖 AI Response:\n{response}")
            
            print(f"\n📋 Detailed Results:")
            for i, food in enumerate(recommendations, 1):
                print(f"\n{i}. {food['food_name'].replace('_', ' ').title()}")
                print(f"   📊 {food['calories']} calories | {food['protein_g']}g protein")
                print(f"   🥗 Category: {food['category']} | Meal: {food['meal_type']}")
                print(f"   ✅ Vegetarian: {food['vegetarian']} | Vegan: {food['vegan']} | Gluten-Free: {food['gluten_free']}")
        else:
            print("❌ No foods found matching your criteria!")
    
    def nutrition_analysis(self):
        """Get detailed nutrition analysis for a food with examples"""
        print("\n📊 NUTRITION ANALYSIS")
        print("\n📝 Example food names to try:")
        print("• palak_paneer")
        print("• biryani") 
        print("• dal_makhani")
        print("• chicken_tikka")
        print("• aloo_gobi")
        
        food_name = input("\nEnter exact food name (use underscores for spaces): ").strip()
        
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
            macros = analysis['macronutrients']
            print(f"\n🥗 Macronutrients:")
            print(f"   - Protein: {macros['protein_g']}g")
            print(f"   - Carbs: {macros['carbs_g']}g")
            print(f"   - Fat: {macros['fat_g']}g")
            print(f"   - Fiber: {macros['fiber_g']}g")
            
            # Dietary info
            dietary = analysis['dietary_info']
            print(f"\n🏷️  Dietary Information:")
            print(f"   - Vegetarian: {'✅' if dietary['vegetarian'] else '❌'}")
            print(f"   - Vegan: {'✅' if dietary['vegan'] else '❌'}")
            print(f"   - Gluten-Free: {'✅' if dietary['gluten_free'] else '❌'}")
            
            # Nutrition score
            print(f"\n🏆 Nutrition Score: {analysis['nutrition_score']}/100")
            
            # Recommendations
            if analysis.get('recommendations'):
                print(f"\n💡 Recommendations:")
                for rec in analysis['recommendations']:
                    print(f"   • {rec}")
        else:
            print("❌ Food not found! Please check the spelling or try one of the examples above.")
    
    # Keep all your other methods (generate_daily_menu, etc.) exactly the same
    def generate_daily_menu(self):
        """Generate a daily menu"""
        print("\n🍱 DAILY MENU GENERATOR")
        
        # Get dietary preferences
        print("Set your dietary preferences (press Enter to skip):")
        preferences = {}
        
        veg = input("Vegetarian only? (y/n): ").strip().lower()
        if veg == 'y':
            preferences['vegetarian'] = True
        
        vegan = input("Vegan only? (y/n): ").strip().lower()
        if vegan == 'y':
            preferences['vegan'] = True
        
        gluten = input("Gluten-free only? (y/n): ").strip().lower()
        if gluten == 'y':
            preferences['gluten_free'] = True
        
        healthy = input("Healthy foods only? (y/n): ").strip().lower()
        if healthy == 'y':
            preferences['healthy'] = True
        
        # Get calorie target
        try:
            calories = input("Daily calorie target (default 2000): ").strip()
            if calories:
                custom_targets = {'calories': float(calories)}
            else:
                custom_targets = None
        except ValueError:
            custom_targets = None
        
        print("\n🔄 Generating your daily menu...")
        menu = self.menu_generator.generate_daily_menu(
            dietary_preferences=preferences if preferences else None,
            custom_targets=custom_targets
        )
        
        self.display_daily_menu(menu)
    
    def display_daily_menu(self, menu):
        """Display the generated daily menu"""
        print(f"\n🍱 DAILY MENU FOR {menu['date']}")
        print("=" * 50)
        
        for meal_type, meal_data in menu['meals'].items():
            print(f"\n🍽️  {meal_type.upper()}")
            print("-" * 20)
            
            if meal_data['foods']:
                for food in meal_data['foods']:
                    print(f"• {food['food_name'].replace('_', ' ').title()}")
                    print(f"  📊 {food['calories']} cal | {food['protein_g']}g protein")
                
                totals = meal_data['totals']
                print(f"\n📊 Meal Total: {totals['calories']} calories")
                print(f"🎯 Target: {meal_data['target_calories']:.0f} calories")
            else:
                print("❌ No foods available for this meal")
        
        # Daily summary
        print(f"\n📊 DAILY SUMMARY")
        print("=" * 30)
        totals = menu['daily_totals']
        targets = menu['targets']
        
        print(f"🔥 Total Calories: {totals['calories']:.0f} / {targets['calories']:.0f}")
        print(f"🥩 Protein: {totals['protein_g']:.1f}g / {targets['protein_g']:.1f}g")
        print(f"🍞 Carbs: {totals['carbs_g']:.1f}g / {targets['carbs_g']:.1f}g")
        print(f"🥑 Fat: {totals['fat_g']:.1f}g / {targets['fat_g']:.1f}g")
        
        # Achievement percentages
        achievement = menu['target_achievement']
        print(f"\n🎯 TARGET ACHIEVEMENT:")
        for nutrient, percent in achievement.items():
            status = "✅" if 80 <= percent <= 120 else "⚠️" if percent > 120 else "❌"
            print(f"   {nutrient}: {percent}% {status}")
    
    def generate_weekly_menu(self):
        """Generate a weekly menu"""
        print("\n📅 WEEKLY MENU GENERATOR")
        print("⏳ This may take a moment...")
        
        # Simple preferences for demo
        preferences = {}
        veg = input("Vegetarian meals only? (y/n): ").strip().lower()
        if veg == 'y':
            preferences['vegetarian'] = True
        
        weekly_menu = self.menu_generator.generate_weekly_menu(
            dietary_preferences=preferences if preferences else None
        )
        
        print(f"\n📅 WEEKLY MENU STARTING {weekly_menu['week_start']}")
        print("=" * 50)
        
        for day_name, daily_menu in weekly_menu['days'].items():
            print(f"\n📅 {day_name.upper()}")
            print("-" * 20)
            
            for meal_type, meal_data in daily_menu['meals'].items():
                if meal_data['foods']:
                    foods = [food['food_name'].replace('_', ' ').title() for food in meal_data['foods']]
                    print(f"{meal_type.capitalize()}: {', '.join(foods)}")
        
        # Weekly summary
        summary = weekly_menu['weekly_summary']
        print(f"\n📊 WEEKLY SUMMARY")
        print("=" * 30)
        print(f"📊 Average Daily Calories: {summary['avg_calories']}")
        print(f"🥩 Average Daily Protein: {summary['avg_protein']}g")
        print(f"🎨 Food Variety: {summary['food_variety']} different foods")
    
    def compare_foods(self):
        """Compare multiple foods"""
        print("\n⚖️  FOOD COMPARISON")
        print("Enter 2-4 food names to compare (use underscores for spaces)")
        
        foods = []
        for i in range(4):
            food = input(f"Food {i+1} (press Enter to finish): ").strip()
            if not food:
                break
            foods.append(food)
        
        if len(foods) < 2:
            print("❌ Please enter at least 2 foods to compare!")
            return
        
        comparison = self.nutrition_engine.compare_foods(foods)
        
        if comparison:
            print(f"\n⚖️  COMPARISON RESULTS")
            print("=" * 40)
            
            # Display each food's stats
            for food_name, data in comparison['foods'].items():
                print(f"\n🍽️  {food_name.replace('_', ' ').title()}")
                print(f"   📊 {data['calories']} cal | {data['protein_g']}g protein")
                print(f"   🏆 Nutrition Score: {data['nutrition_score']}/100")
            
            # Best/worst for each nutrient
            print(f"\n🏆 BEST FOR:")
            for nutrient, food in comparison['best_for'].items():
                print(f"   {nutrient}: {food.replace('_', ' ').title()}")
            
            print(f"\n⚠️  HIGHEST IN:")
            for nutrient, food in comparison['worst_for'].items():
                if nutrient == 'sodium_mg':  # Higher sodium is worse
                    print(f"   {nutrient}: {food.replace('_', ' ').title()}")
        else:
            print("❌ Couldn't compare foods. Please check the food names!")
    
    def random_suggestions(self):
        """Get random food suggestions"""
        print("\n🎲 RANDOM FOOD SUGGESTIONS")
        
        # Filter options
        print("Filter by category (optional):")
        categories = self.nutrition_db.get_food_categories()['categories']
        print("Available categories:", list(categories.keys()))
        
        category = input("Enter category (or press Enter for all): ").strip()
        
        criteria = {}
        if category and category in categories:
            criteria['category'] = category
        
        # Get random foods
        random_foods = self.nutrition_db.get_random_foods(count=5, criteria=criteria if criteria else None)
        
        if random_foods:
            print(f"\n🎲 Here are {len(random_foods)} random suggestions:")
            for i, food_name in enumerate(random_foods, 1):
                food_data = self.nutrition_db.get_nutrition(food_name)
                print(f"{i}. {food_name.replace('_', ' ').title()}")
                print(f"   📊 {food_data['calories']} cal | {food_data['category']} | {food_data['meal_type']}")
        else:
            print("❌ No foods found for the selected criteria!")
    
    def browse_categories(self):
        """Browse food categories"""
        print("\n📋 FOOD CATEGORIES")
        
        categories_info = self.nutrition_db.get_food_categories()
        
        print("\n🥘 FOOD CATEGORIES:")
        for category, count in categories_info['categories'].items():
            print(f"   {category}: {count} foods")
        
        print("\n🍽️  MEAL TYPES:")
        for meal_type, count in categories_info['meal_types'].items():
            print(f"   {meal_type}: {count} foods")
        
        print("\n🏷️  DIETARY OPTIONS:")
        dietary = categories_info['dietary_options']
        print(f"   Vegetarian: {dietary['vegetarian']} foods")
        print(f"   Vegan: {dietary['vegan']} foods")
        print(f"   Gluten-Free: {dietary['gluten_free']} foods")
        
        # Browse specific category
        category = input("\nEnter category name to browse foods (or press Enter to skip): ").strip()
        if category:
            filtered_foods = self.nutrition_db.filter_by_criteria({'category': category})
            if filtered_foods:
                print(f"\n📋 Foods in '{category}' category:")
                for food in filtered_foods[:10]:  # Show first 10
                    print(f"   • {food.replace('_', ' ').title()}")
                if len(filtered_foods) > 10:
                    print(f"   ... and {len(filtered_foods) - 10} more")
            else:
                print(f"❌ No foods found in '{category}' category!")
    
    # Include all your other existing methods here...
    # (I'll keep them the same to save space)
    
    # Image recognition methods (only if available)
    def recognize_food_from_image(self):
        """Recognize food from image file"""
        if not self.image_recognition_available:
            print("❌ Image recognition not available!")
            return
            
        print("\n📸 FOOD IMAGE RECOGNITION")
        print("Supported formats: JPG, PNG, JPEG")
        
        image_path = input("Enter image file path: ").strip()
        
        if not image_path:
            print("❌ Please enter an image path!")
            return
        
        print("🔄 Analyzing image...")
        analysis = self.image_recognizer.analyze_food_image(image_path)
        
        if analysis['success']:
            print(f"\n✅ FOOD DETECTED: {analysis['detected_food']}")
            print(f"🎯 Confidence: {analysis['confidence']}")
            
            # Show nutrition info
            nutrition = analysis['nutrition_info']
            print(f"\n📊 NUTRITION INFORMATION:")
            print(f"   🔥 Calories: {nutrition['calories']}")
            print(f"   🥩 Protein: {nutrition['protein_g']}g")
            print(f"   🍞 Carbs: {nutrition['carbs_g']}g")
            print(f"   🥑 Fat: {nutrition['fat_g']}g")
        else:
            print(f"\n❌ {analysis['message']}")
    
    def run(self):
        """Run the enhanced demo"""
        while True:
            try:
                self.display_menu()
                
                if self.image_recognition_available:
                    choice = input("\nEnter your choice (1-13): ").strip()
                    max_choice = 13
                else:
                    choice = input("\nEnter your choice (1-9): ").strip()
                    max_choice = 9
                
                if choice == '1':
                    self.search_foods()
                elif choice == '2':
                    self.natural_language_query()
                elif choice == '3':
                    self.nutrition_analysis()
                elif choice == '4':
                    self.generate_daily_menu()
                elif choice == '5':
                    self.generate_weekly_menu()  # Fixed: Call actual method
                elif choice == '6':
                    self.compare_foods()  # Fixed: Call actual method
                elif choice == '7':
                    self.random_suggestions()  # Fixed: Call actual method
                elif choice == '8':
                    self.browse_categories()  # Fixed: Call actual method
                elif choice == '9' and self.image_recognition_available:
                    self.recognize_food_from_image()
                elif (choice == '9' and not self.image_recognition_available) or (choice == '13' and self.image_recognition_available):
                    print("\n👋 Thank you for using Enhanced NutriMenu AI Demo!")
                    break
                else:
                    print(f"❌ Invalid choice! Please enter 1-{max_choice}.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Demo interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                input("Press Enter to continue...")

if __name__ == "__main__":
    demo = EnhancedNutriMenuDemo()
    demo.run()