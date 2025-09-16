import pandas as pd
from typing import List, Dict, Optional
import random
from datetime import datetime, timedelta
from nutrition_database import NutritionDatabase
from nutrition_engine import NutritionEngine

class MenuGenerator:
    def __init__(self, database: NutritionDatabase, nutrition_engine: NutritionEngine):
        """Initialize menu generator with database and nutrition engine"""
        self.database = database
        self.nutrition_engine = nutrition_engine
        
        # Daily nutrition targets (example values)
        self.daily_targets = {
            'calories': 2000,
            'protein_g': 150,
            'carbs_g': 250,
            'fat_g': 67,
            'fiber_g': 25,
            'sodium_mg': 2300
        }
        
        # Meal distribution (percentage of daily calories)
        self.meal_distribution = {
            'breakfast': 0.25,
            'lunch': 0.35,
            'dinner': 0.30,
            'snack': 0.10
        }
    
    def set_daily_targets(self, targets: Dict[str, float]):
        """Set custom daily nutrition targets"""
        self.daily_targets.update(targets)
    
    def get_foods_by_meal_type(self, meal_type: str, dietary_preferences: Dict[str, bool] = None) -> pd.DataFrame:
        """Get foods suitable for a specific meal type"""
        all_foods = self.database.get_all_foods()
        
        # Filter by meal type
        meal_foods = all_foods[all_foods['meal_type'] == meal_type]
        
        # Apply dietary preferences if provided
        if dietary_preferences:
            meal_foods = self.nutrition_engine.filter_by_dietary_preferences(meal_foods, dietary_preferences)
        
        return meal_foods
    
    def select_foods_for_meal(self, meal_type: str, target_calories: float, 
                            dietary_preferences: Dict[str, bool] = None, 
                            exclude_foods: List[str] = None) -> List[Dict]:
        """Select foods for a specific meal within calorie target"""
        
        available_foods = self.get_foods_by_meal_type(meal_type, dietary_preferences)
        
        if exclude_foods:
            available_foods = available_foods[~available_foods['food_name'].isin(exclude_foods)]
        
        if available_foods.empty:
            return []
        
        # Calculate nutrition scores for ranking
        nutrition_scores = available_foods.apply(
            lambda row: self.nutrition_engine.calculate_nutrition_score(row.to_dict()), 
            axis=1
        )
        available_foods = available_foods.copy()
        available_foods['nutrition_score'] = nutrition_scores
        
        # Sort by nutrition score
        available_foods = available_foods.sort_values('nutrition_score', ascending=False)
        
        selected_foods = []
        total_calories = 0
        calorie_tolerance = target_calories * 0.2  # 20% tolerance
        
        # Primary selection strategy: try to get close to target calories
        for _, food in available_foods.iterrows():
            food_calories = food['calories']
            
            if total_calories + food_calories <= target_calories + calorie_tolerance:
                food_dict = food.to_dict()
                selected_foods.append(food_dict)
                total_calories += food_calories
                
                # Stop if we're close to target
                if total_calories >= target_calories * 0.8:
                    break
        
        # If we haven't reached minimum calories, add more foods
        if total_calories < target_calories * 0.6 and len(selected_foods) < 3:
            remaining_foods = available_foods[~available_foods['food_name'].isin([f['food_name'] for f in selected_foods])]
            
            for _, food in remaining_foods.head(2).iterrows():
                food_dict = food.to_dict()
                selected_foods.append(food_dict)
                total_calories += food['calories']
        
        return selected_foods
    
    def generate_daily_menu(self, dietary_preferences: Dict[str, bool] = None,
                          custom_targets: Dict[str, float] = None) -> Dict:
        """Generate a complete daily menu"""
        
        if custom_targets:
            targets = {**self.daily_targets, **custom_targets}
        else:
            targets = self.daily_targets.copy()
        
        daily_menu = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'meals': {},
            'daily_totals': {
                'calories': 0,
                'protein_g': 0,
                'carbs_g': 0,
                'fat_g': 0,
                'fiber_g': 0,
                'sodium_mg': 0
            },
            'targets': targets,
            'target_achievement': {}
        }
        
        used_foods = []
        
        # Generate meals
        for meal_type, calorie_percentage in self.meal_distribution.items():
            target_calories = targets['calories'] * calorie_percentage
            
            foods = self.select_foods_for_meal(
                meal_type, 
                target_calories, 
                dietary_preferences, 
                used_foods
            )
            
            # Calculate meal totals
            meal_totals = {
                'calories': sum(food['calories'] for food in foods),
                'protein_g': sum(food['protein_g'] for food in foods),
                'carbs_g': sum(food['carbs_g'] for food in foods),
                'fat_g': sum(food['fat_g'] for food in foods),
                'fiber_g': sum(food['fiber_g'] for food in foods),
                'sodium_mg': sum(food['sodium_mg'] for food in foods)
            }
            
            daily_menu['meals'][meal_type] = {
                'foods': foods,
                'totals': meal_totals,
                'target_calories': target_calories
            }
            
            # Add to daily totals
            for nutrient in daily_menu['daily_totals']:
                daily_menu['daily_totals'][nutrient] += meal_totals[nutrient]
            
            # Add used foods to exclusion list
            used_foods.extend([food['food_name'] for food in foods])
        
        # Calculate target achievement percentages
        for nutrient, target in targets.items():
            actual = daily_menu['daily_totals'].get(nutrient, 0)
            if target > 0:
                achievement = (actual / target) * 100
                daily_menu['target_achievement'][nutrient] = round(achievement, 1)
        
        return daily_menu
    
    def generate_weekly_menu(self, dietary_preferences: Dict[str, bool] = None,
                           custom_targets: Dict[str, float] = None) -> Dict:
        """Generate a weekly menu plan"""
        
        weekly_menu = {
            'week_start': datetime.now().strftime('%Y-%m-%d'),
            'days': {},
            'weekly_summary': {
                'avg_calories': 0,
                'avg_protein': 0,
                'food_variety': 0,
                'target_achievement': {}
            }
        }
        
        all_used_foods = set()
        daily_totals = {
            'calories': 0,
            'protein_g': 0,
            'carbs_g': 0,
            'fat_g': 0,
            'fiber_g': 0,
            'sodium_mg': 0
        }
        
        # Generate menu for each day
        for day in range(7):
            current_date = datetime.now() + timedelta(days=day)
            day_name = current_date.strftime('%A')
            
            daily_menu = self.generate_daily_menu(dietary_preferences, custom_targets)
            daily_menu['date'] = current_date.strftime('%Y-%m-%d')
            daily_menu['day_name'] = day_name
            
            weekly_menu['days'][day_name] = daily_menu
            
            # Collect used foods for variety calculation
            for meal_type, meal_data in daily_menu['meals'].items():
                for food in meal_data['foods']:
                    all_used_foods.add(food['food_name'])
            
            # Add to weekly totals
            for nutrient in daily_totals:
                daily_totals[nutrient] += daily_menu['daily_totals'][nutrient]
        
        # Calculate weekly averages
        weekly_menu['weekly_summary']['avg_calories'] = round(daily_totals['calories'] / 7, 1)
        weekly_menu['weekly_summary']['avg_protein'] = round(daily_totals['protein_g'] / 7, 1)
        weekly_menu['weekly_summary']['food_variety'] = len(all_used_foods)
        
        # Calculate average target achievement
        targets = custom_targets if custom_targets else self.daily_targets
        for nutrient, target in targets.items():
            avg_actual = daily_totals[nutrient] / 7
            if target > 0:
                achievement = (avg_actual / target) * 100
                weekly_menu['weekly_summary']['target_achievement'][nutrient] = round(achievement, 1)
        
        return weekly_menu
    
    def suggest_meal_replacements(self, current_meal: List[str], meal_type: str,
                                dietary_preferences: Dict[str, bool] = None) -> List[Dict]:
        """Suggest alternative foods for a meal"""
        
        # Calculate current meal nutrition
        current_nutrition = {'calories': 0, 'protein_g': 0, 'carbs_g': 0, 'fat_g': 0}
        
        for food_name in current_meal:
            food_data = self.database.get_food_by_name(food_name)
            if food_data is not None:
                for nutrient in current_nutrition:
                    current_nutrition[nutrient] += food_data[nutrient]
        
        # Find alternative foods with similar nutrition profile
        target_calories = current_nutrition['calories']
        alternatives = self.select_foods_for_meal(
            meal_type, 
            target_calories, 
            dietary_preferences, 
            current_meal
        )
        
        # Score alternatives based on nutrition similarity
        scored_alternatives = []
        for food in alternatives:
            similarity_score = self._calculate_nutrition_similarity(
                current_nutrition, 
                food
            )
            food['similarity_score'] = similarity_score
            scored_alternatives.append(food)
        
        # Sort by similarity score
        scored_alternatives.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return scored_alternatives[:5]  # Return top 5 alternatives
    
    def _calculate_nutrition_similarity(self, target_nutrition: Dict, food_data: Dict) -> float:
        """Calculate similarity score between target nutrition and food"""
        
        nutrients = ['calories', 'protein_g', 'carbs_g', 'fat_g']
        total_score = 0
        
        for nutrient in nutrients:
            target_value = target_nutrition.get(nutrient, 0)
            food_value = food_data.get(nutrient, 0)
            
            if target_value > 0:
                # Calculate percentage difference
                diff = abs(target_value - food_value) / target_value
                # Convert to similarity score (1 = identical, 0 = completely different)
                similarity = max(0, 1 - diff)
                total_score += similarity
        
        return total_score / len(nutrients)
    
    def optimize_menu_for_goals(self, menu: Dict, goals: Dict[str, str]) -> Dict:
        """Optimize menu based on specific health goals"""
        
        optimized_menu = menu.copy()
        
        # Goal-based optimization rules
        optimization_rules = {
            'weight_loss': {
                'target_calories_multiplier': 0.8,
                'prefer_high_protein': True,
                'prefer_high_fiber': True
            },
            'muscle_gain': {
                'target_calories_multiplier': 1.2,
                'prefer_high_protein': True,
                'min_protein_per_meal': 20
            },
            'heart_health': {
                'max_sodium_per_day': 1500,
                'prefer_low_sodium': True,
                'prefer_high_fiber': True
            },
            'diabetes_management': {
                'prefer_low_sugar': True,
                'prefer_high_fiber': True,
                'prefer_complex_carbs': True
            }
        }
        
        # Apply optimization rules
        for goal_type, goal_value in goals.items():
            if goal_value and goal_type in optimization_rules:
                rules = optimization_rules[goal_type]
                
                # Adjust calorie targets
                if 'target_calories_multiplier' in rules:
                    multiplier = rules['target_calories_multiplier']
                    optimized_menu['targets']['calories'] *= multiplier
                
                # Apply other rules by regenerating menu with modified preferences
                # This is a simplified implementation
                optimized_menu['optimization_applied'] = goal_type
        
        return optimized_menu