import pandas as pd
from typing import List, Dict, Optional, Tuple
import numpy as np
from nutrition_database import NutritionDatabase
from nlp_processor import NLPProcessor

class NutritionEngine:
    def __init__(self, database: NutritionDatabase):
        """Initialize nutrition engine with database and NLP processor"""
        self.database = database
        self.nlp_processor = NLPProcessor()
        
        # Nutrition scoring weights
        self.nutrition_weights = {
            'protein_density': 0.3,  # protein per calorie
            'fiber_content': 0.2,
            'vitamin_content': 0.2,
            'low_sodium': 0.15,
            'low_sugar': 0.15
        }
    
    def calculate_nutrition_score(self, food_data: Dict) -> float:
        """Calculate a nutrition score for a food item"""
        try:
            calories = food_data.get('calories', 1)
            protein = food_data.get('protein_g', 0)
            fiber = food_data.get('fiber_g', 0)
            sodium = food_data.get('sodium_mg', 0)
            sugar = food_data.get('sugar_g', 0)
            vitamin_a = food_data.get('vitamin_a_mcg', 0)
            vitamin_c = food_data.get('vitamin_c_mg', 0)
            
            # Calculate component scores (0-1 scale)
            protein_density = min((protein / calories) * 100, 1.0) if calories > 0 else 0
            fiber_score = min(fiber / 10, 1.0)  # 10g fiber = max score
            vitamin_score = min((vitamin_a + vitamin_c) / 500, 1.0)  # Normalized vitamin content
            sodium_score = max(1 - (sodium / 2000), 0)  # Lower sodium = higher score
            sugar_score = max(1 - (sugar / 25), 0)  # Lower sugar = higher score
            
            # Calculate weighted score
            total_score = (
                protein_density * self.nutrition_weights['protein_density'] +
                fiber_score * self.nutrition_weights['fiber_content'] +
                vitamin_score * self.nutrition_weights['vitamin_content'] +
                sodium_score * self.nutrition_weights['low_sodium'] +
                sugar_score * self.nutrition_weights['low_sugar']
            )
            
            return round(total_score * 100, 2)  # Return as percentage
            
        except Exception as e:
            print(f"Error calculating nutrition score: {e}")
            return 0.0
    
    def filter_by_dietary_preferences(self, foods: pd.DataFrame, preferences: Dict[str, bool]) -> pd.DataFrame:
        """Filter foods based on dietary preferences"""
        filtered_foods = foods.copy()
        
        if preferences.get('vegetarian', False):
            filtered_foods = filtered_foods[filtered_foods['vegetarian'] == True]
        
        if preferences.get('vegan', False):
            filtered_foods = filtered_foods[filtered_foods['vegan'] == True]
        
        if preferences.get('gluten_free', False):
            filtered_foods = filtered_foods[filtered_foods['gluten_free'] == True]
        
        if preferences.get('healthy', False):
            # Filter for foods with good nutrition scores
            scores = filtered_foods.apply(lambda row: self.calculate_nutrition_score(row.to_dict()), axis=1)
            filtered_foods = filtered_foods[scores >= 60]  # Top 60% nutrition score
        
        if preferences.get('high_protein', False):
            # Filter for high protein foods (>15g protein per serving)
            filtered_foods = filtered_foods[filtered_foods['protein_g'] >= 15]
        
        return filtered_foods
    
    def filter_by_meal_type(self, foods: pd.DataFrame, meal_type: str) -> pd.DataFrame:
        """Filter foods by meal type"""
        if meal_type == 'any':
            return foods
        
        return foods[foods['meal_type'] == meal_type]
    
    def search_foods_by_name(self, query: str, limit: int = 10) -> List[Dict]:
        """Search foods by name with fuzzy matching"""
        all_foods = self.database.get_all_foods()
        food_names = all_foods['food_name'].tolist()
        
        # Find similar food names
        similar_foods = self.nlp_processor.find_similar_food_names(query, food_names)
        
        results = []
        for food_name, similarity in similar_foods[:limit]:
            food_data = self.database.get_food_by_name(food_name)
            if food_data is not None:
                food_dict = food_data.to_dict()
                food_dict['similarity_score'] = similarity
                food_dict['nutrition_score'] = self.calculate_nutrition_score(food_dict)
                results.append(food_dict)
        
        return results
    
    def get_recommendations(self, query: str, limit: int = 10) -> Tuple[List[Dict], Dict]:
        """Get food recommendations based on natural language query"""
        # Parse the query
        query_analysis = self.nlp_processor.parse_nutrition_query(query)
        
        # Get all foods
        all_foods = self.database.get_all_foods()
        
        # Apply filters
        filtered_foods = all_foods.copy()
        
        # Filter by dietary preferences
        preferences = query_analysis.get('food_preferences', {})
        if any(preferences.values()):
            filtered_foods = self.filter_by_dietary_preferences(filtered_foods, preferences)
        
        # Filter by meal type
        meal_type = query_analysis.get('meal_type', 'any')
        filtered_foods = self.filter_by_meal_type(filtered_foods, meal_type)
        
        # Check if query contains specific food names
        processed_query = query_analysis.get('processed_query', '')
        if len(processed_query.split()) <= 3:  # Likely a food name search
            name_results = self.search_foods_by_name(processed_query, limit)
            if name_results:
                return name_results, query_analysis
        
        # Calculate nutrition scores and sort
        if not filtered_foods.empty:
            nutrition_scores = filtered_foods.apply(
                lambda row: self.calculate_nutrition_score(row.to_dict()), 
                axis=1
            )
            filtered_foods['nutrition_score'] = nutrition_scores
            filtered_foods = filtered_foods.sort_values('nutrition_score', ascending=False)
            
            # Convert to list of dictionaries
            recommendations = []
            for _, row in filtered_foods.head(limit).iterrows():
                food_dict = row.to_dict()
                recommendations.append(food_dict)
            
            return recommendations, query_analysis
        
        return [], query_analysis
    
    def get_nutrition_analysis(self, food_name: str) -> Optional[Dict]:
        """Get detailed nutrition analysis for a specific food"""
        food_data = self.database.get_food_by_name(food_name)
        
        if food_data is None:
            return None
        
        food_dict = food_data.to_dict()
        nutrition_score = self.calculate_nutrition_score(food_dict)
        
        # Create analysis
        analysis = {
            'food_name': food_dict['food_name'],
            'nutrition_score': nutrition_score,
            'calorie_breakdown': {
                'total_calories': food_dict['calories'],
                'protein_calories': food_dict['protein_g'] * 4,
                'carb_calories': food_dict['carbs_g'] * 4,
                'fat_calories': food_dict['fat_g'] * 9
            },
            'macronutrients': {
                'protein_g': food_dict['protein_g'],
                'carbs_g': food_dict['carbs_g'],
                'fat_g': food_dict['fat_g'],
                'fiber_g': food_dict['fiber_g']
            },
            'micronutrients': {
                'sodium_mg': food_dict['sodium_mg'],
                'calcium_mg': food_dict['calcium_mg'],
                'iron_mg': food_dict['iron_mg'],
                'vitamin_a_mcg': food_dict['vitamin_a_mcg'],
                'vitamin_c_mg': food_dict['vitamin_c_mg']
            },
            'dietary_info': {
                'vegetarian': food_dict['vegetarian'],
                'vegan': food_dict['vegan'],
                'gluten_free': food_dict['gluten_free']
            },
            'recommendations': self._generate_nutrition_recommendations(food_dict)
        }
        
        return analysis
    
    def _generate_nutrition_recommendations(self, food_data: Dict) -> List[str]:
        """Generate nutrition recommendations based on food data"""
        recommendations = []
        
        calories = food_data.get('calories', 0)
        protein = food_data.get('protein_g', 0)
        sodium = food_data.get('sodium_mg', 0)
        sugar = food_data.get('sugar_g', 0)
        fiber = food_data.get('fiber_g', 0)
        
        if protein > 20:
            recommendations.append("Excellent source of protein - great for muscle building and satiety")
        elif protein > 10:
            recommendations.append("Good source of protein")
        
        if fiber > 5:
            recommendations.append("High in fiber - supports digestive health")
        elif fiber > 3:
            recommendations.append("Good source of fiber")
        
        if sodium > 800:
            recommendations.append("High in sodium - consider limiting if you have blood pressure concerns")
        elif sodium < 300:
            recommendations.append("Low in sodium - heart-healthy choice")
        
        if sugar > 20:
            recommendations.append("High in sugar - enjoy in moderation")
        elif sugar < 5:
            recommendations.append("Low in sugar - good for blood sugar management")
        
        if calories < 200:
            recommendations.append("Low calorie option - great for weight management")
        elif calories > 400:
            recommendations.append("Higher calorie food - consider portion size")
        
        return recommendations
    
    def compare_foods(self, food_names: List[str]) -> Optional[Dict]:
        """Compare nutrition profiles of multiple foods"""
        if len(food_names) < 2:
            return None
        
        comparison_data = {}
        
        for food_name in food_names:
            food_data = self.database.get_food_by_name(food_name)
            if food_data is not None:
                food_dict = food_data.to_dict()
                food_dict['nutrition_score'] = self.calculate_nutrition_score(food_dict)
                comparison_data[food_name] = food_dict
        
        if len(comparison_data) < 2:
            return None
        
        # Identify best and worst for each nutrient
        nutrients = ['calories', 'protein_g', 'carbs_g', 'fat_g', 'fiber_g', 'sodium_mg']
        comparison_summary = {
            'foods': comparison_data,
            'best_for': {},
            'worst_for': {}
        }
        
        for nutrient in nutrients:
            values = {name: data.get(nutrient, 0) for name, data in comparison_data.items()}
            
            if nutrient == 'sodium_mg':  # Lower is better for sodium
                comparison_summary['best_for'][nutrient] = min(values, key=values.get)
                comparison_summary['worst_for'][nutrient] = max(values, key=values.get)
            else:  # Higher is generally better for other nutrients
                comparison_summary['best_for'][nutrient] = max(values, key=values.get)
                comparison_summary['worst_for'][nutrient] = min(values, key=values.get)
        
        return comparison_summary