import pandas as pd
from typing import Dict, Optional, List, Union
import json

class NutritionDatabase:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.df.set_index('food_name', inplace=True)
        
        self.scalable_fields = [
            'calories', 'protein_g', 'carbs_g', 'fat_g', 
            'fiber_g', 'sugar_g', 'sodium_mg', 'calcium_mg', 
            'iron_mg', 'vitamin_a_mcg', 'vitamin_c_mg'
        ]
    
    def get_nutrition(self, food_name: str, portion_size: Optional[float] = None) -> Optional[Dict]:
        if food_name not in self.df.index:
            return None
        
        nutrition = self.df.loc[food_name].to_dict()
        
        if portion_size is not None and portion_size > 0:
            serving_size = nutrition['serving_size_g']
            multiplier = portion_size / serving_size
            
            for field in self.scalable_fields:
                if field in nutrition and pd.notna(nutrition[field]):
                    nutrition[field] = round(nutrition[field] * multiplier, 2)
        
        return nutrition
    
    def get_all_foods(self) -> pd.DataFrame:
        """Return all foods as DataFrame"""
        return self.df.reset_index()
    
    def get_food_by_name(self, food_name: str) -> Optional[pd.Series]:
        """Get a specific food by name"""
        if food_name in self.df.index:
            return self.df.loc[food_name]
        return None
    
    def get_multiple_nutrition(self, food_items: List[Dict]) -> List[Dict]:
        results = []
        for item in food_items:
            food_name = item.get('food_name')
            portion_size = item.get('portion_size')
            
            nutrition = self.get_nutrition(food_name, portion_size)
            if nutrition:
                nutrition['requested_portion'] = portion_size
                results.append(nutrition)
        
        return results
    
    def search_foods(self, query: str, limit: int = 10) -> List[str]:
        matching_foods = []
        query_lower = query.lower()
        
        for food_name in self.df.index:
            if query_lower in food_name.lower():
                matching_foods.append(food_name)
        
        return matching_foods[:limit]
    
    def filter_by_criteria(self, criteria: Dict) -> List[str]:
        filtered_df = self.df.copy()
        
        if 'vegetarian' in criteria and criteria['vegetarian']:
            filtered_df = filtered_df[filtered_df['vegetarian'] == True]
        
        if 'vegan' in criteria and criteria['vegan']:
            filtered_df = filtered_df[filtered_df['vegan'] == True]
        
        if 'gluten_free' in criteria and criteria['gluten_free']:
            filtered_df = filtered_df[filtered_df['gluten_free'] == True]
        
        if 'category' in criteria:
            categories = criteria['category'] if isinstance(criteria['category'], list) else [criteria['category']]
            filtered_df = filtered_df[filtered_df['category'].isin(categories)]
        
        if 'meal_type' in criteria:
            meal_types = criteria['meal_type'] if isinstance(criteria['meal_type'], list) else [criteria['meal_type']]
            filtered_df = filtered_df[filtered_df['meal_type'].isin(meal_types)]
        
        if 'max_calories' in criteria:
            filtered_df = filtered_df[filtered_df['calories'] <= criteria['max_calories']]
        
        if 'min_protein' in criteria:
            filtered_df = filtered_df[filtered_df['protein_g'] >= criteria['min_protein']]
        
        if 'max_sodium' in criteria:
            filtered_df = filtered_df[filtered_df['sodium_mg'] <= criteria['max_sodium']]
        
        return filtered_df.index.tolist()
    
    def get_nutrition_summary(self, food_items: List[Dict]) -> Dict:
        total_nutrition = {field: 0 for field in self.scalable_fields}
        total_nutrition['serving_size_g'] = 0
        
        valid_items = []
        
        for item in food_items:
            nutrition = self.get_nutrition(item.get('food_name'), item.get('portion_size'))
            if nutrition:
                valid_items.append({**item, **nutrition})
                
                for field in self.scalable_fields:
                    if field in nutrition and pd.notna(nutrition[field]):
                        total_nutrition[field] += nutrition[field]
                
                if 'portion_size' in item:
                    total_nutrition['serving_size_g'] += item['portion_size']
                else:
                    total_nutrition['serving_size_g'] += nutrition['serving_size_g']
        
        return {
            'total_nutrition': total_nutrition,
            'individual_items': valid_items,
            'item_count': len(valid_items)
        }
    
    def calculate_meal_balance(self, food_items: List[Dict]) -> Dict:
        summary = self.get_nutrition_summary(food_items)
        total_nutrition = summary['total_nutrition']
        
        total_calories = total_nutrition['calories']
        
        if total_calories == 0:
            return {
                'macronutrient_distribution': {'carbs': 0, 'protein': 0, 'fat': 0},
                'balance_score': 0,
                'recommendations': ['No valid food items found']
            }
        
        protein_calories = total_nutrition['protein_g'] * 4
        carb_calories = total_nutrition['carbs_g'] * 4
        fat_calories = total_nutrition['fat_g'] * 9
        
        protein_percent = (protein_calories / total_calories) * 100
        carb_percent = (carb_calories / total_calories) * 100
        fat_percent = (fat_calories / total_calories) * 100
        
        ideal_ranges = {
            'protein': (25, 35),
            'carbs': (35, 50),
            'fat': (20, 35)
        }
        
        balance_scores = []
        recommendations = []
        
        for macro, (min_val, max_val) in ideal_ranges.items():
            if macro == 'protein':
                current = protein_percent
            elif macro == 'carbs':
                current = carb_percent
            else:
                current = fat_percent
            
            if current < min_val:
                score = current / min_val
                recommendations.append(f"Increase {macro} intake")
            elif current > max_val:
                score = max_val / current
                recommendations.append(f"Reduce {macro} intake")
            else:
                score = 1.0
            
            balance_scores.append(score)
        
        overall_balance = sum(balance_scores) / len(balance_scores)
        
        return {
            'macronutrient_distribution': {
                'carbs': round(carb_percent, 1),
                'protein': round(protein_percent, 1),
                'fat': round(fat_percent, 1)
            },
            'balance_score': round(overall_balance, 2),
            'recommendations': recommendations if recommendations else ['Well-balanced meal'],
            'total_nutrition': total_nutrition
        }
    
    def get_food_categories(self) -> Dict:
        categories = self.df['category'].value_counts().to_dict()
        meal_types = self.df['meal_type'].value_counts().to_dict()
        
        return {
            'categories': categories,
            'meal_types': meal_types,
            'dietary_options': {
                'vegetarian': len(self.df[self.df['vegetarian'] == True]),
                'vegan': len(self.df[self.df['vegan'] == True]),
                'gluten_free': len(self.df[self.df['gluten_free'] == True])
            }
        }
    
    def get_random_foods(self, count: int = 5, criteria: Optional[Dict] = None) -> List[str]:
        if criteria:
            filtered_foods = self.filter_by_criteria(criteria)
            available_foods = filtered_foods
        else:
            available_foods = self.df.index.tolist()
        
        if not available_foods:
            return []
        
        return list(pd.Series(available_foods).sample(n=min(count, len(available_foods))))