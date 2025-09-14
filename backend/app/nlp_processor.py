import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
from typing import List, Dict, Tuple
import difflib

class NLPProcessor:
    def __init__(self):
        """Initialize NLP processor with required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Food-related keywords
        self.food_keywords = {
            'vegetarian': ['vegetarian', 'veg', 'veggie'],
            'vegan': ['vegan', 'plant-based'],
            'gluten_free': ['gluten-free', 'gluten free', 'no gluten'],
            'healthy': ['healthy', 'nutritious', 'low-calorie', 'diet'],
            'protein': ['protein', 'high-protein'],
            'dessert': ['sweet', 'dessert', 'cake', 'pudding'],
            'spicy': ['spicy', 'hot', 'chili'],
            'meal_type': ['breakfast', 'lunch', 'dinner', 'snack']
        }
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text input"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces and hyphens
        text = re.sub(r'[^a-zA-Z0-9\s\-]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_stem(self, text: str) -> List[str]:
        """Tokenize text and apply stemming"""
        tokens = word_tokenize(text)
        
        # Remove stopwords and apply stemming
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                stemmed = self.stemmer.stem(token)
                processed_tokens.append(stemmed)
        
        return processed_tokens
    
    def extract_food_preferences(self, query: str) -> Dict[str, bool]:
        """Extract dietary preferences from user query"""
        processed_query = self.preprocess_text(query)
        preferences = {
            'vegetarian': False,
            'vegan': False,
            'gluten_free': False,
            'healthy': False,
            'high_protein': False
        }
        
        for pref, keywords in self.food_keywords.items():
            for keyword in keywords:
                if keyword in processed_query:
                    if pref == 'protein':
                        preferences['high_protein'] = True
                    else:
                        preferences[pref] = True
                    break
        
        return preferences
    
    def extract_meal_type(self, query: str) -> str:
        """Extract meal type from query"""
        processed_query = self.preprocess_text(query)
        
        meal_types = ['breakfast', 'lunch', 'dinner', 'snack', 'dessert']
        for meal in meal_types:
            if meal in processed_query:
                return meal
        
        return 'any'
    
    def find_similar_food_names(self, query: str, food_names: List[str], threshold: float = 0.6) -> List[Tuple[str, float]]:
        """Find food names similar to the query using fuzzy matching"""
        processed_query = self.preprocess_text(query)
        
        matches = []
        for food_name in food_names:
            # Calculate similarity ratio
            similarity = difflib.SequenceMatcher(None, processed_query, food_name.replace('_', ' ')).ratio()
            
            if similarity >= threshold:
                matches.append((food_name, similarity))
        
        # Sort by similarity score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:10]  # Return top 10 matches
    
    def parse_nutrition_query(self, query: str) -> Dict[str, any]:
        """Parse user query to extract nutrition-related information"""
        processed_query = self.preprocess_text(query)
        
        result = {
            'food_preferences': self.extract_food_preferences(query),
            'meal_type': self.extract_meal_type(query),
            'nutrition_focus': [],
            'original_query': query,
            'processed_query': processed_query
        }
        
        # Extract nutrition focus
        nutrition_terms = {
            'calories': ['calorie', 'energy'],
            'protein': ['protein'],
            'carbs': ['carb', 'carbohydrate'],
            'fat': ['fat'],
            'fiber': ['fiber', 'fibre'],
            'vitamin': ['vitamin'],
            'calcium': ['calcium'],
            'iron': ['iron']
        }
        
        for nutrient, terms in nutrition_terms.items():
            for term in terms:
                if term in processed_query:
                    result['nutrition_focus'].append(nutrient)
                    break
        
        return result
    
    def generate_response_text(self, food_recommendations: List[Dict], query_analysis: Dict) -> str:
        """Generate natural language response based on recommendations"""
        if not food_recommendations:
            return "I couldn't find any foods matching your criteria. Please try a different search."
        
        response_parts = []
        
        # Add greeting based on meal type
        meal_type = query_analysis.get('meal_type', 'any')
        if meal_type != 'any':
            response_parts.append(f"Here are some great {meal_type} options for you:")
        else:
            response_parts.append("Here are some nutritious food recommendations:")
        
        # Add top recommendations
        for i, food in enumerate(food_recommendations[:3], 1):
            food_name = food['food_name'].replace('_', ' ').title()
            calories = food['calories']
            protein = food['protein_g']
            response_parts.append(f"{i}. {food_name} - {calories} calories, {protein}g protein")
        
        # Add dietary preference confirmation
        preferences = query_analysis.get('food_preferences', {})
        active_prefs = [pref for pref, active in preferences.items() if active]
        if active_prefs:
            pref_text = ', '.join(active_prefs).replace('_', ' ')
            response_parts.append(f"All recommendations are {pref_text}.")
        
        return '\n'.join(response_parts)