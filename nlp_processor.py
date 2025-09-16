import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
from typing import List, Dict, Tuple
import difflib

class NLPProcessor:
    def __init__(self):
        """Initialize NLP processor with pre-trained models"""
        print("🤖 Loading AI models...")
        
        # Intent classification pipeline
        self.intent_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        # Sentiment analysis for preference strength
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        
        # Define possible intents/preferences
        self.dietary_labels = [
            "vegetarian food request",
            "vegan food request", 
            "non-vegetarian food request",
            "gluten-free food request",
            "healthy low-calorie food request",
            "high-protein food request",
            "general food request"
        ]
        
        self.meal_labels = [
            "breakfast request",
            "lunch request", 
            "dinner request",
            "snack request",
            "dessert request"
        ]
        
        self.nutrition_labels = [
            "calorie focused",
            "protein focused",
            "low-carb focused",
            "vitamin focused",
            "fiber focused"
        ]
        
        print("✅ AI models loaded successfully!")
    
    def extract_food_preferences(self, query: str) -> Dict[str, bool]:
        """Extract dietary preferences using AI classification"""
        try:
            # Classify dietary preferences
            dietary_result = self.intent_classifier(query, self.dietary_labels)
            
            preferences = {
                'vegetarian': False,
                'vegan': False,
                'gluten_free': False,
                'healthy': False,
                'high_protein': False,
                'non_vegetarian': False
            }
            
            # Get top prediction with confidence
            top_prediction = dietary_result['labels'][0]
            confidence = dietary_result['scores'][0]
            
            # Map predictions to preferences (only if confidence > 0.3)
            if confidence > 0.3:
                if "vegetarian" in top_prediction and "non-vegetarian" not in top_prediction:
                    preferences['vegetarian'] = True
                elif "vegan" in top_prediction:
                    preferences['vegan'] = True
                elif "non-vegetarian" in top_prediction:
                    preferences['non_vegetarian'] = True
                elif "gluten-free" in top_prediction:
                    preferences['gluten_free'] = True
                elif "healthy" in top_prediction or "low-calorie" in top_prediction:
                    preferences['healthy'] = True
                elif "high-protein" in top_prediction:
                    preferences['high_protein'] = True
            
            return preferences
            
        except Exception as e:
            print(f"⚠️ AI model error, falling back to keyword matching: {e}")
            return self._fallback_preference_extraction(query)
    
    def extract_meal_type(self, query: str) -> str:
        """Extract meal type using AI classification"""
        try:
            result = self.intent_classifier(query, self.meal_labels)
            
            # Get top prediction
            top_prediction = result['labels'][0]
            confidence = result['scores'][0]
            
            if confidence > 0.3:
                for meal in ['breakfast', 'lunch', 'dinner', 'snack', 'dessert']:
                    if meal in top_prediction:
                        return meal
            
            return 'any'
            
        except Exception as e:
            print(f"⚠️ AI model error: {e}")
            return self._fallback_meal_extraction(query)
    
    def extract_nutrition_focus(self, query: str) -> List[str]:
        """Extract nutrition focus using AI"""
        try:
            result = self.intent_classifier(query, self.nutrition_labels)
            
            focus_areas = []
            for label, score in zip(result['labels'], result['scores']):
                if score > 0.3:  # Confidence threshold
                    if "calorie" in label:
                        focus_areas.append("calories")
                    elif "protein" in label:
                        focus_areas.append("protein")
                    elif "carb" in label:
                        focus_areas.append("carbs")
                    elif "vitamin" in label:
                        focus_areas.append("vitamins")
                    elif "fiber" in label:
                        focus_areas.append("fiber")
            
            return focus_areas
            
        except Exception as e:
            print(f"⚠️ AI model error: {e}")
            return []
    
    def parse_nutrition_query(self, query: str) -> Dict[str, any]:
        """Enhanced query parsing with AI"""
        try:
            preferences = self.extract_food_preferences(query)
            meal_type = self.extract_meal_type(query)
            nutrition_focus = self.extract_nutrition_focus(query)
            
            return {
                'food_preferences': preferences,
                'meal_type': meal_type,
                'nutrition_focus': nutrition_focus,
                'original_query': query,
                'processed_query': query.lower().strip()
            }
            
        except Exception as e:
            print(f"⚠️ Error in AI processing: {e}")
            return self._fallback_query_parsing(query)
    
    def generate_response_text(self, food_recommendations: List[Dict], query_analysis: Dict) -> str:
        """Generate natural response using the recommendations"""
        if not food_recommendations:
            return "I couldn't find any foods matching your criteria. Please try a different search."
        
        response_parts = []
        
        # Analyze the query sentiment for response tone
        meal_type = query_analysis.get('meal_type', 'any')
        preferences = query_analysis.get('food_preferences', {})
        
        # Generate contextual greeting
        if meal_type != 'any':
            response_parts.append(f"Here are some excellent {meal_type} recommendations:")
        else:
            response_parts.append("Here are some great food suggestions for you:")
        
        # Add top recommendations with smart descriptions
        for i, food in enumerate(food_recommendations[:3], 1):
            food_name = food['food_name'].replace('_', ' ').title()
            calories = food['calories']
            protein = food['protein_g']
            
            # Add contextual information based on preferences
            description = f"{food_name} - {calories} calories, {protein}g protein"
            
            if preferences.get('healthy', False) and calories < 200:
                description += " (low-calorie option)"
            elif preferences.get('high_protein', False) and protein > 15:
                description += " (high-protein)"
            
            response_parts.append(f"{i}. {description}")
        
        # Add dietary confirmation
        active_prefs = [pref.replace('_', ' ') for pref, active in preferences.items() if active]
        if active_prefs:
            if len(active_prefs) == 1:
                response_parts.append(f"All options are {active_prefs[0]}.")
            else:
                response_parts.append(f"All options meet your {', '.join(active_prefs)} requirements.")
        
        return '\n'.join(response_parts)
    
    # Fallback methods for when AI models fail
    def _fallback_preference_extraction(self, query: str) -> Dict[str, bool]:
        """Simple keyword-based fallback"""
        query_lower = query.lower()
        
        return {
            'vegetarian': any(word in query_lower for word in ['vegetarian', 'veg only', 'veggie']),
            'vegan': any(word in query_lower for word in ['vegan', 'plant-based']),
            'gluten_free': any(word in query_lower for word in ['gluten-free', 'gluten free']),
            'healthy': any(word in query_lower for word in ['healthy', 'light', 'low-calorie']),
            'high_protein': any(word in query_lower for word in ['protein', 'high-protein']),
            'non_vegetarian': any(word in query_lower for word in ['non-veg', 'nonveg', 'meat', 'chicken', 'fish'])
        }
    
    def _fallback_meal_extraction(self, query: str) -> str:
        """Simple keyword-based meal type extraction"""
        query_lower = query.lower()
        for meal in ['breakfast', 'lunch', 'dinner', 'snack', 'dessert']:
            if meal in query_lower:
                return meal
        return 'any'
    
    def _fallback_query_parsing(self, query: str) -> Dict[str, any]:
        """Fallback query parsing"""
        return {
            'food_preferences': self._fallback_preference_extraction(query),
            'meal_type': self._fallback_meal_extraction(query),
            'nutrition_focus': [],
            'original_query': query,
            'processed_query': query.lower().strip()
        }
    
    def find_similar_food_names(self, query: str, food_names: List[str], threshold: float = 0.6) -> List[Tuple[str, float]]:
        """Find similar food names using fuzzy matching"""
        query_clean = re.sub(r'[^a-zA-Z0-9\s]', '', query.lower())
        
        matches = []
        for food_name in food_names:
            food_clean = food_name.replace('_', ' ').lower()
            similarity = difflib.SequenceMatcher(None, query_clean, food_clean).ratio()
            
            if similarity >= threshold:
                matches.append((food_name, similarity))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)[:10]