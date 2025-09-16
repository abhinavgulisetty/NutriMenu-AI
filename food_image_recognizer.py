import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import cv2
import io
import os
from typing import List, Tuple, Dict, Optional

class FoodImageRecognizer:
    def __init__(self, nutrition_database, model_path='backend/ml_models/final_restarted_model.keras', 
                 label_encoder_path='backend/ml_models/label_encoder.pkl'):
        """Initialize food image recognition system with your trained model"""
        print("🔄 Loading your trained food recognition model...")
        
        self.nutrition_db = nutrition_database
        
        # Load your trained model
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"✅ Model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Load label encoder
        try:
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"✅ Label encoder loaded with {len(self.label_encoder.classes_)} classes")
        except Exception as e:
            print(f"⚠️ Warning: Could not load label encoder: {e}")
            # Create a fallback label encoder based on your database
            self.label_encoder = self._create_fallback_label_encoder()
        
        # Model input shape (based on EfficientNet typical input)
        self.input_shape = (331, 331)  # Update this based on your model's input
        
        # Get available food classes
        self.available_foods = self._get_available_foods()
        
        print("✅ Food recognition system ready!")
    
    def _create_fallback_label_encoder(self):
        """Create label encoder from database if file not found"""
        print("🔄 Creating label encoder from database...")
        
        class LabelEncoder:
            def __init__(self, classes):
                self.classes_ = np.array(classes)
                self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            
            def inverse_transform(self, indices):
                return [self.classes_[idx] for idx in indices]
        
        # Get all unique food names from database
        all_foods = self.nutrition_db.get_all_foods()
        food_names = sorted(all_foods['food_name'].unique())
        
        return LabelEncoder(food_names)
    
    def _get_available_foods(self) -> List[str]:
        """Get list of available food classes"""
        return self.label_encoder.classes_.tolist()
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for your model"""
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize(self.input_shape)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Preprocess based on your model's training
        # EfficientNet preprocessing
        processed_image = tf.keras.applications.efficientnet.preprocess_input(image_array)
        
        return processed_image
    
    def predict_food(self, image: Image.Image, top_k: int = 5) -> List[Dict]:
        """Predict food from image using your trained model"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Add batch dimension
            batch_image = np.expand_dims(processed_image, axis=0)
            
            # Make prediction
            predictions = self.model.predict(batch_image, verbose=0)
            
            # Get top k predictions
            top_indices = np.argsort(predictions[0])[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                # Get food name from label encoder
                food_name = self.label_encoder.inverse_transform([idx])[0]
                confidence = float(predictions[0][idx])
                
                # Get nutrition data
                nutrition_data = self.nutrition_db.get_nutrition(food_name)
                
                if nutrition_data:
                    result = {
                        'food_name': food_name,
                        'confidence': confidence,
                        'confidence_percentage': round(confidence * 100, 2),
                        **nutrition_data
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return []
    
    def recognize_food_from_path(self, image_path: str, top_k: int = 5) -> List[Dict]:
        """Recognize food from image file path"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Load and preprocess image
            image = Image.open(image_path)
            return self.predict_food(image, top_k)
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return []
    
    def recognize_food_from_url(self, image_url: str, top_k: int = 5) -> List[Dict]:
        """Recognize food from image URL"""
        try:
            import requests
            from io import BytesIO
            
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            return self.predict_food(image, top_k)
            
        except Exception as e:
            print(f"❌ Error downloading/processing image: {e}")
            return []
    
    def recognize_food_from_camera(self, top_k: int = 5) -> List[Dict]:
        """Capture image from camera and recognize food"""
        try:
            # Initialize camera
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("❌ Could not open camera")
                return []
            
            print("📸 Camera ready! Press SPACE to capture, ESC to cancel")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Display frame
                cv2.imshow('Food Recognition - Press SPACE to capture', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # SPACE key
                    # Convert BGR to RGB and create PIL Image
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb_frame)
                    
                    cap.release()
                    cv2.destroyAllWindows()
                    
                    return self.predict_food(image, top_k)
                    
                elif key == 27:  # ESC key
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            return []
            
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return []
    
    def analyze_food_image(self, image_path: str) -> Dict:
        """Complete food analysis from image"""
        recognition_results = self.recognize_food_from_path(image_path, top_k=3)
        
        if not recognition_results:
            return {
                'success': False,
                'message': 'No food detected in image',
                'suggestions': [
                    'Try a clearer image with better lighting',
                    'Ensure the food item is the main focus',
                    'Use an image with a single food item',
                    'Check if the food is one of the 80 supported Indian dishes'
                ]
            }
        
        # Get the best match
        best_match = recognition_results[0]
        
        analysis = {
            'success': True,
            'detected_food': best_match['food_name'].replace('_', ' ').title(),
            'confidence': f"{best_match['confidence_percentage']:.1f}%",
            'nutrition_info': {
                'calories': best_match['calories'],
                'protein_g': best_match['protein_g'],
                'carbs_g': best_match['carbs_g'],
                'fat_g': best_match['fat_g'],
                'fiber_g': best_match['fiber_g'],
                'serving_size_g': best_match['serving_size_g']
            },
            'dietary_info': {
                'vegetarian': best_match['vegetarian'],
                'vegan': best_match['vegan'],
                'gluten_free': best_match['gluten_free'],
                'category': best_match['category'],
                'meal_type': best_match['meal_type']
            },
            'alternative_matches': [
                {
                    'name': result['food_name'].replace('_', ' ').title(),
                    'confidence': f"{result['confidence_percentage']:.1f}%",
                    'category': result['category']
                }
                for result in recognition_results[1:3]
            ]
        }
        
        return analysis
    
    def batch_analyze_images(self, image_paths: List[str]) -> List[Dict]:
        """Analyze multiple images at once"""
        results = []
        
        print(f"🔄 Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"   Processing {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            analysis = self.analyze_food_image(image_path)
            analysis['image_path'] = image_path
            analysis['image_name'] = os.path.basename(image_path)
            results.append(analysis)
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_input_shape': self.model.input_shape,
            'model_output_shape': self.model.output_shape,
            'num_classes': len(self.label_encoder.classes_),
            'supported_foods': self.label_encoder.classes_.tolist(),
            'model_summary': str(self.model.summary())
        }
    
    def test_model_with_sample(self) -> Dict:
        """Test the model with a sample prediction"""
        try:
            # Create a dummy image for testing
            dummy_image = Image.new('RGB', self.input_shape, color='red')
            
            # Test prediction
            results = self.predict_food(dummy_image, top_k=3)
            
            return {
                'test_status': 'success',
                'predictions_count': len(results),
                'top_prediction': results[0] if results else None
            }
            
        except Exception as e:
            return {
                'test_status': 'failed',
                'error': str(e)
            }
    
    def validate_food_database_alignment(self) -> Dict:
        """Check alignment between model classes and nutrition database"""
        model_foods = set(self.label_encoder.classes_)
        db_foods = set(self.nutrition_db.get_all_foods()['food_name'].unique())
        
        common_foods = model_foods.intersection(db_foods)
        model_only = model_foods - db_foods
        db_only = db_foods - model_foods
        
        return {
            'total_model_classes': len(model_foods),
            'total_db_foods': len(db_foods),
            'common_foods': len(common_foods),
            'model_only_foods': list(model_only),
            'db_only_foods': list(db_only),
            'alignment_percentage': (len(common_foods) / len(model_foods)) * 100
        }