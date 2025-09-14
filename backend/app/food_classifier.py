import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import io
from typing import Tuple, List, Dict

class FoodClassifier:
    def __init__(self, model_path: str, label_encoder_path: str):
        self.model = tf.keras.models.load_model(model_path)
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.input_shape = (331, 331)
        
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize(self.input_shape)
        image_array = np.array(image)
        
        processed_image = tf.keras.applications.efficientnet.preprocess_input(image_array)
        
        return processed_image
    
    def predict_food(self, image_data: bytes, top_k: int = 5) -> Dict:
        processed_image = self.preprocess_image(image_data)
        
        predictions = self.model.predict(np.expand_dims(processed_image, axis=0))
        
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            food_name = self.label_encoder.inverse_transform([idx])[0]
            confidence = float(predictions[0][idx])
            results.append({
                'food_name': food_name,
                'confidence': confidence,
                'confidence_percentage': round(confidence * 100, 2)
            })
        
        return {
            'top_prediction': results[0],
            'all_predictions': results,
            'raw_predictions': predictions[0].tolist()
        }
    
    def batch_predict(self, image_data_list: List[bytes]) -> List[Dict]:
        results = []
        for image_data in image_data_list:
            result = self.predict_food(image_data)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict:
        return {
            'model_input_shape': self.model.input_shape,
            'model_output_shape': self.model.output_shape,
            'num_classes': len(self.label_encoder.classes_),
            'class_names': self.label_encoder.classes_.tolist()
        }