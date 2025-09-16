import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import json
from pathlib import Path
import cv2
import time

class ModelTester:
    def __init__(self, model_path, label_encoder_path, test_data_dir, nutrition_csv_path):
        """
        Initialize the model tester with paths to model, labels, and test data
        
        Args:
            model_path: Path to the trained .h5 or .keras model
            label_encoder_path: Path to the label encoder pickle file
            test_data_dir: Directory containing test images
            nutrition_csv_path: Path to nutrition database CSV
        """
        # Load model and label encoder
        self.model = tf.keras.models.load_model(model_path)
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load nutrition database
        self.nutrition_db = pd.read_csv(nutrition_csv_path)
        self.nutrition_db.set_index('food_name', inplace=True)
        
        # Get class names
        self.class_names = list(self.label_encoder.classes_)
        self.num_classes = len(self.class_names)
        
        print(f"Model loaded successfully with {self.num_classes} classes")
        print(f"Class names: {self.class_names[:5]}...")  # Show first 5

    def preprocess_image(self, image_path, target_size=(331, 331)):
        """
        Preprocess an image for EfficientNet prediction
        
        Args:
            image_path: Path to the image file
            target_size: Target size for resizing
            
        Returns:
            Preprocessed image array
        """
        # Read and decode image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, target_size)
        
        # EfficientNet specific preprocessing
        image = tf.keras.applications.efficientnet.preprocess_input(image)
        return image.numpy()

    def predict_single_image(self, image_path, top_k=5):
        """
        Make prediction on a single image
        
        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        # Preprocess image
        image = self.preprocess_image(image_path)
        
        # Add batch dimension and predict
        predictions = self.model.predict(np.expand_dims(image, axis=0), verbose=0)
        
        # Get top K predictions
        top_k_indices = np.argsort(predictions[0])[-top_k:][::-1]
        top_k_confidences = predictions[0][top_k_indices]
        top_k_classes = self.label_encoder.inverse_transform(top_k_indices)
        
        # Get nutrition info for top prediction
        top_food = top_k_classes[0]
        nutrition_info = self.get_nutrition_info(top_food)
        
        return {
            'top_predictions': list(zip(top_k_classes, top_k_confidences)),
            'predicted_class': top_k_classes[0],
            'confidence': float(top_k_confidences[0]),
            'nutrition_info': nutrition_info,
            'all_predictions': predictions[0]
        }

    def get_nutrition_info(self, food_name, portion_size=250):
        """
        Get nutrition information for a food item
        
        Args:
            food_name: Name of the food item
            portion_size: Portion size in grams
            
        Returns:
            Dictionary with nutrition information
        """
        if food_name not in self.nutrition_db.index:
            return {"error": "Food not found in database"}
        
        nutrition = self.nutrition_db.loc[food_name].to_dict()
        
        # Scale nutrition based on portion size
        serving_size = nutrition['serving_size_g']
        multiplier = portion_size / serving_size
        
        scalable_fields = ['calories', 'protein_g', 'carbs_g', 'fat_g', 
                          'fiber_g', 'sugar_g', 'sodium_mg', 'calcium_mg', 
                          'iron_mg', 'vitamin_a_mcg', 'vitamin_c_mg']
        
        for field in scalable_fields:
            if field in nutrition:
                nutrition[field] = round(nutrition[field] * multiplier, 2)
        
        nutrition['portion_size_g'] = portion_size
        return nutrition

    def evaluate_on_test_generator(self, test_generator):
        """
        Comprehensive evaluation using a test generator
        
        Args:
            test_generator: Keras ImageDataGenerator for test data
            
        Returns:
            Comprehensive evaluation results
        """
        print("🚀 Starting comprehensive evaluation...")
        
        # 1. Basic evaluation
        start_time = time.time()
        test_results = self.model.evaluate(test_generator, verbose=1, return_dict=True)
        eval_time = time.time() - start_time
        
        print(f"✅ Evaluation completed in {eval_time:.2f} seconds")
        
        # 2. Get predictions for confusion matrix and detailed analysis
        print("📊 Generating predictions for detailed analysis...")
        y_true = test_generator.classes
        y_pred_proba = self.model.predict(test_generator, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 3. Calculate top-k accuracy
        top_3_accuracy = tf.keras.metrics.top_k_categorical_accuracy(
            tf.one_hot(y_true, self.num_classes), y_pred_proba, k=3
        ).numpy().mean()
        
        top_5_accuracy = tf.keras.metrics.top_k_categorical_accuracy(
            tf.one_hot(y_true, self.num_classes), y_pred_proba, k=5
        ).numpy().mean()
        
        # 4. Generate classification report
        clf_report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # 5. Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # 6. Calculate per-class accuracy
        per_class_accuracy = {}
        for i, class_name in enumerate(self.class_names):
            class_mask = (y_true == i)
            if np.sum(class_mask) > 0:
                accuracy = np.mean(y_pred[class_mask] == i)
                per_class_accuracy[class_name] = accuracy
        
        # Sort classes by accuracy (worst to best)
        sorted_accuracy = sorted(per_class_accuracy.items(), key=lambda x: x[1])
        
        return {
            'basic_metrics': test_results,
            'top_3_accuracy': float(top_3_accuracy),
            'top_5_accuracy': float(top_5_accuracy),
            'classification_report': clf_report,
            'confusion_matrix': cm,
            'per_class_accuracy': per_class_accuracy,
            'worst_performing_classes': sorted_accuracy[:10],  # Top 10 worst
            'best_performing_classes': sorted_accuracy[-10:],  # Top 10 best
            'evaluation_time_seconds': eval_time
        }

    def plot_confusion_matrix(self, cm, save_path='confusion_matrix.png'):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(20, 18))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                   cbar=False, xticklabels=False, yticklabels=False)
        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Confusion matrix saved to {save_path}")

    def plot_class_accuracy(self, per_class_accuracy, save_path='class_accuracy.png'):
        """Plot per-class accuracy"""
        # Prepare data
        classes, accuracies = zip(*sorted(per_class_accuracy.items(), key=lambda x: x[1]))
        
        plt.figure(figsize=(15, 20))
        plt.barh(range(len(classes)), accuracies)
        plt.yticks(range(len(classes)), classes, fontsize=8)
        plt.xlabel('Accuracy')
        plt.title('Per-Class Accuracy')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Class accuracy plot saved to {save_path}")

    def generate_detailed_report(self, evaluation_results, save_path='model_evaluation_report.json'):
        """Generate detailed JSON report"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_performance': {
                'test_loss': float(evaluation_results['basic_metrics']['loss']),
                'test_accuracy': float(evaluation_results['basic_metrics']['accuracy']),
                'top_3_accuracy': evaluation_results['top_3_accuracy'],
                'top_5_accuracy': evaluation_results['top_5_accuracy'],
                'evaluation_time_seconds': evaluation_results['evaluation_time_seconds']
            },
            'class_performance': {
                'worst_performing_classes': [
                    {'class': cls, 'accuracy': acc} 
                    for cls, acc in evaluation_results['worst_performing_classes']
                ],
                'best_performing_classes': [
                    {'class': cls, 'accuracy': acc} 
                    for cls, acc in evaluation_results['best_performing_classes']
                ]
            },
            'confusion_matrix_shape': evaluation_results['confusion_matrix'].shape
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Detailed report saved to {save_path}")
        return report

    def test_speed(self, test_image_path, num_iterations=100):
        """Test inference speed"""
        print("⏱️  Testing inference speed...")
        
        # Warmup
        for _ in range(5):
            self.predict_single_image(test_image_path)
        
        # Actual timing
        start_time = time.time()
        for _ in range(num_iterations):
            self.predict_single_image(test_image_path)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_image = total_time / num_iterations
        fps = 1 / avg_time_per_image
        
        print(f"📊 Speed test results ({num_iterations} iterations):")
        print(f"   Average time per image: {avg_time_per_image:.4f} seconds")
        print(f"   Throughput: {fps:.2f} FPS")
        
        return avg_time_per_image, fps

# Example usage and test
if __name__ == "__main__":
    # Initialize paths (UPDATE THESE PATHS)
    MODEL_PATH = 'final_restarted_model.keras'
    LABEL_ENCODER_PATH = 'label_encoder.pkl'
    NUTRITION_CSV_PATH = 'indian_food_nutrition.csv'
    TEST_DATA_DIR = 'path/to/your/test/directory'
    
    # Create test data generator (example)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DATA_DIR,
        target_size=(331, 331),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Initialize tester
    tester = ModelTester(MODEL_PATH, LABEL_ENCODER_PATH, TEST_DATA_DIR, NUTRITION_CSV_PATH)
    
    # Test single image
    print("\n🧪 Testing single image prediction...")
    sample_image_path = 'chapati.png'  # Replace with actual path
    prediction = tester.predict_single_image(sample_image_path)
    
    print(f"📷 Prediction for {sample_image_path}:")
    print(f"   Predicted: {prediction['predicted_class']}")
    print(f"   Confidence: {prediction['confidence']:.2%}")
    print("   Top 5 predictions:")
    for i, (cls, conf) in enumerate(prediction['top_predictions'], 1):
        print(f"     {i}. {cls}: {conf:.2%}")
    
    # Comprehensive evaluation
    print("\n" + "="*50)
    evaluation_results = tester.evaluate_on_test_generator(test_generator)
    
    print("\n📊 Evaluation Results:")
    print(f"   Test Loss: {evaluation_results['basic_metrics']['loss']:.4f}")
    print(f"   Test Accuracy: {evaluation_results['basic_metrics']['accuracy']:.2%}")
    print(f"   Top-3 Accuracy: {evaluation_results['top_3_accuracy']:.2%}")
    print(f"   Top-5 Accuracy: {evaluation_results['top_5_accuracy']:.2%}")
    
    print("\n❌ Worst performing classes:")
    for cls, acc in evaluation_results['worst_performing_classes']:
        print(f"   {cls}: {acc:.2%}")
    
    print("\n✅ Best performing classes:")
    for cls, acc in evaluation_results['best_performing_classes']:
        print(f"   {cls}: {acc:.2%}")
    
    # Generate visualizations and reports
    tester.plot_confusion_matrix(evaluation_results['confusion_matrix'])
    tester.plot_class_accuracy(evaluation_results['per_class_accuracy'])
    detailed_report = tester.generate_detailed_report(evaluation_results)
    
    # Test inference speed
    print("\n" + "="*50)
    avg_time, fps = tester.test_speed(sample_image_path)
    
    print("\n🎯 Testing completed successfully!")