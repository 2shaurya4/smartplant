import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import json
import logging
from datetime import datetime

class PlantWateringClassifier:
    """
    ML classifier for plant watering recommendations
    
    Training data labels:
    0 = Don't water (plant has enough moisture)
    1 = Water soon (soil getting dry)
    2 = Water now (soil very dry)
    """
    
    def __init__(self, model_type='decision_tree'):
        """
        Initialize the classifier
        
        Args:
            model_type: 'decision_tree' or 'random_forest'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'soil_moisture',
            'temperature',
            'humidity',
            'light_level'
        ]
        self.label_names = {
            0: "Don't Water",
            1: "Water Soon",
            2: "Water Now"
        }
    
    def generate_training_data(self, num_samples=500):
        """
        Generate synthetic training data based on plant physiology
        
        Args:
            num_samples: Number of training samples to generate
            
        Returns:
            Features and labels arrays
        """
        X = []
        y = []
        
        for _ in range(num_samples):

            soil_moisture = np.random.normal(55, 20)  
            soil_moisture = np.clip(soil_moisture, 5, 95)  
            
            temperature = np.random.normal(22, 5)  
            temperature = np.clip(temperature, 10, 35)
            
            humidity = np.random.normal(60, 15)  
            humidity = np.clip(humidity, 20, 100)
            
            light_level = np.random.uniform(10, 90)
            
            if soil_moisture < 25:
                label = 2  
            elif soil_moisture < 40:
                label = 1  
            else:
                label = 0  
            
            if temperature > 28 and humidity < 40 and soil_moisture < 50:
                label = max(label, 1)

            if temperature < 15 and humidity > 75:
                label = min(label, 0)
            
            X.append([soil_moisture, temperature, humidity, light_level])
            y.append(label)
        
        unique, counts = np.unique(y, return_counts=True)

        return np.array(X), np.array(y)
    
    def train(self, X=None, y=None, custom_data=False):
        """
        Train the classifier
        
        Args:
            X: Feature matrix (if None, generates synthetic data)
            y: Labels (if None, generates synthetic data)
            custom_data: If True, X and y are provided
        """
        try:
            if X is None or y is None:
                X, y = self.generate_training_data(500)
            
            X_scaled = self.scaler.fit_transform(X)
            
            if self.model_type == 'decision_tree':
                self.model = DecisionTreeClassifier(
                    max_depth=5,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42
                )
            elif self.model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=20,
                    max_depth=5,
                    min_samples_split=10,
                    random_state=42
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            self.model.fit(X_scaled, y)
            self.is_trained = True

            y_pred = self.model.predict(X_scaled)
            accuracy = accuracy_score(y, y_pred)
            
            return True
            
        except Exception as e:
            return False
    
    def predict(self, sensor_reading):
        """
        Predict watering recommendation
        
        Args:
            sensor_reading: Dict with keys:
                - soil_moisture (0-100)
                - temperature (°C)
                - humidity (0-100)
                - light_level (0-100)
        
        Returns:
            Dict with prediction and confidence
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained! Call train() first.")
        
        try:

            features = np.array([[
                sensor_reading.get('soil_moisture', 50),
                sensor_reading.get('temperature', 22),
                sensor_reading.get('humidity', 60),
                sensor_reading.get('light_level', 50)
            ]])
            

            features_scaled = self.scaler.transform(features)
            

            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            

            confidence = probabilities[prediction]
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'prediction': int(prediction),
                'recommendation': self.label_names[prediction],
                'confidence': float(confidence),
                'probabilities': {
                    self.label_names[i]: float(p) 
                    for i, p in enumerate(probabilities)
                },
                'input_features': {
                    name: sensor_reading.get(name)
                    for name in self.feature_names
                }
            }
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'prediction': None,
                'recommendation': 'Error'
            }
    
    def batch_predict(self, readings_list):
        """
        Predict for multiple readings
        
        Args:
            readings_list: List of sensor reading dicts
            
        Returns:
            List of prediction results
        """
        results = []
        for reading in readings_list:
            results.append(self.predict(reading))
        return results
    
    def save_model(self, path):
        """Save trained model and scaler"""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'label_names': self.label_names,
                'model_type': self.model_type
            }
            
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            
            return True
        except Exception as e:

            return False
    
    def load_model(self, path):
        """Load trained model and scaler"""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.label_names = model_data['label_names']
            self.model_type = model_data['model_type']
            self.is_trained = True
            
            return True
        except Exception as e:
            return False
    
    def get_feature_importance(self):
        """Get feature importance (if available)"""
        if not self.is_trained or self.model is None:
            return None
        
        try:
            importances = self.model.feature_importances_
            return {
                name: float(importance)
                for name, importance in zip(self.feature_names, importances)
            }
        except AttributeError:

            return None


def create_advanced_training_data(num_samples=1000):
    """
    Create more realistic training data based on plant physiology
    
    Different plants have different watering needs:
    - Succulents: Need less water, prefer dry
    - Ferns: Need more water, prefer moist
    - Typical houseplants: Middle ground
    """
    
    X = []
    y = []
    
    plant_types = ['typical', 'succulent', 'fern']
    
    for _ in range(num_samples):
        plant_type = np.random.choice(plant_types)
        
        if plant_type == 'succulent':

            soil_moisture = np.random.normal(35, 15)
            temp = np.random.normal(25, 5)
            humidity = np.random.normal(40, 10)
        elif plant_type == 'fern':

            soil_moisture = np.random.normal(70, 15)
            temp = np.random.normal(20, 4)
            humidity = np.random.normal(75, 10)
        else:  
            soil_moisture = np.random.normal(55, 18)
            temp = np.random.normal(22, 5)
            humidity = np.random.normal(60, 15)
        
        soil_moisture = np.clip(soil_moisture, 5, 95)
        temp = np.clip(temp, 10, 35)
        humidity = np.clip(humidity, 20, 100)
        light = np.random.uniform(10, 90)
        

        if soil_moisture < 20:
            label = 2  
        elif soil_moisture < 35:
            label = 1  
        else:
            label = 0  
        
        if temp > 28 and humidity < 40 and soil_moisture < 55:
            label = max(label, 1)
        
        if temp < 15 and humidity > 70 and soil_moisture > 60:
            label = min(label, 0)
        
        X.append([soil_moisture, temp, humidity, light])
        y.append(label)
    
    return np.array(X), np.array(y)


if __name__ == '__main__':
    print("=" * 60)
    print("Plant Watering Classifier - Training")
    print("=" * 60 + "\n")
    
    dt_classifier = PlantWateringClassifier(model_type='decision_tree')
    dt_classifier.train()
    dt_classifier.save_model('/tmp/plant_watering_dt.pkl')
    
    print("\n" + "=" * 60 + "\n")

    rf_classifier = PlantWateringClassifier(model_type='random_forest')
    rf_classifier.train()
    rf_classifier.save_model('/tmp/plant_watering_rf.pkl')
    
    print("\n" + "=" * 60 + "\n")
    
    test_cases = [
        {
            'name': 'Dry soil, hot day',
            'reading': {
                'soil_moisture': 15,
                'temperature': 28,
                'humidity': 35,
                'light_level': 80
            }
        },
        {
            'name': 'Moist soil, cool day',
            'reading': {
                'soil_moisture': 70,
                'temperature': 18,
                'humidity': 70,
                'light_level': 40
            }
        },
        {
            'name': 'Moderate conditions',
            'reading': {
                'soil_moisture': 45,
                'temperature': 22,
                'humidity': 55,
                'light_level': 60
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        print(f"  Soil: {test_case['reading']['soil_moisture']}%")
        print(f"  Temp: {test_case['reading']['temperature']}°C")
        print(f"  Humidity: {test_case['reading']['humidity']}%")
        print(f"  Light: {test_case['reading']['light_level']}%")
        
        result = dt_classifier.predict(test_case['reading'])
        print(f"\n  Recommendation: {result['recommendation']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  All probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"    - {label}: {prob:.2%}")
