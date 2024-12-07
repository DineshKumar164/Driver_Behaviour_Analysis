import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

class DriverBehaviorModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.categories = ['Safe', 'Moderate', 'Risky']
        
    def preprocess_data(self, data):
        """Preprocess the input data"""
        features = data[['speed', 'acceleration', 'brake_intensity']].values
        return self.scaler.fit_transform(features)
    
    def train(self, data, test_size=0.2):
        """Train the model with provided data"""
        X = self.preprocess_data(data)
        y = np.array([self.categories.index(cat) for cat in data['category']])
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        return {
            'train_accuracy': self.model.score(X_train, y_train),
            'val_accuracy': self.model.score(X_val, y_val)
        }
    
    def predict(self, data):
        """Make predictions on new data"""
        X = self.preprocess_data(data)
        predictions = self.model.predict(X)
        return [self.categories[i] for i in predictions]
    
    def evaluate(self, data):
        """Evaluate model performance"""
        X = self.preprocess_data(data)
        y_true = np.array([self.categories.index(cat) for cat in data['category']])
        y_pred = self.model.predict(X)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted')
        }

if __name__ == "__main__":
    # Test the model with dummy data
    from data_generator import DriverDataGenerator
    
    generator = DriverDataGenerator()
    data = generator.generate_batch(1000)
    
    model = DriverBehaviorModel()
    history = model.train(data)
    metrics = model.evaluate(data)
    print("Model Performance:", metrics)
