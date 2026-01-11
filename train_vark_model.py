"""
VARK Learning Style Classification Model
Training Script - FIXED VERSION

Features yang digunakan:
1. Visual score (0-16)
2. Auditory score (0-16)
3. Reading/Writing score (0-16)
4. Kinesthetic score (0-16)

Total: 4 features saja (sesuai input dari Flutter)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

def load_dataset(filepath='dataset/vark_dataset.csv'):
    """Load dan validate dataset"""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    return df

def preprocess_vark_data(df):
    """
    Preprocess raw survey data into VARK scores
    Maps 16 survey questions to 4 VARK categories
    """
    # Question columns mapping based on VARK questionnaire
    # Using exact column names from the CSV file
    
    # Visual questions
    visual_cols = [
        "I learn better by reading than by listening to someone.",
        "I understand things better in class when I participate in role-playing."
    ]
    
    # Auditory questions
    auditory_cols = [
        "When the teacher tells me the instructions I understand better",
        "When someone tells me how to do something in class, I learn it better.",
        "I remember things I have heard in class better than things I have read.",
        "I learn better in class when the teacher gives a lecture."
    ]
    
    # Read/Write questions
    readwrite_cols = [
        "I learn better by reading what the teacher writes on the chalkboard.",
        "When I read instructions, I remember them better.",
        "I understand better when I read instructions.",
        "I learn better by reading than by listening to someone.",
        "I learn more by reading textbooks than by listening to lectures."
    ]
    
    # Kinesthetic questions
    kinesthetic_cols = [
        "I prefer to learn by doing something in class.",
        "When I do things in class, I learn better.",
        "I enjoy learning in class by doing experiments.",
        "I understand things better in class when I participate in role-playing."
    ]
    
    # Calculate aggregate scores
    df['visual'] = df[visual_cols].sum(axis=1)
    df['auditory'] = df[auditory_cols].sum(axis=1)
    df['readwrite'] = df[readwrite_cols].sum(axis=1)
    df['kinesthetic'] = df[kinesthetic_cols].sum(axis=1)
    
    # Determine learning style based on highest score
    def get_learning_style(row):
        scores = {
            'V': row['visual'],
            'A': row['auditory'],
            'R': row['readwrite'],
            'K': row['kinesthetic']
        }
        return max(scores, key=scores.get)
    
    df['learning_style'] = df.apply(get_learning_style, axis=1)
    
    return df

def prepare_data(df):
    """
    Prepare features dan labels
    HANYA GUNAKAN 4 FITUR: visual, auditory, readwrite, kinesthetic
    """
    # Feature columns (hanya 4 fitur)
    feature_columns = ['visual', 'auditory', 'readwrite', 'kinesthetic']
    
    # Pastikan kolom ada di dataset
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")
    
    # Extract features
    X = df[feature_columns].values
    
    # Extract label
    if 'learning_style' not in df.columns:
        raise ValueError("Column 'learning_style' not found in dataset")
    
    y = df['learning_style'].values
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Feature columns: {feature_columns}")
    print(f"Labels shape: {y.shape}")
    print(f"Label distribution:")
    print(pd.Series(y).value_counts())
    
    return X, y, feature_columns

def create_synthetic_dataset():
    """
    Buat synthetic dataset jika file dataset tidak ada
    Berguna untuk development dan testing
    """
    print("Creating synthetic dataset...")
    
    np.random.seed(42)
    n_samples = 500
    
    data = []
    
    # Generate data untuk setiap gaya belajar
    styles = {
        'V': {'visual': (10, 16), 'auditory': (0, 8), 'readwrite': (4, 10), 'kinesthetic': (0, 8)},
        'A': {'visual': (0, 8), 'auditory': (10, 16), 'readwrite': (4, 10), 'kinesthetic': (0, 8)},
        'R': {'visual': (4, 10), 'auditory': (0, 8), 'readwrite': (10, 16), 'kinesthetic': (0, 8)},
        'K': {'visual': (0, 8), 'auditory': (0, 8), 'readwrite': (4, 10), 'kinesthetic': (10, 16)},
    }
    
    for style, ranges in styles.items():
        for _ in range(n_samples // 4):
            row = {
                'visual': np.random.randint(*ranges['visual']),
                'auditory': np.random.randint(*ranges['auditory']),
                'readwrite': np.random.randint(*ranges['readwrite']),
                'kinesthetic': np.random.randint(*ranges['kinesthetic']),
                'learning_style': style
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save dataset
    os.makedirs('dataset', exist_ok=True)
    df.to_csv('dataset/vark_dataset.csv', index=False)
    print(f"Synthetic dataset created: {df.shape}")
    
    return df

def train_models(X_train, X_test, y_train, y_test):
    """
    Train dan compare multiple models
    """
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        )
    }
    
    results = {}
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print(f"Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    
    print("\n" + "="*60)
    print(f"BEST MODEL: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print("="*60)
    
    return best_model, best_model_name, results

def save_model(model, scaler, label_encoder, feature_columns):
    """Save trained model and preprocessors"""
    os.makedirs('model', exist_ok=True)
    
    joblib.dump(model, 'model/vark_random_forest.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(label_encoder, 'model/label_encoder.pkl')
    joblib.dump(feature_columns, 'model/feature_columns.pkl')
    
    print("\nModel artifacts saved:")
    print("- model/vark_random_forest.pkl")
    print("- model/scaler.pkl")
    print("- model/label_encoder.pkl")
    print("- model/feature_columns.pkl")

def main():
    """Main training pipeline"""
    print("="*60)
    print("VARK LEARNING STYLE CLASSIFICATION")
    print("Machine Learning Model Training")
    print("="*60)
    
    # Load or create dataset
    try:
        df = load_dataset()
    except FileNotFoundError:
        print("\nDataset not found. Creating synthetic dataset...")
        df = create_synthetic_dataset()

    # Preprocess data to create VARK aggregate scores
    df = preprocess_vark_data(df)

    # Prepare data
    X, y, feature_columns = prepare_data(df)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\nLabel encoding:")
    for original, encoded in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
        print(f"  {original} -> {encoded}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nFeature scaling completed")
    print(f"Mean: {scaler.mean_}")
    print(f"Std: {scaler.scale_}")
    
    # Train models
    best_model, best_model_name, all_results = train_models(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Save best model
    save_model(best_model, scaler, label_encoder, feature_columns)
    
    # Test saved model
    print("\n" + "="*60)
    print("TESTING SAVED MODEL")
    print("="*60)
    
    loaded_model = joblib.load('model/vark_random_forest.pkl')
    loaded_scaler = joblib.load('model/scaler.pkl')
    loaded_encoder = joblib.load('model/label_encoder.pkl')
    
    # Test dengan sample data
    test_samples = [
        {'visual': 14, 'auditory': 3, 'readwrite': 6, 'kinesthetic': 2},  # Visual
        {'visual': 2, 'auditory': 15, 'readwrite': 5, 'kinesthetic': 3},  # Auditory
        {'visual': 5, 'auditory': 4, 'readwrite': 14, 'kinesthetic': 2},  # Reading
        {'visual': 3, 'auditory': 2, 'readwrite': 6, 'kinesthetic': 14},  # Kinesthetic
    ]
    
    print("\nPrediction tests:")
    for i, sample in enumerate(test_samples, 1):
        sample_array = np.array([[
            sample['visual'],
            sample['auditory'],
            sample['readwrite'],
            sample['kinesthetic']
        ]])
        
        sample_scaled = loaded_scaler.transform(sample_array)
        prediction = loaded_model.predict(sample_scaled)
        probabilities = loaded_model.predict_proba(sample_scaled)
        
        predicted_label = loaded_encoder.inverse_transform(prediction)[0]
        
        print(f"\nTest {i}:")
        print(f"  Input: {sample}")
        print(f"  Predicted: {predicted_label}")
        print(f"  Probabilities: {dict(zip(loaded_encoder.classes_, probabilities[0]))}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()

