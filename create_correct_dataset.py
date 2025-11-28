import pandas as pd
import numpy as np

def create_correct_wine_dataset():
    """Create a proper wine quality dataset without string errors"""
    np.random.seed(42)
    n_samples = 1599
    
    # Create realistic wine features
    data = {
        'fixed_acidity': np.random.uniform(4.0, 16.0, n_samples),
        'volatile_acidity': np.random.uniform(0.1, 1.6, n_samples),
        'citric_acid': np.random.uniform(0.0, 1.0, n_samples),
        'residual_sugar': np.random.uniform(0.5, 15.5, n_samples),
        'chlorides': np.random.uniform(0.01, 0.2, n_samples),
        'free_sulfur_dioxide': np.random.uniform(1, 72, n_samples),
        'total_sulfur_dioxide': np.random.uniform(6, 289, n_samples),
        'density': np.random.uniform(0.98, 1.04, n_samples),
        'pH': np.random.uniform(2.8, 4.0, n_samples),
        'sulphates': np.random.uniform(0.3, 2.0, n_samples),
        'alcohol': np.random.uniform(8.0, 15.0, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create quality score (3-9 like real wine quality ratings)
    # Based on realistic combinations of features
    quality = (
        (df['alcohol'] > 12) * 1 +
        (df['volatile_acidity'] < 0.5) * 1 +
        (df['citric_acid'] > 0.3) * 1 +
        (df['residual_sugar'] < 4) * 1 +
        (df['chlorides'] < 0.08) * 1 +
        3  # base quality
    )
    
    # Add some randomness
    quality += np.random.randint(0, 2, n_samples)
    
    # Ensure quality stays between 3-9
    df['quality'] = np.clip(quality, 3, 9)
    
    # Convert to integer
    df['quality'] = df['quality'].astype(int)
    
    print(f"Created dataset with shape: {df.shape}")
    print(f"Quality value counts:\n{df['quality'].value_counts().sort_index()}")
    
    return df

if __name__ == "__main__":
    df = create_correct_wine_dataset()
    df.to_csv('MLProject/wine_quality_processed.csv', index=False)
    print("✅ Dataset created successfully at MLProject/wine_quality_processed.csv")