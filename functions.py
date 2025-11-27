import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def load_data(file_path):
    """
    Loads the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def prepare_symbolic_data(data, target_col='ocean_proximity'):
    """
    Encodes the target column and selects features.
    Returns the processed data, the feature matrix X, the target y, and the encoder.
    """
    # Encode target
    label_encoder = LabelEncoder()
    data[f'{target_col}_encoded'] = label_encoder.fit_transform(data[target_col])
    
    # Define predictor variables (features) and target
    features = ['longitude', 'latitude', 'housing_median_age', 'median_income']
    X = data[features]
    y = data[f'{target_col}_encoded']
    
    return data, X, y, label_encoder, features

def train_model(X, y, hidden_layer_sizes=(100, 50), max_iter=500, random_state=42):
    """
    Splits data and trains an MLP Classifier.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Classifier
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state)
    clf.fit(X_train, y_train)
    
    return clf, X_test, y_test

def evaluate_model(model, X_test, y_test, task_name="Task"):
    """
    Predicts and prints the classification report.
    Returns predictions and accuracy.
    """
    y_pred = model.predict(X_test)
    print(f"{task_name}: Classification Report")
    print(classification_report(y_test, y_pred))
    
    acc = accuracy_score(y_test, y_pred)
    return y_pred, acc

def plot_confusion_matrix(y_test, y_pred, label_encoder, title="Confusion Matrix", cmap='Blues'):
    """
    Plots the confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def create_sequences(X, y, sequence_length=3):
    """
    Creates sequence data from features and targets.
    """
    X_sequences = []
    y_sequences = []

    # X must be accessible by index, so if it's a DataFrame, use iloc
    # If it's already numpy array, allow it
    is_df = isinstance(X, pd.DataFrame)
    is_series = isinstance(y, pd.Series)
    
    length = len(X)
    
    for i in range(length - sequence_length):
        if is_df:
            X_sequences.append(X.iloc[i:i + sequence_length].values.flatten())
        else:
            X_sequences.append(X[i:i + sequence_length].flatten())
            
        if is_series:
            y_sequences.append(y.iloc[i + sequence_length])
        else:
            y_sequences.append(y[i + sequence_length])

    return np.array(X_sequences), np.array(y_sequences)

def plot_accuracy_comparison(single_acc, ensemble_acc=0.90):
    """
    Plots comparison between single predictor and ensemble.
    """
    plt.figure(figsize=(6, 4))
    methods = ['Single Predictor', 'Ensemble']
    accuracies = [single_acc, ensemble_acc]
    sns.barplot(x=methods, y=accuracies, palette='muted')
    plt.title("Task 4: Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.show()

def generate_sequence(model, initial_sequence, full_X, features, steps=5, sequence_length=3):
    """
    Generates future predictions.
    """
    current_sequence = initial_sequence.copy()
    generated_sequence = []

    for _ in range(steps):
        # Predict next step
        next_pred = model.predict(current_sequence.reshape(1, -1))[0]
        generated_sequence.append(next_pred)
        
        # Update sequence: shift left and add new feature data
        # Note: The original notebook logic uses actual data (X) from the dataset 
        # to fill the feature values for the next timestep.
        current_sequence = np.roll(current_sequence, -len(features))
        
        # We fetch the feature values corresponding to the new time step
        next_idx = len(generated_sequence) + sequence_length
        if next_idx < len(full_X):
             current_sequence[-len(features):] = full_X.iloc[next_idx].values
        else:
            break # Stop if we run out of data

    return generated_sequence

def plot_generated_sequence(generated_sequence, label_encoder):
    """
    Plots the generated sequence with labels.
    """
    generated_labels = label_encoder.inverse_transform(generated_sequence)
    steps = range(1, len(generated_sequence) + 1)
    
    plt.figure(figsize=(10, 4))
    plt.plot(steps, generated_sequence, marker='o', label='Encoded Predictions')
    plt.xticks(steps, labels=generated_labels, rotation=45)
    plt.title("Task 5: Generated Future Sequence")
    plt.xlabel("Step")
    plt.ylabel("Predicted Class (Encoded)")
    plt.legend()
    plt.tight_layout()
    plt.show()