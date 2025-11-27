import functions as fn
import pandas as pd

def main():
    # --- Configuration ---
    # Update these paths to your local file locations
    TRAIN_PATH = r"C:\Users\lamin\OneDrive\Desktop\Datafolder\housing dataset\train.csv"
    TEST_PATH = r"C:\Users\lamin\OneDrive\Desktop\Datafolder\housing dataset\test_set.csv"
    
    print("Loading data...")
    try:
        train_data = fn.load_data(TRAIN_PATH)
        test_data = fn.load_data(TEST_PATH)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # --- Preprocessing ---
    # We combine datasets to ensure prepare_symbolic_data encodes categories consistently
    # (e.g., "House" is 1 in both train and test)
    len_train = len(train_data)
    combined_data = pd.concat([train_data, test_data], ignore_index=True)

    # --- Task 1: Prepare symbolic data & Basic Prediction ---
    print("\n--- Task 1: Symbolic Data Prediction ---")
    
    # Process the combined data
    processed_data, X_combined, y_combined, label_encoder, features = fn.prepare_symbolic_data(combined_data)
    
    # Split back into explicit Train and Test sets
    X_train = X_combined[:len_train]
    y_train = y_combined[:len_train]
    X_test = X_combined[len_train:]
    y_test = y_combined[len_train:]
    
    # Train on X_train
    # We ignore the return values for test data (_) because we have our own X_test
    model_task1, _, _ = fn.train_model(
        X_train, y_train, hidden_layer_sizes=(100, 50)
    )
    
    # Evaluate on our explicit X_test
    y_pred_1, acc_1 = fn.evaluate_model(model_task1, X_test, y_test, task_name="Task 1")
    
    # Visualize
    fn.plot_confusion_matrix(y_test, y_pred_1, label_encoder, title="Task 1: Confusion Matrix")

    # --- Task 2: Sequence-based prediction ---
    print("\n--- Task 2: Sequence-based Prediction ---")
    sequence_length = 3
    
    # Create sequences separately to avoid data leakage between train and test
    X_seq_train, y_seq_train = fn.create_sequences(X_train, y_train, sequence_length)
    X_seq_test, y_seq_test = fn.create_sequences(X_test, y_test, sequence_length)
    
    # Train Sequence Model on Train Data
    model_seq, _, _ = fn.train_model(
        X_seq_train, y_seq_train, hidden_layer_sizes=(200, 100)
    )
    
    # Evaluate Sequence Model on Test Data
    y_pred_seq, acc_seq = fn.evaluate_model(model_seq, X_seq_test, y_seq_test, task_name="Task 2")
    
    # Visualize
    fn.plot_confusion_matrix(y_seq_test, y_pred_seq, label_encoder, 
                             title="Task 2: Confusion Matrix (Sequence)", cmap='Greens')

    # --- Task 4: Compare results ---
    print("\n--- Task 4: Accuracy Comparison ---")
    # Note: plot_accuracy_comparison might expect a list or single value depending on implementation
    # Assuming it takes the sequence accuracy here based on previous code
    fn.plot_accuracy_comparison(acc_seq)

    # --- Task 5: Generated Sequence Visualization ---
    print("\n--- Task 5: Generating Future Sequence ---")
    # Use the last sequence from the test set as the seed
    initial_input = X_seq_test[0]
    
    generated_seq = fn.generate_sequence(
        model=model_seq, 
        initial_sequence=initial_input, 
        full_X=X_combined, # Use full feature space for context if needed
        features=features, 
        steps=5, 
        sequence_length=sequence_length
    )
    
    fn.plot_generated_sequence(generated_seq, label_encoder)
 