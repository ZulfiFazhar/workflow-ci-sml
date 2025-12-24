import argparse
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filepath: str, target_col: str):
    """Load preprocessed dataset"""
    df = pd.read_csv(filepath)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Target distribution:\n{y.value_counts()}\n")
    
    return X, y


def plot_confusion_matrix(y_true, y_pred, labels, save_path="confusion_matrix.png"):
    """Create confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return save_path


def plot_feature_importance(model, feature_names, top_n=20, save_path="feature_importance.png"):
    """Plot top N feature importances"""
    importances = model.feature_importances_
    
    # Adjust top_n jika jumlah features lebih sedikit
    actual_n = min(top_n, len(feature_names))
    indices = np.argsort(importances)[::-1][:actual_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Top {actual_n} Feature Importances')
    plt.barh(range(actual_n), importances[indices])
    plt.yticks(range(actual_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return save_path


def train_model(X_train, X_test, y_train, y_test, feature_names, n_estimators, max_depth):
    """Train model dengan MLflow logging"""
    
    # Disable autolog, gunakan manual logging
    mlflow.sklearn.autolog(disable=True)
    
    with mlflow.start_run(run_name="ci_random_forest"):
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # === MANUAL LOGGING ===
        
        # 1. Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", 42)
        
        # 2. Log metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1_score", test_f1)
        
        overfit_diff = train_accuracy - test_accuracy
        mlflow.log_metric("overfitting_diff", overfit_diff)
        
        print(f"\n{'='*50}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print(f"Overfitting Difference: {overfit_diff:.4f}")
        print(f"{'='*50}\n")
        
        # 3. Log classification report
        class_report = classification_report(y_test, y_test_pred, output_dict=True)
        class_report_path = "classification_report.json"
        with open(class_report_path, 'w') as f:
            json.dump(class_report, f, indent=2)
        mlflow.log_artifact(class_report_path)
        
        print("Classification Report:")
        print(classification_report(y_test, y_test_pred))
        
        # 4. Log confusion matrix
        labels = sorted(y_test.unique())
        cm_path = plot_confusion_matrix(y_test, y_test_pred, labels)
        mlflow.log_artifact(cm_path)
        
        # 5. Log feature importance
        fi_path = plot_feature_importance(model, feature_names)
        mlflow.log_artifact(fi_path)
        
        # 6. Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="CreditScoringCI"
        )
        
        print(f"\n✅ Logged artifacts:")
        print(f"  - {class_report_path}")
        print(f"  - {cm_path}")
        print(f"  - {fi_path}")
        print(f"  - model/")
        
        # Get run ID
        run_id = mlflow.active_run().info.run_id
        print(f"\n🔑 Run ID: {run_id}")
        
        return model, run_id


def main():
    parser = argparse.ArgumentParser(description="Train model untuk MLflow Project CI")
    parser.add_argument("--data", default="Credit_Score_Classification_Dataset_preprocessing.csv")
    parser.add_argument("--target", default="Credit Score")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=20)
    
    args = parser.parse_args()
    
    # Set tracking URI (akan di-override oleh environment variable di CI)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"MLflow Tracking URI: {tracking_uri}")
    
    # Load data
    X, y = load_data(args.data, args.target)
    feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=42,
        stratify=y
    )
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}\n")
    
    # Train model
    model, run_id = train_model(
        X_train, X_test, y_train, y_test, 
        feature_names, args.n_estimators, args.max_depth
    )
    
    print("\nTraining complete!")
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    main()
