import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix, average_precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
from xgboost.callback import EarlyStopping
import json

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load the preprocessed data - MODIFIED to use augmented training data
print("Loading augmented training data and original validation/test data...")
# Use augmented training data
X_train = pd.read_csv('Data/processed/X_train_augmented.csv')
y_train = pd.read_csv('Data/processed/y_train_augmented.csv').iloc[:, 0]

# Keep same validation and test sets for fair comparison
X_val = pd.read_csv('Data/processed/X_val.csv')
y_val = pd.read_csv('Data/processed/y_val.csv').iloc[:, 0]

X_test = pd.read_csv('Data/processed/X_test.csv')
y_test = pd.read_csv('Data/processed/y_test.csv').iloc[:, 0]

print(f"Augmented training data: {X_train.shape}, Validation data: {X_val.shape}, Test data: {X_test.shape}")
print(f"Fraud ratio in augmented training: {y_train.mean():.6f}")
print(f"Original fraud ratio (reference): {pd.read_csv('Data/processed/y_train.csv').iloc[:, 0].mean():.6f}")

# 2. Set hyperparameters for XGBoost
# Recalculate class weights based on augmented data
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"New positive class weight: {scale_pos_weight:.2f}")

params = {
    'objective': 'binary:logistic',
    'eval_metric': ['auc', 'aucpr'],  # Both ROC-AUC and PR-AUC
    'scale_pos_weight': scale_pos_weight,
    'learning_rate': 0.05,
    'max_depth': 5,
    'min_child_weight': 2,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 300,
    'tree_method': 'hist',  # For faster training
    'random_state': 42
}

# 3. Train the model
print("Training XGBoost model on augmented data...")
start_time = time.time()

# Try this version for older XGBoost versions
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric=['auc', 'aucpr'],
    scale_pos_weight=scale_pos_weight,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=300,
    tree_method='hist',
    random_state=42
)

# For older XGBoost versions
eval_set = [(X_train, y_train), (X_val, y_val)]
model.fit(X_train, y_train, 
          eval_set=eval_set,
          verbose=True)

# Try to run for a set number of iterations and manually keep track of best model
best_score = 0
best_iter = 0
best_model = None

for i in range(50):  # 50 iterations max
    model.n_estimators = (i+1) * 10  # Increase trees by 10 each iteration
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    
    # Get validation score
    val_pred = model.predict_proba(X_val)[:, 1]
    val_score = average_precision_score(y_val, val_pred)
    
    print(f"Iteration {i+1}, Trees: {(i+1)*10}, PR-AUC: {val_score:.4f}")
    
    if val_score > best_score:
        best_score = val_score
        best_iter = i
        # Save best model parameters
        best_model = model.get_params()
        best_model['n_estimators'] = (i+1) * 10
    else:
        # No improvement for 3 consecutive iterations, stop
        if i > best_iter + 3:
            print(f"No improvement for 3 iterations, stopping at iteration {i+1}")
            break

print(f"Best iteration: {best_iter+1}, Trees: {(best_iter+1)*10}, Best PR-AUC: {best_score:.4f}")

# Create final model with best parameters
final_model = xgb.XGBClassifier(**best_model)
final_model.fit(X_train, y_train)
model = final_model

train_time = time.time() - start_time
print(f"Training completed in {train_time:.2f} seconds")

# 4. Model evaluation
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Finding the optimal threshold based on F1 score on validation data
val_pred_proba = model.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, val_pred_proba)
f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimal threshold based on F1: {optimal_threshold:.4f}")

# Apply threshold to test predictions
y_pred = (y_pred_proba >= optimal_threshold).astype(int)

# Calculate and display metrics
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

print("\n======= Test Set Performance with Augmented Training Data =======")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"PR-AUC: {average_precision_score(y_test, y_pred_proba):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"Sensitivity/Recall: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Precision: {tp/(tp+fp) if (tp+fp) > 0 else 0:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(conf_matrix)

# 5. Visualizations
plt.figure(figsize=(12, 10))

# 5.1 Feature Importance
plt.subplot(2, 2, 1)
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)[-15:]
plt.barh(X_train.columns[sorted_idx], feature_importance[sorted_idx])
plt.title("Feature Importance (Augmented Model)")
plt.xlabel("XGBoost Importance")

# 5.2 ROC Curve
plt.subplot(2, 2, 2)
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.4f})')

# 5.3 Precision-Recall Curve
plt.subplot(2, 2, 3)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AP = {average_precision_score(y_test, y_pred_proba):.4f})')
plt.axhline(y=y_test.mean(), color='r', linestyle='--', label=f'Baseline = {y_test.mean():.4f}')
plt.legend()

# 5.4 Confusion Matrix Heatmap
plt.subplot(2, 2, 4)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'])
plt.title('Confusion Matrix (Augmented Model)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('xgboost_augmented_results.png', dpi=300)
plt.show()

# 6. Save the model
model.save_model('Models/xgboost_augmented.json')
print("Model saved to 'Models/xgboost_augmented.json'")

# Optional: Save prediction probabilities for later analysis
pd.DataFrame({
    'true_label': y_test,
    'fraud_probability': y_pred_proba
}).to_csv('Results/augmented_predictions.csv', index=False)

# Function to convert NumPy types to native Python types
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

# Create metrics dictionary with converted values
metrics = {
    'roc_auc': convert_to_serializable(roc_auc_score(y_test, y_pred_proba)),
    'pr_auc': convert_to_serializable(average_precision_score(y_test, y_pred_proba)),
    'f1_score': convert_to_serializable(f1_score(y_test, y_pred)),
    'sensitivity_recall': convert_to_serializable(sensitivity),
    'specificity': convert_to_serializable(specificity),
    'precision': convert_to_serializable(tp/(tp+fp) if (tp+fp) > 0 else 0),
    'optimal_threshold': convert_to_serializable(optimal_threshold),
    'confusion_matrix': {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }
}

# Save to JSON file
with open('Results/augmented_model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

# Save ROC curve data points
fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
roc_data = pd.DataFrame({
    'false_positive_rate': fpr,
    'true_positive_rate': tpr,
    'thresholds': roc_thresholds
})
roc_data.to_csv('Results/augmented_roc_curve_data.csv', index=False)

# Save PR curve data points
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
# Account for length difference between precision/recall and thresholds
pr_data = pd.DataFrame({
    'precision': precision[:-1],
    'recall': recall[:-1],
    'thresholds': pr_thresholds
})
pr_data.to_csv('Results/augmented_pr_curve_data.csv', index=False)

# 7. NEW: Add comparison to baseline metrics if available
try:
    with open('Results/model_metrics.json', 'r') as f:
        baseline_metrics = json.load(f)
    
    print("\n======= Comparison with Baseline Model =======")
    print(f"Metric             | Baseline   | Augmented  | Change")
    print(f"--------------------|------------|------------|-------")
    print(f"ROC-AUC            | {baseline_metrics['roc_auc']:.4f}      | {metrics['roc_auc']:.4f}      | {(metrics['roc_auc']-baseline_metrics['roc_auc']):.4f}")
    print(f"PR-AUC             | {baseline_metrics['pr_auc']:.4f}      | {metrics['pr_auc']:.4f}      | {(metrics['pr_auc']-baseline_metrics['pr_auc']):.4f}")
    print(f"F1 Score           | {baseline_metrics['f1_score']:.4f}      | {metrics['f1_score']:.4f}      | {(metrics['f1_score']-baseline_metrics['f1_score']):.4f}")
    print(f"Sensitivity/Recall | {baseline_metrics['sensitivity_recall']:.4f}      | {metrics['sensitivity_recall']:.4f}      | {(metrics['sensitivity_recall']-baseline_metrics['sensitivity_recall']):.4f}")
    print(f"Specificity        | {baseline_metrics['specificity']:.4f}      | {metrics['specificity']:.4f}      | {(metrics['specificity']-baseline_metrics['specificity']):.4f}")
    print(f"Precision          | {baseline_metrics['precision']:.4f}      | {metrics['precision']:.4f}      | {(metrics['precision']-baseline_metrics['precision']):.4f}")
    
    # Create a comparative visualization
    plt.figure(figsize=(15, 8))
    
    # Plot PR curves side by side
    plt.subplot(1, 2, 1)
    try:
        baseline_pr = pd.read_csv('Results/pr_curve_data.csv')
        augmented_pr = pd.read_csv('Results/augmented_pr_curve_data.csv')
        
        plt.plot(baseline_pr['recall'], baseline_pr['precision'], label='Baseline')
        plt.plot(augmented_pr['recall'], augmented_pr['precision'], label='Augmented')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve Comparison')
        plt.legend()
    except:
        print("Could not create PR curve comparison")
    
    # Plot a bar chart of key metrics
    plt.subplot(1, 2, 2)
    metrics_to_plot = ['sensitivity_recall', 'precision', 'f1_score', 'pr_auc']
    labels = ['Recall', 'Precision', 'F1 Score', 'PR-AUC']
    
    baseline_values = [baseline_metrics[m] for m in metrics_to_plot]
    augmented_values = [metrics[m] for m in metrics_to_plot]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, baseline_values, width, label='Baseline')
    plt.bar(x + width/2, augmented_values, width, label='Augmented')
    
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Model Performance Comparison')
    plt.xticks(x, labels)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('xgboost_comparison.png', dpi=300)
    plt.show()
    
except Exception as e:
    print(f"Couldn't compare with baseline: {e}")