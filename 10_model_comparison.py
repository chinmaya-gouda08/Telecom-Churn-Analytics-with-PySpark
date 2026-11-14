
# ==========================================
# CUSTOMER CHURN MODEL COMPARISON
# ==========================================

print("\n========== CLASSIFICATION MODEL COMPARISON ==========")

# Results collected from previous experiments

models = {
    "Logistic Regression": {
        "Accuracy": 0.8074,
        "Precision": 0.7982,
        "Weighted Recall": 0.8074,
        "F1 Score": 0.7987,
        "ROC-AUC": 0.8413,
        "Churn Recall": 196 / (196 + 173)
    },

    "Random Forest": {
        "Accuracy": 0.7985,
        "Precision": 0.7876,
        "Weighted Recall": 0.7985,
        "F1 Score": 0.7872,
        "ROC-AUC": 0.8395,
        "Churn Recall": 182 / (182 + 187)
    }
}

# Display comparison
print("\n{:<23} {:>12} {:>12}".format(
    "Metric",
    "Logistic Reg.",
    "Random Forest"
))

print("-" * 50)

metrics = [
    "Accuracy",
    "Precision",
    "Weighted Recall",
    "F1 Score",
    "ROC-AUC",
    "Churn Recall"
]

for metric in metrics:

    lr_value = models["Logistic Regression"][metric]
    rf_value = models["Random Forest"][metric]

    print("{:<23} {:>11.2%} {:>12.2%}".format(
        metric,
        lr_value,
        rf_value
    ))

# Identify the best model for ROC-AUC
best_model = max(
    models,
    key=lambda model: models[model]["ROC-AUC"]
)

print("\nBest model based on ROC-AUC:", best_model)

# Identify the best model for churn recall
best_churn_model = max(
    models,
    key=lambda model: models[model]["Churn Recall"]
)

print("Best model based on Churn Recall:", best_churn_model)

print("\nClassification comparison completed!")


# ==========================================
# CLUSTERING MODEL COMPARISON
# ==========================================

print("\n========== CLUSTERING MODEL COMPARISON ==========")

clustering_models = {
    "K-Means": {
        "Best K": 2,
        "Silhouette Score": 0.4417
    },

    "Bisecting K-Means": {
        "Best K": 2,
        "Silhouette Score": 0.4417
    }
}

print("\n{:<25} {:>12} {:>20}".format(
    "Algorithm",
    "Best K",
    "Silhouette Score"
))

print("-" * 60)

for name, results in clustering_models.items():

    print("{:<25} {:>12} {:>19.4f}".format(
        name,
        results["Best K"],
        results["Silhouette Score"]
    ))

print("\nClustering comparison completed!")
