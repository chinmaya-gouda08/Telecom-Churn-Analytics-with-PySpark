
import os
import json
import math

# ==================================================
# 1. LOAD SAVED MODEL
# ==================================================

MODEL_PATH = os.path.join(
    "saved_models",
    "final_churn_model.json"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

with open(MODEL_PATH, "r", encoding="utf-8") as file:
    model = json.load(file)

print("\nModel loaded successfully!")

# ==================================================
# 2. EXTRACT MODEL INFORMATION
# ==================================================

numeric_features = model["numeric_features"]
categorical_features = model["categorical_features"]

categorical_mappings = model["categorical_mappings"]
category_sizes = model["category_sizes"]

coefficients = [
    float(value) for value in model["coefficients"]
]

intercept = float(model["intercept"])
threshold = float(model["threshold"])

drop_last = model["drop_last"]

print("Algorithm: Logistic Regression")
print("Threshold:", threshold)
print("Expected feature count:", len(coefficients))

# ==================================================
# 3. VALIDATE SAVED FEATURE DIMENSIONS
# ==================================================

if len(categorical_features) != len(category_sizes):
    raise ValueError(
        "Categorical feature count does not match "
        "the saved category sizes."
    )

expected_dimension = (
    len(numeric_features) + sum(category_sizes)
)

if expected_dimension != len(coefficients):
    raise ValueError(
        f"Saved metadata mismatch!\n"
        f"Expected dimension from metadata: {expected_dimension}\n"
        f"Model coefficient count: {len(coefficients)}\n"
        "Check the Step 15 model export."
    )

print("Saved feature dimensions validated!")

# ==================================================
# 4. SAMPLE CUSTOMER DATA
# ==================================================

customers = [

    {
        "customerID": "NEW001",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.85,
        "TotalCharges": 179.70
    },

    {
        "customerID": "NEW002",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 65.50,
        "TotalCharges": 3930.00
    },

    {
        "customerID": "NEW003",
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.00,
        "TotalCharges": 1140.00
    }
]

# ==================================================
# 5. PREPARE CUSTOMER FEATURES
# ==================================================

def create_feature_vector(customer):

    features = []

    # ----------------------------------------------
    # Numeric features
    # ----------------------------------------------

    for column in numeric_features:

        if column not in customer:
            raise ValueError(
                f"Missing numeric feature: {column}"
            )

        value = float(customer[column])

        if not math.isfinite(value):
            raise ValueError(
                f"Invalid value for {column}"
            )

        features.append(value)

    # ----------------------------------------------
    # Categorical features
    # ----------------------------------------------

    for i, column in enumerate(categorical_features):

        if column not in customer:
            raise ValueError(
                f"Missing categorical feature: {column}"
            )

        labels = categorical_mappings[column]["labels"]

        value = str(customer[column])

        # Reproduce StringIndexer mapping
        if value in labels:
            category_index = labels.index(value)
        else:
            # Unseen or invalid category
            category_index = len(labels)

        # Use the saved output dimension directly.
        # The Step 15 export stores the dimensions
        # needed to reproduce the trained feature vector.
        output_size = int(category_sizes[i])

        encoded = [0.0] * output_size

        # With dropLast=True, the dropped category
        # is represented by an all-zero vector.
        if category_index < output_size:
            encoded[category_index] = 1.0

        features.extend(encoded)

    # ----------------------------------------------
    # Final dimension validation
    # ----------------------------------------------

    if len(features) != len(coefficients):
        raise ValueError(
            f"Feature dimension mismatch!\n"
            f"Generated features: {len(features)}\n"
            f"Model coefficients: {len(coefficients)}"
        )

    return features

# ==================================================
# 6. PREDICT CUSTOMER CHURN
# ==================================================

def predict_churn(customer):

    features = create_feature_vector(customer)

    # Logistic Regression decision function
    z = intercept + sum(
        feature * coefficient
        for feature, coefficient in zip(
            features, coefficients
        )
    )

    # Numerically stable sigmoid
    if z >= 0:
        probability = 1.0 / (1.0 + math.exp(-z))
    else:
        exp_value = math.exp(z)
        probability = exp_value / (1.0 + exp_value)

    # Apply selected threshold
    prediction = (
        "Yes"
        if probability >= threshold
        else "No"
    )

    return probability, prediction

# ==================================================
# 7. GENERATE PREDICTIONS
# ==================================================

print("\n" + "=" * 55)
print("          CUSTOMER CHURN PREDICTIONS")
print("=" * 55)

for customer in customers:

    probability, prediction = predict_churn(customer)

    print("\nCustomer ID:", customer["customerID"])

    print(
        "Churn Probability:",
        f"{probability * 100:.2f}%"
    )

    print("Prediction:", prediction)

    if prediction == "Yes":
        print("Status: Customer is likely to churn")
    else:
        print("Status: Customer is likely to stay")

    print("-" * 55)

print("\nPrediction process completed successfully!")
