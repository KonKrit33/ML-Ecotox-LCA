# The following code implements an integrated approach for predicting the classes of inorganic elements, followed by estimating their precise CF ΕCOTOX values. The results are compiled into an Excel file named "Predicted_CF_values.xlsx," ensuring a structured and clear presentation of the CF value predictions.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import numpy as np

np.random.seed(1999999999)

# Load the model's training set
my_data_path = r"C:\Users\Supplementary Material 1.xlsx"
my_data = pd.read_excel(my_data_path, sheet_name="Classification Training Set")

# Prepare features and target for classification
X_train = my_data.drop("Class", axis=1)
y_train = my_data["Class"]

# Load the unseen dataset
unseen_data_path = r"C:\Users\Unseen dataset.xlsx"
unseen_data = pd.read_excel(unseen_data_path)

# Add an empty 'Class' column for consistency
unseen_data["Class"] = pd.NA

# Initialize the classifier model
best_model = RandomForestClassifier(random_state=1999999999)

# Train the classifier
best_model.fit(X_train, y_train)

# Predict the class for the unseen data (excluding the empty 'Class' column)
class_predictions = best_model.predict(unseen_data.drop(columns=["Class"]))

# Print the predictions

print("Predicted Classes for the Unseen Dataset:")
print(class_predictions)

# Load the overall training dataset for the regression task. This dataset is located in Supplementary Material 1, under the tab named "Regression full set."

regression_data_path = r"C:\Users\Supplementary Material 1.xlsx"
df_train = pd.read_excel(regression_data_path, sheet_name="Regression full set")

# Function to execute the regression model based on the class assigned to the new, unseen elements by the classification model

def run_regression(df_train, indices, unseen_data, element_name):
    X_train_reg = df_train.iloc[indices].drop(columns=["EF3.1 CF"])
    y_train_reg = df_train.iloc[indices]["EF3.1 CF"]
    
    # The ml model
    model = DecisionTreeRegressor(random_state=1000000000)
    
    # Fitting the model to the training data
    model.fit(X_train_reg, y_train_reg)
    
    # Predict on user's new datapoints
    y_unseen_pred = model.predict(unseen_data.drop(columns=["Class"]))
    
    # Create the dataframe with the following specified structure

    export_data = pd.DataFrame({
        "SMILES": [element_name] * len(y_unseen_pred),
        "EF3.1 CF": y_unseen_pred})
    
    return export_data

# List to store all predictions
all_predictions = []

# Set the column names to represent the specific elements in the unseen dataset e.g., Element_1 == Magnesium

elements = ["[Element_1]", "[Element_2]"]

# Loop through each prediction and run the appropriate set of datapoints

for i, (pred_class, element) in enumerate(zip(class_predictions, elements)):
    if pred_class == 0:
        indices = range(0, 100)
    elif pred_class == 1:
        indices = range(100, 200)
    elif pred_class == 2:
        indices = range(200, 300)
    elif pred_class == 3:
        indices = range(300, 400)
    elif pred_class == 4:
        indices = range(400, 500)
    elif pred_class == 5:
        indices = range(500, 600)
    
    result = run_regression(df_train, indices, unseen_data.iloc[[i]], element)
    all_predictions.append(result)

# Combine all predictions into a single dataframe
combined_predictions = pd.concat(all_predictions, ignore_index=True)

# Export the predictions to an Excel file
combined_predictions.to_excel(r"C:\Users\Predicted_CF_values.xlsx", index=False)
