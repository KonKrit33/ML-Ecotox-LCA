from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

np.random.seed(1999999999)

# Load the model's training set. See it in Supplementary Material 1, under the tab "Classification Training Set"
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

### Procedure for Applying the Yeo-Johnson Transformation to New, Unseen Datapoints ###

# First, calculate all the molecular descriptors for the new, unseen elements using the four toolkits: PaDEL, RDKit, Mordred, and Pybel, and save the results in an Excel file.
# Then, apply the short code block below to the generated Excel file and export the new Excel file named "filtered_untransformed_dataset".

# Path to the dataset containing all molecular descriptors for the new unseen elements.
dataset_with_all_descriptors_path = r"C:\Users\Dataset_with_all_descriptors.xlsx"

# Read the Excel file into a dataframe
produced_dataset_df = pd.read_excel(dataset_with_all_descriptors_path)

# Remove columns where all rows have the same value
filtered_dataset_df = produced_dataset_df.loc[:, produced_dataset_df.nunique() != 1]

# Export the filtered dataframe to a new Excel file
filtered_output_path = r"C:\Users\Filtered_untransformed_dataset.xlsx"
filtered_dataset_df.to_excel(filtered_output_path, index=False)

# Display message confirming the export to the Excel file
print(f"Filtered untransformed dataset saved to: {filtered_output_path}")

# Subsequently, to effectively apply the Yeo-Johnson transformation, first add the Filtered untransformed dataset to the dataset containing the 51 elements and the 190 molecular descriptors (see Suppl. Mat. 1), and name the new Excel file "final_filtered_untransformed_dataset". Before doing this, remove the columns named "XF" and "FF" if these values are not available for the unseen elements, as these descriptors cannot be calculated using the toolkits mentioned earlier and require specific external data sources for their determination. Additionally, calculate the RF descriptor for the unseen elements using the straightforward method outlined in the paper.

# The Yeo-Johnson transformation

from sklearn.preprocessing import PowerTransformer

# Path to the filtered_untransformed_dataset
final_filtered_untransformed_dataset_path = r"C:\Users\Final_filtered_untransformed_dataset.xlsx"

# Read the filtered untransformed dataset
final_df = pd.read_excel(final_filtered_untransformed_dataset_path)

# Apply the Yeo-Johnson power transform to descriptors
power = PowerTransformer(method='yeo-johnson')
transformed_dataset = pd.DataFrame(
    power.fit_transform(final_df), 
    columns=final_df.columns)

# Remove the first 51 rows (from index 0 to 50 inclusive), representing the 51 elements from the EF v.3.1 database
transformed_dataset = transformed_dataset.iloc[51:]

# Keep only the eight molecular descriptors
columns_to_keep = ['RF', 'Sv', 'qed', 'NumValenceElectrons', 'Chi0v', 'Kappa2', 'VMcGowan', 'logP']
final_transformed_dataset = transformed_dataset[columns_to_keep]

# Path to save the final transformed dataset
final_output_excel_path = r"C:\Users\Final_transformed_dataset.xlsx"

# Export the dataframe to Excel
final_transformed_dataset.to_excel(final_output_excel_path, index=False)

# Display message confirming the export
print("Final transformed dataset exported to Excel successfully.")

### End of the Yeo-Johnson Transformation procedure 
