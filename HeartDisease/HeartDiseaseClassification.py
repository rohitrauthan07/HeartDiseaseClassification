import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the Data
dataset = pd.read_csv('HeartDisease/Heart_Disease.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# Step 2: Data Preprocessing
# No preprocessing needed in this example. If required, you can handle missing values and convert categorical variables.
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encoding the Independent Variable
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# X = np.array(ct.fit_transform(X))

#Encoding the Dependent Variable
# le = LabelEncoder()
# y = le.fit_transform(y)


# Step 3: Feature Selection and Target Variable
X = dataset.drop('class', axis=1)  # Input features
y = dataset['class']              # Target variable

# Step 4: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Hyperparameter Tuning with GridSearchCV
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],    # Regularization parameter
    'solver': ['liblinear', 'lbfgs', 'saga']  # Optimization algorithm
}

log_reg = LogisticRegression(max_iter=1000)
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_

# Step 6: Train the Model with Best Hyperparameters
best_log_reg = LogisticRegression(max_iter=1000, **best_params)
best_log_reg.fit(X_train, y_train)

# Step 7: Evaluate the Model
y_pred = best_log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Best Hyperparameters:", best_params)
print("Accuracy:", accuracy)
print("Classification Report:", report)

# Step 9: Make Predictions on New Data
# Replace 'test_data.csv' with the filename or path of your new data (test data)
new_data = pd.read_csv('HeartDisease/test_data.csv')
# Separate features and target variable from the new data
new_data_features = new_data.drop('class', axis=1)
# new_data_target = new_data['class']

# Create a new SimpleImputer for handling missing values in the new data
new_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
new_imputer.fit(new_data_features)
new_data_features = pd.DataFrame(new_imputer.transform(new_data_features), columns=new_data_features.columns)

# Make predictions on the new data features
new_predictions = best_log_reg.predict(new_data_features)
# Display the predictions
print("Predictions on New Data:", new_predictions)
