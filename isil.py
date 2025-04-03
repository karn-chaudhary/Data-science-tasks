import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r"D:\inter\Iris.csv")

# Display the first few rows of the data
print("First five rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# If an 'Id' column exists, drop it as it's not useful for classification
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# Separate features (X) and target (y)
X = df.drop('Species', axis=1)
y = df['Species']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance with a classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Optional: Visualize the feature distributions with a pair plot
sns.pairplot(df, hue="Species", diag_kind="kde")
plt.show()
