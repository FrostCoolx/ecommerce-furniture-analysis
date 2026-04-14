import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("1. Loading dataset...")
# Trying CSV first, checking for Excel as a backup
try:
    df = pd.read_csv('ecommerce_furniture_dataset_2024.csv')
except FileNotFoundError:
    try:
        df = pd.read_excel('ecommerce_furniture_dataset_2024.xlsx')
    except Exception as e:
        print("Error: Could not find the file. Please ensure the exact name matches.")
        exit()

print("2. Cleaning and preprocessing data...")
# Clean the 'price' column by removing '$' and ',' and converting to float
if df['price'].dtype == 'O': # Check if it's an object (string) before replacing
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)

# Handle missing values
df = df.dropna(subset=['price', 'sold', 'tagText'])

# Simplify 'tagText' 
df['tagText'] = df['tagText'].apply(lambda x: x if x in ['Free shipping', '+Shipping: $5.09'] else 'others')

# Encode categorical 'tagText' into numbers
le = LabelEncoder()
df['tagText'] = le.fit_transform(df['tagText'])

print("3. Engineering features (converting text to numbers)...")
# Convert 'productTitle' into numeric features using TF-IDF 
tfidf = TfidfVectorizer(max_features=100)
productTitle_tfidf = tfidf.fit_transform(df['productTitle'])
productTitle_tfidf_df = pd.DataFrame(productTitle_tfidf.toarray(), columns=tfidf.get_feature_names_out())

# Combine new features with the main dataset 
df = pd.concat([df.reset_index(drop=True), productTitle_tfidf_df.reset_index(drop=True)], axis=1)
df = df.drop(['productTitle', 'originalPrice'], axis=1, errors='ignore')

print("4. Training Machine Learning models... (This might take a few seconds)")
# Splitting data into features (X) and target (y) to predict 'sold'
X = df.drop('sold', axis=1)
y = df['sold']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train models
lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

print("5. Evaluating models...")
y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Print evaluation results
print("\n" + "="*40)
print("             MODEL RESULTS")
print("="*40)
print(f"Linear Regression MSE: {mean_squared_error(y_test, y_pred_lr):.2f}")
print(f"Linear Regression R2:  {r2_score(y_test, y_pred_lr):.2f}")
print("-" * 40)
print(f"Random Forest MSE:     {mean_squared_error(y_test, y_pred_rf):.2f}")
print(f"Random Forest R2:      {r2_score(y_test, y_pred_rf):.2f}")
print("="*40 + "\n")

print("6. Generating EDA Visualizations...")

# Set the visual style
sns.set_theme(style="whitegrid")

# Chart 1: Distribution of Furniture Prices
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=50, kde=True, color='blue')
plt.title('Distribution of Furniture Prices')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.savefig('chart1_price_distribution.png')
plt.close()

# Chart 2: Price vs. Units Sold
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='sold', data=df, color='orange', alpha=0.6)
plt.title('Impact of Price on Units Sold')
plt.xlabel('Price ($)')
plt.ylabel('Units Sold')
plt.savefig('chart2_price_vs_sold.png')
plt.close()

print("Success! The script has finished running and charts are saved.")