import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Load and clean data
df = pd.read_csv(r"C:/Users/Manisha/OneDrive/Desktop/Projects/youtube_ad_revenue_dataset.csv", encoding='latin-1')
df.drop_duplicates(inplace=True)
df.fillna({'likes':df['views'].mean() , 'comments': df['views'].mean(), 'watch_time_minutes': df['watch_time_minutes'].mean()}, inplace=True)


# Feature engineering
df['engagement_rate'] = (df['likes'] + df['comments']) / df['views']
df['subscriber_value'] = df['ad_revenue_usd'] / (df['subscribers'] )

# Separate features and target
X = df.drop(columns=['date', 'ad_revenue_usd','video_id'],errors='ignore')
y = df['ad_revenue_usd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    'ElasticNet':ElasticNet(alpha=1.0, l1_ratio=0.5)
}

# Identify categorical & numeric columns
cat_cols = X_train.select_dtypes(include=['object']).columns
num_cols = X_train.select_dtypes(exclude=['object']).columns




# Categorical preprocessing pipeline



categorical_transformer = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine into ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', num_cols),
    ('cat', categorical_transformer, cat_cols)
])



# Store results
results = []

# Loop through models
for name, model in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results.append([name, mae, rmse, r2])

# Convert to DataFrame
results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"])
print(results_df)

# Columns
num_features = ["views", "likes", "comments", "watch_time_minutes",
                "video_length_minutes", "subscribers",
                "engagement_rate", "subscriber_value"]

cat_features = ["category", "device", "country"]

# Preprocessor
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])
# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "Linear Regression.pkl")
print("✅ Final pipeline saved as Linear Regression.pkl")


