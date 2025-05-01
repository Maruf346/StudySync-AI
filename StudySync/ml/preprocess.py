import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import joblib

# 1. Load the dataset
df = pd.read_csv("ml/data/studysync_training_dataset.csv")

# 2. Drop Name column (not useful for model)
df = df.drop(columns=['Name'])

# 3. Encode categorical columns
label_cols = ['StudyTime', 'SubjectInterest', 'SkillLevel', 'LearningStyle', 'GoalType']
encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # Save encoder for later use (during predictions)

# 4. Normalize GPA (0 to 1 scale)
scaler = MinMaxScaler()
df['GPA'] = scaler.fit_transform(df[['GPA']])

# Final preprocessed dataset
print(df.head())

# (Optional) Save the encoders and scaler for later use with test/user-uploaded data

os.makedirs("ml/encoders", exist_ok=True)
for col, le in encoders.items():
    joblib.dump(le, f"ml/encoders/{col}_encoder.pkl")
joblib.dump(scaler, "ml/encoders/gpa_scaler.pkl")
