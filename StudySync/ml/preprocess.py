import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

print("🛠️  preprocess.py is running!")
print("🔄 Preprocessing data...")
# Paths
BASE_DIR = "C:\\Users\\user\\OneDrive\\Documents\\My Projects\\StudySync-AI\\StudySync\\ml"
DATA_PATH = "C:\\Users\\user\\OneDrive\\Documents\\My Projects\\StudySync-AI\\StudySync\\ml\\data\\studysync_training_dataset.csv"
ENCODER_DIR = "C:\\Users\\user\\OneDrive\\Documents\\My Projects\\StudySync-AI\\StudySync\\ml\\encoders"

def load_and_preprocess():
    print(f"✅ Base dir:   {BASE_DIR}")
    print(f"✅ Data path:  {DATA_PATH}")
    print(f"✅ Encoder dir:{ENCODER_DIR}")

    # Check data file exists
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"🗒  Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Drop Name
    df = df.drop(columns=["Name"], errors='ignore')
    print("✂️  Dropped column: Name")

    # Categorical columns
    cat_cols = ["StudyTime", "SubjectInterest", "LearningStyle", "GoalType", "SkillLevel"]
    print(f"🔤 Encoding columns: {cat_cols}  +  SkillLevel (ordered)")

    # 2) One‑Hot + MinMaxScaler in a pipeline
    onehot = ColumnTransformer(
        [("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore", categories="auto"), cat_cols)],
        remainder="passthrough"
    )
    scaler = MinMaxScaler()
    pipeline = Pipeline([
        ("onehot", onehot),
        ("scale", scaler)
    ])

    print("🔄 Fitting pipeline...")
    data_processed = pipeline.fit_transform(df)
    pipeline_path = os.path.join(ENCODER_DIR, "full_preprocessing_pipeline.pkl")
    joblib.dump(pipeline, pipeline_path)
    print(f"💾 Saved full pipeline → {pipeline_path}")

    print(f"🎉 Preprocessing complete. Processed shape: {data_processed.shape}")
    return data_processed

if __name__ == "__main__":
    try:
        load_and_preprocess()
    except Exception as e:
        print(f"❗ Error during preprocessing:\n{e}")
        raise
