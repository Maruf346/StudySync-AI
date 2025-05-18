import os
import joblib
import pandas as pd
# from sklearn.cluster import KMeans
from ml.manual_kmeans import ManualKMeans


# Paths (adjust if needed)
BASE_DIR = "C:\\Users\\user\\OneDrive\\Documents\\My Projects\\StudySync-AI\\StudySync\\ml"
ENCODER_DIR = "C:\\Users\\user\\OneDrive\\Documents\\My Projects\\StudySync-AI\\StudySync\\ml\\encoders"
PIPELINE_PATH = "C:\\Users\\user\\OneDrive\\Documents\\My Projects\\StudySync-AI\\StudySync\\ml\\encoders\\full_preprocessing_pipeline.pkl"




def predict_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw DataFrame (with columns Name, StudyTime, SubjectInterest,
    SkillLevel, LearningStyle, GPA, AvailabilityDays, GoalType) and returns
    the processed numpy array ready for clustering or model input.
    """
    # Drop Name if present
    df = df.drop(columns=["Name"], errors="ignore")

    # Load the preprocessing pipeline
    pipeline = joblib.load(PIPELINE_PATH)

    # Run transform (not fit_transform!)
    processed = pipeline.transform(df)
    return processed


# Path where the trained KMeans will be stored
KMEANS_PATH = os.path.join(ENCODER_DIR, "kmeans_model.pkl")

def train_kmeans(
    n_clusters: int = 5,
    random_state: int = 42,
    **kmeans_kwargs
) -> ManualKMeans:
    """
    Loads your training CSV, preprocesses it, fits a KMeans model,
    saves it to disk, and returns the fitted model.
    """
    # 1. Load raw data
    data_raw = pd.read_csv("C:\\Users\\user\\OneDrive\\Documents\\My Projects\\StudySync-AI\\StudySync\\ml\\data\\studysync_training_dataset.csv")

    # 2. Preprocess (reuse your pipeline!)
    X = predict_preprocess(data_raw)

    # 3. Fit KMeans
    print(f"⚙️  Training KMeans with k={n_clusters}…")
    kmeans = ManualKMeans(n_clusters=n_clusters, random_state=random_state)

    kmeans.fit(X)

    # 4. Save it
    os.makedirs(ENCODER_DIR, exist_ok=True)
    joblib.dump(kmeans, KMEANS_PATH)
    print(f"💾 Saved KMeans model → {KMEANS_PATH}")

    return kmeans

def predict_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw DataFrame, runs preprocessing, applies the saved
    KMeans model, and returns the original DataFrame with an extra
    'cluster' column.
    """
    # 1. Preprocess
    X = predict_preprocess(df)

    # 2. Load the trained model
    kmeans = joblib.load(KMEANS_PATH)

    # 3. Predict clusters
    labels = kmeans.predict(X)

    # 4. Return DataFrame with labels
    result = df.copy().reset_index(drop=True)
    result["cluster"] = labels
    return result

if __name__ == "__main__":
    # simple CLI entrypoint
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Train or test StudySync utilities")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Fit and save a new KMeans model"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=5,
        help="Number of clusters for KMeans"
    )
    args = parser.parse_args()

    if args.train:
        train_kmeans(n_clusters=args.n_clusters)
    else:
        print("No action specified. Use --train to fit a KMeans model.")
