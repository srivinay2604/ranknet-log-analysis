import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import time
import io
import sys

# --- UI IMPORTS ---
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML # <-- Added HTML

# --- CONSTANTS AND CONFIGURATION ---
# DATASET_PATH is now set by the UI
RANDOM_STATE = 42
TARGET_COL = 'relevance_score'
PAIRWISE_LABEL_COL = 'preference_label'

# --- 1. DATA LOADING AND INITIAL CLEANING ---

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads the log data from a CSV file and performs initial cleaning.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        print("Please make sure 'train.csv' is uploaded to your Colab session.")
        return pd.DataFrame()

    # Convert timestamp to datetime
    df['time_iso8601'] = pd.to_datetime(df['time_iso8601'], errors='coerce')
    df.dropna(subset=['time_iso8601'], inplace=True)

    # Convert numeric columns
    numeric_cols = ['request_length', 'bytes_sent', 'request_time',
                    'upstream_connect_time', 'upstream_header_time',
                    'upstream_response_time', 'status', 'upstream_status']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with critical missing data
    df.dropna(subset=['request_time', 'status', 'upstream_response_time'], inplace=True)

    print(f"Loaded and cleaned {len(df)} records.")
    return df

# --- 2. FEATURE ENGINEERING ---

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates relevant features from the raw log data for ranking.
    """
    # Time-based features
    df['hour_of_day'] = df['time_iso8601'].dt.hour
    df['day_of_week'] = df['time_iso8601'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Request features
    df['request_method'] = df['request'].apply(lambda x: str(x).split(' ')[0])
    df['request_path'] = df['request'].apply(lambda x: str(x).split(' ')[1] if len(str(x).split(' ')) > 1 else 'N/A')
    df['path_depth'] = df['request_path'].apply(lambda x: len([p for p in x.split('/') if p]))

    # Status code categorization
    def categorize_status(status):
        if 200 <= status < 300: return '2xx_Success'
        elif 300 <= status < 400: return '3xx_Redirection'
        elif 400 <= status < 500: return '4xx_Client_Error'
        elif 500 <= status < 600: return '5xx_Server_Error'
        else: return 'Other'
    df['status_category'] = df['status'].apply(categorize_status)

    # User Agent feature
    df['is_bot'] = df['http_user_agent'].apply(lambda x: 1 if 'bot' in str(x).lower() or 'spider' in str(x).lower() else 0)

    # IP feature
    df['is_internal_ip'] = df['remote_addr'].apply(lambda x: 1 if str(x).startswith('192.168') or str(x).startswith('10.') else 0)

    # Fill NaNs
    df['request_time'].fillna(df['request_time'].mean(), inplace=True)
    df['upstream_response_time'].fillna(df['upstream_response_time'].mean(), inplace=True)

    print("Engineered features.")
    return df

# --- 3. RELEVANCE SCORING AND PAIRWISE DATA GENERATION ---

def assign_relevance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns a pseudo-relevance score based on log heuristics.
    """
    df['status_score'] = df['status_category'].map({
        '2xx_Success': 3.0,
        '3xx_Redirection': 2.0,
        '4xx_Client_Error': 1.0,
        '5xx_Server_Error': 0.0,
        'Other': 0.5
    }).fillna(0.0)

    # Normalize request time: smaller time -> higher score.
    max_time = df['request_time'].max()
    min_time = df['request_time'].min()
    df['time_score'] = (max_time - df['request_time']) / (max_time - min_time + 1e-6)

    # Path depth score
    max_depth = df['path_depth'].max()
    df['path_score'] = df['path_depth'] / (max_depth + 1e-6)

    # Final Relevance Score
    df[TARGET_COL] = (
        0.5 * df['status_score'] +
        0.4 * df['time_score'] +
        0.1 * df['path_score']
    )

    # Discretize the score into 'grades'
    bins = np.linspace(df[TARGET_COL].min(), df[TARGET_COL].max(), 6)
    df['relevance_grade'] = pd.cut(df[TARGET_COL], bins=bins, labels=False, include_lowest=True) + 1
    df['relevance_grade'] = df['relevance_grade'].astype(int)

    print(f"Assigned relevance scores (grades from 1 to 5).")
    return df

def create_pairwise_data(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Generates training data in a pairwise format, as required by RankNet.
    """
    print("Generating pairwise training data. This may take a moment...")
    pairwise_data = []
    MAX_PAIRS_PER_GRADE_DIFF = 100000

    # Group records by relevance grade
    graded_groups = df.groupby('relevance_grade')[features + ['relevance_grade']].apply(lambda x: x.to_dict('records')).to_dict()
    grade_levels = sorted(graded_groups.keys(), reverse=True)

    start_time = time.time()
    for i in range(len(grade_levels)):
        grade_a = grade_levels[i]
        items_a = graded_groups[grade_a]

        for j in range(i + 1, len(grade_levels)):
            grade_b = grade_levels[j]
            items_b = graded_groups[grade_b]

            if grade_a > grade_b:
                num_pairs = 0
                random.shuffle(items_a)
                random.shuffle(items_b)

                sample_limit = min(2000, len(items_a), len(items_b))

                for item_a in items_a[:sample_limit]:
                    for item_b in items_b[:sample_limit]:
                        if num_pairs >= MAX_PAIRS_PER_GRADE_DIFF:
                            break

                        # 1. Pair A > B (Label 1)
                        pair_instance_ab = {}
                        for f in features:
                            pair_instance_ab[f'{f}_diff'] = item_a[f] - item_b[f]
                        pair_instance_ab[PAIRWISE_LABEL_COL] = 1
                        pairwise_data.append(pair_instance_ab)

                        # 2. Pair B < A (Label 0)
                        pair_instance_ba = {}
                        for f in features:
                            pair_instance_ba[f'{f}_diff'] = item_b[f] - item_a[f]
                        pair_instance_ba[PAIRWISE_LABEL_COL] = 0
                        pairwise_data.append(pair_instance_ba)

                        num_pairs += 2

                    if num_pairs >= MAX_PAIRS_PER_GRADE_DIFF:
                        break

    end_time = time.time()
    print(f"Pair generation complete. Total pairs: {len(pairwise_data)}. Time taken: {end_time - start_time:.2f}s")
    return pd.DataFrame(pairwise_data)

# --- 4. RANKNET MODEL TEMPLATE ---

class RankNetTemplate:
    """
    A template class to demonstrate the structure of a RankNet project
    using LogisticRegression as a STAND-IN.
    """
    def __init__(self, features):
        self.features = features
        self.pipeline = self._create_pipeline()
        print("RankNet Template Initialized.")

    def _create_pipeline(self):
        """Creates the feature preprocessing and model pipeline."""
        numeric_features = [f'{f}_diff' for f in self.features]

        preprocessor = ColumnTransformer(
            transformers=[('num', StandardScaler(), numeric_features)],
            remainder='passthrough'
        )

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=RANDOM_STATE, solver='liblinear'))
        ])
        return model

    def train(self, X_train, y_train):
        """Trains the pairwise classification model."""
        print("Starting training of the RankNet Template (Logistic Regression STAND-IN)...")
        self.pipeline.fit(X_train, y_train)
        print("Training complete.")

    def predict_scores(self, X_original):
        """
        Predicts the raw relevance scores for original (non-pairwise) items.
        """
        numeric_features = [f for f in self.features]

        score_preprocessor = ColumnTransformer(
            transformers=[('num', StandardScaler(), numeric_features)],
            remainder='passthrough'
        )

        # Train a simple classifier on original data to proxy the RankNet's scoring function
        score_model = Pipeline(steps=[
            ('preprocessor', score_preprocessor),
            ('regressor', LogisticRegression(random_state=RANDOM_STATE, solver='liblinear'))
        ])

        print("Training a dummy score model for prediction...")
        try:
            X_score = X_original[numeric_features]
            y_score = X_original['relevance_grade']
            score_model.fit(X_score, y_score)

            # Use probability of the highest class as the score proxy
            num_classes = len(np.unique(y_score))
            return score_model.predict_proba(X_score)[:, num_classes - 1]
        except Exception as e:
            print(f"Warning: Dummy score prediction failed: {e}. Returning zeros.")
            return np.zeros(len(X_original))

    def evaluate(self, X_test_pairwise, y_test_pairwise):
        """Evaluates the model on the pairwise test set (classification accuracy)."""
        accuracy = self.pipeline.score(X_test_pairwise, y_test_pairwise)
        print(f"Pairwise Classification Accuracy: {accuracy:.4f}")
        return accuracy

# --- MAIN PROJECT EXECUTION ---

def run_project(file_path: str):
    """
    Main function to execute the log analysis and RankNet template project.
    """
    print("--- RankNet Log Analysis Project Start ---")

    # 1. Load and Clean Data
    df = load_and_clean_data(file_path)
    if df.empty:
        print("--- Project Halted: Could not load data. ---")
        return

    # 2. Feature Engineering
    df = engineer_features(df)

    # 3. Assign Relevance Score
    df = assign_relevance(df)

    # Define the final numerical features for the model
    final_features = [
        'request_length', 'bytes_sent', 'request_time',
        'upstream_response_time', 'hour_of_day', 'day_of_week',
        'is_weekend', 'path_depth', 'is_bot', 'is_internal_ip',
        'status_score', 'time_score', 'path_score'
    ]

    # 4. Generate Pairwise Training Data
    # Use a small sample (500 records) to manage memory and execution time
    df_sample = df.sample(n=min(500, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
    df_pairwise = create_pairwise_data(df_sample, final_features)

    if df_pairwise.empty or len(df_pairwise) == 0:
        print("Error: No pairwise data generated. Exiting.")
        print("--- Project Halted: No data to train on. ---")
        return

    # Extract features and labels for the pairwise model
    X_pairwise = df_pairwise.drop(columns=[PAIRWISE_LABEL_COL])
    y_pairwise = df_pairwise[PAIRWISE_LABEL_COL]

    # Split the pairwise dataset
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X_pairwise, y_pairwise, test_size=0.2, random_state=RANDOM_STATE, stratify=y_pairwise
    )
    print(f"Pairwise Train Samples: {len(X_train_p)}, Test Samples: {len(X_test_p)}")

    # 5. Initialize and Train RankNet Template
    ranknet_model = RankNetTemplate(final_features)
    ranknet_model.train(X_train_p, y_train_p)

    # 6. Evaluate
    ranknet_model.evaluate(X_test_p, y_test_p)

    # 7. Predict and Analyze
    df_predict_sample = df.sample(n=min(50, len(df)), random_state=RANDOM_STATE+1).reset_index(drop=True)
    df_predict_sample['predicted_relevance_score'] = ranknet_model.predict_scores(df_predict_sample)

    # Sort and show the top ranked items based on the predicted score
    ranked_results = df_predict_sample.sort_values(
        by='predicted_relevance_score', ascending=False
    )[['timestamp', 'remote_addr', 'request', 'status', TARGET_COL, 'predicted_relevance_score']].head(10)

    print("\n--- Top 10 Ranked Log Entries (Based on Predicted Score) ---")

    # --- ***THIS IS THE FIX*** ---
    # Convert the DataFrame to an HTML table and display it
    # This renders a clean, formatted table in the ipywidgets.Output
    html_table = ranked_results.to_html(index=False, border=1)
    display(HTML(html_table))
    # --- ***END OF FIX*** ---

    print("\n--- RankNet Log Analysis Project End ---")


# --- UI CREATION ---

# 1. Create UI Components
file_path_input = widgets.Text(
    value='train.csv',
    description='Dataset Path:',
    style={'description_width': 'initial'}
)

run_button = widgets.Button(
    description='Run RankNet Analysis',
    button_style='success', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click to start the project pipeline',
    icon='play'
)

# 2. Create an Output widget to capture print statements
output_area = widgets.Output()

# 3. Define the function to be called on button click
def on_run_button_clicked(b):
    # Clear previous output
    output_area.clear_output(wait=True)

    # Run the project inside the output_area context
    # This will redirect all print() and display() calls to this widget
    with output_area:
        # Get the file path from the text box
        file_path = file_path_input.value
        run_project(file_path)

# 4. Link the button click to the function
run_button.on_click(on_run_button_clicked)

# 5. Display the UI
print("--- RankNet Project UI ---")
print("Please ensure your 'train.csv' file is uploaded to the Colab session.")
print("Click the button below to start the analysis.")

display(file_path_input, run_button, output_area)
