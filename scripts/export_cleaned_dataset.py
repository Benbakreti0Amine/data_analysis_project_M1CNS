import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Source file path (from your environment - matches notebook)
src = r"C:\Users\user\.cache\kagglehub\datasets\muratkokludataset\pumpkin-seeds-dataset\versions\1\Pumpkin_Seeds_Dataset\Pumpkin_Seeds_Dataset.xlsx"

out_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(out_dir, exist_ok=True)

if not os.path.isfile(src):
    raise SystemExit(f"Source dataset not found at: {src}\nPlease update the path in this script to where your dataset actually is.")

print('Reading raw dataset from:', src)
df = pd.read_excel(src)
print('Raw dataset shape:', df.shape)

# If the label column name differs, adjust here
label_col = 'Class' if 'Class' in df.columns else df.columns[-1]

# Separate features and labels
if label_col in df.columns:
    X = df.drop(columns=[label_col]).copy()
    y = df[label_col].copy()
else:
    X = df.copy()
    y = pd.Series(dtype='int')

# Drop columns with a single unique value (constant columns)
const_cols = [c for c in X.columns if X[c].nunique() <= 1]
if const_cols:
    print('Dropping constant columns:', const_cols)
    X = X.drop(columns=const_cols)

# Label encode the target if present
if not y.empty:
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_name = label_col + '_encoded'
    print('Encoded label classes:', list(le.classes_))
else:
    y_enc = None
    y_name = None

# Scale the features (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Combine into a single DataFrame to export
if y_enc is not None:
    out_df = X_scaled_df.copy()
    out_df[y_name] = y_enc
else:
    out_df = X_scaled_df.copy()

out_path = os.path.join(out_dir, 'cleaned_pumpkin_seeds_scaled.csv')
out_df.to_csv(out_path, index=False)
print('Saved cleaned (scaled) dataset to:', out_path)

# Also save an unscaled version (raw features after dropping constant columns + labels)
if y_enc is not None:
    out_unscaled = X.copy()
    out_unscaled[label_col] = y
    out_path2 = os.path.join(out_dir, 'cleaned_pumpkin_seeds_unscaled.csv')
    out_unscaled.to_csv(out_path2, index=False)
    print('Saved cleaned (unscaled) dataset to:', out_path2)

print('Done.')
