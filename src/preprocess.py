import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Columns that are IDs, geo-data, or would leak the target
_DROP_COLS = [
    "CustomerID", "Count", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude",
    "Churn Label",   # text copy of target  → leakage
    "Churn Score",   # derived from target  → leakage
    "CLTV",          # derived metric       → leakage
    "Churn Reason",  # only known post-churn → leakage
]


def _is_string_col(series: pd.Series) -> bool:
    """Return True for both pandas 'object' and newer 'str' / StringDtype."""
    return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)


def preprocess(path: str):
    """
    Load and clean the raw dataset.

    Returns
    -------
    df           : cleaned, fully-numeric DataFrame
    target_col   : name of the target column
    encoders     : dict {col_name: fitted LabelEncoder}
    feature_cols : list of feature column names (in model order)
    """

    # FIX 1: file is Excel despite the .csv extension
    ext = path.lower(); df = pd.read_csv(path) if ext.endswith(".csv") else pd.read_excel(path)

    print("Columns:", df.columns.tolist())
    print("Initial shape:", df.shape)

    # FIX 2: handle spaces + mixed case in column names
    target_col = None
    for col in df.columns:
        if col.lower().replace(" ", "") in ("churnvalue", "churn", "target", "exited"):
            target_col = col
            break

    if target_col is None:
        raise ValueError(
            "No target column found. Expected one of: "
            "'Churn Value', 'Churn', 'Target', 'Exited'.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    print(f"Target column: '{target_col}'")

    # FIX 5: drop leakage / ID columns
    cols_to_drop = [c for c in _DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # FIX 4: Total Charges has 11 blank strings — coerce, don't drop rows
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(
            df["Total Charges"], errors="coerce"
        ).fillna(0.0)

    # FIX 3: targeted dropna — only drop rows still genuinely missing
    before = len(df)
    df = df.dropna()
    print(f"Rows after cleaning: {len(df)} (dropped {before - len(df)})")

    feature_cols = [c for c in df.columns if c != target_col]

    # FIX 6 + 7: encode string columns; detect both 'object' and 'str' dtypes
    encoders: dict = {}
    for col in feature_cols:
        if _is_string_col(df[col]):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Ensure target is int
    df[target_col] = df[target_col].astype(int)

    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Categorical encoded: {list(encoders.keys())}")

    return df, target_col, encoders, feature_cols