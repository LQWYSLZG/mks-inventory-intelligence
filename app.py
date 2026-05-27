import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import io
from scipy import stats as scipy_stats

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Inventory Intelligence for MKS SPORTS INDUSTRIES",
    page_icon="🏸",
    layout="wide",
)

# ── Shared CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global font & background ── */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a2e14 0%, #145a2e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e6f4ec !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #2e7d4f;
    }

    /* ── Main header banner ── */
    .mks-header {
        background: linear-gradient(135deg, #0a2e14 0%, #145a2e 60%, #0a2e14 100%);
        border-left: 6px solid #f0c040;
        border-radius: 10px;
        padding: 28px 36px;
        margin-bottom: 24px;
    }
    .mks-header h1 {
        color: #ffffff;
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: 0.5px;
        line-height: 1.2;
    }
    .mks-header p {
        color: #f0c040;
        font-size: 0.9rem;
        margin: 0;
        font-weight: 500;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    /* ── Tab bar ── */
    [data-testid="stTabs"] [role="tablist"] {
        border-bottom: 2px solid #f0c040;
        gap: 4px;
    }
    [data-testid="stTabs"] [role="tab"] {
        background: #edf7f1;
        border-radius: 6px 6px 0 0;
        color: #0a2e14;
        font-weight: 600;
        padding: 6px 16px;
        border: none;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        background: #0a2e14;
        color: #f0c040 !important;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0a2e14;
        margin-top: 1rem;
        border-left: 3px solid #f0c040;
        padding-left: 8px;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: #ffffff;
        border: 1px solid #c8e6d0;
        border-top: 3px solid #f0c040;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }

    /* ── ABC badges ── */
    .badge-A { background:#e8f5ee; color:#1a5c30; padding:2px 8px; border-radius:4px; font-weight:700; border:1px solid #a8d5b8; }
    .badge-B { background:#fdf6e3; color:#7a5c00; padding:2px 8px; border-radius:4px; font-weight:700; border:1px solid #e0cc80; }
    .badge-C { background:#fdf0f0; color:#8c2020; padding:2px 8px; border-radius:4px; font-weight:700; border:1px solid #e0a8a8; }

    /* ── Primary buttons ── */
    [data-testid="stButton"] > button[kind="primary"] {
        background: #0a2e14;
        color: #f0c040;
        border: 1px solid #f0c040;
        font-weight: 600;
        border-radius: 6px;
    }
    [data-testid="stButton"] > button[kind="primary"]:hover {
        background: #145a2e;
        color: #ffe080;
    }

    /* ── Dataframe header ── */
    [data-testid="stDataFrame"] thead tr th {
        background-color: #0a2e14 !important;
        color: #f0c040 !important;
    }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def categorize_item(name: str) -> str:
    """Legacy fallback — used only if auto-categorization is not yet run."""
    return "Uncategorized"

def auto_categorize_items(item_names: pd.Series, n_clusters: int = 4) -> pd.Series:
    """
    Automatically categorize items using TF-IDF on item names + KMeans clustering.
    Returns a Series mapping item name → category label (most representative word).
    Works for any product dataset regardless of industry.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    import re

    names = item_names.unique().tolist()

    if len(names) < n_clusters:
        # Not enough items to cluster — assign all to one group
        return pd.Series({n: "All Items" for n in names})

    # Clean names: remove numbers, sizes, special chars — keep meaningful words
    def clean(text):
        text = re.sub(r'\b\d+(\.\d+)?\s*(mm|cm|g|kg|ml|l|inch|in|oz|lb|pcs|pc|set|pair|no|no\.|#)?\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        return text.lower().strip()

    cleaned = [clean(n) for n in names]

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        min_df=1,
        stop_words='english',
        max_features=500
    )
    try:
        X = vectorizer.fit_transform(cleaned)
    except Exception:
        return pd.Series({n: "All Items" for n in names})

    # KMeans clustering
    n_clusters = min(n_clusters, len(names))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # Label each cluster with its top TF-IDF term
    feature_names = vectorizer.get_feature_names_out()
    cluster_labels = {}
    for cluster_id in range(n_clusters):
        center = km.cluster_centers_[cluster_id]
        top_idx = center.argsort()[::-1][:3]
        top_words = [feature_names[i].title() for i in top_idx if feature_names[i].strip()]
        cluster_labels[cluster_id] = top_words[0] if top_words else f"Group {cluster_id + 1}"

    return pd.Series({name: cluster_labels[label] for name, label in zip(names, labels)})

def abc_label(pct: float) -> str:
    if pct <= 80:   return "A"
    elif pct <= 95: return "B"
    return "C"

def xyz_label(cv: float) -> str:
    if cv <= 0.5:  return "X"
    elif cv <= 1.0: return "Y"
    return "Z"

def build_abc_df(df: pd.DataFrame, item_type: str | None = None, value_col: str = "Quantity") -> pd.DataFrame:
    subset = df if item_type is None else df[df["Item Type"] == item_type]
    # Drop rows where the value column is NaN (e.g. missing price in revenue mode)
    subset = subset.dropna(subset=[value_col]) if value_col in subset.columns else subset
    ranked = (subset.groupby("Item Name")[value_col]
              .sum().sort_values(ascending=False).reset_index())
    ranked.columns = ["Item Name", value_col]
    ranked["Rank"] = range(1, len(ranked) + 1)
    total = ranked[value_col].sum()
    ranked["Cumulative Demand %"] = (ranked[value_col].cumsum() / total * 100).round(2)
    ranked["ABC"] = ranked["Cumulative Demand %"].apply(abc_label)
    return ranked

def build_xyz_series(df: pd.DataFrame) -> pd.DataFrame:
    """Compute XYZ classification per SKU based on monthly demand CV."""
    if "Invoice Date" not in df.columns:
        return pd.DataFrame(columns=["Item Name", "CV", "XYZ"])
    df2 = df.copy()
    df2["Month"] = df2["Invoice Date"].dt.to_period("M")
    monthly = df2.groupby(["Item Name", "Month"])["Quantity"].sum().reset_index()
    cv_data = []
    for item, grp in monthly.groupby("Item Name"):
        mean_q = grp["Quantity"].mean()
        # Use sample std (ddof=1) — more accurate for small samples
        std_q  = grp["Quantity"].std(ddof=1) if len(grp) > 1 else 0.0
        cv = (std_q / mean_q) if mean_q > 0 else 0.0
        cv_data.append({"Item Name": item, "CV": round(cv, 3), "XYZ": xyz_label(cv)})
    return pd.DataFrame(cv_data)

def pareto_chart(abc_df, title, bar_color, top_n=20, value_col="Quantity"):
    top = abc_df.head(top_n)
    actual_n = len(top)
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(top["Item Name"], top[value_col], color=bar_color, alpha=0.85, label=value_col)
    ax1.set_ylabel(value_col, fontsize=11)
    ax1.tick_params(axis="x", rotation=75, labelsize=7)
    ax2 = ax1.twinx()
    ax2.plot(top["Item Name"], top["Cumulative Demand %"], color="crimson",
             marker="o", linewidth=2, markersize=4, label="Cumulative %")
    ax2.axhline(80, color="orange", linestyle="--", linewidth=1, alpha=0.7)
    ax2.axhline(95, color="red",    linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_ylabel("Cumulative %", fontsize=11)
    ax2.set_ylim(0, 110)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig, actual_n

def abc_summary_metrics(abc_df, value_col="Quantity"):
    counts = abc_df["ABC"].value_counts().reindex(["A","B","C"], fill_value=0)
    total_qty = abc_df[value_col].sum()
    is_revenue = value_col not in ("Quantity",)
    cols = st.columns(3)
    info = {"A": ("🟢","top 80%"), "B": ("🟡","80–95%"), "C": ("🔴","tail 5%")}
    for col, cat in zip(cols, ["A","B","C"]):
        qty = abc_df[abc_df["ABC"] == cat][value_col].sum()
        icon, rng = info[cat]
        if is_revenue:
            val_str = f"${qty:,.0f} ({qty/total_qty*100:.1f}%)"
        else:
            val_str = f"{qty:,} units ({qty/total_qty*100:.1f}%)"
        col.metric(f"{icon} Class {cat} ({rng})",
                   f"{counts[cat]} SKUs",
                   val_str,
                   delta_color="off")

def highlight_abc(row):
    c = {"A":"background-color:#e8f5ee; color:#1a1a1a",
         "B":"background-color:#fdf6e3; color:#1a1a1a",
         "C":"background-color:#fdf0f0; color:#1a1a1a"}
    style = c.get(row["ABC"], "")
    return [style] * len(row)

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()

def apply_keyword_rules(name: str, active_rules: list) -> str:
    """Categorize an item name using keyword rules. Returns 'Other Items' if no match."""
    n = str(name).upper()
    for rule in active_rules:
        kws = rule["keywords"] if isinstance(rule["keywords"], list) else [rule["keywords"]]
        if any(kw.upper() in n for kw in kws):
            return rule["category"]
    return "Other Items"

# Known type column names for smart default detection
KNOWN_TYPE_COLS = ["Item Type", "Category", "Product Type", "Type",
                   "Product Category", "Item Category", "Group", "Product Group"]

@st.cache_data
def build_smart_defaults(df: pd.DataFrame, top_n: int = 3) -> list:
    """
    Build default keyword rules intelligently:
    Tier 1 — use existing type column (top N most frequent values)
    Tier 2 — extract last meaningful word from item names (top N)
    Tier 3 — empty, let user define their own
    """
    import re
    from collections import Counter
    STOP_WORDS = {"new","black","white","red","blue","green","grey","gray",
                  "matt","glossy","matte","pro","plus","max","mini","lite",
                  "light","dark","gold","silver","navy","orange","purple",
                  "yellow","pink","brown","beige","clear","set","pair"}

    # Tier 1: existing type column
    type_col = next((c for c in KNOWN_TYPE_COLS if c in df.columns), None)
    if type_col:
        top_types = (df[type_col].dropna().astype(str).str.strip()
                     .value_counts().head(top_n).index.tolist())
        if top_types:
            return [{"category": t, "keywords": t.upper()} for t in top_types]

    # Tier 2: last meaningful word from item names
    if "Item Name" in df.columns:
        last_words = []
        for name in df["Item Name"].dropna().unique():
            words = re.sub(r'[^a-zA-Z\s]', ' ', str(name)).upper().split()
            for word in reversed(words):
                if word.lower() not in STOP_WORDS and len(word) > 2:
                    last_words.append(word.title())
                    break
        if last_words:
            top_words = [w for w, cnt in Counter(last_words).most_common(top_n) if cnt >= 2]
            if top_words:
                return [{"category": w, "keywords": w.upper()} for w in top_words]

    # Tier 3: empty
    return []

ABC_XYZ_RECOMMENDATIONS = {
    "AX": ("🔵 Critical & Stable",    "Tight reorder points, low safety stock. High priority for replenishment."),
    "AY": ("🔵 Critical & Variable",  "Moderate safety stock. Monitor closely and review frequently."),
    "AZ": ("🔴 Critical & Erratic",   "High safety stock. Investigate demand drivers. Frequent review essential."),
    "BX": ("🟢 Important & Stable",   "Standard replenishment. Automate reorder where possible."),
    "BY": ("🟡 Important & Variable", "Moderate safety stock. Regular review cycle."),
    "BZ": ("🟠 Important & Erratic",  "Higher safety stock. Consider demand smoothing or supplier agreements."),
    "CX": ("⚪ Low Value & Stable",   "Minimal stock. Consider consolidating orders to reduce handling cost."),
    "CY": ("⚪ Low Value & Variable", "Low stock. Review if item is worth keeping in range."),
    "CZ": ("⚪ Low Value & Erratic",  "Consider discontinuing or ordering only on demand."),
}

# ── Forecast helper functions ─────────────────────────────────────────────────
def fc_compute_mape(actual, fitted):
    actual, fitted = np.array(actual, dtype=float), np.array(fitted, dtype=float)
    mask = actual > 0
    if mask.sum() == 0:
        return None
    return round(float(np.mean(np.abs((actual[mask] - fitted[mask]) / actual[mask])) * 100), 1)

def fc_compute_mase(actual, fitted):
    actual  = np.array(actual,  dtype=float)
    fitted  = np.array(fitted,  dtype=float)
    model_mae = np.mean(np.abs(actual - fitted))
    naive_mae = np.mean(np.abs(np.diff(actual))) if len(actual) > 1 else None
    if naive_mae is None or naive_mae == 0:
        return None
    return round(float(model_mae / naive_mae), 2)

def fc_accuracy_display(actual, fitted, model_label):
    mape = fc_compute_mape(actual, fitted)
    mase = fc_compute_mase(actual, fitted)
    low_vol = np.mean(np.array(actual, dtype=float)) < 10
    if mase is not None and low_vol:
        if   mase < 0.8:  label, detail = "\U0001f7e2 Excellent fit", f"The model is significantly better than a naive guess (MASE: {mase})."
        elif mase < 1.0:  label, detail = "\U0001f7e1 Good fit",      f"The model beats a naive guess (MASE: {mase})."
        elif mase < 1.5:  label, detail = "\U0001f7e0 Moderate fit",  f"The model is close to a naive guess (MASE: {mase}). Demand may be too erratic to forecast reliably."
        else:             label, detail = "\U0001f534 Unreliable",    f"A naive guess outperforms the model (MASE: {mase}). This item has highly variable demand, so use the forecast as a rough guide only."
        st.info(f"Forecast accuracy: **{label}**. {detail}")
        st.caption(f"Model: {model_label} · Note: MAPE not shown as percentage errors are misleading at low average volume ({np.mean(np.array(actual,dtype=float)):.1f} units/month).")
    elif mape is not None:
        if   mape <= 15: label, detail = "\U0001f7e2 Excellent fit",    f"Average error of {mape}% per month. Forecasts are highly reliable."
        elif mape <= 30: label, detail = "\U0001f7e1 Good fit",         f"Average error of {mape}% per month. Forecasts are reasonably reliable."
        elif mape <= 50: label, detail = "\U0001f7e0 Moderate fit",     f"Average error of {mape}% per month. Demand is variable, so use forecasts as a directional guide."
        else:            label, detail = "\U0001f534 High uncertainty", (
            f"Average error of {mape}% per month. "
            "This item has highly erratic demand that is difficult to predict accurately. "
            "Use the forecast as a rough order-of-magnitude estimate and apply extra safety stock to compensate."
        )
        mase_note = f" \u00b7 MASE: {mase} ({'better' if mase is not None and mase < 1 else 'worse'} than naive)" if mase is not None else ""
        st.info(f"Forecast accuracy: **{label}**. {detail}")
        st.caption(f"Model: {model_label}{mase_note}")

def fc_check_erratic(forecast_vals, historical_vals):
    hist_std  = np.std(historical_vals)
    fc_std    = np.std(forecast_vals)
    hist_mean = np.mean(np.abs(historical_vals))
    if hist_std > 0 and fc_std > 5 * hist_std and hist_mean > 0 and fc_std > hist_mean:
        st.warning(
            "\u26a0\ufe0f The forecast shows unusually high variability compared to historical demand. "
            "This can happen with strong trends or limited data. "
            "Treat these numbers as directional estimates only."
        )
        return True
    return False

def fc_detect_growth_plateau(vals):
    if len(vals) < 8:
        return False
    arr = np.array(vals, dtype=float)
    mid = len(arr) // 2
    early_slope = np.polyfit(np.arange(mid), arr[:mid], 1)[0]
    late_slope  = np.polyfit(np.arange(len(arr)-mid), arr[mid:], 1)[0]
    return early_slope > 1.0 and abs(late_slope) < early_slope * 0.3

# ── Croston's method for intermittent demand ──────────────────────────────────
def croston_forecast(series, forecast_periods, alpha=0.1):
    """
    Croston's method for intermittent demand forecasting.
    series: list/array of demand values (can include zeros)
    Returns list of forecast_periods forecasted values (constant level)
    """
    demand = [x for x in series if x > 0]
    intervals = []
    last_idx = -1
    for i, x in enumerate(series):
        if x > 0:
            if last_idx >= 0:
                intervals.append(i - last_idx)
            last_idx = i
    if not demand or not intervals:
        return [0.0] * forecast_periods
    z = demand[0]
    p = intervals[0] if intervals else 1
    for d, iv in zip(demand[1:], intervals[1:]):
        z = alpha * d + (1 - alpha) * z
        p = alpha * iv + (1 - alpha) * p
    rate = z / p if p > 0 else 0
    return [max(0, round(rate, 1))] * forecast_periods

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR – file upload
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏸 MKS SPORTS INDUSTRIES")
    st.markdown("**Inventory Intelligence Platform**")
    st.markdown("---")
    st.markdown("Upload your demand history Excel files to start and unlock all features.")
    uploaded = st.file_uploader("📂 Upload Excel file(s)", type=["xlsx","xls"],
                                accept_multiple_files=True)

st.markdown("""
<div class="mks-header">
    <h1>🏸 Inventory Intelligence for MKS SPORTS INDUSTRIES</h1>
    <p>Badminton Equipment Manufacturing · Supply Chain Demand Analytics Dashboard</p>
</div>
""", unsafe_allow_html=True)

if not uploaded:
    st.markdown("""
    <div style="background:#edf7f1; border:1px solid #c8e6d0; border-left:4px solid #f0c040;
                border-radius:8px; padding:20px 24px; margin-top:10px;">
        <h4 style="color:#0a2e14; margin:0 0 8px 0;">👈 Get Started</h4>
        <p style="color:#1a5c30; margin:0;">Upload one or more demand history Excel files from the sidebar to begin analysis. Multiple files will be merged automatically.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load & validate ───────────────────────────────────────────────────────────
@st.cache_data
def load_data(files):
    """
    Load one or more Excel files, harmonise columns, and merge into one dataframe.
    Each file can have multiple sheets. Columns with common variant names are standardised.
    """
    # Column name variants → standard name
    COL_MAP = {
        "product name": "Item Name", "product": "Item Name", "item": "Item Name",
        "sku name": "Item Name", "description": "Item Name",
        "qty": "Quantity", "units": "Quantity", "sales qty": "Quantity",
        "order qty": "Quantity", "quantity sold": "Quantity",
        "date": "Invoice Date", "order date": "Invoice Date",
        "sale date": "Invoice Date", "transaction date": "Invoice Date",
        "customer": "Customer Name", "client": "Customer Name",
        "buyer": "Customer Name", "client name": "Customer Name",
        "price": "Unit Price", "unit cost": "Unit Price",
        "selling price": "Unit Price", "sale price": "Unit Price",
        "revenue": "Revenue", "sales": "Revenue", "amount": "Revenue",
        "total": "Revenue", "total sales": "Revenue",
        "type": "Item Type", "category": "Item Type",
        "product type": "Item Type", "product category": "Item Type",
        "item category": "Item Type", "group": "Item Type",
    }
    SKIP_SHEETS = {"dropdowndata","dropdown","lookup","reference","config","settings","readme"}

    # Normalise to list
    if not isinstance(files, list):
        files = [files]

    all_frames = []

    for file in files:
        file_name = getattr(file, "name", "Unknown")
        try:
            xl = pd.ExcelFile(file)
        except Exception:
            continue

        for sheet in xl.sheet_names:
            if sheet.strip().lower() in SKIP_SHEETS:
                continue
            try:
                sdf = xl.parse(sheet)
                sdf.columns = sdf.columns.str.strip()
                # Drop unnamed and .1/.2 duplicate columns
                sdf = sdf.loc[:, ~sdf.columns.str.startswith("Unnamed")]
                sdf = sdf.loc[:, ~sdf.columns.str.contains(r'\.\d+$', regex=True)]

                # Harmonise column names
                rename_map = {}
                for col in sdf.columns:
                    std = COL_MAP.get(col.lower().strip())
                    if std and std not in sdf.columns:
                        rename_map[col] = std
                if rename_map:
                    sdf = sdf.rename(columns=rename_map)

                if "Item Name" not in sdf.columns or "Quantity" not in sdf.columns:
                    continue

                sdf = sdf.dropna(subset=["Item Name", "Quantity"])
                sdf["_Sheet"] = sheet
                sdf["_File"]  = file_name
                all_frames.append(sdf)
            except Exception:
                continue

    if not all_frames:
        return pd.DataFrame()

    # Merge all frames — missing columns filled with NaN
    df = pd.concat(all_frames, ignore_index=True, sort=False)

    if "Invoice Date" in df.columns:
        df["Invoice Date"] = pd.to_datetime(df["Invoice Date"], errors="coerce")

    # Deduplicate across files/sheets
    if "Invoice Number" in df.columns:
        dedup_cols = [c for c in ["Invoice Number", "Item Name", "Quantity"] if c in df.columns]
        df = df.drop_duplicates(subset=dedup_cols)

    return df

df = load_data(uploaded)

if df.empty:
    st.error("Could not load any data from the uploaded file(s). Please check that your files contain `Item Name` and `Quantity` columns.")
    st.stop()

# ── Data cleaning & quality report ───────────────────────────────────────────
def clean_data(df: pd.DataFrame):
    """
    Auto-fix common data quality issues and return cleaned df + quality report.
    """
    report = []
    original_len = len(df)

    # 1. Strip whitespace from Item Name
    if "Item Name" in df.columns:
        before = df["Item Name"].nunique()
        df["Item Name"] = df["Item Name"].astype(str).str.strip()
        after = df["Item Name"].nunique()
        if before != after:
            report.append(("✅ Fixed", f"Trimmed whitespace from Item Name, reducing unique SKUs from {before} to {after}"))

    # 2. Convert Quantity to numeric
    if "Quantity" in df.columns:
        non_numeric = pd.to_numeric(df["Quantity"], errors="coerce").isna().sum()
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
        if non_numeric > 0:
            report.append(("✅ Fixed", f"Converted Quantity to numeric: {non_numeric} non-numeric value(s) set to NaN"))

    # 3. Drop rows with missing Item Name or Quantity
    missing_rows = df[df["Item Name"].isna() | df["Quantity"].isna()]
    if len(missing_rows) > 0:
        df = df.dropna(subset=["Item Name", "Quantity"])
        report.append(("✅ Fixed", f"Removed {len(missing_rows)} row(s) with missing Item Name or Quantity"))

    # 4. Remove zero quantity rows
    zero_rows = (df["Quantity"] == 0).sum()
    if zero_rows > 0:
        df = df[df["Quantity"] != 0]
        report.append(("✅ Fixed", f"Removed {zero_rows} row(s) with zero Quantity"))

    # 5. Remove negative quantity rows
    neg_rows = (df["Quantity"] < 0).sum()
    if neg_rows > 0:
        df = df[df["Quantity"] >= 0]
        report.append(("✅ Fixed", f"Removed {neg_rows} row(s) with negative Quantity (likely returns/credits)"))

    # 6. Flag unparseable Invoice Dates
    if "Invoice Date" in df.columns:
        bad_dates = df["Invoice Date"].isna().sum()
        if bad_dates > 0:
            report.append(("⚠️ Warning", f"{bad_dates} row(s) have unparseable Invoice Date and are excluded from date-based analysis"))

    # 7. Flag duplicate rows (after dedup in load_data, any remaining are soft dupes)
    if "Invoice Number" in df.columns:
        dupes = df.duplicated(subset=["Invoice Number","Item Name","Quantity"], keep=False).sum()
        if dupes > 0:
            report.append(("⚠️ Warning", f"{dupes} row(s) appear to be duplicate invoice lines. Please review your source data"))

    # 8. All good message
    rows_removed = original_len - len(df)
    if rows_removed > 0:
        report.append(("Summary", f"Dataset reduced from {original_len:,} to {len(df):,} rows after cleaning ({rows_removed:,} removed)"))
    else:
        report.append(("Summary", f"No issues found. All {original_len:,} rows are clean"))

    return df, report

df, quality_report = clean_data(df)

# ── Location / Warehouse detection ───────────────────────────────────────────
LOCATION_COLS = ["Location", "Warehouse", "Store", "Branch", "Site", "Region"]
location_col = next((c for c in LOCATION_COLS if c in df.columns), None)
all_locations = sorted(df[location_col].dropna().unique().tolist()) if location_col else []

# Keep a pre-filter copy for the overview breakdown table
df_all_locations = df.copy()

# ── Sidebar continued: location filter + required columns + footer ────────────
# Rendered after data loads so location options are known.
# Streamlit appends sidebar widgets in execution order, so this block
# appears directly below the file uploader.
with st.sidebar:
    if location_col and all_locations:
        st.markdown("---")
        st.markdown("**📍 Filter by Location**")
        selected_location = st.selectbox(
            "Warehouse / Location",
            ["All Locations"] + all_locations,
            key="location_filter"
        )
    else:
        selected_location = "All Locations"

missing = {"Item Name","Quantity"} - set(df.columns)
if missing:
    st.error(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    st.stop()

# ── Apply location filter — df is now scoped to selected location for all tabs ──
if location_col and selected_location != "All Locations":
    df = df[df[location_col] == selected_location].copy()
    if df.empty:
        st.warning(f"No data found for location: {selected_location}")
        st.stop()

if location_col and selected_location != "All Locations":
    st.info(f"📍 Showing data for: **{selected_location}**. Switch location in the sidebar.")

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 ABC Inventory Analysis",
    "📈 Demand Forecast",
    "🔔 Inventory Alerts & Optimization",
    "👥 Customer Segmentation",
    "📉 Demand Health",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – ABC RANKING + XYZ ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("📊 ABC Inventory Analysis",
              help="ABC analysis ranks your items by total demand contribution so you can focus stock management effort where it matters most. Class A items drive 80% of demand, B items the next 15%, and C items the remaining 5%. The XYZ layer adds demand stability: X is stable, Y is variable, and Z is erratic.")

    with st.expander("🔍 Dataset Overview", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Total rows", f"{len(df):,}")
        c2.metric("Unique items", f"{df['Item Name'].nunique():,}")
        c3.metric("Total quantity", f"{df['Quantity'].sum():,}")
        if "Invoice Date" in df.columns:
            date_min = df['Invoice Date'].min().strftime('%d %b %Y')
            date_max = df['Invoice Date'].max().strftime('%d %b %Y')
            st.markdown(f"**📅 Date range:** {date_min} — {date_max}")

        for status, message in quality_report:
            if status == "✅ Fixed":
                st.success(f"{status}: {message}")
            elif status == "⚠️ Warning":
                st.warning(f"{status}: {message}")
            else:
                st.info(f"🧹 Data Quality Report: {message}")

        st.markdown("**Preview (first 10 rows, oldest first)**")
        preview_df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed") or c in ("_Sheet","_File") or df[c].isna().all()])
        if "Invoice Date" in preview_df.columns:
            preview_df = preview_df.sort_values("Invoice Date", ascending=True)
        st.dataframe(preview_df.head(10), use_container_width=True)

        # ── Multi-file source summary ─────────────────────────────────────
        if "_File" in df.columns and df["_File"].nunique() > 1:
            st.markdown("**📁 Loaded from multiple files:**")
            file_summary = df.groupby("_File").agg(
                Rows=("Item Name","count"),
                Items=("Item Name","nunique")
            ).reset_index()
            file_summary.columns = ["File", "Rows", "Unique Items"]
            st.dataframe(file_summary, use_container_width=True, hide_index=True)

        # ── Location breakdown (always uses full unfiltered data) ────────
        if location_col and len(all_locations) > 1:
            st.markdown("**📍 Demand by Location:**")
            loc_summary = df_all_locations.groupby(location_col).agg(
                Rows=("Item Name","count"),
                Items=("Item Name","nunique"),
                Total_Qty=("Quantity","sum")
            ).reset_index()
            loc_summary.columns = ["Location","Rows","Unique Items","Total Quantity"]
            st.dataframe(loc_summary, use_container_width=True, hide_index=True)
            if selected_location != "All Locations":
                st.caption(f"All tabs are currently filtered to: {selected_location}. Change in the sidebar to switch.")

    # ── Item Categorization ───────────────────────────────────────────────────
    # Smart defaults built from the combined dataset at module level
    DEFAULT_RULES = build_smart_defaults(df)

    with st.expander("✏️ Define Item Categories", expanded=False):
        st.caption("Define your own category names and the keywords that identify them from the dataset (Items that match no rule are placed in Other Items).")

        # Number of keyword rules
        n_rules = st.number_input("Number of categories to define", min_value=1, max_value=15,
                                   value=max(1, len(DEFAULT_RULES)), step=1, key="n_rules")

        rules = []
        for i in range(int(n_rules)):
            default = DEFAULT_RULES[i] if i < len(DEFAULT_RULES) else {"category": f"Category {i+1}", "keywords": ""}
            c1, c2 = st.columns([1, 2])
            cat_name = c1.text_input(
                f"Category name #{i+1}",
                value=default["category"],
                key=f"rule_cat_{i}"
            )
            kw_input = c2.text_input(
                f"Keywords (comma-separated) #{i+1}",
                value=default["keywords"],
                placeholder="e.g. GRIP, TAPE, HANDLE",
                key=f"rule_kw_{i}"
            )
            keywords = [k.strip().upper() for k in kw_input.split(",") if k.strip()]
            if cat_name.strip() and keywords:
                rules.append({"category": cat_name.strip(), "keywords": keywords})

        apply_rules = st.button("✅ Apply Keyword Rules", type="primary", key="apply_rules")

    # Apply rules (either defaults on first load or user-defined after button press)
    # Use session state to persist applied rules across reruns
    # Reset smart defaults when uploaded files change
    uploaded_key = str(sorted([getattr(f, "name", "") for f in uploaded]))
    if st.session_state.get("_last_upload_key") != uploaded_key:
        st.session_state._last_upload_key = uploaded_key
        st.session_state.active_rules = DEFAULT_RULES
        st.session_state.user_defined_order = None
        st.session_state.rules_customised = False
        # Clear all widget state that depends on the dataset
        for key in ["abc_mode", "abc_date_range", "top_n", "n_rules"]:
            if key in st.session_state:
                del st.session_state[key]
        # Clear data cache so new file is fully reloaded
        st.cache_data.clear()
        # Clear cached reorder results so optimization models don't show stale data
        for _k in ["rop_df", "rop_display", "stock_df_cached", "last_opt_run",
                   "opt1_open", "opt2_open", "opt3_open", "opt4_open",
                   "slow_df", "slow_monthly", "stock_slow"]:
            st.session_state.pop(_k, None)
        st.rerun()

    if "active_rules" not in st.session_state:
        st.session_state.active_rules = DEFAULT_RULES
    if "user_defined_order" not in st.session_state:
        st.session_state.user_defined_order = None
    if not apply_rules and not st.session_state.get("rules_customised", False):
        st.session_state.active_rules = DEFAULT_RULES
    if apply_rules and rules:
        st.session_state.active_rules = [{"category": r["category"], "keywords": r["keywords"]} for r in rules]
        st.session_state.user_defined_order = [r["category"] for r in rules]
        st.session_state.rules_customised = True

    # Item Type and category ordering are applied after df_abc is created below
    # CATEGORIES and BAR_COLORS are also built after df_abc is created below

    # ── Analysis controls ─────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([1.5, 1.5, 3])

    # Initialise value_col and value_label defaults
    value_col   = "Quantity"
    value_label = "Quantity"

    # Demand vs Revenue toggle
    has_revenue = any(c in df.columns for c in ["Revenue","Price","Unit Price","Sales","Amount"])
    revenue_col = next((c for c in ["Revenue","Price","Unit Price","Sales","Amount"] if c in df.columns), None)

    # Check partial coverage upfront so we can warn before the toggle
    if has_revenue and revenue_col:
        missing_price_pct = df[revenue_col].isna().mean()
        partial_coverage = missing_price_pct > 0.05  # more than 5% missing
    else:
        partial_coverage = False

    if has_revenue:
        if partial_coverage:
            missing_n = df[revenue_col].isna().sum()
            ctrl1.warning(
                f"⚠️ {missing_n:,} of {len(df):,} rows have no price data. "
                "Revenue mode will exclude those items."
            )
        abc_mode = ctrl1.radio("Classify by", ["📋 Demand", f"💰 Revenue"],
                               horizontal=True, key="abc_mode")
        if "Revenue" in abc_mode:
            value_col   = revenue_col
            value_label = revenue_col
    else:
        ctrl1.info("⚙️ Currently classifying by Demand. Add a Unit Price or Revenue column to enable revenue-based analysis.")

    # Date range filter — build df_abc first
    if "Invoice Date" in df.columns:
        date_min_d = df["Invoice Date"].min().date()
        date_max_d = df["Invoice Date"].max().date()
        date_range = ctrl3.date_input("Filter date range", value=(date_min_d, date_max_d),
                                      min_value=date_min_d, max_value=date_max_d, key="abc_date_range")
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            df_abc = df[(df["Invoice Date"].dt.date >= date_range[0]) &
                        (df["Invoice Date"].dt.date <= date_range[1])].copy()
            if df_abc.empty:
                st.warning("No data in selected date range.")
                df_abc = df.copy()
        else:
            df_abc = df.copy()
    else:
        df_abc = df.copy()

    # Compute revenue value column on df_abc after it's defined
    if has_revenue and value_col != "Quantity":
        df_abc["_abc_value"] = (df_abc["Quantity"] * df_abc[revenue_col]).where(df_abc[revenue_col].notna())
        # Warn if partial coverage
        missing_price = df_abc[revenue_col].isna().sum()
        if missing_price > 0:
            st.warning(
                f"⚠️ {missing_price:,} of {len(df_abc):,} rows have no price data "
                "and are excluded from revenue analysis. "
                "Switch to **📋 Demand** mode to include all items."
            )
        value_col = "_abc_value"

    # Apply category rules to df_abc (now that it exists) and to df for other tabs
    df_abc["Item Type"] = df_abc["Item Name"].apply(
        lambda x: apply_keyword_rules(x, st.session_state.active_rules)
    )
    df["Item Type"] = df["Item Name"].apply(
        lambda x: apply_keyword_rules(x, st.session_state.active_rules)
    )

    # Final category list and ordering
    final_cats = df_abc["Item Type"].unique().tolist()
    user_order = st.session_state.get("user_defined_order", None)
    if user_order:
        ordered_cats = [c for c in user_order if c in final_cats]
        ordered_cats += sorted([c for c in final_cats if c not in ordered_cats and c != "Other Items"])
    else:
        ordered_cats = sorted([c for c in final_cats if c != "Other Items"])
    if "Other Items" in final_cats:
        ordered_cats.append("Other Items")

    cat_icons = ["📌","📌","📌","📌","📌","📌","📌","📌","📌","📌"]
    CATEGORIES = {"🏷️ All Items": None}
    for i, cat in enumerate(ordered_cats):
        icon = "📦" if cat == "Other Items" else cat_icons[i % len(cat_icons)]
        CATEGORIES[f"{icon} {cat}"] = cat

    palette = ["#145a2e","#2e9e5e","#f0c040","#7ab898","#4a90d9","#e07b39","#9b59b6","#e74c3c","#1abc9c","#f39c12"]
    BAR_COLORS = {None: palette[0]}
    for i, cat in enumerate(ordered_cats):
        BAR_COLORS[cat] = palette[(i + 1) % len(palette)]
    max_items = df_abc["Item Name"].nunique()
    top_n = ctrl2.slider("Items shown in Pareto chart", min_value=5,
                         max_value=max(5, min(max_items, 50)),
                         value=max(5, min(20, max_items)), key="top_n")

    # XYZ computed on filtered data so it reflects the selected period
    xyz_df_abc = build_xyz_series(df_abc) if "Invoice Date" in df_abc.columns else pd.DataFrame(columns=["Item Name","CV","XYZ"])
    if not xyz_df_abc.empty:
        xyz_df_abc["XYZ"] = xyz_df_abc["XYZ"].where(xyz_df_abc["CV"] > 0, other="N/A")

    st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

    cat_tabs = st.tabs(list(CATEGORIES.keys()))
    for ctab, (label, cat) in zip(cat_tabs, CATEGORIES.items()):
        with ctab:
            abc_df = build_abc_df(df_abc, cat, value_col=value_col)
            # Rename internal _abc_value column to readable label for display
            if value_col == "_abc_value" and "_abc_value" in abc_df.columns:
                abc_df = abc_df.rename(columns={"_abc_value": value_label})
            if abc_df.empty:
                st.warning(f"No items found for {label}.")
                continue
            if cat is not None:
                abc_summary_metrics(abc_df, value_col=value_label)
            actual_n = min(top_n, len(abc_df))
            st.subheader(f"Pareto Chart – Top {actual_n} {'All' if cat is None else cat} Items",
                        help="The Pareto Chart ranks items from highest to lowest demand (bars), while the line shows the running cumulative percentage of total demand. The dashed lines mark the 80% and 95% thresholds used to classify items into ABC classes.")
            fig, _ = pareto_chart(abc_df, f"{'Overall' if cat is None else cat} Demand Pareto",
                                  BAR_COLORS.get(cat, palette[0]), top_n=top_n, value_col=value_label)
            st.pyplot(fig); plt.close(fig)

            # ── ABC-XYZ matrix (All Items tab only) ──────────────────────────
            if cat is None and not xyz_df_abc.empty:
                st.subheader("ABC-XYZ Matrix",
                            help="The ABC-XYZ Matrix combines two classifications: ABC ranks items by their total demand contribution (A = top 80%, B = next 15%, C = bottom 5%), while XYZ measures how stable or erratic that demand is (X = stable, Y = variable, Z = highly irregular). The result is a 9-cell grid below that tells you both how important an item is and how predictable it is.")

                abc_xyz = abc_df.merge(xyz_df_abc[["Item Name","CV","XYZ"]], on="Item Name", how="left")
                # Items with no date data get N/A, not Z
                abc_xyz["XYZ"] = abc_xyz["XYZ"].fillna("N/A")
                abc_xyz["CV"]  = abc_xyz["CV"].fillna(0.0)
                abc_xyz["ABC-XYZ"] = abc_xyz.apply(
                    lambda r: r["ABC"] + r["XYZ"] if r["XYZ"] != "N/A" else "N/A", axis=1)

                # Heatmap (exclude N/A from matrix)
                matrix_data = abc_xyz[abc_xyz["XYZ"] != "N/A"].groupby(["ABC","XYZ"]).size().unstack(fill_value=0)
                for col_xyz in ["X","Y","Z"]:
                    if col_xyz not in matrix_data.columns:
                        matrix_data[col_xyz] = 0
                matrix_data = matrix_data[["X","Y","Z"]].reindex(["A","B","C"], fill_value=0)

                fig_m, ax_m = plt.subplots(figsize=(6, 3.5))
                cmap = plt.cm.YlGn
                im = ax_m.imshow(matrix_data.values, cmap=cmap, aspect="auto", vmin=0)
                ax_m.set_xticks([0,1,2]); ax_m.set_xticklabels(["X (Stable)","Y (Variable)","Z (Irregular)"], fontsize=10)
                ax_m.set_yticks([0,1,2]); ax_m.set_yticklabels(["A (High Value)","B (Mid Value)","C (Low Value)"], fontsize=10)
                ax_m.set_title("ABC-XYZ SKU Count Heatmap", fontweight="bold", pad=10)
                for i in range(3):
                    for j in range(3):
                        val = matrix_data.values[i, j]
                        ax_m.text(j, i, str(val), ha="center", va="center",
                                  fontsize=14, fontweight="bold",
                                  color="white" if (matrix_data.values.max() > 0 and val > matrix_data.values.max() * 0.6) else "#0a2e14")
                plt.colorbar(im, ax=ax_m, label="SKU Count")
                plt.tight_layout()
                st.pyplot(fig_m); plt.close(fig_m)

                # ── ABC-XYZ Recommendations ───────────────────────────────
                st.subheader("ABC-XYZ Action Guide",
                            help="Recommended stocking and review strategy for each classification cell based on both value contribution (ABC) and demand stability (XYZ).")
                rec_cols = st.columns(3)
                for idx, (cell, (badge, rec)) in enumerate(ABC_XYZ_RECOMMENDATIONS.items()):
                    count = int(matrix_data.loc[cell[0], cell[1]]) if cell[0] in matrix_data.index and cell[1] in matrix_data.columns else 0
                    rec_cols[idx % 3].markdown(
                        f"**{cell}** {badge} (*{count} SKUs*)  \n{rec}"
                    )
                st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

                st.subheader("Full ABC-XYZ Ranking Table",
                            help="Items are ranked by total demand quantity (highest to lowest), which determines their ABC class. The XYZ class and CV (Coefficient of Variation) are additional attributes showing demand stability and do not affect the ranking order.")
                cols_show_xyz = ["Rank","Item Name",value_label,"Cumulative Demand %","ABC","CV","XYZ","ABC-XYZ"]
                cols_show_xyz = [c for c in cols_show_xyz if c in abc_xyz.columns]
                st.dataframe(abc_xyz[cols_show_xyz], use_container_width=True, height=400)
                st.download_button(
                    "⬇️ Download ABC-XYZ table",
                    to_excel_bytes(abc_xyz[cols_show_xyz]),
                    "abc_xyz_all.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                st.subheader("Full Ranking Table",
                            help="Items are ranked by total demand quantity from highest to lowest. The ABC class shows each item's contribution to overall demand (A = top 80%, B = next 15%, C = bottom 5%). XYZ shows how stable or erratic the demand is over time.")
                if not xyz_df_abc.empty:
                    abc_df = abc_df.merge(xyz_df_abc[["Item Name","CV","XYZ"]], on="Item Name", how="left")
                    abc_df["XYZ"] = abc_df["XYZ"].fillna("N/A")
                    abc_df["ABC-XYZ"] = abc_df.apply(
                        lambda r: r["ABC"] + r["XYZ"] if r["XYZ"] != "N/A" else "N/A", axis=1)
                    cols_show = ["Rank","Item Name",value_label,"Cumulative Demand %","ABC","XYZ","ABC-XYZ"]
                else:
                    cols_show = ["Rank","Item Name",value_label,"Cumulative Demand %","ABC"]
                cols_show = [c for c in cols_show if c in abc_df.columns]

                if cat is not None:
                    styled = abc_df[cols_show].style.apply(highlight_abc, axis=1)
                    st.dataframe(styled, use_container_width=True, height=400)
                else:
                    st.dataframe(abc_df[cols_show], use_container_width=True, height=400)

                if cat is not None:
                    safe_name = cat.replace(" ", "_").replace("/", "_")
                    st.download_button(
                        f"⬇️ Download {cat} ABC table",
                        to_excel_bytes(abc_df),
                        f"abc_{safe_name.lower()}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – ADAPTIVE DEMAND FORECAST
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("📈 Demand Forecast",
              help="Select an item and how many months ahead you want to predict. We will automatically pick the most suitable forecasting method and model based on your data. No manual setup needed.")

    if "Invoice Date" not in df.columns:
        st.warning("Demand forecasting requires an `Invoice Date` column.")
    else:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            all_items = sorted(df["Item Name"].unique().tolist())
            selected_item = st.selectbox("Select Item / SKU", all_items)
        with col_b:
            forecast_months = st.slider("Forecast horizon (months)", 1, 24, 3)
            st.caption("⚠️ Accuracy decreases for longer horizons, especially with limited history data.")

        if st.button("\U0001f52e Run Forecast", type="primary"):
            item_df = df[df["Item Name"] == selected_item].copy()
            item_df = item_df.dropna(subset=["Invoice Date"])
            monthly = (item_df.groupby(item_df["Invoice Date"].dt.to_period("M"))["Quantity"]
                       .sum().reset_index())
            monthly["Invoice Date"] = monthly["Invoice Date"].dt.to_timestamp()
            monthly.columns = ["ds", "y"]
            monthly = monthly.sort_values("ds").reset_index(drop=True)

            # Fill missing months with 0 to handle data gaps
            if len(monthly) >= 2:
                full_range = pd.date_range(start=monthly["ds"].min(),
                                           end=monthly["ds"].max(), freq="MS")
                monthly = (monthly.set_index("ds")
                           .reindex(full_range, fill_value=0)
                           .reset_index()
                           .rename(columns={"index": "ds"}))

            if len(monthly) < 2:
                st.warning(
                    "Not enough history (need at least 2 months) to forecast this item. "
                    f"Only **{len(monthly)} month(s)** of data found."
                )
            else:
                n_months   = len(monthly)
                raw_vals   = monthly["y"].values.copy().astype(float)
                avg_qty    = float(np.mean(raw_vals))
                zero_pct   = float(np.sum(raw_vals == 0) / n_months)
                nonzero_ct = int(np.sum(raw_vals > 0))

                # ── Outlier detection & smoothing ─────────────────────────
                outlier_notes = []
                if n_months >= 4:
                    nz = raw_vals[raw_vals > 0]
                    if len(nz) > 1:
                        mean_v, std_v = np.mean(nz), np.std(nz)
                        cap = mean_v + 2.5 * std_v if std_v > 0 else np.inf
                        outlier_mask = raw_vals > cap
                        if outlier_mask.any():
                            n_out = int(outlier_mask.sum())
                            raw_vals[outlier_mask] = cap
                            monthly["y"] = raw_vals
                            outlier_notes.append(
                                f"\U0001f9f9 {n_out} outlier month(s) were smoothed before forecasting "
                                f"(capped at {cap:.0f} units). This prevents extreme one-off orders from distorting the forecast."
                            )

                series_vals = monthly["y"].values.tolist()

                # ── Model selection ───────────────────────────────────────
                is_intermittent = (zero_pct > 0.4) or (zero_pct > 0.25 and avg_qty < 5)
                has_plateau     = fc_detect_growth_plateau(series_vals)

                forecast_blocked = False
                model_choice = "prophet"
                model_reason = ""

                if is_intermittent and nonzero_ct < 2:
                    st.warning(
                        f"This item only has **{nonzero_ct} month(s) with actual sales**. "
                        "There is not enough demand history to produce a meaningful forecast. "
                        "Try again once more sales data is available."
                    )
                    forecast_blocked = True
                elif is_intermittent:
                    model_choice = "croston"
                    model_reason = (
                        f"Using **Croston\u2019s method** \u2014 "
                        f"{zero_pct*100:.0f}% of months have zero demand "
                        f"(avg {avg_qty:.1f} units/month). "
                        "Designed for sporadic, intermittent demand patterns."
                    )
                elif has_plateau and n_months >= 8:
                    model_choice = "prophet_logistic"
                    model_reason = (
                        "Using **Prophet (logistic growth)** \u2014 "
                        "demand shows a growth-then-plateau pattern. "
                        "Logistic mode prevents the forecast from extrapolating growth indefinitely."
                    )
                elif n_months < 6 or avg_qty < 10:
                    model_choice = "ets"
                    if avg_qty < 10 and n_months >= 6:
                        model_reason = (
                            f"Using **Holt\u2019s / ETS Exponential Smoothing** \u2014 "
                            f"average demand is low ({avg_qty:.1f} units/month). "
                            "Holt\u2019s method handles low-volume stable items more reliably than Prophet."
                        )
                    else:
                        model_reason = (
                            f"Using **Holt\u2019s / ETS Exponential Smoothing** \u2014 "
                            f"only {n_months} months of history available. "
                            "This method works well for short data series."
                        )
                else:
                    model_choice = "prophet"
                    model_reason = (
                        f"Using **Prophet** \u2014 {n_months} months of history available. "
                        + ("Yearly seasonality enabled." if n_months >= 12
                           else "Trend-based forecast (yearly seasonality needs 12+ months).")
                    )

                if not forecast_blocked:
                    st.info(f"\U0001f916 {model_reason}")
                for note in outlier_notes:
                    st.caption(note)

                # ── Forecast helpers are defined at module level (fc_* prefix) ──

                BAND_CAPTION = (
                    "The shaded area shows the **uncertainty range**. "
                    "Demand is expected to fall within this band most of the time. "
                    "A wider band means less certainty; a narrower band means more confidence."
                )

                def moving_average_fallback(series_vals, forecast_months, future_dates, item_name):
                    """Simple moving average fallback for when all models produce erratic results."""
                    window = min(3, len(series_vals))
                    ma_val = float(np.mean(series_vals[-window:]))
                    fc     = [max(0, int(round(ma_val)))] * forecast_months
                    std_v  = float(np.std(series_vals[-window:])) if window > 1 else ma_val * 0.2
                    lower  = [max(0, int(round(ma_val - 1.5 * std_v)))] * forecast_months
                    upper  = [int(round(ma_val + 1.5 * std_v))] * forecast_months

                    st.warning(
                        "\U0001f4ca All advanced models produced unreliable results for this item. "
                        f"Showing a **{window}-month moving average** as a rough estimate instead. "
                        "This is a simple baseline \u2014 treat it as an order-of-magnitude guide only."
                    )
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.bar(range(len(series_vals)), series_vals, color="#145a2e", alpha=0.7, label="Actual")
                    ax.axhline(ma_val, color="#f0c040", linewidth=2, linestyle="--", label=f"{window}-month MA: {ma_val:.0f}")
                    ax.set_ylabel("Quantity")
                    ax.set_title(f"Demand Forecast (Moving Average) \u2013 {item_name}", fontweight="bold")
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig); plt.close(fig)

                    return pd.DataFrame({
                        "Month":        [d.strftime("%b %Y") for d in future_dates],
                        "Forecast Qty": fc,
                        "Lower Bound":  lower,
                        "Upper Bound":  upper,
                        "Model":        f"{window}-month Moving Average",
                    })

                if not forecast_blocked:
                    with st.spinner("Training forecast model…"):
                        try:
                            last_date    = monthly["ds"].max()
                            future_dates = pd.date_range(
                                start=last_date + pd.DateOffset(months=1),
                                periods=forecast_months, freq="MS"
                            )
                            future_only  = None

                            # ── CROSTON ──────────────────────────────────
                            if model_choice == "croston":
                                forecast_vals = croston_forecast(series_vals, forecast_months)
                                all_labels = [d.strftime("%b %Y") for d in monthly["ds"]] +                                              [d.strftime("%b %Y") for d in future_dates]
                                all_x = list(range(len(monthly) + forecast_months))
                                hist_x = list(range(len(monthly)))
                                fc_x   = list(range(len(monthly), len(monthly) + forecast_months))
                                fig, ax = plt.subplots(figsize=(12, 5))
                                ax.bar(hist_x, monthly["y"], color="#145a2e", alpha=0.7, label="Actual (monthly)")
                                ax.plot(fc_x, forecast_vals, color="#f0c040", linewidth=2.5,
                                        marker="o", markersize=6, label="Croston Forecast")
                                lower = [max(0, v * 0.7) for v in forecast_vals]
                                upper = [v * 1.3 for v in forecast_vals]
                                ax.fill_between(fc_x, lower, upper, alpha=0.2, color="#f0c040", label="Uncertainty range")
                                ax.axvline(len(monthly) - 0.5, color="gray", linestyle="--", linewidth=1)
                                ax.set_xticks(all_x)
                                ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=7)
                                ax.set_ylabel("Quantity")
                                ax.set_title(f"Demand Forecast (Croston) – {selected_item}", fontweight="bold")
                                ax.legend(); plt.tight_layout()
                                st.pyplot(fig); plt.close(fig)
                                st.caption(BAND_CAPTION)
                                future_only = pd.DataFrame({
                                    "Month":        [d.strftime("%b %Y") for d in future_dates],
                                    "Forecast Qty": [int(v) for v in forecast_vals],
                                    "Lower Bound":  [max(0, int(v * 0.7)) for v in forecast_vals],
                                    "Upper Bound":  [int(v * 1.3) for v in forecast_vals],
                                    "Model":        "Croston",
                                })

                            # ── ETS / HOLT ────────────────────────────────
                            elif model_choice == "ets":
                                from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
                                y_vals = monthly["y"].values.astype(float)
                                if np.std(y_vals) == 0:
                                    y_vals = y_vals + np.random.normal(0, 0.01, len(y_vals))
                                model_label = "Simple ETS"
                                try:
                                    holt_fit  = Holt(y_vals, initialization_method="estimated").fit(optimized=True)
                                    holt_mape = fc_compute_mape(y_vals, holt_fit.fittedvalues)
                                    ets_fit2  = SimpleExpSmoothing(y_vals, initialization_method="estimated").fit(optimized=True)
                                    ets_mape2 = fc_compute_mape(y_vals, ets_fit2.fittedvalues)
                                    if (holt_mape if holt_mape is not None else 999) <= (ets_mape2 if ets_mape2 is not None else 999):
                                        fitted_vals   = holt_fit.fittedvalues
                                        forecast_vals = np.maximum(0, np.round(holt_fit.forecast(forecast_months))).astype(int)
                                        resid_std     = max(np.std(y_vals - fitted_vals), 1.0)
                                        model_label   = "Holt’s (trend)"
                                    else:
                                        fitted_vals   = ets_fit2.fittedvalues
                                        forecast_vals = np.maximum(0, np.round(ets_fit2.forecast(forecast_months))).astype(int)
                                        resid_std     = max(np.std(y_vals - fitted_vals), 1.0)
                                except Exception:
                                    ets_fit       = SimpleExpSmoothing(y_vals, initialization_method="estimated").fit(optimized=True)
                                    fitted_vals   = ets_fit.fittedvalues
                                    forecast_vals = np.maximum(0, np.round(ets_fit.forecast(forecast_months))).astype(int)
                                    resid_std     = max(np.std(y_vals - fitted_vals), 1.0)
                                fc_accuracy_display(y_vals, fitted_vals, model_label)
                                lower_b = np.maximum(0, np.array(forecast_vals, dtype=float) - 1.5 * resid_std)
                                upper_b = np.array(forecast_vals, dtype=float) + 1.5 * resid_std
                                fig, ax = plt.subplots(figsize=(12, 5))
                                ax.scatter(monthly["ds"], monthly["y"], color="#f0c040", zorder=5, s=50, label="Actual")
                                ax.plot(monthly["ds"], fitted_vals, color="#145a2e", linewidth=2, linestyle="--", label=f"{model_label} Fitted")
                                ax.plot(future_dates, forecast_vals, color="#145a2e", linewidth=2.5, marker="o", markersize=6, label=f"{model_label} Forecast")
                                ax.fill_between(future_dates, lower_b, upper_b, alpha=0.2, color="#145a2e", label="Uncertainty range")
                                ax.axvline(last_date, color="gray", linestyle="--", linewidth=1)
                                ax.set_ylabel("Quantity")
                                ax.set_title(f"Demand Forecast ({model_label}) – {selected_item}", fontweight="bold")
                                ax.legend(); plt.tight_layout()
                                st.pyplot(fig); plt.close(fig)
                                st.caption(BAND_CAPTION)
                                future_only = pd.DataFrame({
                                    "Month":        [d.strftime("%b %Y") for d in future_dates],
                                    "Forecast Qty": forecast_vals.tolist() if hasattr(forecast_vals, "tolist") else list(forecast_vals),
                                    "Lower Bound":  np.maximum(0, np.round(lower_b)).astype(int).tolist(),
                                    "Upper Bound":  np.maximum(0, np.round(upper_b)).astype(int).tolist(),
                                    "Model":        model_label,
                                })

                            # ── PROPHET LOGISTIC ──────────────────────────
                            elif model_choice == "prophet_logistic":
                                from prophet import Prophet
                                cap_val = float(monthly["y"].max() * 2.0)
                                monthly_lg = monthly.copy()
                                monthly_lg["cap"]   = cap_val
                                monthly_lg["floor"] = 0.0
                                m = Prophet(growth="logistic", yearly_seasonality=n_months >= 12,
                                            weekly_seasonality=False, daily_seasonality=False,
                                            seasonality_mode="additive")
                                m.fit(monthly_lg)
                                future = m.make_future_dataframe(periods=forecast_months, freq="MS")
                                future["cap"] = cap_val; future["floor"] = 0.0
                                forecast = m.predict(future)
                                cutoff = monthly["ds"].max()
                                hist_forecast = forecast[forecast["ds"] <= cutoff]["yhat"].values
                                fc_accuracy_display(monthly["y"].values, hist_forecast, "Prophet (logistic)")
                                future_fc_vals = forecast[forecast["ds"] > cutoff]["yhat"].clip(lower=0).values
                                is_erratic = fc_check_erratic(future_fc_vals, monthly["y"].values)
                                if not is_erratic:
                                    fig, ax = plt.subplots(figsize=(12, 5))
                                    ax.fill_between(forecast["ds"], forecast["yhat_lower"].clip(lower=0),
                                                    forecast["yhat_upper"], alpha=0.2, color="#145a2e", label="Uncertainty range")
                                    ax.plot(forecast["ds"], forecast["yhat"], color="#145a2e", linewidth=2, label="Forecast")
                                    ax.scatter(monthly["ds"], monthly["y"], color="#f0c040", zorder=5, s=40, label="Actual")
                                    ax.axhline(cap_val, color="gray", linestyle=":", linewidth=1, label=f"Growth cap ({cap_val:.0f})")
                                    ax.axvline(cutoff, color="gray", linestyle="--", linewidth=1)
                                    ax.set_ylabel("Quantity")
                                    ax.set_title(f"Demand Forecast (Prophet Logistic) – {selected_item}", fontweight="bold")
                                    ax.legend(); plt.tight_layout()
                                    st.pyplot(fig); plt.close(fig)
                                    st.caption(BAND_CAPTION)
                                    future_only = forecast[forecast["ds"] > cutoff][["ds","yhat","yhat_lower","yhat_upper"]].copy()
                                    future_only.columns = ["Month","Forecast Qty","Lower Bound","Upper Bound"]
                                    future_only["Month"] = future_only["Month"].dt.strftime("%b %Y")
                                    future_only[["Forecast Qty","Lower Bound","Upper Bound"]] = \
                                        future_only[["Forecast Qty","Lower Bound","Upper Bound"]].round(0).astype(int)
                                    future_only["Forecast Qty"] = future_only["Forecast Qty"].clip(lower=0).astype(int)
                                    future_only["Lower Bound"]  = future_only["Lower Bound"].clip(lower=0).astype(int)
                                    future_only["Model"] = "Prophet (logistic)"
                                else:
                                    future_only = moving_average_fallback(series_vals, forecast_months, future_dates, selected_item)

                            # ── PROPHET (standard) ────────────────────────
                            else:
                                from prophet import Prophet
                                use_yearly = n_months >= 12
                                m = Prophet(yearly_seasonality=use_yearly, weekly_seasonality=False,
                                            daily_seasonality=False, seasonality_mode="additive")
                                m.fit(monthly)
                                future   = m.make_future_dataframe(periods=forecast_months, freq="MS")
                                forecast = m.predict(future)
                                cutoff   = monthly["ds"].max()
                                hist_forecast = forecast[forecast["ds"] <= cutoff]["yhat"].values
                                fc_accuracy_display(monthly["y"].values, hist_forecast, "Prophet")
                                future_fc_vals = forecast[forecast["ds"] > cutoff]["yhat"].clip(lower=0).values
                                is_erratic = fc_check_erratic(future_fc_vals, monthly["y"].values)
                                if not is_erratic:
                                    fig, ax = plt.subplots(figsize=(12, 5))
                                    ax.fill_between(forecast["ds"], forecast["yhat_lower"].clip(lower=0),
                                                    forecast["yhat_upper"], alpha=0.2, color="#145a2e", label="Uncertainty range")
                                    ax.plot(forecast["ds"], forecast["yhat"], color="#145a2e", linewidth=2, label="Forecast")
                                    ax.scatter(monthly["ds"], monthly["y"], color="#f0c040", zorder=5, s=40, label="Actual")
                                    ax.axvline(cutoff, color="gray", linestyle="--", linewidth=1)
                                    ax.set_ylabel("Quantity")
                                    ax.set_title(f"Demand Forecast (Prophet) – {selected_item}", fontweight="bold")
                                    ax.legend(); plt.tight_layout()
                                    st.pyplot(fig); plt.close(fig)
                                    st.caption(BAND_CAPTION)
                                    future_only = forecast[forecast["ds"] > cutoff][["ds","yhat","yhat_lower","yhat_upper"]].copy()
                                    future_only.columns = ["Month","Forecast Qty","Lower Bound","Upper Bound"]
                                    future_only["Month"] = future_only["Month"].dt.strftime("%b %Y")
                                    future_only[["Forecast Qty","Lower Bound","Upper Bound"]] = \
                                        future_only[["Forecast Qty","Lower Bound","Upper Bound"]].round(0).astype(int)
                                    future_only["Forecast Qty"] = future_only["Forecast Qty"].clip(lower=0).astype(int)
                                    future_only["Lower Bound"]  = future_only["Lower Bound"].clip(lower=0).astype(int)
                                    future_only["Model"] = "Prophet"
                                else:
                                    tbats_ok = False
                                    try:
                                        from tbats import TBATS as TBATSModel
                                        estimator = TBATSModel(seasonal_periods=None, use_arma_errors=True,
                                                               use_box_cox=True, n_jobs=1)
                                        tbats_fit = estimator.fit(monthly["y"].values)
                                        tbats_fc, tbats_conf = tbats_fit.forecast(steps=forecast_months, confidence_level=0.8)
                                        tbats_fc    = np.maximum(0, np.round(tbats_fc)).astype(int)
                                        tbats_lower = np.maximum(0, np.round(tbats_conf["lower_bound"])).astype(int)
                                        tbats_upper = np.maximum(0, np.round(tbats_conf["upper_bound"])).astype(int)
                                        tbats_fitted = tbats_fit.y_hat
                                        fc_accuracy_display(monthly["y"].values, tbats_fitted, "TBATS")
                                        fig, ax = plt.subplots(figsize=(12, 5))
                                        ax.scatter(monthly["ds"], monthly["y"], color="#f0c040", zorder=5, s=50, label="Actual")
                                        ax.plot(future_dates, tbats_fc, color="#145a2e", linewidth=2.5,
                                                marker="o", markersize=6, label="TBATS Forecast")
                                        ax.fill_between(future_dates, tbats_lower, tbats_upper,
                                                        alpha=0.2, color="#145a2e", label="Uncertainty range")
                                        ax.axvline(last_date, color="gray", linestyle="--", linewidth=1)
                                        ax.set_ylabel("Quantity")
                                        ax.set_title(f"Demand Forecast (TBATS) – {selected_item}", fontweight="bold")
                                        ax.legend(); plt.tight_layout()
                                        st.pyplot(fig); plt.close(fig)
                                        st.caption(BAND_CAPTION)
                                        st.info(
                                            "🔄 Prophet produced an erratic forecast. "
                                            "**TBATS** was used instead as it handles complex patterns better."
                                        )
                                        future_only = pd.DataFrame({
                                            "Month":        [d.strftime("%b %Y") for d in future_dates],
                                            "Forecast Qty": tbats_fc.tolist(),
                                            "Lower Bound":  tbats_lower.tolist(),
                                            "Upper Bound":  tbats_upper.tolist(),
                                            "Model":        "TBATS",
                                        })
                                        tbats_ok = True
                                    except Exception as tbats_err:
                                        st.caption(f"ℹ️ TBATS could not run ({type(tbats_err).__name__}). Using moving average fallback.")
                                    if not tbats_ok:
                                        future_only = moving_average_fallback(
                                            series_vals, forecast_months, future_dates, selected_item)

                            # ── Forecast summary table ────────────────────
                            st.subheader("Forecast Summary",
                                        help="Forecast Qty is the model's best estimate. Lower / Upper Bound show the range demand is likely to fall within. Plan your stock based on the Upper Bound if you want to avoid stockouts.")
                            st.dataframe(future_only, use_container_width=True, hide_index=True)
                            st.download_button(
                                "⬇️ Download forecast",
                                to_excel_bytes(future_only),
                                "forecast.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

                        except Exception as e:
                            st.error(f"Forecast failed: {e}")
                            import traceback
                            st.code(traceback.format_exc())




# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – REORDER ALERTS (with Statistical Safety Stock)
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    # Initialise stock_df at tab level so all sections can access it
    stock_df = None

    # ── Stock file upload — shown at top so user can upload before running anything ──
    has_stock_in_df = "Current Stock" in df.columns
    has_lt_in_df    = "Lead Time (days)" in df.columns

    if has_stock_in_df:
        st.info("✅ `Current Stock` column detected in your dataset and will be used automatically. You can still upload a separate stock file to override it.")
    if has_lt_in_df:
        st.info("✅ `Lead Time (days)` column detected in your dataset and will be used for per-item lead times automatically.")
    if not has_stock_in_df:
        st.markdown("**Tip:** upload a stock levels file with `Item Name`, `Current Stock`, and optionally `Lead Time (days)` for per-item lead times. If your main dataset already contains these columns, no upload is needed.")
    stock_file = st.file_uploader("📂 Stock levels file", type=["xlsx","xls"], key="stock")
    if stock_file:
        stock_df = pd.read_excel(stock_file)
        stock_df.columns = stock_df.columns.str.strip()
        st.success(f"✅ Stock file loaded: {len(stock_df):,} items.")

    st.markdown("---")
    st.header("🔔 Reorder Point Alert",
              help="Calculates the reorder point (ROP) for each item based on your demand history and lead time. Upload a stock levels file above to see which items need restocking now.")

    if "Invoice Date" not in df.columns:
        st.warning("Reorder alerts require an `Invoice Date` column.")
    else:
        n_months_data = df["Invoice Date"].dt.to_period("M").nunique()

        # ── Settings ──────────────────────────────────────────────────────
        with st.expander("⚙️ Settings", expanded=False):
            col1, col2, col3 = st.columns(3)
            lead_time    = col1.number_input("Default lead time (days)", min_value=1, max_value=180, value=14,
                                             help="How many days it takes to receive stock after placing an order. Can be overridden per item in the stock file.")
            demand_window = col2.selectbox("Demand calculation window",
                                           ["Last 30 days","Last 60 days","Last 90 days","Full history"],
                                           index=2,
                                           help="Recent demand is more accurate for trending items. Use Full history for stable items.")
            replenish_days = col3.number_input("Suggested order covers (days)", min_value=7, max_value=180, value=30,
                                               help="How many days of stock the suggested order quantity should cover.")

            stat_mode = st.toggle("📐 Statistical Safety Stock (recommended)", value=n_months_data >= 3,
                                  help="Uses demand variability to calculate safety stock. More accurate than simple days-of-cover.")
            if stat_mode:
                c1, c2 = st.columns(2)
                service_level = c1.selectbox("Service Level", ["90%", "95%", "99%"], index=1,
                                             help="Higher service level = more safety stock = fewer stockouts but higher holding cost.")
                safety_days   = c2.number_input("Simple safety stock (days)",
                                                min_value=0, max_value=60, value=7,
                                                help="Used as a fallback when there is insufficient demand history to calculate statistical safety stock.")
                z_map = {"90%": 1.28, "95%": 1.65, "99%": 2.33}
                z_val = z_map[service_level]
                st.caption(f"Statistical Safety Stock = Z({z_val}) × σ_demand × √(Lead Time) | Service level: {service_level}")
            else:
                safety_days = st.number_input("Safety stock (days of cover)", min_value=0, max_value=60, value=7)
                z_val = 1.65  # default, not used in simple mode

            # ── What-if scenario modelling ────────────────────────────────
            st.markdown("**What-if Scenario**",
                        help="Adjust these sliders to model how changes in lead time or demand would affect your reorder points, without changing your actual settings.")
            sc1, sc2 = st.columns(2)
            lead_time_factor = sc1.slider("Lead time change (%)", min_value=-50, max_value=100, value=0, step=5,
                                          help="Simulate what happens if your lead time increases or decreases. 0% = no change.")
            demand_factor    = sc2.slider("Demand change (%)", min_value=-50, max_value=100, value=0, step=5,
                                          help="Simulate what happens if demand grows or drops. 0% = no change.")
            scenario_active = lead_time_factor != 0 or demand_factor != 0
            if scenario_active:
                st.caption(f"Scenario active: lead time ×{1 + lead_time_factor/100:.2f}, demand ×{1 + demand_factor/100:.2f}. Results will show scenario impact alongside baseline.")

        if st.button("Calculate Reorder Points", type="primary"):

            # ── Demand calculation with configurable window ───────────────
            df_demand = df.dropna(subset=["Invoice Date"]).copy()
            if demand_window != "Full history":
                days_back = {"Last 30 days": 30, "Last 60 days": 60, "Last 90 days": 90}[demand_window]
                cutoff = df_demand["Invoice Date"].max() - pd.Timedelta(days=days_back)
                df_demand = df_demand[df_demand["Invoice Date"] >= cutoff]
                window_days = days_back
            else:
                window_days = max((df_demand["Invoice Date"].max() - df_demand["Invoice Date"].min()).days, 1)

            if df_demand.empty:
                st.warning("No data in the selected demand window. Try a wider window.")
                st.stop()

            # ── Improvement 4: Trend-adjusted demand ──────────────────────
            # Compute monthly demand per item and fit a linear trend
            df_trend = df.dropna(subset=["Invoice Date"]).copy()
            df_trend["Month"] = df_trend["Invoice Date"].dt.to_period("M")
            monthly_trend = df_trend.groupby(["Item Name","Month"])["Quantity"].sum().reset_index()
            monthly_trend["Month_idx"] = monthly_trend.groupby("Item Name").cumcount()

            trend_map = {}
            for item, grp in monthly_trend.groupby("Item Name"):
                if len(grp) >= 3:
                    slope = np.polyfit(grp["Month_idx"], grp["Quantity"], 1)[0]
                    last_month_avg = grp["Quantity"].iloc[-1]
                    # Trend-adjusted daily demand: last month + half a slope step, converted to daily
                    trend_adj = max(0, (last_month_avg + slope * 0.5) / 30)
                    trend_map[item] = round(trend_adj, 4)

            # Flat avg daily demand (used as fallback)
            avg_daily = (df_demand.groupby("Item Name")["Quantity"].sum() / window_days).reset_index()
            avg_daily.columns = ["Item Name", "Avg Daily Demand"]
            # Apply trend adjustment where available
            avg_daily["Trend Adj Daily Demand"] = avg_daily["Item Name"].map(trend_map)
            avg_daily["Effective Daily Demand"] = avg_daily["Trend Adj Daily Demand"].combine_first(avg_daily["Avg Daily Demand"])

            # Monthly demand std for statistical safety stock (use full history for stability)
            df_monthly = df.dropna(subset=["Invoice Date"]).copy()
            df_monthly["Month"] = df_monthly["Invoice Date"].dt.to_period("M")
            monthly_item = df_monthly.groupby(["Item Name","Month"])["Quantity"].sum().reset_index()
            monthly_std = monthly_item.groupby("Item Name")["Quantity"].std(ddof=1).reset_index()
            monthly_std.columns = ["Item Name","Monthly Std"]
            monthly_std["Daily Std"] = monthly_std["Monthly Std"] / np.sqrt(30)

            # Build ROP table
            rop_df = avg_daily.copy()
            rop_df = rop_df.merge(monthly_std[["Item Name","Daily Std"]], on="Item Name", how="left")
            rop_df["Daily Std"] = rop_df["Daily Std"].fillna(0)

            # Merge per-item lead time
            if stock_df is not None and "Lead Time (days)" in stock_df.columns:
                lt_map = stock_df.set_index("Item Name")["Lead Time (days)"].to_dict()
                rop_df["Lead Time"] = rop_df["Item Name"].map(lt_map).fillna(lead_time).astype(float)
            elif has_lt_in_df:
                lt_map = df.groupby("Item Name")["Lead Time (days)"].first().to_dict()
                rop_df["Lead Time"] = rop_df["Item Name"].map(lt_map).fillna(lead_time).astype(float)
            else:
                rop_df["Lead Time"] = float(lead_time)

            # ── Improvement 3: ABC-linked service levels ──────────────────
            abc_all = build_abc_df(df, None)[["Item Name","ABC"]]
            rop_df = rop_df.merge(abc_all, on="Item Name", how="left")
            rop_df["ABC"] = rop_df["ABC"].fillna("C")

            if stat_mode:
                abc_z = {"A": 2.33, "B": 1.65, "C": 1.28}  # 99% / 95% / 90%
                rop_df["Effective Z"] = rop_df["ABC"].map(abc_z)
                abc_sl = {"A": "99%", "B": "95%", "C": "90%"}
                rop_df["Safety Stock Method"] = rop_df.apply(
                    lambda r: f"Statistical (ABC-{r['ABC']}, {abc_sl[r['ABC']]})" if r["Daily Std"] > 0 else "Simple (fallback)", axis=1
                )
                rop_df["Stat Safety Stock"] = (rop_df["Effective Z"] * rop_df["Daily Std"] * np.sqrt(rop_df["Lead Time"])).round(1)
                rop_df["Simple Safety Stock"] = (rop_df["Effective Daily Demand"] * safety_days).round(1)
                rop_df["Safety Stock"] = rop_df.apply(
                    lambda r: r["Stat Safety Stock"] if r["Daily Std"] > 0 else r["Simple Safety Stock"], axis=1
                )
            else:
                rop_df["Simple Safety Stock"] = (rop_df["Effective Daily Demand"] * safety_days).round(1)
                rop_df["Stat Safety Stock"]   = (z_val * rop_df["Daily Std"] * np.sqrt(rop_df["Lead Time"])).round(1)
                rop_df["Safety Stock"] = rop_df["Simple Safety Stock"]
                rop_df["Safety Stock Method"] = "Simple"

            rop_df["ROP (units)"]         = (rop_df["Effective Daily Demand"] * rop_df["Lead Time"] + rop_df["Safety Stock"]).round(1)
            rop_df["Monthly Demand"]      = (rop_df["Effective Daily Demand"] * 30).round(0).astype(int)
            rop_df["Suggested Order Qty"] = (rop_df["Effective Daily Demand"] * replenish_days).round(0).astype(int)

            # ── What-if scenario ROP ──────────────────────────────────────
            if scenario_active:
                scenario_lt     = rop_df["Lead Time"] * (1 + lead_time_factor / 100)
                scenario_demand = rop_df["Effective Daily Demand"] * (1 + demand_factor / 100)
                rop_df["Scenario ROP"] = (scenario_demand * scenario_lt + rop_df["Safety Stock"]).round(1)

            # ── Improvement 6: EOQ ────────────────────────────────────────
            has_price = any(c in df.columns for c in ["Unit Price","Price","Revenue","Amount"])
            price_col = next((c for c in ["Unit Price","Price"] if c in df.columns), None)
            if price_col:
                price_map = df.groupby("Item Name")[price_col].mean()
                rop_df["Unit Price"]   = rop_df["Item Name"].map(price_map)
                rop_df["_unit_price"]  = rop_df["Unit Price"].fillna(0)   # used by cost section
                annual_demand = rop_df["Effective Daily Demand"] * 365
                order_cost    = 50.0   # default $50 per order
                holding_pct   = 0.25   # default 25% of unit price per year
                eoq_vals = np.where(
                    (rop_df["_unit_price"] > 0) & (annual_demand > 0),
                    np.sqrt(2 * annual_demand * order_cost / (rop_df["Unit Price"].fillna(1) * holding_pct)),
                    np.nan
                )
                rop_df["EOQ"] = pd.Series(eoq_vals, index=rop_df.index).fillna(rop_df["Suggested Order Qty"]).round(0).astype(int)
            else:
                rop_df["EOQ"] = rop_df["Suggested Order Qty"]

            # ── Reorder frequency (after EOQ is defined) ──────────────────
            rop_df["Reorder Every (days)"] = (
                rop_df["EOQ"] / rop_df["Effective Daily Demand"].replace(0, np.nan)
            ).round(0).fillna(0).astype(int)

            # Merge stock levels
            if stock_df is not None and "Item Name" in stock_df.columns and "Current Stock" in stock_df.columns:
                rop_df = rop_df.merge(stock_df[["Item Name","Current Stock"]], on="Item Name", how="left")
            elif has_stock_in_df:
                stock_from_df = df.groupby("Item Name")["Current Stock"].first().reset_index()
                rop_df = rop_df.merge(stock_from_df, on="Item Name", how="left")

            if "Current Stock" in rop_df.columns:
                rop_df["Current Stock"] = pd.to_numeric(rop_df["Current Stock"], errors="coerce").fillna(0)

                rop_df["Days of Stock"] = (rop_df["Current Stock"] / rop_df["Effective Daily Demand"].replace(0, np.nan)).round(0)
                rop_df["Days of Stock"] = rop_df["Days of Stock"].fillna(0).astype(int)

                # ── Improvement 1: Overstock detection ───────────────────
                overstock_threshold = 3.0  # flag if stock > 3× ROP
                def get_status(r):
                    if r["Current Stock"] <= r["ROP (units)"]:
                        return "🔴 Reorder Now"
                    elif r["Current Stock"] <= r["ROP (units)"] * 1.5:
                        return "🟡 Low Stock"
                    elif r["Current Stock"] > r["ROP (units)"] * overstock_threshold and r["ROP (units)"] > 0:
                        return "⚠️ Overstocked"
                    else:
                        return "🟢 OK"
                rop_df["Status"] = rop_df.apply(get_status, axis=1)

                # Summary metrics
                reorder_n   = (rop_df["Status"] == "🔴 Reorder Now").sum()
                low_n       = (rop_df["Status"] == "🟡 Low Stock").sum()
                ok_n        = (rop_df["Status"] == "🟢 OK").sum()
                overstock_n = (rop_df["Status"] == "⚠️ Overstocked").sum()
                no_stock    = (rop_df["Current Stock"] == 0).sum()

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("🔴 Reorder Now",  reorder_n)
                m2.metric("🟡 Low Stock",    low_n)
                m3.metric("🟢 OK",           ok_n)
                m4.metric("⚠️ Overstocked",  overstock_n)
                m5.metric("⬛ Zero Stock",    no_stock)

                # Status bar chart
                fig_s, ax_s = plt.subplots(figsize=(8, 3))
                label_map   = {"🔴 Reorder Now": "Reorder Now", "🟡 Low Stock": "Low Stock",
                               "🟢 OK": "OK", "⚠️ Overstocked": "Overstocked"}
                status_counts = rop_df["Status"].value_counts()
                plain_labels  = [label_map.get(s, s) for s in status_counts.index]
                sc_colors     = {"Reorder Now": "#dc3545", "Low Stock": "#ffc107",
                                 "OK": "#28a745", "Overstocked": "#fd7e14"}
                ax_s.barh(plain_labels, status_counts.values,
                          color=[sc_colors.get(l, "gray") for l in plain_labels], alpha=0.85)
                ax_s.set_xlabel("Number of Items")
                ax_s.set_title("Reorder Status Overview", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig_s); plt.close(fig_s)

                # ── Improvement 5: Fill rate ──────────────────────────────
                total_demand = rop_df["Monthly Demand"].sum()
                in_stock_demand = rop_df[rop_df["Current Stock"] > 0]["Monthly Demand"].sum()
                fill_rate = (in_stock_demand / total_demand * 100) if total_demand > 0 else 0
                st.info(
                    f"**Estimated Fill Rate: {fill_rate:.1f}%** "
                    f"Based on current stock levels, approximately {fill_rate:.1f}% of monthly demand "
                    f"can be fulfilled without a stockout. "
                    + ("Consider restocking critical items to improve this." if fill_rate < 90 else "Stock levels are healthy overall.")
                )

                # ── Cost impact summary (when Unit Price available) ───────
                # Reuse the Unit Price already mapped during EOQ calculation
                # Fall back to re-mapping from df if EOQ section didn't run (no price col)
                price_col_cost = next((c for c in ["Unit Price","Price"] if c in df.columns), None)
                if price_col_cost:
                    if "_unit_price" not in rop_df.columns:
                        price_map_cost = df.groupby("Item Name")[price_col_cost].mean()
                        rop_df["_unit_price"] = rop_df["Item Name"].map(price_map_cost).fillna(0)

                    # Only count items that actually have a price (exclude $0 from totals)
                    priced = rop_df[rop_df["_unit_price"] > 0]
                    inv_value = (priced["Current Stock"] * priced["_unit_price"]).sum()

                    reorder_priced = priced[priced["Status"] == "🔴 Reorder Now"]
                    order_value = (reorder_priced["EOQ"] * reorder_priced["_unit_price"]).sum()

                    overstock_priced = priced[priced["Status"] == "⚠️ Overstocked"]
                    overstock_val = overstock_priced.apply(
                        lambda r: max(0, r["Current Stock"] - r["ROP (units)"] * 1.5) * r["_unit_price"], axis=1
                    ).sum()

                    cv1, cv2, cv3 = st.columns(3)
                    cv1.metric("💰 Current Inventory Value", f"${inv_value:,.0f}",
                               help="Total value of current stock across all items with known prices (Current Stock × Unit Price).")
                    cv2.metric("🛒 Immediate Order Cost",    f"${order_value:,.0f}",
                               help="Estimated cost to restock all Reorder Now items using EOQ quantities.")
                    cv3.metric("💸 Overstock Value",         f"${overstock_val:,.0f}",
                               help="Estimated value of excess stock in overstocked items. Consider reducing future orders to free up this cash.")

                    missing_prices = (rop_df["_unit_price"] == 0).sum()
                    if missing_prices > 0:
                        st.caption(f"Note: {missing_prices} item(s) have no price data and are excluded from the value calculations above. The totals shown may be understated.")

                # ── Scenario impact ───────────────────────────────────────
                if scenario_active and "Scenario ROP" in rop_df.columns and "Current Stock" in rop_df.columns:
                    scenario_reorder = (rop_df["Current Stock"] <= rop_df["Scenario ROP"]).sum()
                    baseline_reorder = (rop_df["Status"] == "🔴 Reorder Now").sum()
                    delta = scenario_reorder - baseline_reorder
                    delta_str = f"+{delta}" if delta > 0 else str(delta)
                    st.warning(
                        f"**Scenario impact:** Under your what-if scenario, **{scenario_reorder} items** would need reordering "
                        f"({delta_str} vs baseline of {baseline_reorder}). "
                        + ("Lead time increase raises your reorder points." if lead_time_factor > 0 else "")
                        + (" Demand growth means you need more safety stock." if demand_factor > 0 else "")
                    )

                # Recommended Action column
                def get_action(r):
                    freq = f", reorder every ~{int(r['Reorder Every (days)'])} days" if "Reorder Every (days)" in r.index and r["Reorder Every (days)"] > 0 else ""
                    if r["Status"] == "🔴 Reorder Now":
                        return f"Order {int(r['EOQ'])} units now (stock below reorder point{freq})"
                    elif r["Status"] == "🟡 Low Stock":
                        return f"Plan to order {int(r['EOQ'])} units soon ({int(r['Days of Stock'])} days left{freq})"
                    elif r["Status"] == "⚠️ Overstocked":
                        excess = int(r["Current Stock"] - r["ROP (units)"] * 1.5)
                        return f"Consider reducing next order. Excess stock of ~{excess} units"
                    else:
                        return f"No action needed ({int(r['Days of Stock'])} days of stock remaining{freq})"
                rop_df["Recommended Action"] = rop_df.apply(get_action, axis=1)

                # Sort: Reorder Now → Low Stock → Overstocked → OK
                order_map = {"🔴 Reorder Now": 0, "🟡 Low Stock": 1, "⚠️ Overstocked": 2, "🟢 OK": 3}
                rop_df["_sort"] = rop_df["Status"].map(order_map)
                rop_display = rop_df.sort_values(["_sort","Days of Stock"]).drop(columns=["_sort"]).copy()

            else:
                rop_df["Current Stock"] = "—"
                rop_df["Days of Stock"] = "—"
                rop_df["Status"] = "Upload a stock levels file (or add a Current Stock column to your dataset) to see alerts"
                rop_display = rop_df.sort_values("Monthly Demand", ascending=False).copy()

            rop_df["Avg Daily Demand"]       = rop_df["Avg Daily Demand"].round(3)
            rop_df["Effective Daily Demand"] = rop_df["Effective Daily Demand"].round(3)

            # Display columns
            display_cols = ["Item Name","ABC","Status","Recommended Action","Current Stock","Days of Stock",
                            "ROP (units)","Scenario ROP","Safety Stock","Safety Stock Method","EOQ",
                            "Reorder Every (days)","Suggested Order Qty","Monthly Demand",
                            "Effective Daily Demand","Lead Time"]
            display_cols = [c for c in display_cols if c in rop_display.columns]

            st.subheader(f"Reorder Report: {demand_window}",
                        help="Status tells you what to do: Reorder Now means order immediately. Low Stock means plan an order soon. Overstocked means you have too much stock tying up cash. OK means healthy. EOQ is the economically optimal order quantity. Effective Daily Demand uses trend adjustment where enough history exists.")
            st.dataframe(rop_display[display_cols], use_container_width=True, height=500, hide_index=True)
            st.download_button(
                "⬇️ Download reorder report",
                to_excel_bytes(rop_display[display_cols]),
                "reorder_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # ── Improvement 2: Supplier grouping ─────────────────────────
            SUPPLIER_COLS = ["Supplier", "Supplier Name", "Vendor", "Vendor Name", "Manufacturer"]
            supplier_col = next((c for c in SUPPLIER_COLS if c in df.columns), None)
            # Also check stock file
            if supplier_col is None and stock_df is not None:
                supplier_col = next((c for c in SUPPLIER_COLS if c in stock_df.columns), None)
                if supplier_col:
                    sup_map = stock_df.set_index("Item Name")[supplier_col].to_dict()
                    rop_display["Supplier"] = rop_display["Item Name"].map(sup_map)
                    supplier_col = "Supplier"

            if supplier_col and supplier_col in df.columns:
                sup_map = df.groupby("Item Name")[supplier_col].first().to_dict()
                rop_display["Supplier"] = rop_display["Item Name"].map(sup_map)
                supplier_col = "Supplier"

            if "Supplier" in rop_display.columns:
                action_items = rop_display[rop_display["Status"].isin(["🔴 Reorder Now","🟡 Low Stock"])]
                if not action_items.empty:
                    st.subheader("Purchase Orders by Supplier",
                                help="Items that need restocking grouped by supplier so you can consolidate into one purchase order per supplier.")

                    # Download all POs as one Excel file
                    po_cols_all = ["Supplier","Item Name","Status","EOQ","Recommended Action","Days of Stock","ROP (units)","Lead Time"]
                    po_cols_all = [c for c in po_cols_all if c in action_items.columns]
                    st.download_button(
                        "⬇️ Download all purchase orders",
                        to_excel_bytes(action_items[po_cols_all].sort_values("Supplier")),
                        "purchase_orders.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                    for supplier, grp in action_items.groupby("Supplier"):
                        supplier_label = supplier if pd.notna(supplier) else "Unknown Supplier"
                        total_eoq = grp["EOQ"].sum() if "EOQ" in grp.columns else grp["Suggested Order Qty"].sum()
                        with st.expander(f"📦 {supplier_label} ({len(grp)} item(s) to order)", expanded=False):
                            po_cols = ["Item Name","Status","EOQ","Recommended Action","Days of Stock"]
                            po_cols = [c for c in po_cols if c in grp.columns]
                            st.dataframe(grp[po_cols], use_container_width=True, hide_index=True)
                            st.caption(f"Total units to order from {supplier_label}: {int(total_eoq):,}")

            # Save reorder results to session state so optimization models persist across reruns
            st.session_state["rop_df"]      = rop_df.copy()
            st.session_state["rop_display"] = rop_display.copy()
            st.session_state["stock_df_cached"] = stock_df.copy() if stock_df is not None else None

    st.markdown("---")
    st.header("🔔 Stockout Risk Alert",
              help="Scores every SKU by stockout risk using demand patterns and current stock levels. Each item gets a risk level (High / Medium / Low), a plain-English reason, and an estimated days-to-stockout so you know exactly which items need attention and how urgently.")

    if "Invoice Date" not in df.columns:
        st.warning("Stockout Risk requires an `Invoice Date` column.")
    elif st.button("Run Stockout Risk Analysis", type="primary"):
        with st.spinner("Engineering features and training classifier…"):
            try:
                # ── Feature Engineering ───────────────────────────────────
                df3 = df.copy()
                df3["Month"] = df3["Invoice Date"].dt.to_period("M")
                monthly_sku = df3.groupby(["Item Name","Month"])["Quantity"].sum().reset_index()

                feature_rows = []
                for item, grp in monthly_sku.groupby("Item Name"):
                    grp = grp.sort_values("Month").reset_index(drop=True)
                    vals = grp["Quantity"].values.astype(float)
                    n = len(vals)
                    mean_q = vals.mean()
                    std_q  = vals.std(ddof=0)
                    cv     = (std_q / mean_q) if mean_q > 0 else 0.0
                    zero_pct = (vals == 0).sum() / n
                    peak_val = vals.max()
                    peak_to_avg = (peak_val / mean_q) if mean_q > 0 else 1.0
                    x_idx = np.arange(n)
                    slope = np.polyfit(x_idx, vals, 1)[0] if n >= 2 else 0.0
                    avg_daily = mean_q / 30.0

                    feature_rows.append({
                        "Item Name":          item,
                        "avg_monthly_demand": round(mean_q, 2),
                        "avg_daily_demand":   round(avg_daily, 4),
                        "demand_cv":          round(cv, 3),
                        "demand_trend_slope": round(slope, 3),
                        "months_of_data":     n,
                        "zero_demand_pct":    round(zero_pct, 3),
                        "peak_to_avg_ratio":  round(peak_to_avg, 3),
                    })

                feat_df = pd.DataFrame(feature_rows)

                # ── Merge current stock (from stock_df or main dataset) ───
                if stock_df is not None and "Item Name" in stock_df.columns and "Current Stock" in stock_df.columns:
                    cs = stock_df[["Item Name","Current Stock"]].copy()
                    cs["Current Stock"] = pd.to_numeric(cs["Current Stock"], errors="coerce")
                    feat_df = feat_df.merge(cs, on="Item Name", how="left")
                elif "Current Stock" in df.columns:
                    cs = df.groupby("Item Name")["Current Stock"].first().reset_index()
                    cs["Current Stock"] = pd.to_numeric(cs["Current Stock"], errors="coerce")
                    feat_df = feat_df.merge(cs, on="Item Name", how="left")
                else:
                    feat_df["Current Stock"] = np.nan

                # ── Days to stockout ──────────────────────────────────────
                feat_df["Days to Stockout"] = (
                    feat_df["Current Stock"] / feat_df["avg_daily_demand"].replace(0, np.nan)
                ).round(0)
                # Convert to nullable integer so it shows as 4 not 4.0
                feat_df["Days to Stockout"] = feat_df["Days to Stockout"].astype("Int64")

                # ── Urgency tier based on days to stockout ────────────────
                def urgency_tier(row):
                    d = row["Days to Stockout"]
                    if pd.isna(d) or pd.isna(row["Current Stock"]):
                        return "Unknown"
                    d = int(d)
                    if d <= 7:
                        return "🔴 Critical (under 7 days)"
                    elif d <= 30:
                        return "🟠 Warning (7–30 days)"
                    elif d <= 90:
                        return "🟡 Monitor (30–90 days)"
                    else:
                        return "🟢 Healthy (90+ days)"
                feat_df["Stock Urgency"] = feat_df.apply(urgency_tier, axis=1)

                # ── Merge ABC / XYZ class ─────────────────────────────────
                abc_all = build_abc_df(df, None)[["Item Name","ABC"]]
                feat_df = feat_df.merge(abc_all, on="Item Name", how="left")
                feat_df["ABC"] = feat_df["ABC"].fillna("C")
                feat_df["abc_class"] = feat_df["ABC"].map({"A": 3, "B": 2, "C": 1}).fillna(1).astype(int)

                xyz_all = build_xyz_series(df)[["Item Name","XYZ"]]
                feat_df = feat_df.merge(xyz_all, on="Item Name", how="left")
                feat_df["XYZ"] = feat_df["XYZ"].fillna("Z")
                feat_df["xyz_class"] = feat_df["XYZ"].map({"X": 1, "Y": 2, "Z": 3}).fillna(3).astype(int)

                # ── Rule-based label generation (self-supervised) ─────────
                def rule_label(row):
                    d = row["Days to Stockout"]
                    # If we have stock data, days-to-stockout drives the label
                    if not pd.isna(d):
                        if d <= 7:   return 2  # High
                        elif d <= 30: return 1  # Medium
                        else:         return 0  # Low
                    # Fallback: demand-pattern based
                    if (row["abc_class"] == 3 and
                            (row["demand_cv"] > 0.8 or
                             row["zero_demand_pct"] > 0.2 or
                             row["demand_trend_slope"] < -1)):
                        return 2
                    elif (row["abc_class"] == 1 or
                          (row["demand_cv"] < 0.3 and row["demand_trend_slope"] > 0)):
                        return 0
                    else:
                        return 1

                feat_df["risk_label"] = feat_df.apply(rule_label, axis=1)

                FEATURE_COLS = ["avg_monthly_demand","demand_cv","demand_trend_slope",
                                "abc_class","xyz_class","months_of_data",
                                "zero_demand_pct","peak_to_avg_ratio"]
                X_feat   = feat_df[FEATURE_COLS].values
                y_labels = feat_df["risk_label"].values

                label_names = {0: "🟢 Low", 1: "🟡 Medium", 2: "🔴 High"}

                # ── XGBoost classifier ────────────────────────────────────
                xgb_available = False
                try:
                    import xgboost as xgb
                    xgb_available = True
                except ImportError:
                    st.warning(
                        "⚠️ `xgboost` is not installed. Showing rule-based labels only. "
                        "Install with: `pip install xgboost==2.1.4`"
                    )

                if xgb_available:
                    clf = xgb.XGBClassifier(
                        n_estimators=100, max_depth=4, random_state=42,
                        eval_metric="mlogloss", use_label_encoder=False
                    )
                    clf.fit(X_feat, y_labels)
                    pred_labels = clf.predict(X_feat)
                    pred_proba  = clf.predict_proba(X_feat)

                    feat_df["Risk"] = [label_names[p] for p in pred_labels]
                    feat_df["P(Low)"]    = (pred_proba[:, 0] * 100).round(1)
                    feat_df["P(Medium)"] = (pred_proba[:, 1] * 100).round(1)
                    feat_df["P(High)"]   = (pred_proba[:, 2] * 100).round(1)
                    model_note = "XGBoost classifier (self-supervised labels)"
                else:
                    feat_df["Risk"] = [label_names[l] for l in y_labels]
                    feat_df["P(Low)"]    = feat_df["risk_label"].apply(lambda x: 100.0 if x == 0 else 0.0)
                    feat_df["P(Medium)"] = feat_df["risk_label"].apply(lambda x: 100.0 if x == 1 else 0.0)
                    feat_df["P(High)"]   = feat_df["risk_label"].apply(lambda x: 100.0 if x == 2 else 0.0)
                    model_note = "Rule-based labels (XGBoost not available)"

                # ── Plain-English risk reason per item ────────────────────
                def risk_reason(row):
                    reasons = []
                    d = row["Days to Stockout"]
                    if not pd.isna(d):
                        d_int = int(d)
                        if d_int <= 7:
                            reasons.append(f"only {d_int} days of stock left")
                        elif d_int <= 30:
                            reasons.append(f"{d_int} days of stock remaining")
                    if row["demand_trend_slope"] < -1:
                        reasons.append("demand is declining")
                    elif row["demand_trend_slope"] > 1:
                        reasons.append("demand is growing")
                    if row["demand_cv"] > 0.8:
                        reasons.append("highly erratic demand")
                    elif row["demand_cv"] > 0.4:
                        reasons.append("variable demand")
                    if row["zero_demand_pct"] > 0.3:
                        reasons.append(f"no sales in {int(row['zero_demand_pct']*100)}% of months")
                    if row["abc_class"] == 3:
                        reasons.append("high-value item (Class A)")
                    if not reasons:
                        if row["Risk"] == "🟢 Low":
                            return "Stable demand and healthy stock levels"
                        else:
                            return "Moderate demand variability"
                    return ", ".join(reasons).capitalize()

                feat_df["Risk Reason"] = feat_df.apply(risk_reason, axis=1)

                # ── Recommended Action per item ───────────────────────────
                def recommended_action(row):
                    risk  = row["Risk"]
                    d     = row["Days to Stockout"]
                    daily = row["avg_daily_demand"]
                    cover_30  = int(round(daily * 30))  if daily > 0 else 0
                    cover_60  = int(round(daily * 60))  if daily > 0 else 0

                    if risk == "🔴 High":
                        if not pd.isna(d) and int(d) <= 7:
                            return f"Order immediately. Stock runs out in {int(d)} day(s). Suggested order: {cover_30} units (30-day cover)"
                        elif not pd.isna(d):
                            return f"Order urgently. Only {int(d)} days of stock left. Suggested order: {cover_30} units (30-day cover)"
                        else:
                            return f"High demand risk. Review stock levels. Suggested order: {cover_30} units (30-day cover)"
                    elif risk == "🟡 Medium":
                        if not pd.isna(d) and int(d) <= 30:
                            return f"Plan an order this week. {int(d)} days of stock remaining. Suggested order: {cover_30} units"
                        else:
                            return f"Monitor closely. Variable demand. Consider ordering {cover_30} units if stock drops below reorder point"
                    else:
                        if not pd.isna(d):
                            return f"No action needed. {int(d)} days of stock remaining. Review again next month"
                        else:
                            return "No action needed. Demand is stable and predictable"

                feat_df["Recommended Action"] = feat_df.apply(recommended_action, axis=1)

                # ── Summary metrics ───────────────────────────────────────
                high_n     = (feat_df["Risk"] == "🔴 High").sum()
                med_n      = (feat_df["Risk"] == "🟡 Medium").sum()
                low_n      = (feat_df["Risk"] == "🟢 Low").sum()
                critical_n = (feat_df["Stock Urgency"] == "🔴 Critical (under 7 days)").sum()
                warning_n  = (feat_df["Stock Urgency"] == "🟠 Warning (7–30 days)").sum()
                has_stock_data = feat_df["Current Stock"].notna().any()

                if has_stock_data:
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("🔴 High Risk",            high_n)
                    c2.metric("🟡 Medium Risk",          med_n)
                    c3.metric("🟢 Low Risk",             low_n)
                    c4.metric("🔴 Critical (< 7 days)",  critical_n)
                    c5.metric("🟠 Warning (7–30 days)",  warning_n)
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🔴 High Risk",   high_n)
                    c2.metric("🟡 Medium Risk", med_n)
                    c3.metric("🟢 Low Risk",    low_n)
                    st.caption("Upload a stock levels file above to unlock Days to Stockout and urgency tiers.")

                st.caption(f"Model: {model_note}")

                # ── Risk distribution chart ───────────────────────────────
                if has_stock_data:
                    fig_r, (ax_r, ax_u) = plt.subplots(1, 2, figsize=(12, 3))
                else:
                    fig_r, ax_r = plt.subplots(figsize=(6, 3))
                    ax_u = None

                # Strip emoji from labels — matplotlib can't render them
                risk_counts  = feat_df["Risk"].value_counts()
                risk_label_map  = {"🔴 High": "High", "🟡 Medium": "Medium", "🟢 Low": "Low"}
                risk_colors_map = {"High": "#dc3545", "Medium": "#ffc107", "Low": "#28a745"}
                plain_risk_labels = [risk_label_map.get(r, r) for r in risk_counts.index]
                ax_r.bar(plain_risk_labels, risk_counts.values,
                         color=[risk_colors_map.get(l, "gray") for l in plain_risk_labels], alpha=0.85)
                ax_r.set_ylabel("SKU Count")
                ax_r.set_title("Stockout Risk Distribution", fontweight="bold")

                if ax_u is not None:
                    urgency_order  = ["🔴 Critical (under 7 days)","🟠 Warning (7–30 days)",
                                      "🟡 Monitor (30–90 days)","🟢 Healthy (90+ days)","Unknown"]
                    urgency_label_map = {
                        "🔴 Critical (under 7 days)": "Critical (<7 days)",
                        "🟠 Warning (7–30 days)":     "Warning (7-30 days)",
                        "🟡 Monitor (30–90 days)":    "Monitor (30-90 days)",
                        "🟢 Healthy (90+ days)":      "Healthy (90+ days)",
                        "Unknown":                     "Unknown",
                    }
                    urgency_colors = {
                        "Critical (<7 days)":   "#dc3545",
                        "Warning (7-30 days)":  "#fd7e14",
                        "Monitor (30-90 days)": "#ffc107",
                        "Healthy (90+ days)":   "#28a745",
                        "Unknown":              "#adb5bd",
                    }
                    urg_counts = feat_df["Stock Urgency"].value_counts().reindex(urgency_order).dropna()
                    plain_urg_labels = [urgency_label_map.get(u, u) for u in urg_counts.index]
                    ax_u.barh(plain_urg_labels, urg_counts.values,
                              color=[urgency_colors.get(l, "gray") for l in plain_urg_labels], alpha=0.85)
                    ax_u.set_xlabel("SKU Count")
                    ax_u.set_title("Days-to-Stockout Urgency", fontweight="bold")

                plt.tight_layout()
                st.pyplot(fig_r)
                plt.close(fig_r)

                # ── Permutation-based feature importance ──────────────────
                if xgb_available:
                    from sklearn.inspection import permutation_importance
                    perm = permutation_importance(
                        clf, X_feat, y_labels,
                        n_repeats=10, random_state=42, scoring="accuracy"
                    )
                    # Friendly display names
                    FEATURE_LABELS = {
                        "avg_monthly_demand":  "Avg Monthly Demand",
                        "demand_cv":           "Demand Variability (CV)",
                        "demand_trend_slope":  "Demand Trend",
                        "abc_class":           "ABC Class",
                        "xyz_class":           "XYZ Class",
                        "months_of_data":      "Months of History",
                        "zero_demand_pct":     "Months with No Sales",
                        "peak_to_avg_ratio":   "Peak-to-Avg Ratio",
                    }
                    fi_df = pd.DataFrame({
                        "Feature":    [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS],
                        "Importance": perm.importances_mean,
                        "Std":        perm.importances_std,
                    }).sort_values("Importance", ascending=True)

                    fig_fi, ax_fi = plt.subplots(figsize=(8, 4))
                    bars = ax_fi.barh(
                        fi_df["Feature"], fi_df["Importance"],
                        xerr=fi_df["Std"], color="#145a2e", alpha=0.85,
                        error_kw={"ecolor": "#f0c040", "capsize": 3}
                    )
                    ax_fi.axvline(0, color="gray", linewidth=0.8, linestyle="--")
                    ax_fi.set_xlabel("Mean accuracy drop when feature is shuffled")
                    ax_fi.set_title("What drives the risk score (Permutation Importance)", fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig_fi)
                    plt.close(fig_fi)
                    st.caption(
                        "Each bar shows how much model accuracy drops when that feature is randomly shuffled. "
                        "A larger drop means the feature matters more. "
                        "Error bars show variability across 10 shuffles. "
                        "Features near zero have little impact on the prediction."
                    )

                # ── Per-SKU risk table ────────────────────────────────────
                st.subheader("Per-SKU Risk Table",
                             help="Items sorted by highest risk first. 'Recommended Action' tells you exactly what to do for each item. 'Days to Stockout' shows how long current stock will last at the current demand rate. 'Risk Reason' explains in plain English why each item received its risk score.")

                # Friendly column names for display
                display_df = feat_df.copy()
                display_df = display_df.rename(columns={
                    "avg_monthly_demand":  "Avg Monthly Demand",
                    "demand_cv":           "Demand Variability (CV)",
                    "demand_trend_slope":  "Trend (slope)",
                    "zero_demand_pct":     "Months with No Sales",
                    "months_of_data":      "Months of History",
                })
                display_df["Months with No Sales"] = (display_df["Months with No Sales"] * 100).round(1).astype(str) + "%"

                display_cols = ["Item Name","Risk","Recommended Action","Risk Reason","Stock Urgency",
                                "Current Stock","Days to Stockout",
                                "ABC","XYZ","Avg Monthly Demand",
                                "Demand Variability (CV)","Trend (slope)",
                                "Months with No Sales","Months of History",
                                "P(High)","P(Medium)","P(Low)"]
                display_cols = [c for c in display_cols if c in display_df.columns]

                risk_display = display_df[display_cols].sort_values(
                    ["P(High)","Days to Stockout"],
                    ascending=[False, True],
                    na_position="last"
                ).reset_index(drop=True)

                st.dataframe(risk_display, use_container_width=True, height=450, hide_index=True)
                st.download_button(
                    "⬇️ Download stockout risk table",
                    to_excel_bytes(risk_display),
                    "stockout_risk.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except Exception as e:
                st.error(f"Stockout risk analysis failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    # ══════════════════════════════════════════════════════════════════════
    # 🧮 OPTIMIZATION MODELS — independent section after Stockout Risk
    # Reads from session state so it persists when sub-buttons are clicked
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("— Optimization Models —",
              help="Four optimization models that go beyond simple reorder rules to help you make smarter, cost-efficient inventory decisions. Run Calculate Reorder Points first to unlock these.")

    if "rop_df" not in st.session_state:
        st.info("Run **Calculate Reorder Points** above first. The optimization models use the demand, safety stock, and EOQ values calculated there.")
    else:
        _rop_df   = st.session_state["rop_df"]
        _rop_disp = st.session_state["rop_display"]
        _stock_df = st.session_state.get("stock_df_cached", None)

        # Track each expander's open state independently
        # Set to True when user runs that model; stays True until new file uploaded
        for _k in ["opt1_open","opt2_open","opt3_open","opt4_open"]:
            if _k not in st.session_state:
                st.session_state[_k] = False

        with st.expander("📐 Safety Stock Portfolio Optimization",
                         expanded=st.session_state["opt1_open"]):
            st.markdown(
                "**What it does:** Your current safety stock is calculated item by item using a fixed formula. "
                "This model looks at your entire product range at once and asks: *how can we redistribute safety stock "
                "across all items to hit a target fill rate at the lowest possible holding cost?*\n\n"
                "In practice, it reduces safety stock on stable, low-value items (where you rarely stock out anyway) "
                "and increases it on high-value or erratic items (where a stockout is costly). "
                "The result is the same or better service level, but with less cash tied up in inventory overall.\n\n"
                "**How to use it:** Set your target fill rate (e.g. 95% means you want to be able to fulfil 95% of "
                "monthly demand without a stockout), then click Run. The table shows which items get more or less "
                "safety stock compared to the current calculation."
            )
            opt1_target = st.slider("Target portfolio fill rate (%)", 80, 99, 95, step=1,
                help="The minimum percentage of total monthly demand you want to be able to fulfil without a stockout across all items.",
                key="opt1_target",
                on_change=lambda: st.session_state.update({"opt1_open": True}))
            if st.button("Run Safety Stock Optimization", key="opt1_run",
                         on_click=lambda: st.session_state.update({"opt1_open": True})):
                try:
                    from scipy.optimize import minimize
                    from scipy.stats import norm as sp_norm
                    items_opt  = _rop_df["Item Name"].tolist()
                    demand_opt = _rop_df["Effective Daily Demand"].values.astype(float)
                    daily_std  = _rop_df["Daily Std"].values.astype(float)
                    lead_opt   = _rop_df["Lead Time"].values.astype(float)
                    abc_opt    = _rop_df["ABC"].values
                    hold_cost  = (_rop_df["_unit_price"].values.astype(float) * 0.25 / 365).clip(min=0.01) \
                                 if "_unit_price" in _rop_df.columns else np.ones(len(items_opt)) * 0.01
                    monthly_demand_opt = demand_opt * 30
                    total_monthly      = monthly_demand_opt.sum()
                    if total_monthly == 0:
                        st.warning("No demand data found. Run Calculate Reorder Points with a wider demand window.")
                    else:
                        target_fill = opt1_target / 100.0
                        def objective(ss):
                            return np.dot(ss, hold_cost)
                        def fill_rate_constraint(ss):
                            sigma_lt = np.where(daily_std > 0, daily_std * np.sqrt(np.maximum(lead_opt, 1)), 1e-6)
                            return np.dot(monthly_demand_opt / (total_monthly + 1e-9), sp_norm.cdf(ss / sigma_lt)) - target_fill
                        ss0 = _rop_df["Safety Stock"].values.astype(float).clip(min=0)
                        with st.spinner("Optimizing safety stock levels…"):
                            result = minimize(objective, ss0, method="SLSQP",
                                              bounds=[(0, None)] * len(items_opt),
                                              constraints=[{"type": "ineq", "fun": fill_rate_constraint}],
                                              options={"maxiter": 500, "ftol": 1e-6})
                        if result.success or result.status == 9:
                            if result.status == 9:
                                st.warning("Optimizer reached the iteration limit. Results are the best found so far but may not be fully optimal. Try lowering the target fill rate.")
                            ss_opt = np.maximum(0, result.x).round(1)
                            opt1_df = pd.DataFrame({
                                "Item Name": items_opt, "ABC": abc_opt,
                                "Current Safety Stock": ss0.round(1),
                                "Optimized Safety Stock": ss_opt,
                                "Change": (ss_opt - ss0).round(1),
                                "Monthly Demand": monthly_demand_opt.round(0).astype(int),
                            })
                            opt1_df["Direction"] = opt1_df["Change"].apply(
                                lambda x: "⬆️ Increase" if x > 0.5 else ("⬇️ Reduce" if x < -0.5 else "➡️ No change"))
                            sigma_lt = np.where(daily_std > 0, daily_std * np.sqrt(np.maximum(lead_opt, 1)), 1e-6)
                            achieved = float(np.dot(monthly_demand_opt / (total_monthly + 1e-9), sp_norm.cdf(ss_opt / sigma_lt)))
                            saving   = np.dot(ss0, hold_cost) - np.dot(ss_opt, hold_cost)
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Target Fill Rate",    f"{opt1_target}%")
                            c2.metric("Achieved Fill Rate",  f"{achieved*100:.1f}%")
                            c3.metric("Holding Cost Change", f"{'−' if saving >= 0 else '+'}{abs(saving):.2f} units·cost")
                            st.dataframe(opt1_df.sort_values("Change").reset_index(drop=True),
                                         use_container_width=True, height=400, hide_index=True)
                            st.download_button("⬇️ Download optimized safety stock", to_excel_bytes(opt1_df),
                                               "optimized_safety_stock.xlsx",
                                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                               key="dl_opt1")
                        else:
                            st.warning(f"Optimizer could not find a feasible solution at {opt1_target}% fill rate. Try lowering the target.")
                except Exception as _e1:
                    st.error(f"Safety stock optimization failed: {_e1}")

        with st.expander("💰 Budget-Constrained Replenishment",
                         expanded=st.session_state["opt2_open"]):
            st.markdown(
                "**What it does:** When you can't afford to restock everything at once, this model helps you "
                "decide *which items to restock first* with the money you have available.\n\n"
                "It ranks every item that needs restocking by two things: how urgent it is (items running out "
                "sooner rank higher) and how important it is (Class A items rank higher than C). "
                "It then works through the list from most to least urgent, allocating your budget until it runs out. "
                "Each item either gets fully funded, partially funded, or skipped.\n\n"
                "**How to use it:** Enter your total restocking budget and click Run. The table shows exactly "
                "how many units of each item you can afford to order, what it costs, and which items had to be skipped."
            )
            if "_unit_price" not in _rop_df.columns or (_rop_df["_unit_price"] == 0).all():
                st.info("This optimization requires Unit Price data. Add a Unit Price column to your demand history file to enable it.")
            else:
                budget_input = st.number_input("Total restocking budget ($)", min_value=100, max_value=10_000_000,
                    value=10_000, step=500, key="opt2_budget",
                    help="Enter the total amount available to spend on restocking orders.",
                    on_change=lambda: st.session_state.update({"opt2_open": True}))
                if st.button("Run Budget Optimization", key="opt2_run",
                             on_click=lambda: st.session_state.update({"opt2_open": True})):
                    try:
                        budget_df = _rop_df[
                            _rop_df["Status"].isin(["🔴 Reorder Now","🟡 Low Stock"]) &
                            (_rop_df["_unit_price"] > 0)
                        ].copy()
                        if budget_df.empty:
                            st.info("No items currently need restocking, or no price data available.")
                        else:
                            # ── Build priority scores ─────────────────────
                            abc_weight = {"A": 3, "B": 2, "C": 1}
                            budget_df["abc_w"]     = budget_df["ABC"].map(abc_weight).fillna(1)
                            max_days               = budget_df["Days of Stock"].replace(0, np.nan).max()
                            budget_df["urgency_w"] = 1 - (budget_df["Days of Stock"] / (max_days + 1))
                            budget_df["priority"]  = budget_df["abc_w"] * (1 + budget_df["urgency_w"])
                            budget_df = budget_df.reset_index(drop=True)

                            prices  = budget_df["_unit_price"].values.astype(float)
                            eoqs    = budget_df["EOQ"].values.astype(int)
                            prios   = budget_df["priority"].values.astype(float)
                            n_items = len(budget_df)

                            # ── Integer Linear Programming (ILP) ──────────
                            # Maximise: sum(priority_i × units_i / eoq_i)  [weighted coverage]
                            # Subject to: sum(price_i × units_i) <= budget
                            #             0 <= units_i <= eoq_i  (integer)
                            # We use LP relaxation via HiGHS then round down
                            # (true ILP would need PuLP/CBC — LP relaxation is near-optimal here
                            #  since items are divisible in practice)
                            from scipy.optimize import linprog

                            # Negate priority for minimisation
                            c_obj  = -(prios / np.maximum(eoqs, 1))
                            # Budget constraint: prices · units <= budget
                            A_ub   = prices.reshape(1, -1)
                            b_ub   = np.array([float(budget_input)])
                            bounds = [(0, int(e)) for e in eoqs]

                            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                                          bounds=bounds, method="highs")

                            if res.success:
                                # Round down to integers (conservative — never overspend)
                                alloc_units = np.floor(res.x).astype(int)
                                # Use remaining budget to top up highest-priority items
                                spent = float(np.dot(alloc_units, prices))
                                order = np.argsort(-prios)
                                for i in order:
                                    gap = eoqs[i] - alloc_units[i]
                                    if gap > 0 and prices[i] > 0:
                                        extra = min(gap, int((budget_input - spent) // prices[i]))
                                        alloc_units[i] += extra
                                        spent += extra * prices[i]
                            else:
                                # Fallback to greedy if LP fails
                                alloc_units = np.zeros(n_items, dtype=int)
                                remaining   = float(budget_input)
                                for i in np.argsort(-prios):
                                    if prices[i] > 0:
                                        units = min(eoqs[i], int(remaining // prices[i]))
                                        alloc_units[i] = units
                                        remaining -= units * prices[i]

                            alloc_cost = (alloc_units * prices).round(2)
                            budget_df["Allocated Units"] = alloc_units
                            budget_df["Allocated Cost"]  = alloc_cost
                            budget_df["Fully Funded"]    = budget_df.apply(
                                lambda r: "✅ Yes" if r["Allocated Units"] >= r["EOQ"] else
                                          ("⚠️ Partial" if r["Allocated Units"] > 0 else "❌ No"), axis=1)

                            total_spent   = float(alloc_cost.sum())
                            items_partial = (budget_df["Fully Funded"] == "⚠️ Partial").sum()
                            items_skipped = (budget_df["Fully Funded"] == "❌ No").sum()

                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Budget",             f"${budget_input:,.0f}")
                            c2.metric("Allocated",          f"${total_spent:,.0f}")
                            c3.metric("Remaining",          f"${budget_input - total_spent:,.0f}")
                            c4.metric("Items Fully Funded", (budget_df["Fully Funded"] == "✅ Yes").sum())
                            if items_partial > 0:
                                st.caption(f"{items_partial} item(s) partially funded, {items_skipped} item(s) skipped due to budget.")

                            budget_df["Priority Score"] = budget_df["priority"].round(2)
                            budget_df = budget_df.rename(columns={"_unit_price": "Unit Price ($)"})
                            show_cols = [c for c in ["Item Name","ABC","Status","Priority Score","EOQ",
                                         "Allocated Units","Allocated Cost","Fully Funded",
                                         "Days of Stock","Unit Price ($)"] if c in budget_df.columns]
                            st.dataframe(budget_df[show_cols].sort_values("Priority Score", ascending=False)
                                         .reset_index(drop=True),
                                         use_container_width=True, height=400, hide_index=True)
                            st.download_button("⬇️ Download budget allocation plan",
                                               to_excel_bytes(budget_df[show_cols]), "budget_allocation.xlsx",
                                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                               key="dl_opt2")
                    except Exception as _e2:
                        st.error(f"Budget optimization failed: {_e2}")

        with st.expander("🧮 Order Schedule Optimization",
                         expanded=st.session_state["opt3_open"]):
            st.markdown(
                "**What it does:** Instead of just telling you *which* items to reorder, this model tells you "
                "*when* to place each order, down to the exact date.\n\n"
                "It works backwards from your current stock: for each item, it calculates how many days until "
                "your stock drops to the reorder point, then subtracts the lead time so the order arrives just "
                "in time. Items are grouped into urgency tiers: Order Today, Order This Week, Order This Month, "
                "and Not Yet.\n\n"
                "**How to use it:** Click Generate Order Schedule. The table gives you a full calendar of order "
                "dates and expected arrival dates for every item, sorted from most urgent to least. "
                "Download it and use it as your purchasing calendar."
            )
            if "Current Stock" not in _rop_df.columns or _rop_df["Current Stock"].dtype == object:
                st.info("Upload a stock levels file to enable order scheduling.")
            else:
                if st.button("Generate Order Schedule", key="opt3_run",
                             on_click=lambda: st.session_state.update({"opt3_open": True})):
                    try:
                        sched_df = _rop_df[_rop_df["Effective Daily Demand"] > 0].copy()
                        sched_df["Days Until ROP"] = (
                            (sched_df["Current Stock"] - sched_df["ROP (units)"]) /
                            sched_df["Effective Daily Demand"]
                        ).clip(lower=0).round(0).astype(int)
                        today = pd.Timestamp.today().normalize()
                        sched_df["Order By Date"]    = sched_df["Days Until ROP"].apply(
                            lambda d: today + pd.Timedelta(days=max(0, d)))
                        sched_df["Expected Arrival"] = sched_df.apply(
                            lambda r: r["Order By Date"] + pd.Timedelta(days=int(r["Lead Time"])), axis=1)
                        sched_df["Urgency"] = sched_df["Days Until ROP"].apply(
                            lambda d: "🔴 Order Today"      if d == 0  else
                                      "🟠 Order This Week"  if d <= 7  else
                                      "🟡 Order This Month" if d <= 30 else "🟢 Not Yet")
                        sched_df = sched_df.sort_values("Order By Date").reset_index(drop=True)
                        sched_df["Order By Date"]    = sched_df["Order By Date"].dt.strftime("%d %b %Y")
                        sched_df["Expected Arrival"] = sched_df["Expected Arrival"].dt.strftime("%d %b %Y")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("🔴 Order Today",      (sched_df["Urgency"] == "🔴 Order Today").sum())
                        c2.metric("🟠 Order This Week",  (sched_df["Urgency"] == "🟠 Order This Week").sum())
                        c3.metric("🟡 Order This Month", (sched_df["Urgency"] == "🟡 Order This Month").sum())
                        sched_cols = [c for c in ["Item Name","ABC","Urgency","Order By Date","Expected Arrival",
                                      "Days Until ROP","Current Stock","ROP (units)","EOQ","Lead Time"]
                                      if c in sched_df.columns]
                        st.dataframe(sched_df[sched_cols], use_container_width=True, height=450, hide_index=True)
                        st.download_button("⬇️ Download order schedule", to_excel_bytes(sched_df[sched_cols]),
                                           "order_schedule.xlsx",
                                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                           key="dl_opt3")
                    except Exception as _e3:
                        st.error(f"Order schedule generation failed: {_e3}")

        with st.expander("🏭 Supplier Allocation Optimization",
                         expanded=st.session_state["opt4_open"]):
            st.markdown(
                "**What it does:** If you buy the same item from more than one supplier, each with a different "
                "price or lead time, this model finds the cheapest way to split your order across them.\n\n"
                "For example, if you need 500 units of an item and Supplier A charges $10 while Supplier B "
                "charges $8 but has a longer lead time, the model works out the optimal split that minimises "
                "your total cost while staying within your acceptable lead time.\n\n"
                "**How to use it:** Upload a stock file where the same item appears on multiple rows with "
                "different Supplier, Unit Price, and Lead Time (days) values. Then select the item, set your "
                "maximum acceptable lead time, and click Optimize. The table shows how many units to order "
                "from each supplier and the total cost saving vs ordering everything from the most expensive one."
            )
            multi_supplier_available = False
            if _stock_df is not None and "Supplier" in _stock_df.columns and "Item Name" in _stock_df.columns:
                supplier_counts = _stock_df.groupby("Item Name")["Supplier"].nunique()
                multi_supplier_available = (supplier_counts > 1).any()
            if not multi_supplier_available:
                st.info(
                    "No multi-supplier data detected. To use this feature, upload a stock file where "
                    "the same item appears on multiple rows with different Supplier, Unit Price, and "
                    "Lead Time (days) values."
                )
            else:
                multi_items = supplier_counts[supplier_counts > 1].index.tolist()
                selected_alloc_item = st.selectbox("Select item to optimize supplier split",
                                                   multi_items, key="opt4_item",
                                                   on_change=lambda: st.session_state.update({"opt4_open": True}))
                col_lead, col_max = st.columns(2)
                max_lead = col_lead.number_input("Maximum acceptable lead time (days)",
                                           min_value=1, max_value=180, value=30, key="opt4_lead",
                                           help="Only suppliers with lead time at or below this value will be considered.",
                                           on_change=lambda: st.session_state.update({"opt4_open": True}))
                max_alloc_pct = col_max.slider("Max allocation per supplier (%)", 30, 100, 70, step=5,
                                           key="opt4_max_pct",
                                           help="Caps how much of the total order any single supplier can receive. Reduces supply chain risk by spreading orders across multiple suppliers.",
                                           on_change=lambda: st.session_state.update({"opt4_open": True}))
                if st.button("Optimize Supplier Split", key="opt4_run",
                             on_click=lambda: st.session_state.update({"opt4_open": True})):
                    try:
                        item_suppliers = _stock_df[_stock_df["Item Name"] == selected_alloc_item].copy()
                        price_col_sup  = next((c for c in ["Unit Price","Price"] if c in item_suppliers.columns), None)
                        lt_col_sup     = "Lead Time (days)" if "Lead Time (days)" in item_suppliers.columns else None
                        if price_col_sup is None:
                            st.warning("No Unit Price column found in the stock file.")
                        else:
                            if lt_col_sup:
                                item_suppliers = item_suppliers[item_suppliers[lt_col_sup] <= max_lead]
                            if len(item_suppliers) < 2:
                                st.warning(f"Only one supplier meets the lead time requirement of {max_lead} days.")
                            else:
                                from scipy.optimize import linprog
                                eoq_val = int(_rop_df[_rop_df["Item Name"] == selected_alloc_item]["EOQ"].values[0]) \
                                          if selected_alloc_item in _rop_df["Item Name"].values else 100
                                prices  = item_suppliers[price_col_sup].values.astype(float)
                                n_sup   = len(prices)
                                max_qty = float(eoq_val) * (max_alloc_pct / 100.0)
                                res = linprog(
                                    prices,
                                    A_eq=np.ones((1, n_sup)),
                                    b_eq=np.array([float(eoq_val)]),
                                    bounds=[(0, max_qty)] * n_sup,
                                    method="highs"
                                )
                                if res.success:
                                    alloc_qty = np.round(res.x).astype(int)
                                    diff = eoq_val - alloc_qty.sum()
                                    if diff != 0:
                                        idx_cheapest = int(np.argmin(prices))
                                        alloc_qty[idx_cheapest] = max(0, alloc_qty[idx_cheapest] + diff)
                                    alloc_cost   = alloc_qty * prices
                                    alloc_result = item_suppliers[["Supplier"]].copy().reset_index(drop=True)
                                    if lt_col_sup:
                                        alloc_result["Lead Time (days)"] = item_suppliers[lt_col_sup].values
                                    alloc_result["Unit Price ($)"]     = prices.round(2)
                                    alloc_result["Allocated Units"]    = alloc_qty
                                    alloc_result["Allocated Cost ($)"] = alloc_cost.round(2)
                                    alloc_result["% of Order"]         = (alloc_qty / eoq_val * 100).round(1)
                                    total_cost = float(alloc_cost.sum())
                                    saving     = eoq_val * prices.max() - total_cost
                                    wt_lt = float(np.dot(alloc_qty / max(eoq_val, 1),
                                                         item_suppliers[lt_col_sup].values.astype(float))) \
                                            if lt_col_sup else None
                                    c1, c2, c3, c4 = st.columns(4)
                                    c1.metric("Total Order Qty",           f"{eoq_val} units")
                                    c2.metric("Optimized Cost",            f"${total_cost:,.2f}")
                                    c3.metric("Saving vs Single Supplier", f"${saving:,.2f}",
                                              help="Compared to ordering everything from the most expensive supplier.")
                                    if wt_lt is not None:
                                        c4.metric("Weighted Avg Lead Time", f"{wt_lt:.1f} days",
                                                  help="Average lead time weighted by the units allocated to each supplier.")
                                    st.dataframe(alloc_result.reset_index(drop=True),
                                                 use_container_width=True, hide_index=True)
                                    st.download_button("⬇️ Download supplier allocation",
                                                       to_excel_bytes(alloc_result), "supplier_allocation.xlsx",
                                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                                       key="dl_opt4")
                                else:
                                    st.warning(
                                        f"Could not find a feasible split with max {max_alloc_pct}% per supplier. "
                                        "Try increasing the max allocation percentage or the lead time limit."
                                    )
                    except Exception as _e4:
                        st.error(f"Supplier allocation optimization failed: {_e4}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – CUSTOMER SEGMENTATION (RFM + Quintile Scoring + Adaptive Clustering)
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("👥 Customer Segmentation",
              help="This tab analyses your customers using the RFM method: Recency (how recently they ordered), Frequency (how often they order), and Monetary (how much they spend). Each customer is automatically scored and placed into a meaningful group like Champion, At Risk, or Lost, so you know exactly who to focus on and what to do next.")

    required_rfm = {"Customer Name","Invoice Date","Quantity"}
    if not required_rfm.issubset(df.columns):
        st.warning(f"Customer Segmentation requires columns: {required_rfm}. Missing: {required_rfm - set(df.columns)}")
    else:
        if st.button("🔍 Run Segmentation", type="primary"):
            with st.spinner("Segmenting customers…"):
                snapshot = df["Invoice Date"].max()

                # ── Monetary: use Revenue if available, else Quantity ─────
                revenue_col_rfm = next((c for c in ["Revenue","Unit Price"] if c in df.columns), None)
                df_rfm = df.copy()  # work on a copy to avoid mutating module-level df
                if revenue_col_rfm == "Unit Price":
                    df_rfm["_rfm_monetary"] = df_rfm["Quantity"] * df_rfm["Unit Price"].fillna(0)
                    monetary_label = "Revenue ($)"
                elif revenue_col_rfm == "Revenue":
                    df_rfm["_rfm_monetary"] = df_rfm["Revenue"].fillna(0)
                    monetary_label = "Revenue ($)"
                else:
                    df_rfm["_rfm_monetary"] = df_rfm["Quantity"]
                    monetary_label = "Total Quantity"

                rfm = df_rfm.groupby("Customer Name").agg(
                    Recency   = ("Invoice Date",  lambda x: (snapshot - x.max()).days),
                    Frequency = ("Invoice Number" if "Invoice Number" in df_rfm.columns else "Invoice Date",
                                 "nunique"),
                    Monetary  = ("_rfm_monetary", "sum"),
                    First_Order = ("Invoice Date", "min"),
                    Last_Order  = ("Invoice Date", "max"),
                ).reset_index()

                # ── Customer trend + avg order interval + overdue detection ─
                df_trend = df_rfm.copy()
                df_trend["Month"] = df_trend["Invoice Date"].dt.to_period("M")
                mid_date = df_trend["Invoice Date"].min() + (df_trend["Invoice Date"].max() - df_trend["Invoice Date"].min()) / 2
                trend_rows = []
                for cust, grp in df_trend.groupby("Customer Name"):
                    early = grp[grp["Invoice Date"] <= mid_date]["_rfm_monetary"].sum()
                    late  = grp[grp["Invoice Date"] >  mid_date]["_rfm_monetary"].sum()
                    if early > 0 and late > early * 1.1:
                        trend = "📈 Growing"
                    elif late < early * 0.9 and early > 0:
                        trend = "📉 Declining"
                    else:
                        trend = "➡️ Stable"

                    # Avg order interval (days between orders)
                    dates = grp["Invoice Date"].sort_values().drop_duplicates().values
                    if len(dates) >= 2:
                        gaps = [int((dates[i+1] - dates[i]) / np.timedelta64(1, 'D'))
                                for i in range(len(dates)-1)]
                        avg_gap = int(np.mean(gaps))
                    else:
                        avg_gap = None

                    trend_rows.append({
                        "Customer Name": cust,
                        "Trend": trend,
                        "Avg Order Interval (days)": avg_gap,
                    })

                trend_df = pd.DataFrame(trend_rows)
                rfm = rfm.merge(trend_df, on="Customer Name", how="left")

                # Overdue flag: recency > avg order interval × 1.5
                def overdue_flag(row):
                    gap = row["Avg Order Interval (days)"]
                    if gap is None or pd.isna(gap) or gap == 0:
                        return "Unknown"
                    overdue_by = row["Recency"] - gap
                    if overdue_by > gap * 0.5:
                        return f"⚠️ {int(overdue_by)} days overdue"
                    elif overdue_by > 0:
                        return f"🟡 {int(overdue_by)} days late"
                    else:
                        return "🟢 On schedule"
                rfm["Order Status"] = rfm.apply(overdue_flag, axis=1)

                # ── CLV estimate (12-month forward projection) ────────────
                # CLV = (Monetary / Frequency) × (365 / avg_order_interval) × 1 year
                # Falls back to Monetary × 1.2 if interval unknown
                date_span_days = max((df["Invoice Date"].max() - df["Invoice Date"].min()).days, 1)
                def calc_clv(row):
                    freq = row["Frequency"]
                    mon  = row["Monetary"]
                    gap  = row["Avg Order Interval (days)"]
                    if freq == 0 or mon == 0:
                        return 0.0
                    avg_order_value = mon / freq
                    if gap and not pd.isna(gap) and gap > 0:
                        orders_per_year = 365 / gap
                    else:
                        orders_per_year = (freq / date_span_days) * 365
                    return round(avg_order_value * orders_per_year, 1)
                rfm["Est. Annual Value (12 months)"] = rfm.apply(calc_clv, axis=1)

                # ── RFM Quintile Scoring ──────────────────────────────────
                try:
                    rfm["R_Score"] = pd.qcut(rfm["Recency"].rank(method="first"),
                                             q=5, labels=[5,4,3,2,1]).astype(int)
                    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"),
                                             q=5, labels=[1,2,3,4,5]).astype(int)
                    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"),
                                             q=5, labels=[1,2,3,4,5]).astype(int)
                except Exception:
                    rfm["R_Score"] = 3
                    rfm["F_Score"] = 3
                    rfm["M_Score"] = 3

                rfm["RFM_Score"] = (
                    (rfm["R_Score"] * 0.3 + rfm["F_Score"] * 0.3 + rfm["M_Score"] * 0.4)
                    * (20 / 3)
                ).round(1)

                # ── Behaviour-based segment labels (RFM pattern) ──────────
                def rfm_segment_label(row):
                    r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
                    if r >= 4 and f >= 4 and m >= 4:
                        return "🏆 Champion"
                    elif r >= 3 and f >= 3 and m >= 3:
                        return "⭐ Loyal"
                    elif r >= 4 and f <= 2:
                        return "🆕 New Customer"
                    elif r >= 3 and m >= 4:
                        return "💰 High Spender"
                    elif r <= 2 and f >= 3 and m >= 3:
                        return "⚠️ At Risk"
                    elif r <= 2 and f >= 4:
                        return "😴 Lapsed Loyal"
                    elif r == 1 and f >= 2:
                        return "💀 Lost"
                    elif m <= 2 and f <= 2:
                        return "🌱 Low Value"
                    else:
                        return "🔄 Occasional"

                rfm["Segment"] = rfm.apply(rfm_segment_label, axis=1)

                # ── Churn / at-risk flag ──────────────────────────────────
                avg_order_gap = rfm["Recency"].median()
                rfm["Churn Risk"] = rfm["Recency"].apply(
                    lambda r: "🔴 High" if r > avg_order_gap * 2
                              else ("🟡 Medium" if r > avg_order_gap * 1.3 else "🟢 Low")
                )

                # ── Recommended action per segment ────────────────────────
                SEGMENT_ACTIONS = {
                    "🏆 Champion":      "Reward with exclusive offers. Ask for referrals or reviews.",
                    "⭐ Loyal":         "Upsell premium products. Offer loyalty discounts.",
                    "🆕 New Customer":  "Onboard with a welcome offer. Encourage a second purchase.",
                    "💰 High Spender":  "Offer premium or bundle deals. Prioritise for account management.",
                    "⚠️ At Risk":       "Send a win-back campaign. Offer a discount or check in personally.",
                    "😴 Lapsed Loyal":  "Re-engage with a personalised offer based on past purchases.",
                    "💀 Lost":          "Last-chance re-engagement. If no response, deprioritise.",
                    "🌱 Low Value":     "Nurture with low-cost touchpoints. Look for upsell opportunities.",
                    "🔄 Occasional":    "Increase purchase frequency with targeted promotions.",
                }
                rfm["Recommended Action"] = rfm["Segment"].map(SEGMENT_ACTIONS).fillna("Review manually.")

                # ── Adaptive clustering: auto-optimal K-Means vs GMM vs DBSCAN ──
                # K and GMM: auto-select optimal number of clusters (2–6) using silhouette
                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans, DBSCAN
                from sklearn.metrics import silhouette_score
                from sklearn.neighbors import NearestNeighbors

                scaler = StandardScaler()
                X = scaler.fit_transform(rfm[["Recency","Frequency","Monetary"]])

                max_k = min(6, max(2, len(X) - 1))

                # Auto-find best K for K-Means
                best_km_sil, best_km_labels, best_k = 0.0, None, 2
                for k in range(2, max_k + 1):
                    try:
                        km_try = KMeans(n_clusters=k, random_state=42, n_init=10)
                        lbl    = km_try.fit_predict(X)
                        sil    = silhouette_score(X, lbl) if len(set(lbl)) > 1 else 0.0
                        if sil > best_km_sil:
                            best_km_sil, best_km_labels, best_k = sil, lbl, k
                    except Exception:
                        pass
                # Fallback: if K-Means failed entirely, use k=2
                if best_km_labels is None:
                    km_fb = KMeans(n_clusters=2, random_state=42, n_init=10)
                    best_km_labels = km_fb.fit_predict(X)
                    best_km_sil = 0.0
                km_sil, km_labels = best_km_sil, best_km_labels

                # Auto-find best K for GMM
                gmm_sil, gmm_labels = 0.0, None
                try:
                    from sklearn.mixture import GaussianMixture
                    best_gmm_sil, best_gmm_labels = 0.0, None
                    for k in range(2, max_k + 1):
                        gmm_try = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
                        lbl     = gmm_try.fit_predict(X)
                        sil     = silhouette_score(X, lbl) if k > 1 else 0.0
                        if sil > best_gmm_sil:
                            best_gmm_sil, best_gmm_labels = sil, lbl
                    gmm_sil, gmm_labels = best_gmm_sil, best_gmm_labels
                except Exception:
                    pass

                # DBSCAN — auto-tune eps
                dbscan_sil, dbscan_labels = 0.0, None
                try:
                    k_nn = max(2, min(5, len(X) - 1))
                    nbrs = NearestNeighbors(n_neighbors=k_nn).fit(X)
                    distances, _ = nbrs.kneighbors(X)
                    eps_auto = float(np.percentile(distances[:, -1], 90))
                    db = DBSCAN(eps=eps_auto, min_samples=max(2, len(X) // 20))
                    db_raw = db.fit_predict(X)
                    n_db_clusters = len(set(db_raw) - {-1})
                    if 2 <= n_db_clusters <= max_k + 2:
                        if -1 in db_raw:
                            from sklearn.neighbors import KNeighborsClassifier
                            mask = db_raw != -1
                            if mask.sum() > 0:
                                knn = KNeighborsClassifier(n_neighbors=1)
                                knn.fit(X[mask], db_raw[mask])
                                db_raw[~mask] = knn.predict(X[~mask])
                        dbscan_labels = db_raw
                        dbscan_sil = silhouette_score(X, dbscan_labels) if len(set(dbscan_labels)) > 1 else 0.0
                except Exception:
                    pass

                # Pick the best model
                best_sil, best_labels, best_method = km_sil, km_labels, f"K-Means with {best_k} clusters (silhouette: {km_sil:.2f})"
                if gmm_labels is not None and gmm_sil > best_sil:
                    best_sil, best_labels, best_method = gmm_sil, gmm_labels, f"GMM (silhouette: {gmm_sil:.2f})"
                if dbscan_labels is not None and dbscan_sil > best_sil:
                    best_sil, best_labels, best_method = dbscan_sil, dbscan_labels, f"DBSCAN (silhouette: {dbscan_sil:.2f})"

                rfm["Cluster"] = best_labels
                st.success(f"Best clustering model selected: **{best_method}**")
                st.caption(
                    f"K-Means: {km_sil:.2f} · "
                    f"GMM: {gmm_sil:.2f} · "
                    f"DBSCAN: {f'{dbscan_sil:.2f}' if dbscan_labels is not None else 'N/A'} "
                    "Scores closer to 1.0 mean the groups are more clearly separated. "
                    "The customer labels below are independent of this grouping."
                )

                # ── Silhouette plot ───────────────────────────────────────
                from sklearn.metrics import silhouette_samples
                sil_vals   = silhouette_samples(X, best_labels)
                unique_cls = sorted(set(best_labels))
                fig_sil, ax_sil = plt.subplots(figsize=(8, max(3, len(unique_cls) * 0.8)))
                y_lower = 10
                cluster_colors = plt.cm.Set2.colors
                for i, cl in enumerate(unique_cls):
                    cl_sil = np.sort(sil_vals[best_labels == cl])
                    size   = cl_sil.shape[0]
                    y_upper = y_lower + size
                    ax_sil.fill_betweenx(np.arange(y_lower, y_upper), 0, cl_sil,
                                         facecolor=cluster_colors[i % len(cluster_colors)],
                                         alpha=0.7, label=f"Cluster {i+1} ({size})")
                    y_lower = y_upper + 5
                ax_sil.axvline(x=best_sil, color="red", linestyle="--", linewidth=1.5,
                               label=f"Avg silhouette: {best_sil:.2f}")
                ax_sil.set_xlabel("Silhouette coefficient (higher = better separated)")
                ax_sil.set_ylabel("Customers")
                ax_sil.set_title("Cluster Quality: Silhouette Plot", fontweight="bold")
                ax_sil.legend(fontsize=8, loc="lower right")
                plt.tight_layout()
                st.pyplot(fig_sil); plt.close(fig_sil)
                st.caption(
                    "Each bar represents one customer. "
                    "Longer bars mean the customer fits their group well. "
                    "Bars to the left of the red line may fit better in a different group."
                )

                # ── Summary metrics ───────────────────────────────────────
                champions  = (rfm["Segment"] == "🏆 Champion").sum()
                at_risk    = (rfm["Segment"] == "⚠️ At Risk").sum()
                lost       = (rfm["Segment"] == "💀 Lost").sum()
                churn_high = (rfm["Churn Risk"] == "🔴 High").sum()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🏆 Champions",       champions)
                c2.metric("⚠️ At Risk",          at_risk)
                c3.metric("💀 Lost",             lost)
                c4.metric("🔴 High Churn Risk",  churn_high)

                # ── Customer Lifetime Value summary ───────────────────────
                total_clv     = rfm["Est. Annual Value (12 months)"].sum()
                top3_clv      = rfm.nlargest(3, "Est. Annual Value (12 months)")["Est. Annual Value (12 months)"].sum()
                top3_pct      = (top3_clv / total_clv * 100) if total_clv > 0 else 0
                overdue_n     = rfm["Order Status"].str.startswith("⚠️").sum()

                cv1, cv2, cv3, cv4 = st.columns(4)
                cv1.metric("💰 Est. Annual Customer Value",
                           f"${total_clv:,.0f}",
                           help="Total estimated revenue from all customers over the next 12 months, based on each customer's order frequency and average order value.")
                cv2.metric("🎯 Top 3 Customers Share",
                           f"{top3_pct:.0f}% of total",
                           help="How much of your total projected revenue comes from just your top 3 customers. A high percentage means your business is heavily dependent on a small number of customers.")
                cv3.metric("⚠️ Overdue Orders",
                           overdue_n,
                           help="Customers who haven't ordered in longer than their usual gap. They may need a follow-up.")
                cv4.metric("📈 Growing Customers",
                           (rfm["Trend"] == "📈 Growing").sum(),
                           help="Customers whose spending in the second half of the date range was more than 10% higher than the first half.")

                # ── Revenue concentration warning ─────────────────────────
                if top3_pct > 50:
                    st.warning(
                        f"Your top 3 customers account for **{top3_pct:.0f}%** of projected revenue. "
                        "This is a high concentration risk. If any of these customers reduce orders, "
                        "it will have a significant impact on your business."
                    )

                # ── Segment summary table ─────────────────────────────────
                st.subheader("Segment Summary",
                             help="Each row is a customer group. Avg Recency shows how many days since their last order. Avg Orders is how many times they ordered on average. The Recommended Action column tells you what to do with each group.")
                summary = rfm.groupby("Segment").agg(
                    Customers       = ("Customer Name", "count"),
                    Avg_Recency     = ("Recency",       "mean"),
                    Avg_Orders      = ("Frequency",     "mean"),
                    Avg_Monetary    = ("Monetary",      "mean"),
                    Total_Monetary  = ("Monetary",      "sum"),
                    Avg_RFM_Score   = ("RFM_Score",     "mean"),
                ).round(1).reset_index()
                summary.columns = ["Segment","# Customers","Avg Recency (days)",
                                   "Avg Orders",f"Avg {monetary_label}",
                                   f"Total {monetary_label}","Avg RFM Score"]
                # Add recommended action to summary
                summary["Recommended Action"] = summary["Segment"].map(SEGMENT_ACTIONS).fillna("")
                st.dataframe(summary.sort_values("Avg RFM Score", ascending=False),
                             use_container_width=True, hide_index=True)

                # ── Scatter plots (emoji-free labels) ────────────────────
                seg_label_map = {s: s.split(" ", 1)[1] if " " in s else s for s in rfm["Segment"].unique()}
                rfm["_seg_plain"] = rfm["Segment"].map(seg_label_map)

                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                colors = plt.cm.Set2.colors
                for i, (seg_plain, grp) in enumerate(rfm.groupby("_seg_plain")):
                    axes[0].scatter(grp["Recency"], grp["Monetary"],
                                    label=seg_plain, alpha=0.7, s=50, color=colors[i % len(colors)])
                    axes[1].scatter(grp["Frequency"], grp["Monetary"],
                                    label=seg_plain, alpha=0.7, s=50, color=colors[i % len(colors)])
                axes[0].set_xlabel("Recency (days since last order)")
                axes[0].set_ylabel(monetary_label)
                axes[0].set_title("Recency vs Value")
                axes[1].set_xlabel("Order Frequency")
                axes[1].set_ylabel(monetary_label)
                axes[1].set_title("Frequency vs Value")
                axes[0].legend(fontsize=8)
                plt.tight_layout()
                st.pyplot(fig); plt.close(fig)

                # ── Top items per segment ─────────────────────────────────
                st.subheader("Top Items by Segment",
                             help="The most purchased items within each customer group. Use this to tailor what you stock and promote for each type of customer.")
                # Use df_trend which already has _rfm_monetary and avoids mutating df
                seg_item = df_trend.merge(rfm[["Customer Name","Segment"]], on="Customer Name", how="left")
                top_items = (seg_item.groupby(["Segment","Item Name"])["_rfm_monetary"]
                             .sum().reset_index()
                             .sort_values(["Segment","_rfm_monetary"], ascending=[True, False]))
                top_items = top_items.groupby("Segment").head(5).reset_index(drop=True)
                top_items.columns = ["Segment","Item Name", monetary_label]
                top_items[monetary_label] = top_items[monetary_label].round(1)
                st.dataframe(top_items, use_container_width=True, hide_index=True)

                # ── Customer detail table ─────────────────────────────────
                st.subheader("Customer Detail",
                             help="Every customer with their group, churn risk, trend, and recommended action. Sort any column to find the customers that need attention first.")
                detail_cols = ["Customer Name","Segment","Churn Risk","Order Status","Trend",
                               "Recommended Action","Recency","Avg Order Interval (days)",
                               "Est. Annual Value (12 months)","Frequency","Monetary",
                               "R_Score","F_Score","M_Score","RFM_Score"]
                detail_df = rfm[detail_cols].copy()
                detail_df = detail_df.rename(columns={"Monetary": monetary_label})
                detail_df[monetary_label] = detail_df[monetary_label].round(1)
                st.dataframe(
                    detail_df.sort_values("RFM_Score", ascending=False),
                    use_container_width=True, height=450, hide_index=True
                )
                st.download_button(
                    "⬇️ Download RFM table",
                    to_excel_bytes(detail_df),
                    "rfm_segments.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 – SLOW MOVERS / DEAD STOCK (Seasonal Adjustment + Changepoint Detection)
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.header("📉 Demand Health",
              help="Identifies items whose demand is declining or has stopped completely. Uses seasonal adjustment to remove normal seasonal patterns before flagging items, so you only see genuine declines. Changepoint detection pinpoints the exact month demand dropped.")

    # ── Stock file upload — provides Current Stock and Unit Price ─────────────
    has_stock_in_df5  = "Current Stock" in df.columns
    has_price_in_df5  = any(c in df.columns for c in ["Unit Price","Price"])

    if not has_stock_in_df5 or not has_price_in_df5:
        st.markdown("**Tip:** upload a stock levels file with `Item Name`, `Current Stock`, and `Unit Price` to unlock 'Months Until Zero' and 'Inventory Value' features. Upload is optional.")
    stock_file_slow = st.file_uploader("📂 Stock levels file", type=["xlsx","xls"], key="stock_slow")
    stock_df_slow = None
    if stock_file_slow:
        stock_df_slow = pd.read_excel(stock_file_slow)
        stock_df_slow.columns = stock_df_slow.columns.str.strip()
        st.success(f"✅ Stock file loaded: {len(stock_df_slow):,} items.")

    st.markdown("---")

    if "Invoice Date" not in df.columns:
        st.warning("Slow mover detection requires an `Invoice Date` column.")
    else:
        col_x, col_y = st.columns(2)
        decline_threshold = col_x.slider(
            "Decline threshold (%)",
            10, 90, 30,
            help="Flag an item if its most recent month's demand is below this percentage of its peak month. Lower = stricter (flags more items). 30% is a good starting point.")
        min_history = col_y.slider(
            "Minimum months of history",
            2, 6, 3,
            help="Only include items with at least this many months of sales data. Items with very short history are excluded to avoid false positives.")

        if st.button("🔍 Detect Slow Movers", type="primary"):
            df2 = df.copy()
            df2["Month"] = df2["Invoice Date"].dt.to_period("M")
            monthly_item = (df2.groupby(["Item Name","Month"])["Quantity"]
                            .sum().reset_index())
            monthly_item["Month_dt"] = monthly_item["Month"].dt.to_timestamp()

            # Check seasonal decomposition availability
            total_months = df2["Month"].nunique()
            seasonal_available = total_months >= 12

            if seasonal_available:
                st.success("Seasonal adjustment applied (12+ months of data detected).")
            else:
                st.info(f"Seasonal adjustment needs 12+ months of data (you have {total_months}). Using raw demand series.")

            # Check ruptures availability
            ruptures_available = False
            try:
                import ruptures as rpt
                ruptures_available = True
            except ImportError:
                st.caption("Changepoint detection not available. Install `ruptures` to enable it: `pip install ruptures==1.1.9`")

            # Unit price for inventory value calculation
            price_col_slow = next((c for c in ["Unit Price","Price"] if c in df.columns), None)
            price_map_slow = df.groupby("Item Name")[price_col_slow].mean().to_dict() if price_col_slow else {}
            # Override/supplement with stock file if uploaded
            if stock_df_slow is not None:
                price_col_sf = next((c for c in ["Unit Price","Price"] if c in stock_df_slow.columns), None)
                if price_col_sf:
                    sf_price = stock_df_slow.set_index("Item Name")[price_col_sf].to_dict()
                    price_map_slow = {**price_map_slow, **sf_price}  # stock file takes priority

            # Current stock for months-until-zero calculation
            has_stock_slow = "Current Stock" in df.columns
            stock_map_slow = df.groupby("Item Name")["Current Stock"].first().to_dict() if has_stock_slow else {}
            # Override/supplement with stock file if uploaded
            if stock_df_slow is not None and "Current Stock" in stock_df_slow.columns:
                sf_stock = stock_df_slow.set_index("Item Name")["Current Stock"].to_dict()
                stock_map_slow = {**stock_map_slow, **sf_stock}  # stock file takes priority

            # ABC class map from demand ranking
            abc_class_map = {}
            try:
                abc_ranked = build_abc_df(df, None)[["Item Name","ABC"]]
                abc_class_map = abc_ranked.set_index("Item Name")["ABC"].to_dict()
            except Exception:
                pass

            # Recommended action function — defined once outside the loop
            def get_slow_action(s, pct, muz, iv, abc):
                prefix = "High priority (Class A). " if abc == "A" and s in ("📉 Declining","💀 Dead Stock","😴 Stagnant") else ""
                if s == "💀 Dead Stock":
                    val = f" (${iv:,.0f} tied up)" if iv else ""
                    return f"{prefix}Consider clearance sale or write-off{val}. No demand recorded recently."
                elif s == "📉 Declining":
                    if muz is not None and muz < 3:
                        return f"{prefix}Stock runs out in ~{muz} months at current rate. Stop reordering and clear remaining stock."
                    return f"{prefix}Reduce reorder quantities. Investigate cause of decline before next order."
                elif s == "😴 Stagnant":
                    val = f" (${iv:,.0f} in stock)" if iv else ""
                    return f"{prefix}Demand has plateaued{val}. Consider a promotion to move stock or reduce safety stock."
                elif s == "🔄 Recovering":
                    return "Demand is recovering. Monitor closely before resuming normal reorder quantities."
                else:
                    return "No action needed."

            results = []
            for item, grp in monthly_item.groupby("Item Name"):
                grp = grp.sort_values("Month_dt").reset_index(drop=True)
                if len(grp) < min_history:
                    continue

                raw_series = grp["Quantity"].values.astype(float)

                # ── Seasonal adjustment ───────────────────────────────────
                if seasonal_available and len(grp) >= 12:
                    try:
                        from statsmodels.tsa.seasonal import seasonal_decompose
                        decomp = seasonal_decompose(raw_series, model="additive", period=12, extrapolate_trend="freq")
                        adjusted_series = decomp.trend + decomp.resid
                        adjusted_series = np.nan_to_num(adjusted_series, nan=np.nanmean(adjusted_series))
                    except Exception:
                        adjusted_series = raw_series
                else:
                    adjusted_series = raw_series

                peak        = adjusted_series.max()
                last_qty    = adjusted_series[-1]
                avg_qty     = adjusted_series.mean()
                pct_of_peak = last_qty / peak * 100 if peak > 0 else 0

                x     = np.arange(len(adjusted_series))
                slope = np.polyfit(x, adjusted_series, 1)[0]

                if pct_of_peak <= decline_threshold:
                    if last_qty <= 0:
                        status = "💀 Dead Stock"
                    elif slope < -0.5:
                        status = "📉 Declining"
                    else:
                        status = "😴 Stagnant"
                else:
                    status = "✅ Active"

                # ── Changepoint detection ─────────────────────────────────
                demand_shift_month = "N/A"
                if status != "✅ Active" and ruptures_available and len(adjusted_series) >= 4:
                    try:
                        signal = adjusted_series.reshape(-1, 1)
                        algo = rpt.Pelt(model="rbf").fit(signal)
                        change_points = algo.predict(pen=10)
                        if len(change_points) > 1:
                            cp_idx = change_points[-2] - 1
                            if 0 <= cp_idx < len(grp):
                                demand_shift_month = grp.iloc[cp_idx]["Month_dt"].strftime("%b %Y")
                    except Exception:
                        demand_shift_month = "N/A"

                # ── Months until zero stock ───────────────────────────────
                current_stock = stock_map_slow.get(item, None)
                if current_stock is not None and last_qty > 0:
                    months_until_zero = round(float(current_stock) / last_qty, 1)
                elif current_stock is not None and last_qty <= 0:
                    months_until_zero = 0.0
                else:
                    months_until_zero = None

                # ── Inventory value at risk ───────────────────────────────
                unit_price = price_map_slow.get(item, None)
                inv_value  = round(float(current_stock) * float(unit_price), 2) \
                             if current_stock is not None and unit_price is not None else None

                # ── Human-readable trend label ────────────────────────────
                slope_abs = abs(slope)
                if slope_abs < 0.5:
                    trend_label = "Flat"
                elif slope > 0:
                    trend_label = f"+{slope_abs:.1f} units/month"
                else:
                    trend_label = f"-{slope_abs:.1f} units/month"

                # ── Recovery detection ────────────────────────────────────
                if len(adjusted_series) >= 6:
                    recent_slope = np.polyfit(np.arange(3), adjusted_series[-3:], 1)[0]
                    if slope < -0.5 and recent_slope > 0.5:
                        status = "🔄 Recovering"

                # ── ABC class ─────────────────────────────────────────────
                abc_class = abc_class_map.get(item, "C")

                # ── Recommended action ────────────────────────────────────
                action = get_slow_action(status, pct_of_peak, months_until_zero, inv_value, abc_class)

                results.append({
                    "Item Name":            item,
                    "ABC":                  abc_class,
                    "Status":               status,
                    "Recommended Action":   action,
                    "Peak Qty/Month":       int(round(peak)),
                    "Last Month Qty":       int(round(last_qty)),
                    "Avg Qty/Month":        round(avg_qty, 1),
                    "% of Peak":            round(pct_of_peak, 1),
                    "Trend":                trend_label,
                    "Months Tracked":       len(grp),
                    "Demand Shift Month":   demand_shift_month,
                    "Current Stock":        current_stock,
                    "Months Until Zero":    months_until_zero,
                    "Inventory Value ($)":  inv_value,
                })

            slow_df = pd.DataFrame(results)
            # Save to session state so chart selectbox doesn't collapse the tab on rerun
            st.session_state["slow_df"]       = slow_df
            st.session_state["slow_monthly"]  = monthly_item
            if slow_df.empty:
                st.info("No items matched the criteria. Try lowering the decline threshold or minimum history.")
            else:
                dead     = (slow_df["Status"] == "💀 Dead Stock").sum()
                decline  = (slow_df["Status"] == "📉 Declining").sum()
                stagnant = (slow_df["Status"] == "😴 Stagnant").sum()
                active   = (slow_df["Status"] == "✅ Active").sum()
                recovering_n = (slow_df["Status"] == "🔄 Recovering").sum()

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("💀 Dead Stock",   dead)
                m2.metric("📉 Declining",    decline)
                m3.metric("😴 Stagnant",     stagnant)
                m4.metric("🔄 Recovering",   recovering_n)
                m5.metric("✅ Active",        active)

                # ── Status distribution chart (emoji-free labels) ─────────
                fig, ax = plt.subplots(figsize=(7, 3))
                status_label_map = {
                    "💀 Dead Stock": "Dead Stock",
                    "📉 Declining":  "Declining",
                    "😴 Stagnant":   "Stagnant",
                    "🔄 Recovering": "Recovering",
                    "✅ Active":     "Active",
                }
                status_counts  = slow_df["Status"].value_counts()
                plain_statuses = [status_label_map.get(s, s) for s in status_counts.index]
                colors_map     = {"Dead Stock": "#dc3545", "Declining": "#fd7e14",
                                  "Stagnant": "#ffc107", "Recovering": "#4a90d9",
                                  "Active": "#28a745"}
                bar_colors     = [colors_map.get(l, "gray") for l in plain_statuses]
                ax.barh(plain_statuses, status_counts.values, color=bar_colors, alpha=0.85)
                ax.set_xlabel("Number of Items")
                ax.set_title("Item Status Distribution", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig); plt.close(fig)

                # ── Items needing attention ───────────────────────────────
                problem_df = slow_df[
                    slow_df["Status"].isin(["💀 Dead Stock","📉 Declining","😴 Stagnant"])
                ].sort_values("% of Peak")
                st.subheader(f"Items Needing Attention ({len(problem_df)})",
                             help="Items flagged as Dead Stock, Declining, or Stagnant, sorted by how far they've fallen from their peak demand. Items at the top need the most urgent attention.")

                show_cols = ["Item Name","ABC","Status","Recommended Action","% of Peak",
                             "Last Month Qty","Avg Qty/Month","Trend","Months Until Zero",
                             "Inventory Value ($)","Demand Shift Month","Months Tracked"]
                show_cols = [c for c in show_cols if c in problem_df.columns]
                st.dataframe(problem_df[show_cols], use_container_width=True, height=400, hide_index=True)

                # ── Recovering items ──────────────────────────────────────
                if recovering_n > 0:
                    rec_df = slow_df[slow_df["Status"] == "🔄 Recovering"].sort_values("% of Peak", ascending=False)
                    st.subheader(f"Recovering Items ({recovering_n})",
                                 help="Items that were declining but whose demand has turned positive in the last 3 months. Worth monitoring as they may be returning to health.")
                    st.dataframe(rec_df[show_cols], use_container_width=True, height=250, hide_index=True)

                # ── All items ─────────────────────────────────────────────
                st.subheader("All Items",
                             help="Full list of all items sorted by % of peak demand. Active items are at the top, dead stock at the bottom.")
                all_cols = ["Item Name","ABC","Status","% of Peak","Last Month Qty",
                            "Avg Qty/Month","Trend","Months Tracked",
                            "Demand Shift Month","Current Stock","Months Until Zero"]
                all_cols = [c for c in all_cols if c in slow_df.columns]
                st.dataframe(slow_df[all_cols].sort_values("% of Peak", ascending=False),
                             use_container_width=True, height=300, hide_index=True)

                st.download_button(
                    "⬇️ Download slow mover report",
                    to_excel_bytes(slow_df),
                    "slow_movers.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        # ── Demand history chart — outside button block so selectbox doesn't collapse tab ──
        if "slow_df" in st.session_state:
            _slow_df      = st.session_state["slow_df"]
            _slow_monthly = st.session_state["slow_monthly"]

            st.subheader("Demand History Chart",
                         help="Select any item to see its full demand history. The red dotted line marks where demand started shifting (changepoint). The dashed trend line shows the overall direction.")
            chart_items = _slow_df["Item Name"].tolist()
            selected_chart_item = st.selectbox("Select item to view demand history",
                                               chart_items, key="slow_chart_item")
            if selected_chart_item:
                item_hist = _slow_monthly[_slow_monthly["Item Name"] == selected_chart_item].sort_values("Month_dt")
                item_row  = _slow_df[_slow_df["Item Name"] == selected_chart_item].iloc[0]

                # Strip emoji from status and trend for matplotlib title
                status_plain = item_row["Status"].split(" ", 1)[1] if " " in item_row["Status"] else item_row["Status"]
                trend_plain  = item_row.get("Trend", "")

                fig_h, ax_h = plt.subplots(figsize=(12, 4))
                ax_h.bar(range(len(item_hist)), item_hist["Quantity"].values,
                         color="#145a2e", alpha=0.7, label="Monthly Demand")

                # Trend line
                x_arr      = np.arange(len(item_hist))
                trend_line = np.polyval(np.polyfit(x_arr, item_hist["Quantity"].values, 1), x_arr)
                trend_color = "#28a745" if trend_line[-1] >= trend_line[0] else "#dc3545"
                ax_h.plot(x_arr, trend_line, color=trend_color, linewidth=2,
                          linestyle="--", label="Trend")

                # Changepoint marker
                cp_month = item_row.get("Demand Shift Month", "N/A")
                if cp_month != "N/A":
                    try:
                        cp_dt  = pd.to_datetime(cp_month, format="%b %Y")
                        cp_mask = item_hist["Month_dt"] <= cp_dt
                        if cp_mask.any():
                            cp_pos = int(cp_mask.sum()) - 1
                            ax_h.axvline(cp_pos, color="red", linewidth=1.5,
                                         linestyle=":", label=f"Demand shift: {cp_month}")
                    except Exception:
                        pass

                # X-axis labels
                labels = [d.strftime("%b %Y") for d in item_hist["Month_dt"]]
                tick_positions = list(range(0, len(labels), max(1, len(labels) // 12)))
                ax_h.set_xticks(tick_positions)
                ax_h.set_xticklabels([labels[i] for i in tick_positions],
                                     rotation=45, ha="right", fontsize=8)
                ax_h.set_ylabel("Quantity")
                ax_h.set_title(f"{selected_chart_item}  |  {status_plain}  |  {trend_plain}",
                               fontweight="bold")
                ax_h.legend(fontsize=9)
                plt.tight_layout()
                st.pyplot(fig_h); plt.close(fig_h)
