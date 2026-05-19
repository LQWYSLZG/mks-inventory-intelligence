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
        else:             label, detail = "\U0001f534 Unreliable",    f"A naive guess outperforms the model (MASE: {mase}). This item has highly variable demand \u2014 use the forecast as a rough guide only."
        st.info(f"\U0001f4ca Forecast accuracy: **{label}** \u2014 {detail}")
        st.caption(f"Model: {model_label} \u00b7 Note: MAPE not shown \u2014 percentage errors are misleading at low average volume ({np.mean(np.array(actual,dtype=float)):.1f} units/month).")
    elif mape is not None:
        if   mape <= 15: label, detail = "\U0001f7e2 Excellent fit",    f"Average error of {mape}% per month \u2014 forecasts are highly reliable."
        elif mape <= 30: label, detail = "\U0001f7e1 Good fit",         f"Average error of {mape}% per month \u2014 forecasts are reasonably reliable."
        elif mape <= 50: label, detail = "\U0001f7e0 Moderate fit",     f"Average error of {mape}% per month \u2014 demand is variable. Use forecasts as a directional guide."
        else:            label, detail = "\U0001f534 High uncertainty", (
            f"Average error of {mape}% per month. "
            "This item has highly erratic demand that is difficult to predict accurately. "
            "Use the forecast as a rough order-of-magnitude estimate and apply extra safety stock to compensate."
        )
        mase_note = f" \u00b7 MASE: {mase} ({'better' if mase is not None and mase < 1 else 'worse'} than naive)" if mase is not None else ""
        st.info(f"\U0001f4ca Forecast accuracy: **{label}** \u2014 {detail}")
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
    st.markdown("---")
    st.markdown("**Required Basic Columns**<br>**for Demand History Files:**", unsafe_allow_html=True)
    st.markdown("- `Item Name`\n- `Quantity`\n- `Invoice Date`\n- `Customer Name`")
    st.markdown("---")
    st.markdown("<small style='color:#7ab898;'>Battalion Bikes · MKS SPORTS INDUSTRIES<br>Feedback Contact: lqwyslzg@gmail.com</small>", unsafe_allow_html=True)

st.markdown("""
<div class="mks-header">
    <h1>🏸 Inventory Intelligence for MKS SPORTS INDUSTRIES</h1>
    <p>Badminton Equipment Manufacturing · Demand Analytics Dashboard</p>
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
            report.append(("✅ Fixed", f"Trimmed whitespace from Item Name — reduced unique SKUs from {before} to {after}"))

    # 2. Convert Quantity to numeric
    if "Quantity" in df.columns:
        non_numeric = pd.to_numeric(df["Quantity"], errors="coerce").isna().sum()
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
        if non_numeric > 0:
            report.append(("✅ Fixed", f"Converted Quantity to numeric — {non_numeric} non-numeric value(s) set to NaN"))

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
            report.append(("⚠️ Warning", f"{bad_dates} row(s) have unparseable Invoice Date — these rows are excluded from date-based analysis"))

    # 7. Flag duplicate rows (after dedup in load_data, any remaining are soft dupes)
    if "Invoice Number" in df.columns:
        dupes = df.duplicated(subset=["Invoice Number","Item Name","Quantity"], keep=False).sum()
        if dupes > 0:
            report.append(("⚠️ Warning", f"{dupes} row(s) appear to be duplicate invoice lines — review your source data"))

    # 8. All good message
    rows_removed = original_len - len(df)
    if rows_removed > 0:
        report.append(("📊 Summary", f"Dataset reduced from {original_len:,} to {len(df):,} rows after cleaning ({rows_removed:,} removed)"))
    else:
        report.append(("📊 Summary", f"No issues found — all {original_len:,} rows are clean"))

    return df, report

df, quality_report = clean_data(df)

missing = {"Item Name","Quantity"} - set(df.columns)
if missing:
    st.error(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 ABC Inventory Analysis",
    "📈 Demand Forecast",
    "🔔 Reorder & Stockout Alerts",
    "👥 Customer Segments",
    "📉 Slow Movers",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – ABC RANKING + XYZ ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("ABC Inventory Analysis")

    with st.expander("🔍 Dataset Overview", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Total rows", f"{len(df):,}")
        c2.metric("Unique items", f"{df['Item Name'].nunique():,}")
        c3.metric("Total quantity", f"{df['Quantity'].sum():,}")
        if "Invoice Date" in df.columns:
            date_min = df['Invoice Date'].min().strftime('%d %b %Y')
            date_max = df['Invoice Date'].max().strftime('%d %b %Y')
            st.markdown(f"**📅 Date range:** {date_min} — {date_max}")

        # ── Data Quality Report ───────────────────────────────────────────
        st.markdown("**🧹 Data Quality Report**")
        for status, message in quality_report:
            if status == "✅ Fixed":
                st.success(f"{status} — {message}")
            elif status == "⚠️ Warning":
                st.warning(f"{status} — {message}")
            else:
                st.info(f"{status} — {message}")

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
    if has_revenue:
        abc_mode = ctrl1.radio("Classify by", ["📋 Demand", f"💰 Revenue"],
                               horizontal=True, key="abc_mode")
        if "Revenue" in abc_mode:
            value_col   = revenue_col
            value_label = revenue_col
    else:
        ctrl1.caption("⚙️ Currently classifying by Demand. Add a Unit Price or Revenue column for the dataset to enable revenue-based analysis.")

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
            st.subheader(f"Pareto Chart – Top {actual_n} {'All' if cat is None else cat} Items")
            st.caption("The Pareto Chart ranks items from highest to lowest demand (bars), while the line shows the running cumulative percentage of total demand. The dashed lines mark the 80% and 95% thresholds used to classify items into ABC classes.")
            fig, _ = pareto_chart(abc_df, f"{'Overall' if cat is None else cat} Demand Pareto",
                                  BAR_COLORS.get(cat, palette[0]), top_n=top_n, value_col=value_label)
            st.pyplot(fig); plt.close(fig)

            # ── ABC-XYZ matrix (All Items tab only) ──────────────────────────
            if cat is None and not xyz_df_abc.empty:
                st.subheader("ABC-XYZ Matrix")
                st.caption("The ABC-XYZ Matrix combines two classifications: ABC ranks items by their total demand contribution (A = top 80%, B = next 15%, C = bottom 5%), while XYZ measures how stable or erratic that demand is (X = stable, Y = variable, Z = highly irregular). The result is a 9-cell grid below that tells you both how important an item is and how predictable it is.")

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
                st.subheader("ABC-XYZ Action Guide")
                st.caption("Recommended stocking and review strategy for each classification cell.")
                rec_cols = st.columns(3)
                for idx, (cell, (badge, rec)) in enumerate(ABC_XYZ_RECOMMENDATIONS.items()):
                    count = int(matrix_data.loc[cell[0], cell[1]]) if cell[0] in matrix_data.index and cell[1] in matrix_data.columns else 0
                    rec_cols[idx % 3].markdown(
                        f"**{cell}** {badge} — *{count} SKUs*  \n{rec}"
                    )
                st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

                st.subheader("Full ABC-XYZ Ranking Table")
                st.caption("Items are ranked by total demand quantity (highest to lowest), which determines their ABC class. The XYZ class and CV (Coefficient of Variation) are additional attributes showing demand stability and do not affect the ranking order.")
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
                st.subheader("Full Ranking Table")
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
    st.header("📈 Demand Forecast")
    st.caption("Select an item and how many months ahead you want to predict. We will automatically pick the most suitable forecasting method and model based on your data (no manual setup needed).")

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
                    "The shaded area shows the **uncertainty range** \u2014 "
                    "demand is expected to fall within this band most of the time. "
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
                    monthly_ds = pd.date_range(start=monthly["ds"].min(), periods=len(series_vals), freq="MS")
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
                                    future_only[["Forecast Qty","Lower Bound","Upper Bound"]] =                                         future_only[["Forecast Qty","Lower Bound","Upper Bound"]].round(0).astype(int)
                                    future_only["Forecast Qty"] = future_only["Forecast Qty"].clip(lower=0)
                                    future_only["Lower Bound"]  = future_only["Lower Bound"].clip(lower=0)
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
                                    future_only[["Forecast Qty","Lower Bound","Upper Bound"]] =                                         future_only[["Forecast Qty","Lower Bound","Upper Bound"]].round(0).astype(int)
                                    future_only["Forecast Qty"] = future_only["Forecast Qty"].clip(lower=0)
                                    future_only["Lower Bound"]  = future_only["Lower Bound"].clip(lower=0)
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
                                            "**TBATS** was used instead — it handles complex patterns better."
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
                            st.subheader("Forecast Summary")
                            st.caption(
                                "**Forecast Qty** is the model’s best estimate. "
                                "**Lower / Upper Bound** show the range demand is likely to fall within. "
                                "Plan your stock based on the Upper Bound if you want to avoid stockouts."
                            )
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
    st.header("🔔 Reorder Point")
    st.caption(
        "Calculates the reorder point (ROP) for each item based on your demand history and lead time. "
        "Upload a stock levels file to see which items need restocking now."
    )

    if "Invoice Date" not in df.columns:
        st.warning("Reorder alerts require an `Invoice Date` column.")
    else:
        n_months_data = df["Invoice Date"].dt.to_period("M").nunique()

        # ── Settings ──────────────────────────────────────────────────────
        with st.expander("⚙️ Settings", expanded=True):
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
                safety_days   = c2.number_input("Simple safety stock (days) — fallback if insufficient data",
                                                min_value=0, max_value=60, value=7)
                z_map = {"90%": 1.28, "95%": 1.65, "99%": 2.33}
                z_val = z_map[service_level]
                st.caption(f"Statistical Safety Stock = Z({z_val}) × σ_demand × √(Lead Time) — Service level: {service_level}")
            else:
                safety_days = st.number_input("Safety stock (days of cover)", min_value=0, max_value=60, value=7)
                z_val = 1.65  # default, not used in simple mode

        # ── Stock file upload ─────────────────────────────────────────────
        st.markdown("**Optional:** upload a stock levels file with `Item Name`, `Current Stock`, and optionally `Lead Time (days)` for per-item lead times.")
        stock_file = st.file_uploader("📂 Stock levels file", type=["xlsx","xls"], key="stock")

        stock_df = None
        if stock_file:
            stock_df = pd.read_excel(stock_file)
            stock_df.columns = stock_df.columns.str.strip()
            st.success(f"✅ Stock file loaded — {len(stock_df):,} items.")

        if st.button("🔔 Calculate Reorder Points", type="primary"):

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

            # Avg daily demand per item
            avg_daily = (df_demand.groupby("Item Name")["Quantity"].sum() / window_days).reset_index()
            avg_daily.columns = ["Item Name", "Avg Daily Demand"]

            # Monthly demand std for statistical safety stock (use full history for stability)
            df_monthly = df.dropna(subset=["Invoice Date"]).copy()
            df_monthly["Month"] = df_monthly["Invoice Date"].dt.to_period("M")
            monthly_item = df_monthly.groupby(["Item Name","Month"])["Quantity"].sum().reset_index()
            monthly_std = monthly_item.groupby("Item Name")["Quantity"].std(ddof=1).reset_index()
            monthly_std.columns = ["Item Name","Monthly Std"]
            # Correct conversion: daily std = monthly std / sqrt(30)
            monthly_std["Daily Std"] = monthly_std["Monthly Std"] / np.sqrt(30)

            # Build ROP table
            rop_df = avg_daily.copy()
            rop_df = rop_df.merge(monthly_std[["Item Name","Daily Std"]], on="Item Name", how="left")
            rop_df["Daily Std"] = rop_df["Daily Std"].fillna(0)

            # Merge per-item lead time if provided in stock file
            if stock_df is not None and "Lead Time (days)" in stock_df.columns:
                lt_map = stock_df.set_index("Item Name")["Lead Time (days)"].to_dict()
                rop_df["Lead Time"] = rop_df["Item Name"].map(lt_map).fillna(lead_time).astype(float)
            else:
                rop_df["Lead Time"] = float(lead_time)

            # Safety stock calculation
            rop_df["Simple Safety Stock"] = (rop_df["Avg Daily Demand"] * safety_days).round(1)
            rop_df["Stat Safety Stock"]   = (z_val * rop_df["Daily Std"] * np.sqrt(rop_df["Lead Time"])).round(1)

            if stat_mode:
                # Use statistical where we have enough data, fall back to simple otherwise
                rop_df["Safety Stock"] = rop_df.apply(
                    lambda r: r["Stat Safety Stock"] if r["Daily Std"] > 0 else r["Simple Safety Stock"], axis=1
                )
                rop_df["Safety Stock Method"] = rop_df["Daily Std"].apply(
                    lambda x: f"Statistical ({service_level})" if x > 0 else "Simple (fallback)"
                )
            else:
                rop_df["Safety Stock"] = rop_df["Simple Safety Stock"]
                rop_df["Safety Stock Method"] = "Simple"

            rop_df["ROP (units)"]        = (rop_df["Avg Daily Demand"] * rop_df["Lead Time"] + rop_df["Safety Stock"]).round(1)
            rop_df["Monthly Demand"]     = (rop_df["Avg Daily Demand"] * 30).round(0).astype(int)
            rop_df["Suggested Order Qty"] = (rop_df["Avg Daily Demand"] * replenish_days).round(0).astype(int)

            # Merge stock levels
            if stock_df is not None and "Item Name" in stock_df.columns and "Current Stock" in stock_df.columns:
                rop_df = rop_df.merge(stock_df[["Item Name","Current Stock"]], on="Item Name", how="left")
                rop_df["Current Stock"] = pd.to_numeric(rop_df["Current Stock"], errors="coerce").fillna(0)

                # Days of stock remaining
                rop_df["Days of Stock"] = (rop_df["Current Stock"] / rop_df["Avg Daily Demand"].replace(0, np.nan)).round(0)
                rop_df["Days of Stock"] = rop_df["Days of Stock"].fillna(0).astype(int)

                # Status
                def get_status(r):
                    if r["Current Stock"] <= r["ROP (units)"]:
                        return "🔴 Reorder Now"
                    elif r["Current Stock"] <= r["ROP (units)"] * 1.5:
                        return "🟡 Low Stock"
                    else:
                        return "🟢 OK"
                rop_df["Status"] = rop_df.apply(get_status, axis=1)

                # Summary metrics
                reorder_n = (rop_df["Status"] == "🔴 Reorder Now").sum()
                low_n     = (rop_df["Status"] == "🟡 Low Stock").sum()
                ok_n      = (rop_df["Status"] == "🟢 OK").sum()
                no_stock  = (rop_df["Current Stock"] == 0).sum()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("🔴 Reorder Now", reorder_n)
                m2.metric("🟡 Low Stock",   low_n)
                m3.metric("🟢 OK",          ok_n)
                m4.metric("⬛ Zero Stock",   no_stock)

                # Status bar chart
                fig_s, ax_s = plt.subplots(figsize=(7, 3))
                status_counts = rop_df["Status"].value_counts()
                sc_colors = {"🔴 Reorder Now": "#dc3545", "🟡 Low Stock": "#ffc107", "🟢 OK": "#28a745"}
                ax_s.barh(status_counts.index, status_counts.values,
                          color=[sc_colors.get(s, "gray") for s in status_counts.index], alpha=0.85)
                ax_s.set_xlabel("Number of Items")
                ax_s.set_title("Reorder Status Distribution", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig_s); plt.close(fig_s)

                # Show alert items first, then the rest
                alert_df = rop_df[rop_df["Status"] != "🟢 OK"].sort_values("Days of Stock")
                ok_df    = rop_df[rop_df["Status"] == "🟢 OK"].sort_values("Days of Stock")
                rop_display = pd.concat([alert_df, ok_df], ignore_index=True)

            else:
                rop_df["Current Stock"] = "—"
                rop_df["Days of Stock"] = "—"
                rop_df["Status"] = "Upload stock file to see alerts"
                rop_display = rop_df.sort_values("Monthly Demand", ascending=False)

            rop_df["Avg Daily Demand"] = rop_df["Avg Daily Demand"].round(3)

            # Display columns
            display_cols = ["Item Name","Status","Current Stock","Days of Stock",
                            "ROP (units)","Safety Stock","Safety Stock Method",
                            "Suggested Order Qty","Monthly Demand",
                            "Avg Daily Demand","Lead Time"]
            display_cols = [c for c in display_cols if c in rop_display.columns]

            st.subheader(f"Reorder Report — {demand_window}")
            st.caption(
                "Items are sorted by urgency — **🔴 Reorder Now** and **🟡 Low Stock** appear first. "
                "**Days of Stock** shows how many days of supply remain at current demand rate. "
                "**Suggested Order Qty** covers the configured replenishment period."
            )
            st.dataframe(rop_display[display_cols], use_container_width=True, height=500, hide_index=True)
            st.download_button(
                "⬇️ Download reorder report",
                to_excel_bytes(rop_display[display_cols]),
                "reorder_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    st.markdown("---")
    st.header("🔔 Stockout Risk")
    st.markdown("""
    Classifies each SKU as **🔴 High / 🟡 Medium / 🟢 Low** stockout risk using
    features derived entirely from your uploaded demand history.

    > **How it works:** Features like demand variability, trend slope, ABC/XYZ class,
    > and zero-demand frequency are extracted per SKU. A rule-based heuristic generates
    > training labels (self-supervised), then an **XGBoost classifier** learns the
    > pattern and predicts risk probabilities for all SKUs.
    > This is a *pattern-recognition* exercise — not a held-out test — so the model
    > is trained and predicted on the same dataset to surface structural risk signals.
    """)

    if "Invoice Date" not in df.columns:
        st.warning("Stockout Risk requires an `Invoice Date` column.")
    elif st.button("🚨 Run Stockout Risk Analysis", type="primary"):
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

                    feature_rows.append({
                        "Item Name":           item,
                        "avg_monthly_demand":  round(mean_q, 2),
                        "demand_cv":           round(cv, 3),
                        "demand_trend_slope":  round(slope, 3),
                        "months_of_data":      n,
                        "zero_demand_pct":     round(zero_pct, 3),
                        "peak_to_avg_ratio":   round(peak_to_avg, 3),
                    })

                feat_df = pd.DataFrame(feature_rows)

                # Merge ABC class
                abc_all = build_abc_df(df, None)[["Item Name","ABC"]]
                feat_df = feat_df.merge(abc_all, on="Item Name", how="left")
                feat_df["ABC"] = feat_df["ABC"].fillna("C")
                feat_df["abc_class"] = feat_df["ABC"].map({"A": 3, "B": 2, "C": 1}).fillna(1).astype(int)

                # Merge XYZ class
                xyz_all = build_xyz_series(df)[["Item Name","XYZ"]]
                feat_df = feat_df.merge(xyz_all, on="Item Name", how="left")
                feat_df["XYZ"] = feat_df["XYZ"].fillna("Z")
                feat_df["xyz_class"] = feat_df["XYZ"].map({"X": 1, "Y": 2, "Z": 3}).fillna(3).astype(int)

                # ── Rule-based label generation (self-supervised) ─────────
                def rule_label(row):
                    if (row["abc_class"] == 3 and
                            (row["demand_cv"] > 0.8 or
                             row["zero_demand_pct"] > 0.2 or
                             row["demand_trend_slope"] < -1)):
                        return 2  # High
                    elif (row["abc_class"] == 1 or
                          (row["demand_cv"] < 0.3 and row["demand_trend_slope"] > 0)):
                        return 0  # Low
                    else:
                        return 1  # Medium

                feat_df["risk_label"] = feat_df.apply(rule_label, axis=1)

                FEATURE_COLS = ["avg_monthly_demand","demand_cv","demand_trend_slope",
                                "abc_class","xyz_class","months_of_data",
                                "zero_demand_pct","peak_to_avg_ratio"]
                X_feat = feat_df[FEATURE_COLS].values
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

                    # ── Feature importance chart ──────────────────────────
                    st.subheader("Feature Importance")
                    importances = clf.feature_importances_
                    fi_df = pd.DataFrame({
                        "Feature": FEATURE_COLS,
                        "Importance": importances
                    }).sort_values("Importance", ascending=True)

                    fig_fi, ax_fi = plt.subplots(figsize=(8, 4))
                    ax_fi.barh(fi_df["Feature"], fi_df["Importance"], color="#145a2e", alpha=0.85)
                    ax_fi.set_xlabel("Importance Score")
                    ax_fi.set_title("XGBoost Feature Importance", fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig_fi); plt.close(fig_fi)

                    model_note = "XGBoost classifier (self-supervised labels)"
                else:
                    feat_df["Risk"] = [label_names[l] for l in y_labels]
                    feat_df["P(Low)"]    = feat_df["risk_label"].apply(lambda x: 100.0 if x == 0 else 0.0)
                    feat_df["P(Medium)"] = feat_df["risk_label"].apply(lambda x: 100.0 if x == 1 else 0.0)
                    feat_df["P(High)"]   = feat_df["risk_label"].apply(lambda x: 100.0 if x == 2 else 0.0)
                    model_note = "Rule-based labels (XGBoost not available)"

                # ── Risk summary metrics ──────────────────────────────────
                high_n   = (feat_df["Risk"] == "🔴 High").sum()
                med_n    = (feat_df["Risk"] == "🟡 Medium").sum()
                low_n    = (feat_df["Risk"] == "🟢 Low").sum()

                m1, m2, m3 = st.columns(3)
                m1.metric("🔴 High Risk",   high_n)
                m2.metric("🟡 Medium Risk", med_n)
                m3.metric("🟢 Low Risk",    low_n)

                st.caption(f"Model: {model_note}")

                # ── Risk distribution chart ───────────────────────────────
                fig_r, ax_r = plt.subplots(figsize=(6, 3))
                risk_counts = feat_df["Risk"].value_counts()
                risk_colors = {"🔴 High": "#dc3545", "🟡 Medium": "#ffc107", "🟢 Low": "#28a745"}
                bar_c = [risk_colors.get(r, "gray") for r in risk_counts.index]
                ax_r.bar(risk_counts.index, risk_counts.values, color=bar_c, alpha=0.85)
                ax_r.set_ylabel("SKU Count")
                ax_r.set_title("Stockout Risk Distribution", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig_r); plt.close(fig_r)

                # ── Per-SKU risk table ────────────────────────────────────
                st.subheader("Per-SKU Risk Table")
                display_risk_cols = ["Item Name","Risk","P(High)","P(Medium)","P(Low)",
                                     "ABC","XYZ","avg_monthly_demand","demand_cv",
                                     "demand_trend_slope","zero_demand_pct","months_of_data"]
                display_risk_cols = [c for c in display_risk_cols if c in feat_df.columns]

                risk_display = feat_df[display_risk_cols].sort_values(
                    "P(High)", ascending=False
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

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – CUSTOMER SEGMENTATION (RFM + Quintile Scoring + Adaptive Clustering)
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("👥 Customer Segmentation (RFM)")
    st.markdown("""
    Segments customers by **Recency** (how recently they ordered),
    **Frequency** (how often), and **Monetary** (total quantity ordered).
    Includes **RFM Quintile Scoring** (1–5 per dimension) and **adaptive clustering**
    that auto-selects K-Means or GMM based on silhouette score.
    """)

    required_rfm = {"Customer Name","Invoice Date","Quantity"}
    if not required_rfm.issubset(df.columns):
        st.warning(f"RFM requires columns: {required_rfm}. Missing: {required_rfm - set(df.columns)}")
    else:
        n_clusters = st.slider("Number of segments", 2, 5, 3)

        if st.button("🔍 Run Segmentation", type="primary"):
            with st.spinner("Segmenting customers…"):
                snapshot = df["Invoice Date"].max()
                rfm = df.groupby("Customer Name").agg(
                    Recency   = ("Invoice Date",  lambda x: (snapshot - x.max()).days),
                    Frequency = ("Invoice Number" if "Invoice Number" in df.columns else "Invoice Date",
                                 "nunique"),
                    Monetary  = ("Quantity", "sum")
                ).reset_index()

                # ── RFM Quintile Scoring ──────────────────────────────────
                try:
                    # Recency: lower days = higher score (reverse)
                    rfm["R_Score"] = pd.qcut(rfm["Recency"].rank(method="first"),
                                             q=5, labels=[5,4,3,2,1]).astype(int)
                    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"),
                                             q=5, labels=[1,2,3,4,5]).astype(int)
                    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"),
                                             q=5, labels=[1,2,3,4,5]).astype(int)
                except Exception:
                    # Fallback if not enough distinct values for quintiles
                    rfm["R_Score"] = 3
                    rfm["F_Score"] = 3
                    rfm["M_Score"] = 3

                rfm["RFM_Score"] = (
                    (rfm["R_Score"] * 0.3 + rfm["F_Score"] * 0.3 + rfm["M_Score"] * 0.4)
                    * (20 / 3)
                ).round(1)

                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score

                scaler = StandardScaler()
                X = scaler.fit_transform(rfm[["Recency","Frequency","Monetary"]])

                # ── Adaptive clustering: K-Means vs GMM ──────────────────
                km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                km_labels = km.fit_predict(X)
                km_sil = silhouette_score(X, km_labels) if n_clusters > 1 else 0.0

                gmm_sil = 0.0
                gmm_labels = None
                try:
                    from sklearn.mixture import GaussianMixture
                    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42)
                    gmm_labels = gmm.fit_predict(X)
                    gmm_sil = silhouette_score(X, gmm_labels) if n_clusters > 1 else 0.0
                except Exception:
                    pass

                if gmm_labels is not None and gmm_sil > km_sil:
                    rfm["Cluster"] = gmm_labels
                    cluster_method = f"GMM (silhouette: {gmm_sil:.2f})"
                else:
                    rfm["Cluster"] = km_labels
                    cluster_method = f"K-Means (silhouette: {km_sil:.2f})"

                st.success(f"🤖 Using **{cluster_method}**")

                # Label clusters by average Monetary (high → low)
                cluster_order = rfm.groupby("Cluster")["Monetary"].mean().sort_values(ascending=False)
                labels = ["🥇 High Value","🥈 Mid Value","🥉 Low Value",
                          "⚪ Segment 4","⚪ Segment 5"]
                label_map = {c: labels[i] for i, c in enumerate(cluster_order.index)}
                rfm["Segment"] = rfm["Cluster"].map(label_map)

                # Summary
                summary = rfm.groupby("Segment").agg(
                    Customers   = ("Customer Name","count"),
                    Avg_Recency = ("Recency","mean"),
                    Avg_Freq    = ("Frequency","mean"),
                    Avg_Qty     = ("Monetary","mean"),
                    Total_Qty   = ("Monetary","sum"),
                    Avg_RFM_Score = ("RFM_Score","mean"),
                ).round(1).reset_index()
                summary.columns = ["Segment","# Customers","Avg Recency (days)",
                                   "Avg Orders","Avg Qty","Total Qty","Avg RFM Score"]

                st.subheader("Segment Summary")
                st.dataframe(summary, use_container_width=True, hide_index=True)

                # Scatter plot
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                colors = plt.cm.Set2.colors
                for i, (seg, grp) in enumerate(rfm.groupby("Segment")):
                    axes[0].scatter(grp["Recency"], grp["Monetary"],
                                    label=seg, alpha=0.7, s=40, color=colors[i % len(colors)])
                    axes[1].scatter(grp["Frequency"], grp["Monetary"],
                                    label=seg, alpha=0.7, s=40, color=colors[i % len(colors)])
                axes[0].set_xlabel("Recency (days since last order)")
                axes[0].set_ylabel("Total Quantity")
                axes[0].set_title("Recency vs Quantity")
                axes[1].set_xlabel("Order Frequency")
                axes[1].set_ylabel("Total Quantity")
                axes[1].set_title("Frequency vs Quantity")
                axes[0].legend(fontsize=8)
                plt.tight_layout()
                st.pyplot(fig); plt.close(fig)

                st.subheader("Customer Detail (with RFM Scores)")
                detail_cols = ["Customer Name","Segment","Recency","Frequency","Monetary",
                               "R_Score","F_Score","M_Score","RFM_Score"]
                st.dataframe(
                    rfm[detail_cols].sort_values("Monetary", ascending=False),
                    use_container_width=True, height=400, hide_index=True
                )
                st.download_button(
                    "⬇️ Download RFM table",
                    to_excel_bytes(rfm),
                    "rfm_segments.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 – SLOW MOVERS / DEAD STOCK (Seasonal Adjustment + Changepoint Detection)
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.header("📉 Slow Mover & Dead Stock Detection")
    st.markdown("""
    Identifies items whose demand is **declining** or **stagnant** month-over-month.
    Applies **seasonal decomposition** (12+ months) and **changepoint detection** via `ruptures`.
    """)

    if "Invoice Date" not in df.columns:
        st.warning("Slow mover detection requires an `Invoice Date` column.")
    else:
        col_x, col_y = st.columns(2)
        decline_threshold = col_x.slider(
            "Decline threshold (%): flag if last month < X% of peak month",
            10, 90, 30)
        min_history = col_y.slider("Minimum months of history to include", 2, 6, 3)

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
                st.success("✅ Seasonal adjustment applied (12+ months of data detected).")
            else:
                st.info(f"ℹ️ Insufficient data for seasonal adjustment (need 12+ months, have {total_months}). Using raw series.")

            # Check ruptures availability
            ruptures_available = False
            try:
                import ruptures as rpt
                ruptures_available = True
            except ImportError:
                st.warning("⚠️ `ruptures` library not found. Changepoint detection will show N/A. Install with: `pip install ruptures==1.1.9`")

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

                peak     = adjusted_series.max()
                last_qty = adjusted_series[-1]
                avg_qty  = adjusted_series.mean()
                pct_of_peak = last_qty / peak * 100 if peak > 0 else 0

                x = np.arange(len(adjusted_series))
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
                        # change_points is list of end indices; last is len(series)
                        if len(change_points) > 1:
                            cp_idx = change_points[-2] - 1  # last real changepoint
                            if 0 <= cp_idx < len(grp):
                                demand_shift_month = grp.iloc[cp_idx]["Month_dt"].strftime("%b %Y")
                    except Exception:
                        demand_shift_month = "N/A"

                results.append({
                    "Item Name":          item,
                    "Status":             status,
                    "Peak Qty/Month":     int(round(peak)),
                    "Last Month Qty":     int(round(last_qty)),
                    "Avg Qty/Month":      round(avg_qty, 1),
                    "% of Peak":          round(pct_of_peak, 1),
                    "Trend Slope":        round(slope, 2),
                    "Months Tracked":     len(grp),
                    "Demand Shift Month": demand_shift_month,
                })

            slow_df = pd.DataFrame(results)
            if slow_df.empty:
                st.info("No items matched the criteria.")
            else:
                dead    = (slow_df["Status"] == "💀 Dead Stock").sum()
                decline = (slow_df["Status"] == "📉 Declining").sum()
                stagnant= (slow_df["Status"] == "😴 Stagnant").sum()
                active  = (slow_df["Status"] == "✅ Active").sum()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("💀 Dead Stock",  dead)
                m2.metric("📉 Declining",   decline)
                m3.metric("😴 Stagnant",    stagnant)
                m4.metric("✅ Active",       active)

                fig, ax = plt.subplots(figsize=(7, 3))
                status_counts = slow_df["Status"].value_counts()
                colors_map = {"💀 Dead Stock":"#dc3545","📉 Declining":"#fd7e14",
                              "😴 Stagnant":"#ffc107","✅ Active":"#28a745"}
                bar_colors = [colors_map.get(s,"gray") for s in status_counts.index]
                ax.barh(status_counts.index, status_counts.values, color=bar_colors)
                ax.set_xlabel("Number of Items")
                ax.set_title("Item Status Distribution", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig); plt.close(fig)

                problem_df = slow_df[slow_df["Status"] != "✅ Active"].sort_values("% of Peak")
                st.subheader(f"⚠️ Items Needing Attention ({len(problem_df)})")
                st.dataframe(problem_df, use_container_width=True, height=400, hide_index=True)

                st.subheader("All Items")
                st.dataframe(slow_df.sort_values("% of Peak"),
                             use_container_width=True, height=300, hide_index=True)

                st.download_button(
                    "⬇️ Download slow mover report",
                    to_excel_bytes(slow_df),
                    "slow_movers.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
