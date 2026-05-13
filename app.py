import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import io

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
    n = name.upper()
    if "GRIP" in n:
        return "GRIP"
    elif "UNSTRUNG" in n:
        return "UNSTRUNG"
    return "Others"

def abc_label(pct: float) -> str:
    if pct <= 80:   return "A"
    elif pct <= 95: return "B"
    return "C"

def build_abc_df(df: pd.DataFrame, item_type: str | None = None) -> pd.DataFrame:
    subset = df if item_type is None else df[df["Item Type"] == item_type]
    ranked = (subset.groupby("Item Name")["Quantity"]
              .sum().sort_values(ascending=False).reset_index())
    ranked.columns = ["Item Name", "Quantity"]
    ranked["Rank"] = range(1, len(ranked) + 1)
    ranked["Cumulative Qty"] = ranked["Quantity"].cumsum()
    total = ranked["Quantity"].sum()
    ranked["Cumulative %"] = (ranked["Cumulative Qty"] / total * 100).round(2)
    ranked["ABC"] = ranked["Cumulative %"].apply(abc_label)
    return ranked

def pareto_chart(abc_df, title, bar_color):
    top = abc_df.head(20)
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(top["Item Name"], top["Quantity"], color=bar_color, alpha=0.85, label="Quantity")
    ax1.set_ylabel("Quantity", fontsize=11)
    ax1.tick_params(axis="x", rotation=75, labelsize=7)
    ax2 = ax1.twinx()
    ax2.plot(top["Item Name"], top["Cumulative %"], color="crimson",
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
    return fig

def abc_summary_metrics(abc_df):
    counts = abc_df["ABC"].value_counts().reindex(["A","B","C"], fill_value=0)
    total_qty = abc_df["Quantity"].sum()
    cols = st.columns(3)
    info = {"A": ("🟢","top 80%"), "B": ("🟡","80–95%"), "C": ("🔴","tail 5%")}
    for col, cat in zip(cols, ["A","B","C"]):
        qty = abc_df[abc_df["ABC"] == cat]["Quantity"].sum()
        icon, rng = info[cat]
        col.metric(f"{icon} Class {cat} ({rng})",
                   f"{counts[cat]} SKUs",
                   f"{qty:,} units ({qty/total_qty*100:.1f}%)",
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

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR – file upload
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏸 MKS SPORTS INDUSTRIES")
    st.markdown("**Inventory Intelligence Platform**")
    st.markdown("---")
    st.markdown("Upload your demand history Excel file to start and unlock all features.")
    uploaded = st.file_uploader("📂 Upload Excel file", type=["xlsx","xls"])
    st.markdown("---")
    st.markdown("**Required Columns**<br>**for Demand History Files:**", unsafe_allow_html=True)
    st.markdown("- `Item Name`\n- `Quantity`\n- `Invoice Date`\n- `Customer Name`")
    st.markdown("---")
    st.markdown("<small style='color:#7ab898;'>Battalion Bikes · MKS SPORTS INDUSTRIES<br>Feedback Contact: lqwyslzg@gmail.com</small>", unsafe_allow_html=True)

st.markdown("""
<div class="mks-header">
    <h1>🏸 Inventory Intelligence for MKS SPORTS INDUSTRIES</h1>
    <p>Badminton Equipment Manufacturing · Demand Analytics Dashboard</p>
</div>
""", unsafe_allow_html=True)

if uploaded is None:
    st.markdown("""
    <div style="background:#edf7f1; border:1px solid #c8e6d0; border-left:4px solid #f0c040;
                border-radius:8px; padding:20px 24px; margin-top:10px;">
        <h4 style="color:#0a2e14; margin:0 0 8px 0;">👈 Get Started</h4>
        <p style="color:#1a5c30; margin:0;">Upload your demand history Excel file from the sidebar to begin analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load & validate ───────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    df.columns = df.columns.str.strip()
    if "Invoice Date" in df.columns:
        df["Invoice Date"] = pd.to_datetime(df["Invoice Date"], errors="coerce")
    if "Item Name" in df.columns:
        df["Item Type"] = df["Item Name"].apply(categorize_item)
    return df

df = load_data(uploaded)

missing = {"Item Name","Quantity"} - set(df.columns)
if missing:
    st.error(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 ABC Ranking",
    "📈 Demand Forecast",
    "🔔 Reorder Alerts",
    "👥 Customer Segments",
    "📉 Slow Movers",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – ABC RANKING
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("ABC Analysis & Demand Ranking")

    with st.expander("🔍 Dataset overview", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total rows", f"{len(df):,}")
        c2.metric("Unique items", f"{df['Item Name'].nunique():,}")
        c3.metric("Total quantity", f"{df['Quantity'].sum():,}")
        if "Invoice Date" in df.columns:
            c4.metric("Date range",
                      f"{df['Invoice Date'].min().strftime('%b %Y')} – "
                      f"{df['Invoice Date'].max().strftime('%b %Y')}")
        st.dataframe(df.head(10), use_container_width=True)

    CATEGORIES = {
        "🏸 All Items":  None,
        "✋ Grip":       "GRIP",
        "🎾 Unstrung":  "UNSTRUNG",
        "📦 Others":    "Others",
    }
    BAR_COLORS = {None:"#145a2e","GRIP":"#2e9e5e","UNSTRUNG":"#f0c040","Others":"#7ab898"}

    cat_tabs = st.tabs(list(CATEGORIES.keys()))
    for ctab, (label, cat) in zip(cat_tabs, CATEGORIES.items()):
        with ctab:
            abc_df = build_abc_df(df, cat)
            if abc_df.empty:
                st.warning(f"No items found for {label}.")
                continue
            if cat is not None:
                abc_summary_metrics(abc_df)
            st.subheader(f"Pareto Chart – Top 20 {'All' if cat is None else cat} Items")
            fig = pareto_chart(abc_df, f"{'Overall' if cat is None else cat} Demand Pareto", BAR_COLORS[cat])
            st.pyplot(fig); plt.close(fig)
            st.subheader("Full Ranking Table")
            cols_show = ["Rank","Item Name","Quantity","Cumulative %"]
            if cat is not None:
                cols_show.append("ABC")
                styled = abc_df[cols_show].style.apply(highlight_abc, axis=1)
                st.dataframe(styled, use_container_width=True, height=400)
            else:
                st.dataframe(abc_df[cols_show], use_container_width=True, height=400)
            if cat is not None:
                st.download_button(
                    f"⬇️ Download {cat} ABC table",
                    to_excel_bytes(abc_df),
                    f"abc_{cat.lower()}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – DEMAND FORECAST
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("📈 Demand Forecasting")
    st.markdown("Select an item and forecast horizon. The model uses your historical monthly sales to predict future demand.")

    if "Invoice Date" not in df.columns:
        st.warning("Demand forecasting requires an `Invoice Date` column.")
    else:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            all_items = sorted(df["Item Name"].unique().tolist())
            selected_item = st.selectbox("Select Item / SKU", all_items)
        with col_b:
            forecast_months = st.slider("Forecast horizon (months)", 1, 6, 3)

        if st.button("🔮 Run Forecast", type="primary"):
            item_df = df[df["Item Name"] == selected_item].copy()
            monthly = (item_df.groupby(item_df["Invoice Date"].dt.to_period("M"))["Quantity"]
                       .sum().reset_index())
            monthly["Invoice Date"] = monthly["Invoice Date"].dt.to_timestamp()
            monthly.columns = ["ds", "y"]

            if len(monthly) < 3:
                st.warning("Not enough history (need at least 3 months) to forecast this item.")
            else:
                with st.spinner("Training forecast model…"):
                    try:
                        from prophet import Prophet
                        m = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                                    daily_seasonality=False, seasonality_mode="additive")
                        m.fit(monthly)
                        future = m.make_future_dataframe(periods=forecast_months, freq="MS")
                        forecast = m.predict(future)

                        fig, ax = plt.subplots(figsize=(12, 5))
                        ax.fill_between(forecast["ds"], forecast["yhat_lower"],
                                        forecast["yhat_upper"], alpha=0.2, color="#145a2e", label="Confidence band")
                        ax.plot(forecast["ds"], forecast["yhat"], color="#145a2e",
                                linewidth=2, label="Forecast")
                        ax.scatter(monthly["ds"], monthly["y"], color="#f0c040",
                                   zorder=5, s=40, label="Actual")
                        # shade forecast region
                        cutoff = monthly["ds"].max()
                        ax.axvline(cutoff, color="gray", linestyle="--", linewidth=1)
                        ax.set_ylabel("Quantity")
                        ax.set_title(f"Demand Forecast – {selected_item}", fontweight="bold")
                        ax.legend()
                        plt.tight_layout()
                        st.pyplot(fig); plt.close(fig)

                        # Forecast table
                        future_only = forecast[forecast["ds"] > cutoff][["ds","yhat","yhat_lower","yhat_upper"]].copy()
                        future_only.columns = ["Month","Forecast Qty","Lower Bound","Upper Bound"]
                        future_only["Month"] = future_only["Month"].dt.strftime("%b %Y")
                        future_only[["Forecast Qty","Lower Bound","Upper Bound"]] = \
                            future_only[["Forecast Qty","Lower Bound","Upper Bound"]].round(0).astype(int)
                        future_only["Forecast Qty"] = future_only["Forecast Qty"].clip(lower=0)
                        future_only["Lower Bound"]  = future_only["Lower Bound"].clip(lower=0)

                        st.subheader("Forecast Summary")
                        st.dataframe(future_only, use_container_width=True, hide_index=True)
                        st.download_button("⬇️ Download forecast", to_excel_bytes(future_only),
                                           "forecast.xlsx",
                                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    except Exception as e:
                        st.error(f"Forecast failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – REORDER ALERTS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("🔔 Reorder Point Alerts")
    st.markdown("""
    Enter your current stock levels and lead time.
    The app calculates the **reorder point** (ROP) for each item and flags which ones need restocking.

    > **ROP = Average Daily Demand × Lead Time Days + Safety Stock**
    """)

    if "Invoice Date" not in df.columns:
        st.warning("Reorder alerts require an `Invoice Date` column.")
    else:
        col1, col2, col3 = st.columns(3)
        lead_time   = col1.number_input("Lead time (days)", min_value=1, max_value=90, value=14)
        safety_days = col2.number_input("Safety stock (days of cover)", min_value=0, max_value=60, value=7)
        top_n       = col3.number_input("Show top N items by demand", min_value=10, max_value=500, value=50)

        st.markdown("---")
        st.markdown("**Optional: paste your current stock levels**")
        st.markdown("Upload a two-column Excel with `Item Name` and `Current Stock`.")
        stock_file = st.file_uploader("📂 Stock levels file (optional)", type=["xlsx","xls"], key="stock")

        stock_df = None
        if stock_file:
            stock_df = pd.read_excel(stock_file)
            stock_df.columns = stock_df.columns.str.strip()

        if st.button("🔔 Calculate Reorder Points", type="primary"):
            date_range_days = (df["Invoice Date"].max() - df["Invoice Date"].min()).days or 1
            avg_daily = (df.groupby("Item Name")["Quantity"].sum() / date_range_days).reset_index()
            avg_daily.columns = ["Item Name","Avg Daily Demand"]

            rop_df = avg_daily.copy()
            rop_df["ROP (units)"] = ((rop_df["Avg Daily Demand"] * lead_time) +
                                     (rop_df["Avg Daily Demand"] * safety_days)).round(1)
            rop_df["Monthly Forecast"] = (rop_df["Avg Daily Demand"] * 30).round(0).astype(int)
            rop_df = rop_df.sort_values("Monthly Forecast", ascending=False).head(int(top_n))

            if stock_df is not None and "Item Name" in stock_df.columns and "Current Stock" in stock_df.columns:
                rop_df = rop_df.merge(stock_df[["Item Name","Current Stock"]], on="Item Name", how="left")
                rop_df["Current Stock"] = rop_df["Current Stock"].fillna(0)
                rop_df["Status"] = rop_df.apply(
                    lambda r: "🔴 REORDER NOW" if r["Current Stock"] <= r["ROP (units)"]
                    else ("🟡 LOW STOCK" if r["Current Stock"] <= r["ROP (units)"] * 1.5
                    else "🟢 OK"), axis=1)
                alert_count = (rop_df["Status"] == "🔴 REORDER NOW").sum()
                low_count   = (rop_df["Status"] == "🟡 LOW STOCK").sum()
                m1, m2, m3 = st.columns(3)
                m1.metric("🔴 Reorder Now", alert_count)
                m2.metric("🟡 Low Stock",   low_count)
                m3.metric("🟢 OK",          len(rop_df) - alert_count - low_count)
            else:
                rop_df["Current Stock"] = "—"
                rop_df["Status"] = "Upload stock file to see alerts"

            rop_df["Avg Daily Demand"] = rop_df["Avg Daily Demand"].round(2)
            st.dataframe(rop_df, use_container_width=True, height=450, hide_index=True)
            st.download_button("⬇️ Download reorder report", to_excel_bytes(rop_df),
                               "reorder_report.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – CUSTOMER SEGMENTATION (RFM)
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("👥 Customer Segmentation (RFM)")
    st.markdown("""
    Segments customers by **Recency** (how recently they ordered),
    **Frequency** (how often), and **Monetary** (total quantity ordered).
    K-Means clustering groups them into **High / Medium / Low** value tiers.
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

                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans

                scaler = StandardScaler()
                X = scaler.fit_transform(rfm[["Recency","Frequency","Monetary"]])

                km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                rfm["Cluster"] = km.fit_predict(X)

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
                ).round(1).reset_index()
                summary.columns = ["Segment","# Customers","Avg Recency (days)","Avg Orders","Avg Qty","Total Qty"]

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

                st.subheader("Customer Detail")
                st.dataframe(rfm[["Customer Name","Segment","Recency","Frequency","Monetary"]]
                             .sort_values("Monetary", ascending=False),
                             use_container_width=True, height=400, hide_index=True)
                st.download_button("⬇️ Download RFM table", to_excel_bytes(rfm),
                                   "rfm_segments.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 – SLOW MOVERS / DEAD STOCK
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.header("📉 Slow Mover & Dead Stock Detection")
    st.markdown("""
    Identifies items whose demand is **declining** or **stagnant** month-over-month.
    Useful for deciding which SKUs to discount, bundle, or stop reordering.
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

            results = []
            for item, grp in monthly_item.groupby("Item Name"):
                grp = grp.sort_values("Month_dt")
                if len(grp) < min_history:
                    continue
                peak     = grp["Quantity"].max()
                last_qty = grp.iloc[-1]["Quantity"]
                first_qty= grp.iloc[0]["Quantity"]
                avg_qty  = grp["Quantity"].mean()
                pct_of_peak = last_qty / peak * 100 if peak > 0 else 0

                # Trend: simple slope via numpy polyfit
                x = np.arange(len(grp))
                slope = np.polyfit(x, grp["Quantity"].values, 1)[0]

                if pct_of_peak <= decline_threshold:
                    if last_qty == 0:
                        status = "💀 Dead Stock"
                    elif slope < -0.5:
                        status = "📉 Declining"
                    else:
                        status = "😴 Stagnant"
                else:
                    status = "✅ Active"

                results.append({
                    "Item Name":      item,
                    "Status":         status,
                    "Peak Qty/Month": int(peak),
                    "Last Month Qty": int(last_qty),
                    "Avg Qty/Month":  round(avg_qty, 1),
                    "% of Peak":      round(pct_of_peak, 1),
                    "Trend Slope":    round(slope, 2),
                    "Months Tracked": len(grp),
                })

            slow_df = pd.DataFrame(results)
            if slow_df.empty:
                st.info("No items matched the criteria.")
            else:
                # Summary metrics
                dead    = (slow_df["Status"] == "💀 Dead Stock").sum()
                decline = (slow_df["Status"] == "📉 Declining").sum()
                stagnant= (slow_df["Status"] == "😴 Stagnant").sum()
                active  = (slow_df["Status"] == "✅ Active").sum()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("💀 Dead Stock",  dead)
                m2.metric("📉 Declining",   decline)
                m3.metric("😴 Stagnant",    stagnant)
                m4.metric("✅ Active",       active)

                # Bar chart of status distribution
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

                # Filter to problem items only
                problem_df = slow_df[slow_df["Status"] != "✅ Active"].sort_values("% of Peak")
                st.subheader(f"⚠️ Items Needing Attention ({len(problem_df)})")
                st.dataframe(problem_df, use_container_width=True, height=400, hide_index=True)

                st.subheader("All Items")
                st.dataframe(slow_df.sort_values("% of Peak"),
                             use_container_width=True, height=300, hide_index=True)

                st.download_button("⬇️ Download slow mover report",
                                   to_excel_bytes(slow_df),
                                   "slow_movers.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
