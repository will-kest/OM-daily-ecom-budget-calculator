import pandas as pd
import streamlit as st
from io import BytesIO
import calendar
import re
from typing import Optional, Dict, Any

# ============================================================
# Helpers
# ============================================================

def pick_col(df, candidates):
    """
    Try to find a column in df that matches one of the candidate names.

    Matching priority:
    1) Exact match (case-insensitive)
    2) Substring match (case-insensitive)
    3) Regex match if candidate starts with '^'
    """
    cols_lower = {c.lower(): c for c in df.columns}

    # 1) exact match
    for cand in candidates:
        if not cand.startswith('^'):
            for k, orig in cols_lower.items():
                if k == cand.lower():
                    return orig

    # 2) substring match
    for cand in candidates:
        if not cand.startswith('^'):
            for k, orig in cols_lower.items():
                if cand.lower() in k:
                    return orig

    # 3) regex match (candidates starting with ^)
    for cand in candidates:
        if cand.startswith('^'):
            pat = re.compile(cand, re.I)
            for orig in df.columns:
                if pat.search(orig):
                    return orig

    return None


def coerce_numeric(series):
    """
    Convert a column to numeric safely.

    What it does:
    - Converts to string
    - Removes non-numeric characters (commas, currency symbols, etc.)
    - Fixes cases with too many decimal points
    - Converts to float, invalid -> NaN
    """
    if series is None:
        return None

    s = series.astype(str)
    s = s.str.replace(r"[^0-9\.-]", "", regex=True)
    s = s.apply(lambda x: x if x.count('.') <= 1 else (x.replace('.', '', x.count('.') - 1)))

    out = pd.to_numeric(s, errors='coerce')
    return out


# ============================================================
# Allocation logic (customizable)
# ============================================================

def get_allocation(month: int, year: int, settings: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Create a daily allocation schedule as fractions that sum to ~1.0.

    Default rules (current behavior):
    - Day 1..(DD-1): 4% each
    - DD (double date day, e.g., Oct => day 10): 18%
    - DD+1: 4.5%
    - 24th: 3%
    - 25th: 8%
    - Remaining % split across:
        A) mid-month bucket: (DD+2)..23
        B) after payday bucket: 26..end
    With ratio mid=55%, after=45% by default.

    settings keys (percent units unless noted):
    - pct_day1_to_dd_minus1
    - pct_dd
    - pct_dd_plus1
    - pct_24
    - pct_25
    - mid_bucket_share  (0..1 portion of remaining_pct; after = 1 - mid)
    """
    default_settings = {
        "pct_day1_to_dd_minus1": 4.0,
        "pct_dd": 18.0,
        "pct_dd_plus1": 4.5,
        "pct_24": 3.0,
        "pct_25": 8.0,           # default 8%
        "mid_bucket_share": 0.55
    }

    if settings is None:
        settings = {}
    s = {**default_settings, **settings}

    days = calendar.monthrange(year, month)[1]
    dd = min(month, days)

    alloc_map = {d: None for d in range(1, days + 1)}

    # 1) Day 1..(DD-1)
    for d in range(1, dd):
        alloc_map[d] = float(s["pct_day1_to_dd_minus1"])

    # 2) DD
    alloc_map[dd] = float(s["pct_dd"])

    # 3) DD+1
    if dd + 1 <= days:
        alloc_map[dd + 1] = float(s["pct_dd_plus1"])

    # 4) 24th
    if 24 <= days:
        alloc_map[24] = float(s["pct_24"])

    # 5) 25th
    if 25 <= days:
        alloc_map[25] = float(s["pct_25"])

    # 6) Fill remaining days so total becomes 100%
    total_assigned = sum(v for v in alloc_map.values() if v is not None)
    remaining_days = [d for d, v in alloc_map.items() if v is None]

    mid_days = [d for d in remaining_days if (dd + 2) <= d <= 23]
    after_days = [d for d in remaining_days if d >= 26]

    if remaining_days:
        remaining_pct = 100.0 - total_assigned

        # If specials exceed 100%, impossible allocation (negative remaining)
        if remaining_pct < -1e-9:
            st.error(
                f"Allocation invalid: special-day total is {total_assigned:.2f}% (> 100%). "
                "Reduce special day percentages."
            )
            return pd.DataFrame()

        # If one bucket is empty, dump all remaining into the other
        if len(mid_days) == 0 and len(after_days) > 0:
            mid_share, after_share = 0.0, remaining_pct
        elif len(after_days) == 0 and len(mid_days) > 0:
            mid_share, after_share = remaining_pct, 0.0
        else:
            mid_ratio = float(s["mid_bucket_share"])
            mid_ratio = max(0.0, min(1.0, mid_ratio))  # clamp 0..1
            mid_share = remaining_pct * mid_ratio
            after_share = remaining_pct * (1.0 - mid_ratio)

        if len(mid_days) > 0:
            per_mid = mid_share / len(mid_days)
            for d in mid_days:
                alloc_map[d] = per_mid

        if len(after_days) > 0:
            per_after = after_share / len(after_days)
            for d in after_days:
                alloc_map[d] = per_after

        # Fix rounding residue by dumping into the last bucket day
        used = sum(v for v in alloc_map.values() if v is not None)
        leftover = 100.0 - used
        if abs(leftover) > 1e-9:
            if len(after_days) > 0:
                alloc_map[after_days[-1]] += leftover
            elif len(mid_days) > 0:
                alloc_map[mid_days[-1]] += leftover
            else:
                alloc_map[dd] += leftover

    df = pd.DataFrame({
        "day": range(1, days + 1),
        "percent": [alloc_map[d] / 100.0 for d in range(1, days + 1)]
    })

    total_frac = df["percent"].sum()
    if not (0.999 <= total_frac <= 1.001):
        st.warning(f"Daily allocation sums to {total_frac*100:.2f}% (expected ~100%).")

    return df


# ============================================================
# ROAS target + benchmark helpers
# ============================================================

def buffered_target_roas(roas_kpi):
    """
    Convert ROAS KPI -> Target ROAS (buffered / more ambitious).

    Logic:
    - KPI <= 3        : KPI + 2
    - 3 < KPI <= 10   : KPI * 1.2
    - 10 < KPI <= 25  : KPI * 1.3
    - KPI > 25        : KPI * 1.15
    """
    if roas_kpi is None or (isinstance(roas_kpi, float) and pd.isna(roas_kpi)):
        return None
    try:
        roas_kpi = float(roas_kpi)
    except Exception:
        return None

    if roas_kpi <= 3:
        return round(roas_kpi + 2)
    elif 3 < roas_kpi <= 10:
        return round(roas_kpi * 1.2)
    elif 10 < roas_kpi <= 25:
        return round(roas_kpi * 1.3)
    else:
        return round(roas_kpi * 1.15)


def compute_roas_benchmark(roas_kpi):
    """
    Turn a ROAS KPI into a benchmark range string.

    Rules:
    - KPI missing -> ''
    - KPI = 1 -> "1"
    - KPI = 3 -> "2-3"
    - else -> "{max(KPI-2, 0)}-{KPI}" rounded (never negative)
    """
    if roas_kpi is None or (isinstance(roas_kpi, float) and pd.isna(roas_kpi)):
        return ''
    try:
        rk = float(roas_kpi)
    except (TypeError, ValueError):
        return ''

    if abs(rk - 1) < 1e-9:
        return "1"
    if abs(rk - 3) < 1e-9:
        return "2-3"

    low = max(0, round(rk - 2))
    high = max(0, round(rk))
    # If KPI is very small, low==high, keep it clean
    if low == high:
        return f"{high}"
    return f"{low}-{high}"


# ============================================================
# Core transformation (raw monthly plan -> daily allocation output)
# ============================================================

def create_daily_allocation(
    df_raw: pd.DataFrame,
    month: int,
    year: int,
    include_objective: bool,
    alloc_settings: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Main function to transform raw monthly budget plan into daily budgets.

    Output:
    DataFrame with daily rows:
        Date, Retailer, Store, Brand, (Objective optional),
        Daily Budget, Target ROAS, ROAS KPI, ROAS Benchmark
    """
    allocation = get_allocation(month, year, settings=alloc_settings)
    if allocation.empty:
        return pd.DataFrame()

    # 1) Identify columns from raw file
    col_budget = pick_col(df_raw, ['Media Budget Plan', 'Budget', 'media_budget', 'media budget', '^.*budget.*$'])
    col_roas_plan = pick_col(df_raw, ['ROAS Plan', '^roas plan$', 'roas'])
    col_store = pick_col(df_raw, ['Store'])
    col_brand = pick_col(df_raw, ['Brand'])
    col_retailer = pick_col(df_raw, ['Retailer', 'Platform'])
    col_objective = pick_col(df_raw, ['Objective', 'Obj', '^.*objective.*$'])

    # Hard requirements: need budget + store
    if not all([col_budget, col_store]):
        st.error("Couldn't find required columns: Budget and Store.")
        return pd.DataFrame()

    # If user wants Objective but we can't find it
    if include_objective and not col_objective:
        st.warning("Objective toggle is ON, but no Objective column was found. Output will omit Objective.")
        include_objective = False

    # 2) Clean numeric fields + remove zero budgets
    df_raw[col_budget] = coerce_numeric(df_raw[col_budget])
    if col_roas_plan:
        df_raw[col_roas_plan] = coerce_numeric(df_raw[col_roas_plan])

    df_raw = df_raw[df_raw[col_budget].fillna(0) > 0]
    if df_raw.empty:
        st.warning("No rows with positive budget after cleaning.")
        return pd.DataFrame()

    # 3) Decide which stores keep sub-brand breakdown
    stores_with_subbrands = ['dancow', 'nos', 'bos']
    subbrand_mask = df_raw[col_store].astype(str).str.strip().str.lower().isin(stores_with_subbrands)

    group_brand = pd.Series('', index=df_raw.index, dtype='object')
    if col_brand:
        group_brand[subbrand_mask] = df_raw.loc[subbrand_mask, col_brand].astype(str).fillna('')

    retailer_series = df_raw[col_retailer] if col_retailer else pd.Series('Retailer', index=df_raw.index)

    if include_objective:
        objective_series = df_raw[col_objective].astype(str).fillna('') if col_objective else pd.Series('', index=df_raw.index)

    # 4) Group raw data to monthly totals
    agg_dict = {col_budget: 'sum'}
    if col_roas_plan:
        agg_dict[col_roas_plan] = 'median'

    tmp = df_raw.assign(__group_brand=group_brand, __retailer=retailer_series)

    group_keys = ['__retailer', col_store, '__group_brand']
    if include_objective:
        tmp = tmp.assign(__objective=objective_series)
        group_keys.append('__objective')

    grouped = tmp.groupby(group_keys, dropna=False).agg(agg_dict).reset_index()

    # 5) Add aggregate store rows for Dancow/NOS/BOS
    mask_multi = grouped[col_store].astype(str).str.strip().str.lower().isin(stores_with_subbrands)

    if mask_multi.any():
        agg_store_dict = {col_budget: 'sum'}
        if col_roas_plan:
            agg_store_dict[col_roas_plan] = 'median'

        # Aggregate across ALL objectives => objective NOT in keys
        agg_group_keys = ['__retailer', col_store]
        agg_store = grouped.loc[mask_multi].groupby(agg_group_keys, as_index=False).agg(agg_store_dict)

        agg_store['__group_brand'] = ''  # blank brand for aggregate row
        if include_objective:
            agg_store['__objective'] = ''  # force blank objective for aggregate row

        # Remove any existing blank-brand rows (avoid duplicates)
        to_remove = mask_multi & (grouped['__group_brand'].astype(str).str.strip() == '')
        grouped = grouped.loc[~to_remove]

        grouped = pd.concat([grouped, agg_store], ignore_index=True)

    # 6) Expand each monthly row into daily rows
    rows = []

    for _, g in grouped.iterrows():
        retailer = g['__retailer'] if col_retailer else 'Retailer'
        store = g[col_store]
        store_l = str(store).strip().lower()

        brand_out = g['__group_brand'] if store_l in stores_with_subbrands else ''

        objective_out = ''
        if include_objective:
            objective_out = g.get('__objective', '')

        monthly_budget = float(g[col_budget]) if pd.notna(g[col_budget]) else 0.0
        roas_kpi_val = float(g[col_roas_plan]) if (col_roas_plan and pd.notna(g[col_roas_plan])) else None

        is_blank_brand = (not g['__group_brand']) or (str(g['__group_brand']).strip() == '')

        # Special fixed rules for NOS and Dancow aggregate rows only
        if store_l == 'nos' and is_blank_brand:
            roas_kpi_val = 7
            target_roas_val = 20
        elif store_l == 'dancow' and is_blank_brand:
            roas_kpi_val = 15
            target_roas_val = 24
        else:
            target_roas_val = buffered_target_roas(roas_kpi_val)

        benchmark_val = compute_roas_benchmark(roas_kpi_val)

        for _, a in allocation.iterrows():
            date = pd.Timestamp(year=year, month=month, day=int(a['day']))
            daily_budget = round(monthly_budget * float(a['percent']))

            row = {
                'Date': date.strftime('%Y-%m-%d'),
                'Retailer': retailer,
                'Store': store,
                'Daily Budget': daily_budget,
                'Target ROAS': target_roas_val,
                'ROAS KPI': roas_kpi_val,
                'ROAS Benchmark': benchmark_val if benchmark_val is not None else '',
                'Brand': brand_out,
            }

            if include_objective:
                row['Objective'] = objective_out

            rows.append(row)

    out = pd.DataFrame(rows)
    return out


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title='Daily Allocation Budget Maker', layout='wide')
st.title('📊 Daily Allocation Budget Maker')

file = st.file_uploader('Upload the Budget Plan Excel file', type=['xlsx'])

# Controls
col1, col2 = st.columns([1, 1])
with col1:
    month = st.selectbox('Select Month', list(range(1, 13)), index=pd.Timestamp.now().month - 1)
with col2:
    year = st.number_input('Select Year', min_value=2020, max_value=2100, value=pd.Timestamp.now().year, step=1)

include_objective = st.toggle('Include Objective in output', value=False)

# Advanced allocation settings (override)
advanced_alloc = st.toggle("Advanced allocation settings", value=False)

alloc_settings: Dict[str, Any] = {}  # empty => defaults in get_allocation()

if advanced_alloc:
    st.subheader("Allocation Overrides")

    c1, c2, c3 = st.columns(3)
    with c1:
        pct_day1_to_dd_minus1 = st.number_input(
            "Day 1..(DD-1) % each", min_value=0.0, max_value=100.0, value=4.0, step=0.5
        )
        pct_dd = st.number_input(
            "DD %", min_value=0.0, max_value=100.0, value=18.0, step=0.5
        )
        pct_dd_plus1 = st.number_input(
            "DD+1 %", min_value=0.0, max_value=100.0, value=4.5, step=0.5
        )

    with c2:
        pct_24 = st.number_input(
            "24th %", min_value=0.0, max_value=100.0, value=3.0, step=0.5
        )
        pct_25 = st.number_input(
            "25th %", min_value=0.0, max_value=100.0, value=8.0, step=0.5
        )

    with c3:
        mid_bucket_share = st.slider(
            "Mid-month share of remaining (DD+2..23)", min_value=0, max_value=100, value=55, step=1
        )
        st.caption(f"After-payday share (26..end) will be {100 - mid_bucket_share}%")

    alloc_settings = {
        "pct_day1_to_dd_minus1": float(pct_day1_to_dd_minus1),
        "pct_dd": float(pct_dd),
        "pct_dd_plus1": float(pct_dd_plus1),
        "pct_24": float(pct_24),
        "pct_25": float(pct_25),
        "mid_bucket_share": float(mid_bucket_share) / 100.0,
    }

    # ---- Explain what overrides are doing (sum specials, remaining, bucket days) ----
    days_in_month = calendar.monthrange(int(year), int(month))[1]
    dd_day = min(int(month), days_in_month)

    # Count bucket days (based on the exact same logic as get_allocation)
    remaining_days = [d for d in range(1, days_in_month + 1)]
    # Mark special days as "assigned": day 1..dd-1, dd, dd+1, 24, 25
    assigned = set(range(1, dd_day))  # 1..dd-1
    assigned.add(dd_day)
    if dd_day + 1 <= days_in_month:
        assigned.add(dd_day + 1)
    if 24 <= days_in_month:
        assigned.add(24)
    if 25 <= days_in_month:
        assigned.add(25)

    remaining_days = [d for d in remaining_days if d not in assigned]
    mid_days = [d for d in remaining_days if (dd_day + 2) <= d <= 23]
    after_days = [d for d in remaining_days if d >= 26]

    # Sum specials exactly like they contribute (per-day * count + fixed days)
    special_total = (
        max(dd_day - 1, 0) * alloc_settings["pct_day1_to_dd_minus1"]
        + alloc_settings["pct_dd"]
        + (alloc_settings["pct_dd_plus1"] if (dd_day + 1) <= days_in_month else 0.0)
        + (alloc_settings["pct_24"] if 24 <= days_in_month else 0.0)
        + (alloc_settings["pct_25"] if 25 <= days_in_month else 0.0)
    )
    remaining_pct = 100.0 - special_total

    # How remaining will split (consider bucket empties)
    if remaining_pct < 0:
        mid_share_pct = 0.0
        after_share_pct = 0.0
    else:
        if len(mid_days) == 0 and len(after_days) > 0:
            mid_share_pct, after_share_pct = 0.0, remaining_pct
        elif len(after_days) == 0 and len(mid_days) > 0:
            mid_share_pct, after_share_pct = remaining_pct, 0.0
        elif len(mid_days) == 0 and len(after_days) == 0:
            mid_share_pct, after_share_pct = 0.0, 0.0
        else:
            mid_ratio = alloc_settings["mid_bucket_share"]
            mid_share_pct = remaining_pct * mid_ratio
            after_share_pct = remaining_pct * (1.0 - mid_ratio)

    st.markdown("### Allocation Summary (based on your overrides)")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Special days total", f"{special_total:.2f}%")
    with s2:
        st.metric("Remaining to distribute", f"{remaining_pct:.2f}%")
    with s3:
        st.metric("Mid bucket days (DD+2..23)", f"{len(mid_days)} day(s)")
    with s4:
        st.metric("After bucket days (26..end)", f"{len(after_days)} day(s)")

    st.caption(
        f"Remaining split: mid ≈ **{mid_share_pct:.2f}%** across **{len(mid_days)}** day(s), "
        f"after ≈ **{after_share_pct:.2f}%** across **{len(after_days)}** day(s)."
    )

    if special_total > 100.0:
        st.error("Special days total exceeds 100%. Reduce some special day percentages.")


if file:
    try:
        df_raw = pd.read_excel(file)
    except Exception as e:
        st.error(f"Failed to read Excel: {e}")
        st.stop()

    days = calendar.monthrange(int(year), int(month))[1]
    st.caption(f'Using **{days}-day and month-{month}** allocation pattern (month-aware). 🐔')

    df_out = create_daily_allocation(
        df_raw,
        int(month),
        int(year),
        include_objective=include_objective,
        alloc_settings=alloc_settings if advanced_alloc else None
    )

    if df_out.empty:
        st.info('No output produced. Please check your input columns and budgets.')
    else:
        df_out_preview = pd.concat([df_out.head(50), df_out.tail(50)])

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_out.to_excel(writer, index=False, sheet_name='Output')

        st.markdown(
            '''<style>
            div[data-testid="stDownloadButton"] button{
                background-color:#0072ff;
                color:white;
                font-weight:bold;
                border-radius:8px;
                padding:10px 20px 10px 14px;
                box-shadow:0 0 10px rgba(0,114,255,0.6);}
            div[data-testid="stDownloadButton"] button:hover{
                background-color:#0055cc;
                color:#fff;}
            </style>''',
            unsafe_allow_html=True
        )

        st.download_button(
            '🐤 Download Excel',
            data=buffer.getvalue(),
            file_name=f'daily_allocation_{int(year)}-{int(month):02d}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            type='secondary'
        )

        st.subheader('Preview')
        st.dataframe(df_out_preview, use_container_width=True)
