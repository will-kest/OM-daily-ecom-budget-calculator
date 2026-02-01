import pandas as pd
import streamlit as st
from io import BytesIO
import calendar
import re

# ============================================================
# Helpers
# ============================================================
# This section contains small utility functions used throughout the app.
# The goal is to keep the main logic (grouping + output generation) clean
# and easy to read.

def pick_col(df, candidates):
    """
    Try to find a column in df that matches one of the candidate names.

    Matching priority:
    1) Exact match (case-insensitive)
    2) Substring match (case-insensitive)
    3) Regex match if candidate starts with '^'

    Why this exists:
    - Your raw files sometimes have slightly different column naming
    - This makes the code more resilient without hard-coding one name
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

    Why:
    - Your raw file often contains "30,000,000" or "Rp 30.000.000"
    - pandas won't treat that as numeric without cleaning
    """
    if series is None:
        return None

    s = series.astype(str)
    # Keep only digits, minus sign, dot
    s = s.str.replace(r"[^0-9\.-]", "", regex=True)
    # If multiple '.' exist, remove extras (keep the last structure consistent)
    s = s.apply(lambda x: x if x.count('.') <= 1 else (x.replace('.', '', x.count('.') - 1)))

    out = pd.to_numeric(s, errors='coerce')
    return out


# ============================================================
# Allocation logic
# ============================================================
# This section determines the % allocation for each day of the month.
# Output: DataFrame of day -> percent (fraction, sum ~ 1.0)

def get_allocation(month: int, year: int) -> pd.DataFrame:
    """
    Create a daily allocation schedule as fractions that sum to ~1.0.

    Rules in your current logic:
    - Day 1..(DD-1): 4% each
    - DD (double date day, e.g., Oct => day 10): 18%
    - DD+1: 4.5%
    - 24th: 3%
    - 25th: 8%
    - Remaining % split equally across:
        A) mid-month bucket: (DD+2)..23
        B) after payday bucket: 26..end
        (50/50 split unless one bucket is empty)

    Note:
    - dd = month number (Feb => 2, Mar => 3, etc.)
    - capped by month length for safety
    """
    days = calendar.monthrange(year, month)[1]
    dd = min(month, days)

    # alloc_map[d] = percent for day d (in percent units, e.g. 4.0 means 4%)
    alloc_map = {d: None for d in range(1, days + 1)}

    # 1) Day 1..(DD-1)
    for d in range(1, dd):
        alloc_map[d] = 4.0

    # 2) DD
    alloc_map[dd] = 18.0

    # 3) DD+1
    if dd + 1 <= days:
        alloc_map[dd + 1] = 4.5

    # 4) 24th
    if 24 <= days:
        alloc_map[24] = 3.0

    # 5) 25th
    if 25 <= days:
        alloc_map[25] = 8.0

    # 6) Fill remaining days so total becomes 100%
    total_assigned = sum(v for v in alloc_map.values() if v is not None)
    remaining_days = [d for d, v in alloc_map.items() if v is None]

    # Bucket definitions
    mid_days = [d for d in remaining_days if (dd + 2) <= d <= 23]
    after_days = [d for d in remaining_days if d >= 26]

    if remaining_days:
        remaining_pct = 100.0 - total_assigned

        # Avoid division by zero
        if len(mid_days) == 0 and len(after_days) > 0:
            mid_share, after_share = 0.0, remaining_pct
        elif len(after_days) == 0 and len(mid_days) > 0:
            mid_share, after_share = remaining_pct, 0.0
        else:
            mid_share = remaining_pct * 0.55
            after_share = remaining_pct * 0.45

        # Spread within each bucket evenly
        if len(mid_days) > 0:
            per_mid = mid_share / len(mid_days)
            for d in mid_days:
                alloc_map[d] = per_mid

        if len(after_days) > 0:
            per_after = after_share / len(after_days)
            for d in after_days:
                alloc_map[d] = per_after

        # Fix float rounding residue by dumping it into the last bucket day
        used = sum(v for v in alloc_map.values() if v is not None)
        leftover = 100.0 - used
        if abs(leftover) > 1e-9:
            if len(after_days) > 0:
                alloc_map[after_days[-1]] += leftover
            elif len(mid_days) > 0:
                alloc_map[mid_days[-1]] += leftover

    # Convert % to fraction (0.04 instead of 4.0)
    percent_list = [alloc_map[d] / 100.0 for d in range(1, days + 1)]
    df = pd.DataFrame({"day": range(1, days + 1), "percent": percent_list})

    # Safety warning if sum is off due to rounding
    s = df["percent"].sum()
    if not (0.999 <= s <= 1.001):
        st.warning(f"Daily allocation sums to {s*100:.2f}% (expected ~100%).")

    return df


# ============================================================
# ROAS target + benchmark helpers
# ============================================================
# These functions take ROAS KPI values and compute:
#   - Target ROAS: KPI with buffer logic
#   - Benchmark label: e.g. "5-7"

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
    except:
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
    - else -> "{KPI-2}-{KPI}" rounded

    Output is string because you want it displayed as label like "5-7".
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

    return f"{round(rk - 2)}-{round(rk)}"


# ============================================================
# Core transformation (raw monthly plan -> daily allocation output)
# ============================================================

def create_daily_allocation(df_raw, month, year, include_objective: bool):
    """
    Main function to transform raw monthly budget plan into daily budgets.

    Inputs:
    - df_raw: raw Excel sheet loaded into DataFrame
    - month/year: chosen from UI (controls daily allocation dates)
    - include_objective:
        If True: output includes Objective column AND grouping includes Objective,
                EXCEPT for aggregate rows of BOS/NOS/Dancow where Objective is forced blank.

    Output:
    DataFrame with daily rows:
        Date, Retailer, Store, Brand, (Objective optional), Daily Budget, Target ROAS, ROAS KPI, ROAS Benchmark
    """
    allocation = get_allocation(month, year)

    # ----------------------------
    # 1) Identify columns from raw file
    # ----------------------------
    col_budget = pick_col(df_raw, ['Media Budget Plan', 'Budget', 'media_budget', 'media budget', '^.*budget.*$'])
    col_roas_plan = pick_col(df_raw, ['ROAS Plan', '^roas plan$', 'roas'])
    col_store = pick_col(df_raw, ['Store'])
    col_brand = pick_col(df_raw, ['Brand'])
    col_retailer = pick_col(df_raw, ['Retailer', 'Platform'])
    col_objective = pick_col(df_raw, ['Objective', 'Obj', '^.*objective.*$'])

    # Hard requirements: need budget + store to proceed
    if not all([col_budget, col_store]):
        st.error("Couldn't find required columns: Budget and Store.")
        return pd.DataFrame()

    # If user wants Objective but we can't find it, disable it cleanly
    if include_objective and not col_objective:
        st.warning("Objective toggle is ON, but no Objective column was found. Output will omit Objective.")
        include_objective = False

    # ----------------------------
    # 2) Clean numeric fields + remove zero budgets
    # ----------------------------
    df_raw[col_budget] = coerce_numeric(df_raw[col_budget])
    if col_roas_plan:
        df_raw[col_roas_plan] = coerce_numeric(df_raw[col_roas_plan])

    # Keep only rows with positive budget
    df_raw = df_raw[df_raw[col_budget].fillna(0) > 0]
    if df_raw.empty:
        st.warning("No rows with positive budget after cleaning.")
        return pd.DataFrame()

    # ----------------------------
    # 3) Decide which stores should keep sub-brand breakdown
    # ----------------------------
    # Only these stores will have Brand kept in grouping/output.
    # All other stores will have Brand output as blank.
    stores_with_subbrands = ['dancow', 'nos', 'bos']

    # True for rows whose Store is in the list above
    subbrand_mask = df_raw[col_store].astype(str).str.strip().str.lower().isin(stores_with_subbrands)

    # group_brand becomes a helper column used for grouping
    # - for Dancow/NOS/BOS: keep Brand
    # - for other stores: set '' so they don't split by brand
    group_brand = pd.Series('', index=df_raw.index, dtype='object')
    if col_brand:
        group_brand[subbrand_mask] = df_raw.loc[subbrand_mask, col_brand].astype(str).fillna('')

    # Retailer is taken from "Platform" (or "Retailer" if present)
    retailer_series = df_raw[col_retailer] if col_retailer else pd.Series('Retailer', index=df_raw.index)

    # Objective series only needed if toggle is ON
    if include_objective:
        objective_series = df_raw[col_objective].astype(str).fillna('') if col_objective else pd.Series('', index=df_raw.index)

    # ----------------------------
    # 4) Group raw data to monthly totals
    # ----------------------------
    # We group to get one monthly budget per unique combination.
    # If include_objective=True, objective is part of the group *for normal rows*.
    agg_dict = {col_budget: 'sum'}
    if col_roas_plan:
        agg_dict[col_roas_plan] = 'median'

    tmp = df_raw.assign(__group_brand=group_brand, __retailer=retailer_series)

    # group keys always include retailer + store + brand-group
    group_keys = ['__retailer', col_store, '__group_brand']

    # optionally include objective as a grouping key
    if include_objective:
        tmp = tmp.assign(__objective=objective_series)
        group_keys.append('__objective')

    grouped = tmp.groupby(group_keys, dropna=False).agg(agg_dict).reset_index()

    # ----------------------------
    # 5) Add aggregate store rows for Dancow/NOS/BOS
    # ----------------------------
    # Requirement:
    #   For Dancow/NOS/BOS, add an "aggregate row" with blank Brand that sums across:
    #     - all sub-brands
    #     - all objectives (Retention/Recruitment/etc.)
    #   and if Objective toggle is ON, the aggregate row Objective must be BLANK.
    mask_multi = grouped[col_store].astype(str).str.strip().str.lower().isin(stores_with_subbrands)

    if mask_multi.any():
        agg_store_dict = {col_budget: 'sum'}
        if col_roas_plan:
            agg_store_dict[col_roas_plan] = 'median'

        # Key point: we aggregate across ALL objectives, so objective is NOT in keys
        agg_group_keys = ['__retailer', col_store]

        agg_store = grouped.loc[mask_multi].groupby(agg_group_keys, as_index=False).agg(agg_store_dict)

        # Make it an aggregate row: blank brand
        agg_store['__group_brand'] = ''

        # If objective is included, force blank objective for aggregate rows
        if include_objective:
            agg_store['__objective'] = ''

        # Remove any existing blank-brand rows (avoid duplicates)
        # If include_objective=True, there might be multiple blank-brand rows per objective, remove them all.
        to_remove = mask_multi & (grouped['__group_brand'].astype(str).str.strip() == '')
        grouped = grouped.loc[~to_remove]

        # Append aggregate rows back
        grouped = pd.concat([grouped, agg_store], ignore_index=True)

    # ----------------------------
    # 6) Expand each monthly row into daily rows based on allocation
    # ----------------------------
    rows = []

    for _, g in grouped.iterrows():
        retailer = g['__retailer'] if col_retailer else 'Retailer'
        store = g[col_store]
        store_l = str(store).strip().lower()

        # Output Brand rules:
        # - For Dancow/NOS/BOS: show brand (sub-brand or blank if aggregate)
        # - For other stores: always blank
        brand_out = g['__group_brand'] if store_l in stores_with_subbrands else ''

        # Objective output:
        # - Only when toggle ON
        # - For aggregate rows, it will already be forced to '' above
        objective_out = ''
        if include_objective:
            objective_out = g.get('__objective', '')

        monthly_budget = float(g[col_budget]) if pd.notna(g[col_budget]) else 0.0

        # ROAS KPI from data (if present)
        roas_kpi_val = float(g[col_roas_plan]) if (col_roas_plan and pd.notna(g[col_roas_plan])) else None

        # Detect aggregate row: blank brand
        is_blank_brand = (not g['__group_brand']) or (str(g['__group_brand']).strip() == '')

        # Special fixed rules for NOS and Dancow aggregate rows only
        if store_l == 'nos' and is_blank_brand:
            roas_kpi_val = 7
            target_roas_val = 20
        elif store_l == 'dancow' and is_blank_brand:
            roas_kpi_val = 15
            target_roas_val = 24
        else:
            # Default: compute target from KPI using buffer logic
            target_roas_val = buffered_target_roas(roas_kpi_val)

        benchmark_val = compute_roas_benchmark(roas_kpi_val)

        # Expand to daily
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
                'Brand': brand_out
            }

            if include_objective:
                row['Objective'] = objective_out

            rows.append(row)

    out = pd.DataFrame(rows)

    # Optional: reorder columns so Objective appears next to Brand
    if include_objective and 'Objective' in out.columns:
        cols = [
            'Date', 'Retailer', 'Store', 'Brand', 'Objective',
            'Daily Budget', 'Target ROAS', 'ROAS KPI', 'ROAS Benchmark'
        ]
        out = out[[c for c in cols if c in out.columns]]

    return out


# ============================================================
# Streamlit UI
# ============================================================
# This section is the app interface:
#  - upload file
#  - choose month/year
#  - toggle objective
#  - generate output
#  - preview + download excel

st.set_page_config(page_title='Daily Allocation Budget Maker', layout='wide')
st.title('📊 Daily Allocation Budget Maker')

file = st.file_uploader('Upload the Budget Plan Excel file', type=['xlsx'])

# Put controls in columns so it looks neat
col1, col2 = st.columns([1, 1])
with col1:
    month = st.selectbox('Select Month', list(range(1, 13)), index=pd.Timestamp.now().month - 1)
with col2:
    year = st.number_input('Select Year', min_value=2020, max_value=2100, value=pd.Timestamp.now().year, step=1)
    
include_objective = st.toggle('Include Objective in output', value=False)

if file:
    # Read Excel
    try:
        df_raw = pd.read_excel(file)
    except Exception as e:
        st.error(f"Failed to read Excel: {e}")
        st.stop()

    # Inform user which allocation pattern we are using
    days = calendar.monthrange(year, month)[1]
    st.caption(f'Using **{days}-day and month-{month}** allocation pattern (month-aware). 🐔')

    # Run transform
    df_out = create_daily_allocation(df_raw, month, year, include_objective=include_objective)

    # If nothing produced, tell user
    if df_out.empty:
        st.info('No output produced. Please check your input columns and budgets.')
    else:
        # Preview first/last 50 rows to validate output quickly
        df_out_preview = pd.concat([df_out.head(50), df_out.tail(50)])

        # Save to in-memory Excel so user can download
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_out.to_excel(writer, index=False, sheet_name='Output')

        # Styling only for the download button (optional UI polish)
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

        # Download output Excel
        st.download_button(
            '🐤 Download Excel',
            data=buffer.getvalue(),
            file_name=f'daily_allocation_{year}-{month:02d}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            type='secondary'
        )

        # Display preview table in Streamlit
        st.subheader('Preview')
        st.dataframe(df_out_preview, use_container_width=True)
