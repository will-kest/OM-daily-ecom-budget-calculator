import pandas as pd
import streamlit as st
from io import BytesIO
import calendar
import re

# ───────────────────────────────
# Helpers
# ───────────────────────────────

def pick_col(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if not cand.startswith('^'):
            for k, orig in cols_lower.items():
                if k == cand.lower():
                    return orig
    for cand in candidates:
        if not cand.startswith('^'):
            for k, orig in cols_lower.items():
                if cand.lower() in k:
                    return orig
    for cand in candidates:
        if cand.startswith('^'):
            pat = re.compile(cand, re.I)
            for orig in df.columns:
                if pat.search(orig):
                    return orig
    return None


def coerce_numeric(series):
    if series is None:
        return None
    s = series.astype(str)
    s = s.str.replace(r"[^0-9\.-]", "", regex=True)
    s = s.apply(lambda x: x if x.count('.') <= 1 else (x.replace('.', '', x.count('.')-1)))
    out = pd.to_numeric(s, errors='coerce')
    return out


def get_allocation(month: int, year: int) -> pd.DataFrame:
    days = calendar.monthrange(year, month)[1]
    dd = min(month, days)  # double-date day (e.g., 10 for Oct, 11 for Nov), capped by month length

    # Start with an empty allocation map of day -> percent
    alloc_map = {d: None for d in range(1, days + 1)}
    assigned = set()

    # 1) Day 1 .. (DD-1): 4%
    for d in range(1, dd):
        alloc_map[d] = 4.0
        assigned.add(d)

    # 2) DD: 18%
    alloc_map[dd] = 18.0
    assigned.add(dd)

    # 3) DD+1: 4.5% 
    if dd + 1 <= days:
        alloc_map[dd + 1] = 4.5
        assigned.add(dd + 1)

    # 4) H-1 payday (24th): 3% 
    if 24 <= days:
        alloc_map[24] = 3.0
        assigned.add(24)

    # 5) Payday (25th): 8% 
    if 25 <= days:
        alloc_map[25] = 8.0
        assigned.add(25)

    # 6) Distribute the remainder across all unassigned days so total = 100%
    total_assigned = sum(v for v in alloc_map.values() if v is not None)
    remaining_days = [d for d, v in alloc_map.items() if v is None]

    # Define buckets
    # Mid-month = (DD+2) .. 23 (if they exist)
    mid_days = [d for d in remaining_days if (dd + 2) <= d <= 23]
    # After payday = 26 .. last day (if exist)
    after_days = [d for d in remaining_days if d >= 26]

    if remaining_days:
        remaining_pct = 100.0 - total_assigned

        # If one bucket is empty, give 100% to the other to avoid division by zero
        if len(mid_days) == 0 and len(after_days) > 0:
            mid_share = 0.0
            after_share = remaining_pct
        elif len(after_days) == 0 and len(mid_days) > 0:
            mid_share = remaining_pct
            after_share = 0.0
        else:
            mid_share = remaining_pct * 0.5
            after_share = remaining_pct * 0.5

        # Per-day allocation inside each bucket
        if len(mid_days) > 0:
            per_mid = mid_share / len(mid_days)
            for d in mid_days:
                alloc_map[d] = per_mid
        if len(after_days) > 0:
            per_after = after_share / len(after_days)
            for d in after_days:
                alloc_map[d] = per_after

        # Any leftover (due to float rounding) goes to the last day of the last non-empty bucket
        used = sum(v for v in alloc_map.values() if v is not None)
        leftover = 100.0 - used
        if abs(leftover) > 1e-9:
            if len(after_days) > 0:
                alloc_map[after_days[-1]] += leftover
            elif len(mid_days) > 0:
                alloc_map[mid_days[-1]] += leftover

    # Build DataFrame as fractions
    percent_list = [alloc_map[d] / 100.0 for d in range(1, days + 1)]
    df = pd.DataFrame({"day": range(1, days + 1), "percent": percent_list})

    s = df["percent"].sum()
    if not (0.999 <= s <= 1.001):
        st.warning(f"Daily allocation sums to {s*100:.2f}% (expected ~100%).")

    return df


def buffered_target_roas(roas_kpi):
    """Apply a flexible buffer to ROAS KPI to get target ROAS."""
    if pd.isna(roas_kpi):
        return None
    try:
        roas_kpi = float(roas_kpi)
    except:
        return None

    # General buffer logic
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
    KPI Benchmark rules:
    - default: ROAS KPI - 2
    - if KPI == 1 -> 1 (no minus)
    - if KPI == 3 -> 2 (-1)
    Returns '' if KPI is missing or non-numeric.
    """
    if roas_kpi is None or (isinstance(roas_kpi, float) and pd.isna(roas_kpi)):
        return ''
    try:
        rk = float(roas_kpi)
    except (TypeError, ValueError):
        return ''

    # Handle exact 1 and 3 (with tiny tolerance for floats)
    if abs(rk - 1) < 1e-9:
        return "1"
    if abs(rk - 3) < 1e-9:
        return str("{}-{}".format(2, round(rk)))

    return str("{}-{}".format(round(rk - 2), round(rk)))


def create_daily_allocation(df_raw, month, year):
    days = calendar.monthrange(year, month)[1]
    allocation = get_allocation(month, year)

    # Identify columns
    col_budget = pick_col(df_raw, ['Media Budget Plan', 'Budget', 'media_budget', 'media budget', '^.*budget.*$'])
    col_roas_plan = pick_col(df_raw, ['ROAS Plan', '^roas plan$', 'roas'])
    col_target_roas = pick_col(df_raw, ['Target ROAS', 'target_roas', 'target roas'])
    col_benchmark = pick_col(df_raw, ['ROAS Benchmark', 'benchmark'])
    col_store = pick_col(df_raw, ['Store'])
    col_brand = pick_col(df_raw, ['Brand'])
    col_retailer = pick_col(df_raw, ['Retailer', 'Platform'])

    if not all([col_budget, col_store]):
        st.error("Couldn't find required columns: Budget and Store.")
        return pd.DataFrame()

    # Coerce numeric
    df_raw[col_budget] = coerce_numeric(df_raw[col_budget])
    if col_roas_plan:
        df_raw[col_roas_plan] = coerce_numeric(df_raw[col_roas_plan])
    if col_target_roas:
        df_raw[col_target_roas] = coerce_numeric(df_raw[col_target_roas])
    if col_benchmark:
        df_raw[col_benchmark] = coerce_numeric(df_raw[col_benchmark])

    df_raw = df_raw[df_raw[col_budget].fillna(0) > 0]
    if df_raw.empty:
        st.warning("No rows with positive budget after cleaning.")
        return pd.DataFrame()

    # Grouping logic
    dancow_nos_mask = df_raw[col_store].astype(str).str.lower().isin(['dancow', 'nos'])
    group_brand = pd.Series('', index=df_raw.index, dtype='object')
    if col_brand:
        group_brand[dancow_nos_mask] = df_raw.loc[dancow_nos_mask, col_brand].astype(str).fillna('')
    retailer_series = df_raw[col_retailer] if col_retailer else pd.Series('Retailer', index=df_raw.index)

    agg_dict = {col_budget: 'sum'}
    if col_roas_plan:
        agg_dict[col_roas_plan] = 'median'
    if col_target_roas:
        agg_dict[col_target_roas] = 'median'
    if col_benchmark:
        agg_dict[col_benchmark] = 'median'

    tmp = df_raw.assign(__group_brand=group_brand, __retailer=retailer_series)
    grouped = tmp.groupby(['__retailer', col_store, '__group_brand'], dropna=False).agg(agg_dict).reset_index()

    # ── Add aggregate rows for Dancow & NOS (blank brand = sum of all sub-brands)
    mask_dn = grouped[col_store].astype(str).str.lower().isin(['dancow', 'nos'])
    if mask_dn.any():
        # Build agg dict only with columns that actually exist
        agg_store_dict = {col_budget: 'sum'}
        if col_roas_plan:
            agg_store_dict[col_roas_plan] = 'median'
        if col_target_roas:
            agg_store_dict[col_target_roas] = 'median'
        if col_benchmark:
            agg_store_dict[col_benchmark] = 'median'

        agg_store = grouped.loc[mask_dn].groupby(['__retailer', col_store], as_index=False).agg(agg_store_dict)
        agg_store['__group_brand'] = ''
        # Remove any existing blank-brand rows for these stores to avoid duplicates
        to_remove = mask_dn & (grouped['__group_brand'].astype(str) == '')
        grouped = grouped.loc[~to_remove]
        # Append aggregate rows
        grouped = pd.concat([grouped, agg_store], ignore_index=True)

    rows = []
    for _, g in grouped.iterrows():
        retailer = g['__retailer'] if col_retailer else 'Retailer'
        store = g[col_store]
        store_l = str(store).lower()
        brand_out = g['__group_brand'] if store_l in ['dancow', 'nos'] else ''
        monthly_budget = float(g[col_budget]) if pd.notna(g[col_budget]) else 0.0

        roas_kpi_val = float(g[col_roas_plan]) if (col_roas_plan and pd.notna(g[col_roas_plan])) else None
        # Apply buffer for Target ROAS
        is_blank_brand = (not g['__group_brand']) or (str(g['__group_brand']).strip() == '')
        store_l = str(store).strip().lower()

        if store_l == 'nos' and is_blank_brand:
            # Special rule for NOS aggregate row
            roas_kpi_val = 7
            target_roas_val = 20
        elif store_l == 'dancow' and is_blank_brand:
            # Special rule for Dancow aggregate row
            roas_kpi_val = 15
            target_roas_val = 24
        else:
            # Default behavior: use ROAS KPI from data, then buffer to Target ROAS
            roas_kpi_val = float(g[col_roas_plan]) if (col_roas_plan and pd.notna(g[col_roas_plan])) else None
            target_roas_val = buffered_target_roas(roas_kpi_val)

        benchmark_val = compute_roas_benchmark(roas_kpi_val)

        for _, a in allocation.iterrows():
            date = pd.Timestamp(year=year, month=month, day=int(a['day']))
            daily_budget = round(monthly_budget * float(a['percent']))
            rows.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Retailer': retailer,
                'Store': store,
                'Daily Budget': daily_budget,
                'Target ROAS': target_roas_val,
                'ROAS KPI': roas_kpi_val,
                'ROAS Benchmark': benchmark_val if benchmark_val is not None else '',
                'Brand': brand_out
            })

    return pd.DataFrame(rows)

# ───────────────────────────────
# Streamlit UI
# ───────────────────────────────
st.set_page_config(page_title='Daily Allocation Budget Maker', layout='wide')
st.title('📊 Daily Allocation Budget Maker')

file = st.file_uploader('Upload the Budget Plan Excel file', type=['xlsx'])
col1, col2 = st.columns(2)
with col1:
    month = st.selectbox('Select Month', list(range(1, 13)), index=pd.Timestamp.now().month - 1)
with col2:
    year = st.number_input('Select Year', min_value=2020, max_value=2100, value=pd.Timestamp.now().year, step=1)

if file:
    try:
        df_raw = pd.read_excel(file)
    except Exception as e:
        st.error(f"Failed to read Excel: {e}")
        st.stop()

    days = calendar.monthrange(year, month)[1]
    st.caption(f'Using **{days}-day and month-{month}** allocation pattern (month-aware). 🐔')

    df_out = create_daily_allocation(df_raw, month, year)
    df_out_preview = pd.concat([df_out.head(50), df_out.tail(50)])

    if not df_out.empty:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_out.to_excel(writer, index=False, sheet_name='Output')
            
        st.markdown('''<style> 
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
                    unsafe_allow_html=True)
        
        st.download_button('🐤 Download Excel', data=buffer.getvalue(),
                            file_name=f'daily_allocation_{year}-{month:02d}.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            type='secondary')
        
        st.subheader('Preview')
        st.dataframe(df_out_preview, use_container_width=True)
    else:
        st.info('No output produced. Please check your input columns and budgets.')
