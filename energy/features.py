"""Shared, point-in-time feature definitions for training and inference."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
HISTORY_FEATURES = [f'lag_{i}' for i in range(1, 7)] + [
    'mean_6', 'std_6', 'recent_change', 'trend_6', 'month_sin', 'month_cos']
PROFILE_FEATURES = ['profile_available', 'household_members', 'electricity_meters',
                    'refrigerators', 'washing_machines', 'air_conditioners', 'rice_cookers']
ALL_FEATURES = HISTORY_FEATURES + PROFILE_FEATURES
PROFILE_COUNTS = {
    'refrigerators': r'(?i)refrigerator',
    'washing_machines': r'(?i)washing machine',
    'rice_cookers': r'(?i)rice cooker',
}
LABELS = {**{f'lag_{i}': f'Usage {i} month(s) ago' for i in range(1, 7)},
          'mean_6': 'Six-month average', 'std_6': 'Usage variation',
          'recent_change': 'Recent change', 'trend_6': 'Six-month trend',
          'month_sin': 'Seasonal pattern', 'month_cos': 'Seasonal pattern',
          'profile_available': 'Household details available',
          'household_members': 'Household size', 'electricity_meters': 'Meter count',
          'refrigerators': 'Refrigerator count', 'washing_machines': 'Washing machine count',
          'air_conditioners': 'Air conditioner count', 'rice_cookers': 'Rice cooker count'}


def add_history_features(frame):
    """All summaries use past observations only. lag_1 is the latest month."""
    result = frame.copy()
    lag = result[[f'lag_{i}' for i in range(1, 7)]]
    result['mean_6'] = lag.mean(axis=1)
    result['std_6'] = lag.std(axis=1, ddof=0)
    result['recent_change'] = result['lag_1'] - result['lag_2']
    result['trend_6'] = (result['lag_1'] - result['lag_6']) / 5.0
    months = pd.to_datetime(result['month']).dt.month
    result['month_sin'] = np.sin(2 * np.pi * months / 12)
    result['month_cos'] = np.cos(2 * np.pi * months / 12)
    return result


def build_panel(raw_dir=RAW):
    monthly = pd.read_csv(raw_dir / 'electricity_monthly.csv')
    monthly['month'] = pd.to_datetime(monthly['month'])
    monthly['consumption'] = pd.to_numeric(
        monthly['consumption'].astype(str).str.replace(',', '', regex=False), errors='coerce')
    if monthly.duplicated(['household_ID', 'month']).any():
        raise ValueError('Duplicate household-month observations.')
    # Reindex each household to the full monthly calendar so missing months
    # cannot silently turn a two-month-old reading into a one-month lag.
    grid = pd.MultiIndex.from_product([
        sorted(monthly.household_ID.unique()),
        pd.date_range(monthly.month.min(), monthly.month.max(), freq='ME')],
        names=['household_ID', 'month'])
    panel = monthly.set_index(['household_ID', 'month']).reindex(grid).reset_index()
    for i in range(1, 7):
        panel[f'lag_{i}'] = panel.groupby('household_ID')['consumption'].shift(i)
    panel = add_history_features(panel)
    info = pd.read_csv(raw_dir / 'household_information.csv')
    dates = pd.read_csv(raw_dir / 'survey_dates.csv')
    dates['profile_date'] = pd.to_datetime(dates.wave1, format='%m/%d/%Y', errors='raise')
    profile = info[['household_ID', 'no_of_household_members', 'no_of_electricity_meters']].rename(
        columns={'no_of_household_members': 'household_members',
                 'no_of_electricity_meters': 'electricity_meters'})
    appliances = pd.read_csv(raw_dir / 'appliances.csv')
    roster_households = set(appliances.household_ID)
    for column, pattern in PROFILE_COUNTS.items():
        counts = appliances.loc[appliances.appliance_type.str.contains(pattern, na=False)].groupby('household_ID').size()
        profile[column] = profile.household_ID.map(counts).fillna(0)
        # A completely missing roster is unknown, not confirmed zero appliances.
        profile.loc[~profile.household_ID.isin(roster_households), column] = np.nan
    # AC units are recorded in a separate roster, not the general appliance list.
    ac = pd.read_csv(raw_dir / 'ac_roster.csv')
    profile['air_conditioners'] = profile.household_ID.map(ac.groupby('household_ID').size()).fillna(0)
    profile = profile.merge(dates[['household_ID', 'profile_date']], on='household_ID', validate='one_to_one')
    panel = panel.merge(profile, on='household_ID', how='left', validate='many_to_one')
    panel['origin'] = panel.month - pd.offsets.MonthEnd(1)
    available = panel.profile_date.notna() & (panel.profile_date <= panel.origin)
    panel['profile_available'] = available.astype(float)
    panel.loc[~available, PROFILE_FEATURES[1:]] = np.nan
    # Evaluation targets are actual measured values, never imputed labels.
    usable = panel.consumption.notna() & panel[[f'lag_{i}' for i in range(1, 7)]].notna().all(axis=1)
    return panel.loc[usable].reset_index(drop=True)


def make_row(history, forecast_month, profile=None):
    """history: six kWh readings, oldest to most recent; profile known now."""
    if len(history) != 6 or not all(np.isfinite(x) and 0 <= x <= 10000 for x in history):
        raise ValueError('Provide six finite, non-negative readings up to 10,000 kWh.')
    row = {'month': pd.Timestamp(forecast_month + '-01')}
    row.update({f'lag_{i}': history[-i] for i in range(1, 7)})
    row['profile_available'] = float(profile is not None)
    for column in PROFILE_FEATURES[1:]:
        row[column] = profile.get(column, np.nan) if profile else np.nan
    return add_history_features(pd.DataFrame([row]))[ALL_FEATURES]
