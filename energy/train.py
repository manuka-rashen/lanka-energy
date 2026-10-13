"""Run with python -m energy.train. All selection precedes calibration/test."""
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from energy.features import ALL_FEATURES, HISTORY_FEATURES, RAW, ROOT, build_panel


def metrics(y, prediction):
    y, prediction = np.asarray(y), np.maximum(0, np.asarray(prediction))
    return {'rows': len(y), 'mae_kwh': float(mean_absolute_error(y, prediction)),
            'rmse_kwh': float(np.sqrt(mean_squared_error(y, prediction))),
            'r2': float(r2_score(y, prediction)),
            'wape_percent': float(100 * np.abs(y - prediction).sum() / max(y.sum(), 1))}


def baseline(frame, method):
    return frame['lag_1'].to_numpy() if method == 'last_month' else frame[['lag_1', 'lag_2', 'lag_3']].mean(axis=1).to_numpy()


def train():
    start = time.time()
    models_dir, results_dir = ROOT / 'models', ROOT / 'results'
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    panel = build_panel()
    households = np.array(sorted(panel.household_ID.unique()))
    rng = np.random.default_rng(42)
    rng.shuffle(households)
    unseen_ids = set(households[:math.ceil(len(households) * 0.2)])
    panel['household_split'] = np.where(panel.household_ID.isin(unseen_ids), 'unseen', 'development')
    development = panel.household_split.eq('development')
    masks = {
        'train': development & panel.month.le('2024-03-31'),
        'validation': development & panel.month.between('2024-04-01', '2024-06-30'),
        'calibration': development & panel.month.between('2024-07-01', '2024-07-31'),
        'test': development & panel.month.between('2024-08-01', '2024-10-31'),
        'unseen_test': ~development & panel.month.between('2024-08-01', '2024-10-31')}
    splits = {name: panel.loc[mask].copy() for name, mask in masks.items()}
    assert all(len(frame) > 0 for frame in splits.values())
    print('SPLITS', {name: len(frame) for name, frame in splits.items()}, flush=True)
    records, fitted, configs = [], {}, {}
    val = splits['validation']
    for name in ['last_month', 'mean_3']:
        records.append({'name': name, 'kind': 'baseline', 'features': HISTORY_FEATURES,
                        'validation': metrics(val.consumption, baseline(val, name))})
    ridge = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=100))
    ridge.fit(splits['train'][HISTORY_FEATURES], splits['train'].consumption)
    fitted['ridge_history'] = ridge
    records.append({'name': 'ridge_history', 'kind': 'ridge', 'features': HISTORY_FEATURES,
                    'validation': metrics(val.consumption, ridge.predict(val[HISTORY_FEATURES]))})
    for feature_set, columns in [('history', HISTORY_FEATURES), ('context', ALL_FEATURES)]:
        for depth, trees in [(3, 250), (5, 350)]:
            name = f'xgb_{feature_set}_d{depth}'
            params = dict(objective='reg:absoluteerror', n_estimators=trees,
                          max_depth=depth, learning_rate=0.045, subsample=0.85,
                          colsample_bytree=0.9, min_child_weight=15,
                          reg_lambda=15, random_state=42, n_jobs=4, tree_method='hist')
            model = XGBRegressor(**params)
            model.fit(splits['train'][columns], splits['train'].consumption)
            fitted[name], configs[name] = model, params
            records.append({'name': name, 'kind': 'xgboost', 'features': columns,
                            'validation': metrics(val.consumption, model.predict(val[columns]))})
            print(name, records[-1]['validation'], flush=True)
    # Blend weights are picked exclusively from validation, not calibration/test.
    blends = []
    for record in [r for r in records if r['kind'] == 'xgboost']:
        learned = np.maximum(0, fitted[record['name']].predict(val[record['features']]))
        for weight in [0.25, 0.5, 0.75]:
            score = metrics(val.consumption, weight * learned + (1 - weight) * val.lag_1.to_numpy())
            blends.append({**record, 'name': record['name'] + f'_blend_{weight}',
                           'base_model': record['name'], 'learned_weight': weight, 'validation': score})
    records += blends
    selected = min(records, key=lambda record: record['validation']['mae_kwh'])
    selected.setdefault('base_model', selected['name'])
    selected.setdefault('learned_weight', 1.0)
    final_train = pd.concat([splits['train'], val], ignore_index=True)
    kind, columns, base_name = selected['kind'], selected['features'], selected['base_model']
    if kind == 'xgboost':
        model = XGBRegressor(**configs[base_name])
        model.fit(final_train[columns], final_train.consumption)
        model.save_model(models_dir / 'forecast_model.json')
    elif kind == 'ridge':
        model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=100))
        model.fit(final_train[columns], final_train.consumption)
        joblib.dump(model, models_dir / 'forecast_ridge.joblib')
    else:
        model = None

    def predict(frame):
        if kind == 'baseline':
            return baseline(frame, base_name)
        learned = np.maximum(0, model.predict(frame[columns]))
        return selected['learned_weight'] * learned + (1 - selected['learned_weight']) * frame.lag_1.to_numpy()

    calibration = splits['calibration']
    scores = np.abs(calibration.consumption.to_numpy() - predict(calibration)) / np.maximum(calibration.mean_6.to_numpy(), 30)
    rank = min(len(scores), math.ceil((len(scores) + 1) * 0.9))
    quantile = float(np.sort(scores)[rank - 1])
    test_results = {}
    prediction_frames = []
    for name in ['test', 'unseen_test']:
        frame = splits[name]
        pred = predict(frame)
        width = quantile * np.maximum(frame.mean_6.to_numpy(), 30)
        lower, upper = np.maximum(0, pred - width), pred + width
        result = metrics(frame.consumption, pred)
        result.update({'interval_coverage_percent': float(100 * np.mean((frame.consumption >= lower) & (frame.consumption <= upper))),
                       'mean_interval_width_kwh': float(np.mean(upper - lower)),
                       'last_month_baseline': metrics(frame.consumption, baseline(frame, 'last_month'))})
        test_results[name] = result
        saved = frame[['household_ID', 'month', 'consumption', 'profile_available']].copy()
        saved['prediction'], saved['lower'], saved['upper'], saved['split'] = pred, lower, upper, name
        prediction_frames.append(saved)
    # Context contribution: evaluate the two trained candidates on identical
    # validation rows where the survey was already known at forecast origin.
    eligible = val.profile_available.eq(1)
    ablation = {}
    for name in ['xgb_history_d3', 'xgb_context_d3', 'xgb_history_d5', 'xgb_context_d5']:
        record = next(r for r in records if r['name'] == name)
        ablation[name] = metrics(val.loc[eligible, 'consumption'], fitted[name].predict(val.loc[eligible, record['features']]))
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(RAW.glob('*.csv'))}
    manifest = {
        'project': 'Lanka Energy', 'model_version': '1.0.0', 'seed': 42,
        'target': 'Next monthly consumption in kWh', 'selected': selected,
        'parameters': configs.get(base_name, {}), 'features': columns,
        'interval': {'nominal_coverage': 0.9, 'normalized_residual_quantile': quantile,
                     'calibration_month': '2024-07', 'scale_floor_kwh': 30,
                     'method': 'Split conformal, scaled by past six-month mean; coverage is empirical under temporal shift.'},
        'input_order': 'Six monthly kWh readings, oldest to newest.',
        'data_period': ['2022-10', '2024-10'],
        'training_target_end': '2024-06',
        'supported_forecast_months': ['2023-04', '2024-11'],
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'scikit_learn': sklearn.__version__, 'xgboost': xgboost.__version__},
        'source_sha256': source_hashes}
    (models_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    summary = {
        'usable_rows': len(panel), 'households': len(households),
        'held_out_households': len(unseen_ids),
        'splits': {name: {'rows': len(frame), 'households': int(frame.household_ID.nunique()),
                           'start': frame.month.min().date().isoformat(), 'end': frame.month.max().date().isoformat()}
                   for name, frame in splits.items()},
        'candidates': sorted(records, key=lambda r: r['validation']['mae_kwh']),
        'selected': selected['name'], 'test_results': test_results,
        'context_ablation_validation': ablation,
        'limitations': ['Historical LECO-area sample; not nationally representative.',
                        'Survey variables masked until their recorded collection dates.',
                        'Zero measurements retained; measured targets never imputed.',
                        'Prediction intervals are calibrated on July 2024; future temporal shifts can change coverage.',
                        'This monthly prototype is not an appliance meter or live grid-control system.'],
        'seconds': round(time.time() - start, 2)}
    (results_dir / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    pd.concat(prediction_frames).to_csv(results_dir / 'test_predictions.csv', index=False)
    panel[['household_ID', 'household_split']].drop_duplicates().to_csv(results_dir / 'household_split.csv', index=False)
    # Demo requests use observed histories, not targets, and contain no identifiers.
    examples = []
    for position in [0, 300, 1200]:
        row = splits['test'].iloc[position]
        profile = {key: int(row[key]) for key in ALL_FEATURES if key not in HISTORY_FEATURES and key != 'profile_available' and pd.notna(row[key])} if row.profile_available else None
        examples.append({'name': f'Historical example {len(examples) + 1}',
                         'history': [float(row[f'lag_{i}']) for i in range(6, 0, -1)],
                         'forecast_month': row.month.strftime('%Y-%m'),
                         'profile': profile, 'consumption_limit': 150.0})
    (models_dir / 'examples.json').write_text(json.dumps(examples, indent=2), encoding='utf8')
    print('SELECTED', selected['name'], flush=True)
    print(json.dumps(test_results, indent=2), flush=True)
    plot_results(summary)
    return summary


def plot_results(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    candidates = summary['candidates']
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [r['name'] for r in candidates]
    ax.barh(names[::-1], [r['validation']['mae_kwh'] for r in candidates][::-1], color='#137f71')
    ax.set_xlabel('Validation MAE (kWh; lower is better)')
    ax.set_title('Model selection: April–June 2024, development households only')
    fig.tight_layout()
    fig.savefig(ROOT / 'results' / 'model_comparison.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    train()
