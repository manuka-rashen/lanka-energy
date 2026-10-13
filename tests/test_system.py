import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from energy.features import PROFILE_FEATURES, RAW, ROOT, build_panel, make_row
from energy.predict import Forecaster

client = TestClient(app)


@pytest.fixture
def synthetic_raw(tmp_path):
    months = pd.date_range('2023-01-31', periods=9, freq='ME')
    pd.DataFrame({'household_ID': ['A']*9, 'month': months,
                  'consumption': list(range(10, 100, 10))}).to_csv(tmp_path/'electricity_monthly.csv', index=False)
    pd.DataFrame({'household_ID':['A'], 'no_of_household_members':[4],
                  'no_of_electricity_meters':[1]}).to_csv(tmp_path/'household_information.csv', index=False)
    pd.DataFrame({'household_ID':['A'], 'wave1':['07/15/2023']}).to_csv(tmp_path/'survey_dates.csv', index=False)
    pd.DataFrame({'household_ID':['A'], 'appliance_type':['Refrigerator']}).to_csv(tmp_path/'appliances.csv', index=False)
    pd.DataFrame({'household_ID':['A'], 'ac_ID':['AC1']}).to_csv(tmp_path/'ac_roster.csv', index=False)
    return tmp_path


def test_survey_context_is_hidden_until_forecast_origin(synthetic_raw):
    panel = build_panel(synthetic_raw)
    july = panel.loc[panel.month.eq('2023-07-31')].iloc[0]
    august = panel.loc[panel.month.eq('2023-08-31')].iloc[0]
    assert july.profile_available == 0
    assert july[PROFILE_FEATURES[1:]].isna().all()
    assert august.profile_available == 1 and august.household_members == 4
    assert august.air_conditioners == 1


def test_missing_calendar_month_cannot_become_a_false_lag(synthetic_raw):
    path = synthetic_raw/'electricity_monthly.csv'
    source = pd.read_csv(path)
    source = source.loc[~source.month.str.startswith('2023-04')]
    source.to_csv(path, index=False)
    assert build_panel(synthetic_raw).empty


def test_future_target_changes_do_not_change_earlier_features(synthetic_raw):
    before = build_panel(synthetic_raw)
    path = synthetic_raw/'electricity_monthly.csv'
    source = pd.read_csv(path)
    source.loc[source.month.str.startswith('2023-09'), 'consumption'] = 999
    source.to_csv(path, index=False)
    after = build_panel(synthetic_raw)
    pd.testing.assert_frame_equal(before.loc[before.month.lt('2023-09-01')], after.loc[after.month.lt('2023-09-01')])


def test_history_order_and_summary_match_input():
    row = make_row([10,20,30,40,50,60], '2024-10')
    assert row.lag_1.iloc[0] == 60 and row.lag_6.iloc[0] == 10
    assert row.mean_6.iloc[0] == 35 and row.recent_change.iloc[0] == 10
    assert row.profile_available.iloc[0] == 0


def example_request():
    return json.loads((ROOT/'models'/'examples.json').read_text())[0]


def test_health_loads_actual_artifact():
    response = client.get('/health')
    assert response.status_code == 200 and response.json()['status'] == 'ok'
    assert response.json()['model'] == client.get('/api/metadata').json()['model']


def test_exported_model_reproduces_saved_test_prediction():
    example = example_request()
    data = {key:value for key,value in example.items() if key!='name'}
    prediction = client.post('/api/forecast', json=data)
    assert prediction.status_code == 200
    value = prediction.json()
    saved = pd.read_csv(ROOT/'results'/'test_predictions.csv').iloc[0]
    assert value['prediction_kwh'] == pytest.approx(saved.prediction, abs=0.015)
    assert value['lower_kwh'] <= value['prediction_kwh'] <= value['upper_kwh']


def test_explanations_reconstruct_forecast():
    data = example_request()
    result = Forecaster().predict(data['history'], data['forecast_month'], data['profile'])
    explanation = result['explanation']
    if 'reference_kwh' in explanation:
        total = explanation['reference_kwh'] + sum(f['contribution_kwh'] for f in explanation['factors'])
        total += explanation['nonnegative_adjustment_kwh']
        assert total == pytest.approx(result['prediction_kwh'], abs=0.03)


@pytest.mark.parametrize('updates', [
    {'history':[-1,20,30,40,50,60]}, {'history':[1,2,3]},
    {'forecast_month':'2024-13'}, {'forecast_month':'2026-10'},
    {'consumption_limit':0}, {'profile':{'household_members':0}},
    {'unexpected_property':'ignored?'}])
def test_invalid_request_rejected(updates):
    data = {'history':[10,20,30,40,50,60], 'forecast_month':'2024-10', **updates}
    assert client.post('/api/forecast', json=data).status_code == 422


def test_zero_history_is_supported_without_division_by_zero():
    response = client.post('/api/forecast', json={'history':[0]*6, 'forecast_month':'2024-10'})
    assert response.status_code == 200
    assert response.json()['change_from_last_month_percent'] is None
    assert np.isfinite(response.json()['prediction_kwh'])


def test_nonfinite_history_rejected():
    response = client.post('/api/forecast', content='{"history":[0,0,0,0,0,NaN],"forecast_month":"2024-10"}', headers={'Content-Type':'application/json'})
    assert response.status_code == 422


def test_consumption_limit_warning_changes():
    data = {'history':[150]*6, 'forecast_month':'2024-10', 'consumption_limit':1}
    assert client.post('/api/forecast', json=data).json()['alert']['level'] in ['above','likely_above']
    data['consumption_limit'] = 10000
    assert client.post('/api/forecast', json=data).json()['alert']['level'] == 'within'


def test_validation_selection_and_household_holdout_are_separate():
    result = json.loads((ROOT/'results'/'metrics.json').read_text())
    assert result['selected'] == min(result['candidates'], key=lambda r:r['validation']['mae_kwh'])['name']
    split = pd.read_csv(ROOT/'results'/'household_split.csv')
    predictions = pd.read_csv(ROOT/'results'/'test_predictions.csv')
    development_ids = set(split.loc[split.household_split.eq('development'), 'household_ID'])
    unseen_ids = set(predictions.loc[predictions.split.eq('unseen_test'), 'household_ID'])
    assert not development_ids & unseen_ids
    assert result['splits']['validation']['end'] < result['splits']['calibration']['start']
    assert result['splits']['calibration']['end'] < result['splits']['test']['start']


def test_source_files_match_recorded_training_hashes():
    manifest = json.loads((ROOT/'models'/'manifest.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        assert hashlib.sha256((RAW/name).read_bytes()).hexdigest() == expected
