"""Load the exported training artifacts; never train during an API request."""
import json
from functools import lru_cache

import joblib
import numpy as np
import xgboost as xgb

from energy.features import LABELS, ROOT, make_row


class Forecaster:
    def __init__(self, models_dir=None):
        self.models_dir = models_dir or ROOT / 'models'
        self.manifest = json.loads((self.models_dir / 'manifest.json').read_text(encoding='utf8'))
        self.columns = self.manifest['features']
        self.kind = self.manifest['selected']['kind']
        if self.kind == 'xgboost':
            self.model = xgb.Booster()
            # Loading bytes avoids native filesystem canonicalization failures
            # in restricted Windows environments and also works on Linux.
            self.model.load_model(bytearray((self.models_dir / 'forecast_model.json').read_bytes()))
        elif self.kind == 'ridge':
            self.model = joblib.load(self.models_dir / 'forecast_ridge.joblib')
        else:
            self.model = None

    def predict(self, history, forecast_month, profile=None, consumption_limit=150):
        first, last = self.manifest['supported_forecast_months']
        if not first <= forecast_month <= last:
            raise ValueError(f'This historical-data prototype supports forecast months {first} to {last}.')
        row = make_row(history, forecast_month, profile)
        weight = self.manifest['selected']['learned_weight']
        explanations = []
        if self.kind == 'xgboost':
            matrix = xgb.DMatrix(row[self.columns], feature_names=self.columns)
            raw = float(self.model.predict(matrix)[0])
            contributions = self.model.predict(matrix, pred_contribs=True)[0]
            expected = float(contributions[-1]) * weight
            by_label = {}
            for column, value in zip(self.columns, contributions[:-1]):
                label = LABELS[column]
                by_label[label] = by_label.get(label, 0.0) + float(value) * weight
            by_label['Usage 1 month(s) ago'] = by_label.get('Usage 1 month(s) ago', 0.0) + (1 - weight) * history[-1]
            raw_blend = weight * raw + (1 - weight) * history[-1]
            estimate = weight * max(0, raw) + (1 - weight) * history[-1]
            explanations = [{'factor': label, 'contribution_kwh': round(value, 3)}
                            for label, value in sorted(by_label.items(), key=lambda p: abs(p[1]), reverse=True)]
            explanation = {'method': 'TreeSHAP feature contributions; associations, not causal appliance consumption.',
                           'reference_kwh': round(expected, 3),
                           'unclipped_forecast_kwh': round(raw_blend, 3),
                           'nonnegative_adjustment_kwh': round(estimate - raw_blend, 3),
                           'factors': explanations}
        elif self.kind == 'ridge':
            estimate = max(0, float(self.model.predict(row[self.columns])[0]))
            explanation = {'method': 'Linear history model.', 'factors': []}
        else:
            name = self.manifest['selected']['base_model']
            estimate = history[-1] if name == 'last_month' else float(np.mean(history[-3:]))
            explanation = {'method': 'Measured baseline selected on validation.', 'factors': []}
        scale = max(float(np.mean(history)), self.manifest['interval']['scale_floor_kwh'])
        half_width = scale * self.manifest['interval']['normalized_residual_quantile']
        lower, upper = max(0, estimate - half_width), estimate + half_width
        if lower > consumption_limit:
            alert = {'level': 'likely_above', 'message': 'The whole prediction range is above your consumption limit.'}
        elif estimate > consumption_limit:
            alert = {'level': 'above', 'message': 'The estimate is above your limit; the prediction range overlaps it.'}
        elif upper > consumption_limit:
            alert = {'level': 'possible', 'message': 'The estimate is below your limit, but higher usage is possible within the prediction range.'}
        else:
            alert = {'level': 'within', 'message': 'The estimate and prediction range are below your consumption limit.'}
        warnings = ['Historical LECO-area prototype; applicability to current or nationwide use has not been established.']
        if max(history) > 3025:
            warnings.append('A reading exceeds the largest monthly value observed in the source dataset.')
        return {'forecast_month': forecast_month, 'prediction_kwh': round(estimate, 2),
                'lower_kwh': round(lower, 2), 'upper_kwh': round(upper, 2),
                'nominal_interval_coverage': 0.9, 'consumption_limit_kwh': consumption_limit,
                'change_from_last_month_percent': round(100 * (estimate - history[-1]) / history[-1], 1) if history[-1] else None,
                'model': self.manifest['selected']['name'], 'model_version': self.manifest['model_version'],
                'profile_used': profile is not None and 'profile_available' in self.columns,
                'alert': alert, 'explanation': explanation, 'warnings': warnings}


@lru_cache(maxsize=1)
def get_forecaster():
    return Forecaster()
