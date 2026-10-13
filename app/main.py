import json
import re
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from energy.features import ROOT
from energy.predict import get_forecaster

app = FastAPI(title='Lanka Energy', version='1.0.0', description='Historical household electricity consumption forecasting for Sri Lanka.')
app.mount('/static', StaticFiles(directory=ROOT / 'app' / 'static'), name='static')


@app.exception_handler(RequestValidationError)
async def invalid_request(_request, exc):
    # Do not echo submitted values (including NaN/Infinity) into JSON errors.
    return JSONResponse(status_code=422, content={'detail': [
        {'type': error['type'], 'loc': list(error['loc']), 'msg': error['msg']}
        for error in exc.errors()]})


class HouseholdProfile(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    household_members: Annotated[int, Field(ge=1, le=50)]
    electricity_meters: Annotated[int, Field(ge=1, le=10)] = 1
    refrigerators: Annotated[int | None, Field(ge=0, le=20)] = None
    washing_machines: Annotated[int | None, Field(ge=0, le=20)] = None
    air_conditioners: Annotated[int | None, Field(ge=0, le=20)] = None
    rice_cookers: Annotated[int | None, Field(ge=0, le=20)] = None


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    history: Annotated[list[Annotated[float, Field(ge=0, le=10000)]], Field(min_length=6, max_length=6)]
    forecast_month: str = '2024-11'
    profile: HouseholdProfile | None = None
    consumption_limit: Annotated[float, Field(gt=0, le=10000)] = 150

    @field_validator('forecast_month')
    @classmethod
    def valid_month(cls, value):
        if not re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])', value):
            raise ValueError('Use YYYY-MM with a valid month.')
        return value


@app.get('/', include_in_schema=False)
def index():
    return FileResponse(ROOT / 'app' / 'static' / 'index.html')


@app.get('/health')
def health():
    forecaster = get_forecaster()
    return {'status': 'ok', 'model_version': forecaster.manifest['model_version'],
            'model': forecaster.manifest['selected']['name']}


@app.get('/api/metadata')
def metadata():
    metrics = json.loads((ROOT / 'results' / 'metrics.json').read_text(encoding='utf8'))
    manifest = get_forecaster().manifest
    return {'data_period': manifest['data_period'], 'target': manifest['target'],
            'supported_forecast_months': manifest['supported_forecast_months'],
            'model': metrics['selected'], 'test': metrics['test_results']['test'],
            'unseen_households': metrics['test_results']['unseen_test'],
            'splits': metrics['splits'], 'candidates': metrics['candidates'],
            'context_ablation': metrics['context_ablation_validation'],
            'limitations': metrics['limitations'], 'training_target_end': manifest['training_target_end'],
            'dataset_url': 'https://www.kaggle.com/datasets/lirneasia/sri-lankan-residential-electricity-consumption',
            'paper_url': 'https://arxiv.org/abs/2609.28783'}


@app.get('/api/examples')
def examples():
    return json.loads((ROOT / 'models' / 'examples.json').read_text(encoding='utf8'))


@app.post('/api/forecast')
def forecast(request: ForecastRequest):
    forecaster = get_forecaster()
    try:
        profile = request.profile.model_dump(exclude_none=True) if request.profile else None
        return forecaster.predict(request.history, request.forecast_month, profile, request.consumption_limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
