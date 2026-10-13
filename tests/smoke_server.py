"""Check the committed model through the production Linux startup command."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.platform == 'win32':
        raise SystemExit('This hosting check runs on Linux, matching Azure App Service.')
    port = '8765'
    base = f'http://127.0.0.1:{port}'
    manifest = json.loads((ROOT / 'models' / 'manifest.json').read_text())
    metrics = json.loads((ROOT / 'results' / 'metrics.json').read_text())
    request_data = json.loads((ROOT / 'models' / 'examples.json').read_text())[0]
    request_data.pop('name', None)
    with (ROOT / 'server-smoke.log').open('w') as log:
        process = subprocess.Popen(['sh', 'startup.sh'], cwd=ROOT,
                                   env={**os.environ, 'PORT': port}, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 45
            while True:
                try:
                    with urlopen(base + '/health', timeout=3) as response:
                        health = json.load(response)
                    break
                except (URLError, TimeoutError):
                    if process.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError('Production startup did not become healthy.')
                    time.sleep(0.5)
            assert health['status'] == 'ok'
            assert health['model'] == manifest['selected']['name'] == metrics['selected']
            with urlopen(base, timeout=5) as response:
                assert response.status == 200 and 'Lanka Energy' in response.read().decode()
            with urlopen(base + '/api/metadata', timeout=5) as response:
                metadata = json.load(response)
            assert metadata['test']['mae_kwh'] == metrics['test_results']['test']['mae_kwh']
            request = Request(base + '/api/forecast', data=json.dumps(request_data).encode(),
                              headers={'Content-Type': 'application/json'}, method='POST')
            with urlopen(request, timeout=10) as response:
                result = json.load(response)
            assert math.isfinite(result['prediction_kwh']) and result['prediction_kwh'] >= 0
            assert result['lower_kwh'] <= result['prediction_kwh'] <= result['upper_kwh']
            print('Committed model served successfully through the Azure startup command.')
            print(json.dumps({'model': health['model'], 'prediction_kwh': result['prediction_kwh'],
                              'packaged_test_mae_kwh': metadata['test']['mae_kwh']}))
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == '__main__':
    main()
