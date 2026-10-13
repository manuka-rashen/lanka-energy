"""Download original public files and verify their recorded SHA-256 hashes."""
import hashlib
import io
import json
from pathlib import Path
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = {
    'electricity_monthly.csv': 'data/data/consumption_data/non_smart_meter/monthly_consumption.csv',
    'household_information.csv': 'data/data/survey_data/wave_1/w1_household_information_and_history.csv',
    'appliances.csv': 'data/data/survey_data/wave_1/w1_appliances.csv',
    'ac_roster.csv': 'data/data/survey_data/wave_1/w1_ac_roster.csv',
    'survey_dates.csv': 'data/data/survey_data/survey_dates.csv',
}
DOCUMENTATION = {
    'consumption_dictionary.csv': 'https://raw.githubusercontent.com/LIRNEasia/SL_electricity_consumption/main/documentation/consumption/consumption_data_dictionary.csv',
    'wave1_dictionary.csv': 'https://raw.githubusercontent.com/LIRNEasia/SL_electricity_consumption/main/documentation/survey/data_dictionaries/wave_1_data_dictionary.csv',
}


def download():
    dest = ROOT/'data'/'raw'
    dest.mkdir(exist_ok=True)
    expected = json.loads((ROOT/'models'/'manifest.json').read_text(encoding='utf8'))['source_sha256']
    urls = {name: 'https://www.kaggle.com/api/v1/datasets/download/lirneasia/sri-lankan-residential-electricity-consumption/' + urllib.parse.quote(path, safe='') for name,path in SOURCE_PATHS.items()}
    urls.update(DOCUMENTATION)
    for name, url in urls.items():
        path = dest/name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected[name]:
            print(name, 'already verified')
            continue
        request = urllib.request.Request(url, headers={'User-Agent':'LankaEnergyResearch/1.0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read(10_000_001)
        if len(raw) > 10_000_000:
            raise ValueError('Unexpectedly large source file; inspect the dataset release.')
        if raw.startswith(b'PK'):
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                names = [member for member in archive.namelist() if member.endswith('.csv')]
                if len(names) != 1 or archive.getinfo(names[0]).file_size > 10_000_000:
                    raise ValueError('Expected one small CSV file.')
                raw = archive.read(names[0])
        if hashlib.sha256(raw).hexdigest() != expected[name]:
            raise ValueError(f'{name}: source differs from the recorded release; do not silently replace the training data.')
        path.write_bytes(raw)
        print(name, 'downloaded and verified')


if __name__ == '__main__':
    download()
