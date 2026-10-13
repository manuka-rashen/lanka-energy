# RECON-SL data sources

Downloaded from the public LIRNEasia RECON-SL release on 12 October 2026. CSV bytes are retained unchanged. Local filenames were shortened.

Dataset page: https://www.kaggle.com/datasets/lirneasia/sri-lankan-residential-electricity-consumption

Owner documentation: https://github.com/LIRNEasia/SL_electricity_consumption

Citation: Chanuka Algama, Merl Chandana, Isuruni Fernando, Dinithi Dissanayake, Amanda Ariyaratne, Nipuni Habaragamuwa, Pasindu Ranage, Jesús Ramos and Kasun Amarasinghe. *Residential Electricity Consumption Dataset for Sri Lanka (RECON-SL)*, arXiv:2609.28783 (2026). https://arxiv.org/abs/2609.28783 . Dataset DOI: https://doi.org/10.21227/n1dk-q860 . The dataset was created by LIRNEasia with utility records from LECO and household surveys.

Licence: the dataset paper's datasheet states **Creative Commons Attribution 4.0 International (CC BY 4.0)** for the dataset, allowing reuse and sharing with attribution. See https://creativecommons.org/licenses/by/4.0/ . Local copies preserve original data; engineered features and trained artifacts are project additions and are not the original dataset authors' work.

| Local CSV | Original source path |
|---|---|
| electricity_monthly.csv | data/data/consumption_data/non_smart_meter/monthly_consumption.csv |
| household_information.csv | data/data/survey_data/wave_1/w1_household_information_and_history.csv |
| appliances.csv | data/data/survey_data/wave_1/w1_appliances.csv |
| ac_roster.csv | data/data/survey_data/wave_1/w1_ac_roster.csv |
| survey_dates.csv | data/data/survey_data/survey_dates.csv |

For the files above, the direct download URL is `https://www.kaggle.com/api/v1/datasets/download/lirneasia/sri-lankan-residential-electricity-consumption/` followed by the URL-encoded original source path.

The consumption dictionary and wave-1 dictionary come from the owner's GitHub documentation:

- https://raw.githubusercontent.com/LIRNEasia/SL_electricity_consumption/main/documentation/consumption/consumption_data_dictionary.csv
- https://raw.githubusercontent.com/LIRNEasia/SL_electricity_consumption/main/documentation/survey/data_dictionaries/wave_1_data_dictionary.csv

The paper defines monthly consumption as **kilowatt-hours (kWh)**; dictionary descriptions containing KW are inconsistent with that definition. Survey dates are parsed as month/day/year. Household identifiers are pseudonymized source keys, excluded from model features and public demo requests. Sampling is concentrated in LECO's western coastal service areas and does not support national representativeness.

Missing general appliance rosters are treated as unknown. Appliance features count recorded roster items; absence of an AC entry is interpreted as no recorded AC unit in the baseline roster. Self-reported appliance usage hours are not used as measured appliance consumption.

`models/manifest.json` contains SHA-256 hashes for every training source CSV.
