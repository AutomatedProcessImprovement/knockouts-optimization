from copy import deepcopy
from random import random, randint

import numpy as np
import pandas as pd
import pm4py

from knockout_ios.utils.constants import *


def enrich_log_df(log_df):
    # log_df = pd.read_pickle('../../inputs/synthetic_example_raw_df.pkl')

    log_df = deepcopy(log_df)

    demographic_values = ['demographic_type_1', 'demographic_type_2', 'demographic_type_3']
    vehicle_values = ['vehicle_type_1', 'vehicle_type_2', 'vehicle_type_3', 'vehicle_type_4']

    # First populate df with values that don't fall under any of the knock outs' rules
    log_df['Monthly Income'] = np.random.randint(1000, 4999, size=len(log_df))
    log_df['Total Debt'] = np.random.randint(0, 4999, size=len(log_df))
    log_df['Loan Ammount'] = np.random.randint(0, 9999, size=len(log_df))
    log_df['Vehicle Owned'] = np.random.choice(vehicle_values, size=len(log_df))
    log_df['Demographic'] = np.random.choice(demographic_values, size=len(log_df))
    log_df['External Risk Score'] = np.random.uniform(0, 0.3, size=len(log_df))

    #  Synthetic Example Ground Truth
    #  (K.O. checks and their rejection rules):
    #
    # 'Check Liability':        'Total Debt'     > 5000 ||  'Vehicle Owned' = "N/A"
    # 'Check Risk':             'Loan Ammount'   > 10000
    # 'Check Monthly Income':   'Monthly Income' < 1000
    # 'Assess application':     'External Risk Score' < 0.3

    for i, row in log_df.iterrows():
        ko_activity = row["knockout_activity"]

        if ko_activity == 'Check Liability':
            if random() < 0.5:
                log_df.loc[i, 'Total Debt'] = randint(5000, 30_000)
            else:
                log_df.loc[i, 'Vehicle Owned'] = np.nan
        elif ko_activity == 'Check Risk':
            log_df.loc[i, 'Loan Ammount'] = randint(10_000, 30_000)
        elif ko_activity == 'Check Monthly Income':
            log_df.loc[i, 'Monthly Income'] = randint(0, 999)
        elif ko_activity == 'Assess application':
            log_df.loc[i, 'External Risk Score'] = random() + 0.3

    # Verify that conditions hold
    # assert log_df[log_df['knockout_activity'] == "Check Monthly Income"]['Monthly Income'].min() <= 1000
    # assert log_df[log_df['knockout_activity'] == "Check Risk"]['Loan Ammount'].min() >= 10000
    # assert log_df[log_df['knockout_activity'] == "Assess application"]['External Risk Score'].min() >= 0.3
    # assert that the rows where knockout_activity is Check Liability and Total Debt < 5000, have Vehicle Owned = np.nan
    # assert log_df[(log_df['knockout_activity'] == "Check Liability") & (log_df['Total Debt'] <= 5000)][
    #    'Vehicle Owned'].isna().all()
    # assert that the rows where knockout_activity is Check Liability and Vehicle Owned is not np.nan, have Total Debt > 5000
    # assert log_df[(log_df['knockout_activity'] == "Check Liability") & (log_df['Vehicle Owned'].notna())][
    #           'Total Debt'].max() > 5000

    # formatted = pm4py.format_dataframe(log_df, case_id=SIMOD_LOG_READER_CASE_ID_COLUMN_NAME,
    #                                    activity_key=SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME,
    #                                    timestamp_key=SIMOD_END_TIMESTAMP_COLUMN_NAME,
    #                                    start_timestamp_key=SIMOD_START_TIMESTAMP_COLUMN_NAME)

    # TODO: fix problem with exported .xes; fluxicon disco complains about lack of activity classifier,
    #  apromore does not even recognize it
    # pm4py.write_xes(formatted, '../../inputs/synthetic_example_enriched.xes')

    return log_df
