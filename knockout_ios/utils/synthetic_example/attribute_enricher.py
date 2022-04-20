from copy import deepcopy
from dataclasses import dataclass
from random import random, randint, seed

import numpy as np
import pandas as pd
import pm4py

from knockout_ios.utils.constants import *


def enrich_log_df(log_df):
    # To keep consistency on every call
    seed(0)
    np.random.seed(0)

    log_df = deepcopy(log_df)

    demographic_values = ['demographic_type_1', 'demographic_type_2', 'demographic_type_3']

    # First populate df with values that don't fall under any of the knock outs' rules
    log_df['Monthly Income'] = np.random.randint(1000, 4999, size=len(log_df))
    log_df['Total Debt'] = np.random.randint(0, 4999, size=len(log_df))
    log_df['Loan Ammount'] = np.random.randint(0, 9999, size=len(log_df))
    log_df['Owns Vehicle'] = True
    log_df['Demographic'] = np.random.choice(demographic_values, size=len(log_df))
    log_df['External Risk Score'] = np.random.uniform(0, 0.3, size=len(log_df))

    #  Synthetic Example Ground Truth
    #  (K.O. checks and their rejection rules):
    #
    # 'Check Liability':        'Total Debt'     > 5000 ||  'Owns Vehicle' = False
    # 'Check Risk':             'Loan Ammount'   > 10000
    # 'Check Monthly Income':   'Monthly Income' < 1000
    # 'Assess application':     'External Risk Score' > 0.3

    # TODO: this is slow, consider to_records()
    for i, row in log_df.iterrows():
        ko_activity = row["knockout_activity"]

        if ko_activity == 'Check Liability':
            if random() < 0.5:
                log_df.loc[i, 'Total Debt'] = randint(5000, 30_000)
            else:
                log_df.loc[i, 'Owns Vehicle'] = False
        elif ko_activity == 'Check Risk':
            log_df.loc[i, 'Loan Ammount'] = randint(10_000, 30_000)
        elif ko_activity == 'Check Monthly Income':
            log_df.loc[i, 'Monthly Income'] = randint(0, 999)
        elif ko_activity == 'Assess application':
            log_df.loc[i, 'External Risk Score'] = random() + 0.3

    # TODO: fix problem with exported .xes; fluxicon disco complains about lack of activity classifier,
    #  apromore does not even recognize it
    # formatted = pm4py.format_dataframe(log_df, case_id=SIMOD_LOG_READER_CASE_ID_COLUMN_NAME,
    #                                    activity_key=SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME,
    #                                    timestamp_key=SIMOD_END_TIMESTAMP_COLUMN_NAME,
    #                                    start_timestamp_key=SIMOD_START_TIMESTAMP_COLUMN_NAME)

    # pm4py.write_xes(formatted, '../../inputs/synthetic_example/synthetic_example_enriched.xes')

    # Another way: convert to EventLog object and then try to export (problem missing activity classifier)
    # formatted = log_converter.apply(formatted, variant=log_converter.Variants.TO_EVENT_LOG)
    # formatted.classifiers["concept:name"] = "concept:name"

    # pm4py.write_xes(formatted, './synthetic_example_enriched.xes')

    return log_df


@dataclass
class RuntimeAttribute:
    attribute_name: str
    value_provider_activity: str


def enrich_log_df_with_value_providers(log_df: pd.DataFrame, runtime_attributes: list[RuntimeAttribute]):
    log_df = enrich_log_df(log_df)
    by_case = log_df.sort_values(by=SIMOD_END_TIMESTAMP_COLUMN_NAME).groupby(SIMOD_LOG_READER_CASE_ID_COLUMN_NAME)

    # for every case in by_case, clear the attribute's value in every event that was before the value provider activity:
    for case_id, case_df in by_case:
        case_df = case_df.sort_values(by=SIMOD_END_TIMESTAMP_COLUMN_NAME)
        for runtime_attribute in runtime_attributes:
            value_provider_activity = runtime_attribute.value_provider_activity
            value_provider_activity_index = \
                case_df[case_df[SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME] == value_provider_activity].index[0]
            case_df.loc[:value_provider_activity_index, runtime_attribute.attribute_name] = None
