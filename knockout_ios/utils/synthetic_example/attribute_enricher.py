from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from knockout_ios.utils.constants import *


def enrich_log_df(log_df) -> pd.DataFrame:
    # To keep consistency on every call
    np.random.seed(0)

    log_df = deepcopy(log_df)

    #  Synthetic Example Ground Truth
    #  (K.O. checks and their rejection rules):
    #
    # 'Check Liability':                 'Total Debt'     > 5000 ||  'Owns Vehicle' = False
    # 'Check Risk':                      'Loan Ammount'   > 10000
    # 'Check Monthly Income':            'Monthly Income' < 1000
    # 'Assess application':              'External Risk Score' > 0.3
    # 'Aggregated Risk Score Check':     'Aggregated Risk Score' > 0.5

    # First populate df with values that don't fall under any of the knock outs' rules

    demographic_values = ['demographic_type_1', 'demographic_type_2', 'demographic_type_3']

    log_df.set_index(SIMOD_LOG_READER_CASE_ID_COLUMN_NAME, inplace=True)

    for caseid in tqdm(log_df.index.unique(), desc="Enriching with case attributes"):

        log_df.at[caseid, 'Monthly Income'] = np.random.randint(1000,
                                                                4999)
        log_df.at[caseid, 'Total Debt'] = np.random.randint(0, 4999)
        log_df.at[caseid, 'Loan Ammount'] = np.random.randint(0, 9999)
        log_df.at[caseid, 'Owns Vehicle'] = True
        log_df.at[caseid, 'Demographic'] = np.random.choice(
            demographic_values)
        log_df.at[caseid, 'External Risk Score'] = np.random.uniform(0,
                                                                     0.3)
        log_df.at[caseid, 'Aggregated Risk Score'] = np.random.uniform(
            0, 0.49)

        ko_activity = log_df.loc[caseid]["knockout_activity"].unique()
        if len(ko_activity) > 0:
            ko_activity = ko_activity[0]

        if ko_activity == 'Check Liability':
            if np.random.uniform(0, 1) < 0.5:
                log_df.at[caseid, 'Total Debt'] = np.random.randint(
                    5000, 30_000)
            else:
                log_df.at[caseid, 'Owns Vehicle'] = False
        elif ko_activity == 'Check Risk':
            log_df.at[caseid, 'Loan Ammount'] = np.random.randint(
                10_000, 30_000)
        elif ko_activity == 'Check Monthly Income':
            log_df.at[caseid, 'Monthly Income'] = np.random.randint(0,
                                                                    999)
        elif ko_activity == 'Assess application':
            log_df.at[caseid, 'External Risk Score'] = np.random.uniform(0.3, 1)
        elif ko_activity == 'Aggregated Risk Score Check':
            log_df.at[caseid, 'Aggregated Risk Score'] = np.random.uniform(0.5, 1)

    log_df.reset_index(inplace=True)
    return log_df


@dataclass
class RuntimeAttribute:
    attribute_name: str
    value_provider_activity: str


def enrich_log_df_with_value_providers(log_df: pd.DataFrame,
                                       runtime_attributes: list[RuntimeAttribute]) -> pd.DataFrame:
    log_df = enrich_log_df(log_df)

    return log_df

    # TODO: implement these attributes known only after certain activity..
    # idea: scan every row, if the activity is not the same as the producer, then set the attribute to None

    # by_case = log_df.sort_values(by=SIMOD_END_TIMESTAMP_COLUMN_NAME).groupby(SIMOD_LOG_READER_CASE_ID_COLUMN_NAME)
    #
    # # for every case in by_case, clear the attribute's value in every event that was before the value provider activity:
    # for case_id, case_df in by_case:
    #     case_df = case_df.sort_values(by=SIMOD_END_TIMESTAMP_COLUMN_NAME)
    #     for runtime_attribute in runtime_attributes:
    #         value_provider_activity = runtime_attribute.value_provider_activity
    #         value_provider_activity_index = \
    #             case_df[case_df[SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME] == value_provider_activity].index[0]
    #         case_df.loc[:value_provider_activity_index, runtime_attribute.attribute_name] = None
    #
    # return by_case.to_frame()
