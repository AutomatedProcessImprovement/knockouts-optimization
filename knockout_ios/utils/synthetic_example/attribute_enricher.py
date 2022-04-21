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

    if SIMOD_LOG_READER_CASE_ID_COLUMN_NAME in log_df.columns:
        log_df.set_index(SIMOD_LOG_READER_CASE_ID_COLUMN_NAME, inplace=True)
    elif PM4PY_CASE_ID_COLUMN_NAME in log_df.columns:
        log_df.set_index(PM4PY_CASE_ID_COLUMN_NAME, inplace=True)

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


def enrich_log_df_with_masked_attributes(log_df: pd.DataFrame,
                                         runtime_attributes: list[RuntimeAttribute]) -> pd.DataFrame:
    """
    Emulates attributes known only after certain activity
    """

    log_df = enrich_log_df(log_df)

    activity_col = SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME

    if SIMOD_LOG_READER_CASE_ID_COLUMN_NAME in log_df.columns:
        log_df.set_index(SIMOD_LOG_READER_CASE_ID_COLUMN_NAME, inplace=True)
    elif PM4PY_CASE_ID_COLUMN_NAME in log_df.columns:
        log_df.set_index(PM4PY_CASE_ID_COLUMN_NAME, inplace=True)
        activity_col = PM4PY_ACTIVITY_COLUMN_NAME

    for attribute in runtime_attributes:

        for caseid in tqdm(log_df.index.unique(), desc="Masking case attributes"):
            case = log_df.loc[caseid]

            if attribute.value_provider_activity not in case[activity_col].unique():
                continue

            case = case.sort_values(by="start_timestamp", ascending=True)

            value_producer_end = case[case['task'] == attribute.value_provider_activity]['end_timestamp'][0]

            log_df.at[caseid, attribute.attribute_name] = np.where(case['start_timestamp'] >= value_producer_end,
                                                                   case[attribute.attribute_name],
                                                                   np.nan)

    log_df.reset_index(inplace=True)
    return log_df
