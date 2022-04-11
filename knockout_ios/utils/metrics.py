import pandas as pd
from numpy import nan
from wittgenstein.abstract_ruleset_classifier import AbstractRulesetClassifier

from knockout_ios.utils.constants import *

import swifter


def find_rejection_rates(log_df, ko_activities):
    # for every ko_activity,
    # P = how many cases were knocked out by it / how many cases contain it
    knock_out_counts_by_activity = log_df.drop_duplicates(subset=SIMOD_LOG_READER_CASE_ID_COLUMN_NAME) \
        .groupby('knockout_activity') \
        .count()[SIMOD_LOG_READER_CASE_ID_COLUMN_NAME]

    rates = {}
    for activity in ko_activities:
        cases_knocked_out_by_activity = knock_out_counts_by_activity.get(activity)
        cases_containing_activity = log_df[log_df[SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME] == activity] \
            .drop_duplicates(subset=SIMOD_LOG_READER_CASE_ID_COLUMN_NAME) \
            .count()[SIMOD_LOG_READER_CASE_ID_COLUMN_NAME]

        rates[activity] = round(cases_knocked_out_by_activity / cases_containing_activity, 3)

    return rates


def get_ko_discovery_metrics(activities, expected_kos, computed_kos):
    # Source: https://towardsdatascience.com/evaluating-categorical-models-e667e17987fd

    total_observations = len(activities)

    if total_observations == 0:
        raise Exception("No activities provided")

    # Compute components of metrics

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for act in activities:
        if (act not in computed_kos) and not (act in expected_kos):
            true_negatives += 1
        elif (act not in computed_kos) and (act in expected_kos):
            false_negatives += 1

    for ko in computed_kos:
        if ko in expected_kos:
            true_positives += 1
        else:
            false_positives += 1

    # Compute metrics, with care for divisions by zero

    accuracy = (true_positives + true_negatives) / total_observations

    if (true_positives + false_positives) == 0:
        precision = 1
    else:
        precision = true_positives / (true_positives + false_positives)

    if (true_positives + false_negatives) == 0:
        recall = 1
    else:
        recall = true_positives / (true_positives + false_negatives)

    if (precision + recall) == 0:
        f1_score = nan
    else:
        f1_score = 2 * ((precision * recall) / (precision + recall))

    return {'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'confusion_matrix': {
                'true_positives': true_positives,
                'false_positives': false_positives,
                'true_negatives': true_negatives,
                'false_negatives': false_negatives
            }}


def calc_knockout_ruleset_support(activity: str, ruleset_model: AbstractRulesetClassifier, log: pd.DataFrame,
                                  available_cases_before_ko: int,
                                  processed_with_pandas_dummies=False):
    predicted_ko = ruleset_model.predict(log)
    log['predicted_ko'] = predicted_ko

    if processed_with_pandas_dummies:
        correct_predictions = \
            log[(log['predicted_ko']) & (log[f'knockout_activity_{activity}'])].shape[0]
    else:
        correct_predictions = log[(log['predicted_ko']) & (log['knockout_activity'] == activity)].shape[0]

    if available_cases_before_ko == 0:
        return 0

    support = correct_predictions / available_cases_before_ko

    return support


def calc_knockout_ruleset_confidence(activity: str, ruleset_model: AbstractRulesetClassifier, log: pd.DataFrame,
                                     processed_with_pandas_dummies=False):
    predicted_ko = ruleset_model.predict(log)
    log['predicted_ko'] = predicted_ko

    if processed_with_pandas_dummies:
        correct_predictions = \
            log[(log['predicted_ko']) & (log[f'knockout_activity_{activity}'])].shape[0]
    else:
        correct_predictions = log[(log['predicted_ko']) & (log['knockout_activity'] == activity)].shape[0]
    total_predictions = sum(predicted_ko)

    if total_predictions == 0:
        return 0

    confidence = correct_predictions / total_predictions

    return confidence


def calc_available_cases_before_ko(ko_activities, log_df):
    counts = {}

    # group log_df by caseid and for each activity count how many groups (i.e. cases) contain that activity
    for activity in ko_activities:
        counts[activity] = log_df[log_df[SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME] == activity].groupby(
            SIMOD_LOG_READER_CASE_ID_COLUMN_NAME).size().sum()

    return counts


def calc_processing_waste(ko_activities, log_df):
    counts = {}

    # Approximation, only adding durations of all activities of knocked out cases per ko activity.
    # does not take into account idle time due to resource timetables
    for activity in ko_activities:
        filtered_df = log_df[log_df['knockout_activity'] == activity]
        total_duration = filtered_df[DURATION_COLUMN_NAME].sum()
        counts[activity] = total_duration

    return counts


def calc_overprocessing_waste(ko_activities, log_df):
    counts = {}

    # Basic Cycle time calculation: end time of last activity of a case - start time of first activity of a case
    for activity in ko_activities:
        filtered_df = log_df[log_df['knockout_activity'] == activity]
        aggr = filtered_df.groupby(PM4PY_CASE_ID_COLUMN_NAME).agg(
            {PM4PY_START_TIMESTAMP_COLUMN_NAME: 'min', PM4PY_END_TIMESTAMP_COLUMN_NAME: 'max'})
        total_duration = aggr[PM4PY_END_TIMESTAMP_COLUMN_NAME] - aggr[PM4PY_START_TIMESTAMP_COLUMN_NAME]

        counts[activity] = total_duration.sum().total_seconds()

    return counts


def calc_mean_waiting_time_waste(ko_activities, log_df):
    mean_times = {}

    for activity in ko_activities:
        mean_times[activity] = 0

        log_df = log_df.groupby(PM4PY_CASE_ID_COLUMN_NAME).agg(
            {PM4PY_START_TIMESTAMP_COLUMN_NAME: 'min', PM4PY_END_TIMESTAMP_COLUMN_NAME: 'max',
             'knockout_activity': 'first'})

        # get all rows where start_timestamp is before start_timestamp of any row in aggr_filtered
        knocked_out_cases = log_df[log_df['knockout_activity'] == activity]
        non_knocked_out_cases = log_df[log_df['knockout_activity'] == False]

        def fn(non_ko_case):
            # sum the overlapping time between non_ko_case and every row in kod
            for knocked_out_case in knocked_out_cases.iterrows():

                ko_starts_before_or_with_non_ko_case = knocked_out_case[1][PM4PY_START_TIMESTAMP_COLUMN_NAME] <= \
                                                       non_ko_case[PM4PY_START_TIMESTAMP_COLUMN_NAME]

                if ko_starts_before_or_with_non_ko_case:
                    overlap = (
                            min(non_ko_case[PM4PY_END_TIMESTAMP_COLUMN_NAME],
                                knocked_out_case[1][PM4PY_END_TIMESTAMP_COLUMN_NAME])
                            - max(non_ko_case[PM4PY_START_TIMESTAMP_COLUMN_NAME],
                                  knocked_out_case[1][PM4PY_START_TIMESTAMP_COLUMN_NAME])
                    )

                    return max(0, overlap.total_seconds())

        overlaps = non_knocked_out_cases.swifter.apply(fn, axis=1)
        mean_times[activity] = overlaps.sum()

    return mean_times
