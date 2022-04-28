from typing import List

import numpy as np
import pandas as pd
from datetimerange import DateTimeRange
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


def calc_available_cases_before_ko(ko_activities: List[str], log_df: pd.DataFrame):
    counts = {}

    # group log_df by caseid and for each activity count how many groups (i.e. cases) contain that activity
    for activity in ko_activities:
        counts[activity] = log_df[log_df[SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME] == activity].groupby(
            SIMOD_LOG_READER_CASE_ID_COLUMN_NAME).size().sum()

    return counts


def calc_processing_waste(ko_activities: List[str], log_df: pd.DataFrame):
    counts = {}

    # Approximation, only adding durations of all activities of knocked out cases per ko activity.
    # does not take into account idle time due to resource timetables
    for activity in ko_activities:
        filtered_df = log_df[log_df['knockout_activity'] == activity]
        total_duration = filtered_df[DURATION_COLUMN_NAME].sum()
        counts[activity] = total_duration

    return counts


def calc_overprocessing_waste(ko_activities: List[str], log_df: pd.DataFrame):
    counts = {}

    # Basic Cycle time calculation: end time of last activity of a case - start time of first activity of a case
    for activity in ko_activities:
        filtered_df = log_df[log_df['knockout_activity'] == activity]
        aggr = filtered_df.groupby(PM4PY_CASE_ID_COLUMN_NAME).agg(
            {PM4PY_START_TIMESTAMP_COLUMN_NAME: 'min', PM4PY_END_TIMESTAMP_COLUMN_NAME: 'max'})
        total_duration = aggr[PM4PY_END_TIMESTAMP_COLUMN_NAME] - aggr[PM4PY_START_TIMESTAMP_COLUMN_NAME]

        counts[activity] = total_duration.sum().total_seconds()

    return counts


def calc_overlapping_time_ko_and_non_ko(ko_activities: List[str], log_df: pd.DataFrame):
    """
    - computes overlap between events of ko case and non-ko case,
        not yet overlap between ko case and "empty spaces" between non-ko case events
    - slow, even with swifter - DateTimeRange package comparison slows it down
    """
    waste = {}

    # Aggregate by case and take into account most frequent resource;
    # this allows just a rough estimate, because at the event level, resource varies.
    # Here we just check for resource contention based on the most common resource per case.
    # log_df = log_df.groupby(PM4PY_CASE_ID_COLUMN_NAME).agg(
    #     {PM4PY_START_TIMESTAMP_COLUMN_NAME: 'min', PM4PY_END_TIMESTAMP_COLUMN_NAME: 'max',
    #      'knockout_activity': 'first', '@@startevent_Resource': lambda x: x.value_counts().index[0]})

    for activity in ko_activities:
        waste[activity] = 0

        knocked_out_case_events = log_df[log_df['knockout_activity'] == activity]
        non_knocked_out_case_events = log_df[log_df['knockout_activity'] == False]

        def compute_overlaps(non_ko_case_event):
            # Get the overlapping time between non_ko_case_event and every knocked_out_case by the current activity
            # that shares the same resource

            resource = non_ko_case_event[PM4PY_RESOURCE_COLUMN_NAME]

            knocked_out = knocked_out_case_events[
                (knocked_out_case_events[PM4PY_RESOURCE_COLUMN_NAME] == resource)]

            total_overlap = 0
            time_range1 = DateTimeRange(
                non_ko_case_event[PM4PY_START_TIMESTAMP_COLUMN_NAME],
                non_ko_case_event[PM4PY_END_TIMESTAMP_COLUMN_NAME]
            )

            for knocked_out_case_event in knocked_out.iterrows():
                time_range2 = DateTimeRange(
                    knocked_out_case_event[1][PM4PY_START_TIMESTAMP_COLUMN_NAME],
                    knocked_out_case_event[1][PM4PY_END_TIMESTAMP_COLUMN_NAME]
                )

                try:
                    total_overlap += time_range1.intersection(time_range2).timedelta.total_seconds()
                except TypeError:  # like this we save up 1 call to is_intersection()
                    continue

            return total_overlap

        overlaps = non_knocked_out_case_events.swifter.progress_bar(False).apply(compute_overlaps, axis=1)
        waste[activity] = overlaps.sum()

    return waste


# TODO: extremely slow, check for possible dup index problem
def calc_waiting_time_waste_v2(ko_activities: List[str], log_df: pd.DataFrame):
    waste = {activity: 0 for activity in ko_activities}

    if not (PM4PY_RESOURCE_COLUMN_NAME in log_df.columns):
        print(f"The log does not contain column {PM4PY_RESOURCE_COLUMN_NAME} (to identify resources)")
        return waste

    log_df = log_df.sort_values(by=[PM4PY_END_TIMESTAMP_COLUMN_NAME])
    log_df.set_index(PM4PY_CASE_ID_COLUMN_NAME, inplace=True)

    for caseid in log_df.index.unique():
        log_df.loc[caseid, "next_activity_start"] = \
            (log_df.loc[caseid][PM4PY_START_TIMESTAMP_COLUMN_NAME]).shift(-1)

    log_df.reset_index(inplace=True)

    non_knocked_out_case_events = log_df[log_df['knockout_activity'] == False]

    for activity in ko_activities:
        waste[activity] = 0

        # consider only events knocked out by the current activity
        knocked_out_case_events = log_df[log_df['knockout_activity'] == activity]

        for resource in knocked_out_case_events[PM4PY_RESOURCE_COLUMN_NAME].unique():
            if pd.isnull(resource):
                continue

            # consider only events that share the same resource
            knocked_out = knocked_out_case_events[knocked_out_case_events[PM4PY_RESOURCE_COLUMN_NAME] == resource]
            non_knocked_out = non_knocked_out_case_events[
                non_knocked_out_case_events[PM4PY_RESOURCE_COLUMN_NAME] == resource]

            def compute_overlaps(non_ko_case_event):

                total_overlap = 0

                # handle these cases:
                # - value of NaT in last activity of every case
                # - cases with no "idle time" between its activities
                if (pd.isnull(non_ko_case_event["next_activity_start"])) or (
                        non_ko_case_event["next_activity_start"] <= non_ko_case_event[PM4PY_END_TIMESTAMP_COLUMN_NAME]):
                    return total_overlap

                # Get the overlapping time between idle intervals of non_ko_case_event and knocked out cases

                idle_time = DateTimeRange(
                    non_ko_case_event[PM4PY_END_TIMESTAMP_COLUMN_NAME],
                    non_ko_case_event["next_activity_start"]
                )

                for knocked_out_case_event in knocked_out.iterrows():
                    time_range2 = DateTimeRange(
                        knocked_out_case_event[1][PM4PY_START_TIMESTAMP_COLUMN_NAME],
                        knocked_out_case_event[1][PM4PY_END_TIMESTAMP_COLUMN_NAME]
                    )

                    try:
                        total_overlap += idle_time.intersection(time_range2).timedelta.total_seconds()
                    except TypeError:  # like this we save up 1 call to is_intersection()
                        continue

                return total_overlap

            overlaps = non_knocked_out.swifter.progress_bar(False).apply(compute_overlaps, axis=1)
            waste[activity] = overlaps.sum()
            continue

    return waste
