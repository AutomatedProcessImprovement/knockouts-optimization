import pandas as pd
from numpy import nan
from wittgenstein.abstract_ruleset_classifier import AbstractRulesetClassifier

from knockout_ios.utils.constants import *


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
                                  dummies=False):
    predicted_ko = ruleset_model.predict(log)
    log['predicted_ko'] = predicted_ko

    if dummies:
        freq_x_and_y = \
            log[(log['predicted_ko']) & (log[f'knockout_activity_{activity}'])].shape[0]
    else:
        freq_x_and_y = log[(log['predicted_ko']) & (log['knockout_activity'] == activity)].shape[0]

    N = log.shape[0]

    if N == 0:
        return 0

    support = freq_x_and_y / N

    return support


def calc_knockout_ruleset_confidence(activity: str, ruleset_model: AbstractRulesetClassifier, log: pd.DataFrame,
                                     dummies=False):
    predicted_ko = ruleset_model.predict(log)
    log['predicted_ko'] = predicted_ko

    if dummies:
        freq_x_and_y = \
            log[(log['predicted_ko']) & (log[f'knockout_activity_{activity}'])].shape[0]
    else:
        freq_x_and_y = log[(log['predicted_ko']) & (log['knockout_activity'] == activity)].shape[0]
    freq_x = sum(predicted_ko)

    if freq_x == 0:
        return 0

    confidence = freq_x_and_y / freq_x

    return confidence
