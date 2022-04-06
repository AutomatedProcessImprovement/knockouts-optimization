import pandas as pd
import pytest

from knockout_ios.utils.constants import *

from knockout_ios.utils.metrics import get_ko_discovery_metrics, find_rejection_rates

log = [
    # 1 Knocked out case (contains check_A and did not pass it)
    {
        SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
        SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: "activity_A",
        'knockout_activity': "check_A",
    },
    {
        SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
        SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: "activity_B",
        'knockout_activity': "check_A"
    },
    {
        SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
        SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: "check_A",
        'knockout_activity': "check_A"
    },

    # 1 Non-knocked out case (contains check_A but passed it)
    {
        SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 1,
        SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: "activity_C",
        'knockout_activity': False,
    },
    {
        SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 1,
        SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: "activity_D",
        'knockout_activity': False
    },
    {
        SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 1,
        SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: "check_A",
        'knockout_activity': False
    },
    # Check done twice
    {
        SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 1,
        SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: "check_A",
        'knockout_activity': False
    }
]

log = pd.DataFrame(log)
ko_activities = ['check_A']


def test_single_ko_activity():
    rates = find_rejection_rates(log, ko_activities)
    assert pytest.approx(rates['check_A']) == 0.5


def test_no_kos():
    assert len(find_rejection_rates(log, []).keys()) == 0


def test_correct_kos():
    activities = ['Start', 'ko_1', 'not_ko_1', 'not_ko_2', 'ko_2']
    expected_kos = ['ko_1', 'ko_2']
    computed_kos = ['ko_1', 'ko_2']
    metrics = get_ko_discovery_metrics(activities, expected_kos, computed_kos)

    conf_matrix = metrics['confusion_matrix']

    assert conf_matrix['true_positives'] == 2
    assert conf_matrix['false_positives'] == 0
    assert conf_matrix['true_negatives'] == 3
    assert conf_matrix['false_negatives'] == 0


def test_wrong_kos():
    activities = ['Start', 'ko_1', 'not_ko_1', 'not_ko_2', 'ko_2']
    expected_kos = ['ko_1', 'ko_2']
    computed_kos = ['Start', 'not_ko_1', 'not_ko_2']
    metrics = get_ko_discovery_metrics(activities, expected_kos, computed_kos)

    conf_matrix = metrics['confusion_matrix']

    assert conf_matrix['true_positives'] == 0
    assert conf_matrix['false_positives'] == 3
    assert conf_matrix['true_negatives'] == 0
    assert conf_matrix['false_negatives'] == 2


def test_partially_correct_kos():
    activities = ['Start', 'ko_1', 'not_ko_1', 'not_ko_2', 'ko_2']
    expected_kos = ['ko_1', 'ko_2']
    computed_kos = ['ko_1', 'ko_2', 'not_ko_1']
    metrics = get_ko_discovery_metrics(activities, expected_kos, computed_kos)

    conf_matrix = metrics['confusion_matrix']

    assert conf_matrix['true_positives'] == 2
    assert conf_matrix['false_positives'] == 1
    assert conf_matrix['true_negatives'] == 2
    assert conf_matrix['false_negatives'] == 0
