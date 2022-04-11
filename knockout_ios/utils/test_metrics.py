import pandas as pd
import pytest
from pandas import Timestamp

from knockout_ios.utils.constants import *

from knockout_ios.utils.metrics import get_ko_discovery_metrics, find_rejection_rates, calc_available_cases_before_ko, \
    calc_overprocessing_waste, calc_mean_waiting_time_waste_v1

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


def test_available_cases_before_ko_calculation():
    log_df = pd.read_pickle('test_fixtures/log_df.pkl')

    activities = ["Check Liability", "Check Risk", "Check Monthly Income"]

    counts = calc_available_cases_before_ko(activities, log_df)

    assert counts["Check Liability"] == 1000
    assert counts["Check Risk"] == 500
    assert counts["Check Monthly Income"] == 350


def test_mean_waiting_time_waste_v1():
    events = [
        # Case that will be knocked out
        {
            PM4PY_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': 'ko_1',
            'knocked_out_case': True,
            PM4PY_ACTIVITY_COLUMN_NAME: 'Start',
            PM4PY_RESOURCE_COLUMN_NAME: 'R1',
            PM4PY_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 09:00:00"),
            PM4PY_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 09:05:00"),
        },
        {
            PM4PY_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': 'ko_1',
            'knocked_out_case': True,
            PM4PY_ACTIVITY_COLUMN_NAME: 'Work',
            PM4PY_RESOURCE_COLUMN_NAME: 'R1',
            PM4PY_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 09:05:00"),
            PM4PY_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 14:00:00"),
        },
        {
            PM4PY_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': 'ko_1',
            'knocked_out_case': True,
            PM4PY_ACTIVITY_COLUMN_NAME: 'End',
            PM4PY_RESOURCE_COLUMN_NAME: 'R1',
            PM4PY_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 14:00:00"),
            PM4PY_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 14:05:00"),
        },

        # Non knocked out case that has to wait
        {
            PM4PY_CASE_ID_COLUMN_NAME: 1,
            'knockout_activity': False,
            'knocked_out_case': False,
            PM4PY_ACTIVITY_COLUMN_NAME: 'Start',
            PM4PY_RESOURCE_COLUMN_NAME: 'R1',
            PM4PY_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 13:00:00"),
            PM4PY_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 13:05:00"),
        },
        {
            PM4PY_CASE_ID_COLUMN_NAME: 1,
            'knockout_activity': False,
            'knocked_out_case': False,
            PM4PY_ACTIVITY_COLUMN_NAME: 'Work',
            PM4PY_RESOURCE_COLUMN_NAME: 'R1',
            PM4PY_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 13:05:00"),
            PM4PY_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 16:00:00"),
        },
        {
            PM4PY_CASE_ID_COLUMN_NAME: 1,
            'knockout_activity': False,
            'knocked_out_case': False,
            PM4PY_ACTIVITY_COLUMN_NAME: 'End',
            PM4PY_RESOURCE_COLUMN_NAME: 'R1',
            PM4PY_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 16:00:00"),
            PM4PY_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 16:05:00"),
        }

    ]

    ko_activities = ["ko_1"]

    waste = calc_mean_waiting_time_waste_v1(ko_activities, pd.DataFrame(events))

    assert waste["ko_1"] == pytest.approx(3900)  # 65 mins. between 13:00 (start case 1) and 14:05 (finish case 0)
