import pandas as pd
import pytest
from pandas import Timestamp

from knockout_ios.utils.constants import globalColumnNames

from knockout_ios.utils.metrics import get_ko_discovery_metrics, find_rejection_rates, calc_available_cases_before_ko, \
    calc_overlapping_time_ko_and_non_ko, calc_waiting_time_waste_parallel
from knockout_ios.utils.platform_check import is_windows

log = [
    # 1 Knocked out case (contains check_A and did not pass it)
    {
        globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
        globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: "activity_A",
        'knockout_activity': "check_A",
    },
    {
        globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
        globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: "activity_B",
        'knockout_activity': "check_A"
    },
    {
        globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
        globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: "check_A",
        'knockout_activity': "check_A"
    },

    # 1 Non-knocked out case (contains check_A but passed it)
    {
        globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 1,
        globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: "activity_C",
        'knockout_activity': False,
    },
    {
        globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 1,
        globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: "activity_D",
        'knockout_activity': False
    },
    {
        globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 1,
        globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: "check_A",
        'knockout_activity': False
    },
    # Check done twice
    {
        globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 1,
        globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: "check_A",
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
    if not is_windows():
        return

    log_df = pd.read_pickle('test/test_fixtures/log_df.pkl')

    activities = ["Check Liability", "Check Risk", "Check Monthly Income"]

    counts = calc_available_cases_before_ko(activities, log_df)

    assert counts["Check Liability"] == 1000
    assert counts["Check Risk"] == 500
    assert counts["Check Monthly Income"] == 350


def test_overlapping_time_ko_and_non_ko():
    events = [
        # Case that will be knocked out
        {
            globalColumnNames.PM4PY_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': 'ko_1',
            'knocked_out_case': True,
            globalColumnNames.PM4PY_ACTIVITY_COLUMN_NAME: 'Start',
            globalColumnNames.PM4PY_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.PM4PY_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 09:00:00"),
            globalColumnNames.PM4PY_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 09:05:00"),
        },
        {
            globalColumnNames.PM4PY_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': 'ko_1',
            'knocked_out_case': True,
            globalColumnNames.PM4PY_ACTIVITY_COLUMN_NAME: 'Work',
            globalColumnNames.PM4PY_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.PM4PY_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 09:05:00"),
            globalColumnNames.PM4PY_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 14:00:00"),
        },
        {
            globalColumnNames.PM4PY_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': 'ko_1',
            'knocked_out_case': True,
            globalColumnNames.PM4PY_ACTIVITY_COLUMN_NAME: 'End',
            globalColumnNames.PM4PY_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.PM4PY_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 14:00:00"),
            globalColumnNames.PM4PY_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 14:05:00"),
        },

        # Non knocked out case that has to wait
        {
            globalColumnNames.PM4PY_CASE_ID_COLUMN_NAME: 1,
            'knockout_activity': False,
            'knocked_out_case': False,
            globalColumnNames.PM4PY_ACTIVITY_COLUMN_NAME: 'Start',
            globalColumnNames.PM4PY_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.PM4PY_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 13:00:00"),
            globalColumnNames.PM4PY_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 13:05:00"),
        },
        {
            globalColumnNames.PM4PY_CASE_ID_COLUMN_NAME: 1,
            'knockout_activity': False,
            'knocked_out_case': False,
            globalColumnNames.PM4PY_ACTIVITY_COLUMN_NAME: 'Work',
            globalColumnNames.PM4PY_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.PM4PY_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 13:05:00"),
            globalColumnNames.PM4PY_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 16:00:00"),
        },
        {
            globalColumnNames.PM4PY_CASE_ID_COLUMN_NAME: 1,
            'knockout_activity': False,
            'knocked_out_case': False,
            globalColumnNames.PM4PY_ACTIVITY_COLUMN_NAME: 'End',
            globalColumnNames.PM4PY_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.PM4PY_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 16:00:00"),
            globalColumnNames.PM4PY_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 16:05:00"),
        }

    ]

    ko_activities = ["ko_1"]

    waste = calc_overlapping_time_ko_and_non_ko(ko_activities, pd.DataFrame(events))

    # 65 mins. between 13:00 (start case 1) and 14:05 (finish case 0)
    assert waste["ko_1"] == pytest.approx(3900)


def test_waiting_time_waste_v2_easy():
    events = [
        # Case that will be knocked out
        {
            globalColumnNames.PM4PY_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': 'ko_1',
            'knocked_out_case': True,
            globalColumnNames.PM4PY_ACTIVITY_COLUMN_NAME: 'Start',
            globalColumnNames.PM4PY_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.PM4PY_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 09:00:00"),
            globalColumnNames.PM4PY_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 09:05:00"),
        },
        {
            globalColumnNames.PM4PY_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': 'ko_1',
            'knocked_out_case': True,
            globalColumnNames.PM4PY_ACTIVITY_COLUMN_NAME: 'Work',
            globalColumnNames.PM4PY_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.PM4PY_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 09:05:00"),
            globalColumnNames.PM4PY_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 14:00:00"),
        },
        {
            globalColumnNames.PM4PY_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': 'ko_1',
            'knocked_out_case': True,
            globalColumnNames.PM4PY_ACTIVITY_COLUMN_NAME: 'End',
            globalColumnNames.PM4PY_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.PM4PY_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 14:00:00"),
            globalColumnNames.PM4PY_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 14:05:00"),
        },

        # Non knocked out case that has to wait
        {
            globalColumnNames.PM4PY_CASE_ID_COLUMN_NAME: 1,
            'knockout_activity': False,
            'knocked_out_case': False,
            globalColumnNames.PM4PY_ACTIVITY_COLUMN_NAME: 'Start',
            globalColumnNames.PM4PY_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.PM4PY_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 10:00:00"),
            globalColumnNames.PM4PY_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 10:05:00"),
        },
        {
            globalColumnNames.PM4PY_CASE_ID_COLUMN_NAME: 1,
            'knockout_activity': False,
            'knocked_out_case': False,
            globalColumnNames.PM4PY_ACTIVITY_COLUMN_NAME: 'Work',
            globalColumnNames.PM4PY_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.PM4PY_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 14:05:00"),
            globalColumnNames.PM4PY_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 16:00:00"),
        },
        {
            globalColumnNames.PM4PY_CASE_ID_COLUMN_NAME: 1,
            'knockout_activity': False,
            'knocked_out_case': False,
            globalColumnNames.PM4PY_ACTIVITY_COLUMN_NAME: 'End',
            globalColumnNames.PM4PY_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.PM4PY_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 16:00:00"),
            globalColumnNames.PM4PY_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 16:05:00"),
        },

        # Non knocked out case that has to wait
        {
            globalColumnNames.PM4PY_CASE_ID_COLUMN_NAME: 2,
            'knockout_activity': False,
            'knocked_out_case': False,
            globalColumnNames.PM4PY_ACTIVITY_COLUMN_NAME: 'Start',
            globalColumnNames.PM4PY_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.PM4PY_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 10:00:00"),
            globalColumnNames.PM4PY_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 10:05:00"),
        },
        {
            globalColumnNames.PM4PY_CASE_ID_COLUMN_NAME: 2,
            'knockout_activity': False,
            'knocked_out_case': False,
            globalColumnNames.PM4PY_ACTIVITY_COLUMN_NAME: 'Work',
            globalColumnNames.PM4PY_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.PM4PY_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 14:05:00"),
            globalColumnNames.PM4PY_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 16:00:00"),
        },
        {
            globalColumnNames.PM4PY_CASE_ID_COLUMN_NAME: 2,
            'knockout_activity': False,
            'knocked_out_case': False,
            globalColumnNames.PM4PY_ACTIVITY_COLUMN_NAME: 'End',
            globalColumnNames.PM4PY_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.PM4PY_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 16:00:00"),
            globalColumnNames.PM4PY_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 16:05:00"),
        }

    ]

    ko_activities = ["ko_1"]

    waste = calc_waiting_time_waste_parallel(ko_activities, pd.DataFrame(events))

    # at 10:00 work on cases 1 and 2 is started.
    # at 10:05 work on cases 1 and 2 is stopped to work on case 0.
    # at 14:05 work on cases 1 and 2 is resumed.
    # total waiting time waiting for a KO-d case to be finished is 4h, or 28800 s
    assert waste["ko_1"] == pytest.approx(28800)
