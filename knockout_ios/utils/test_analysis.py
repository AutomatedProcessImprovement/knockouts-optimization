import pandas as pd
import pytest

from knockout_ios.utils.constants import *

from knockout_ios.utils.analysis import find_rejection_rates

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
