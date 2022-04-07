import pytest
import numpy as np

from pathlib import Path

from .pt_cleaning_util import get_processing_time, get_raw_duration, get_log_name

# Example timetable:
# Mon-Fri 09:00 - 17:00
role_timetable = [
    {
        "fromTime": np.datetime64("2021-01-01T09:00:00.000000"),
        "toTime": np.datetime64("2021-01-01T17:00:00.000000"),
        "fromWeekDay": 0,
        "toWeekDay": 0,
    },
    {
        "fromTime": np.datetime64("2021-01-01T09:00:00.000000"),
        "toTime": np.datetime64("2021-01-01T17:00:00.000000"),
        "fromWeekDay": 1,
        "toWeekDay": 1,
    },
    {
        "fromTime": np.datetime64("2021-01-01T09:00:00.000000"),
        "toTime": np.datetime64("2021-01-01T17:00:00.000000"),
        "fromWeekDay": 2,
        "toWeekDay": 2,
    },
    {
        "fromTime": np.datetime64("2021-01-01T09:00:00.000000"),
        "toTime": np.datetime64("2021-01-01T17:00:00.000000"),
        "fromWeekDay": 3,
        "toWeekDay": 3,
    },
    {
        "fromTime": np.datetime64("2021-01-01T09:00:00.000000"),
        "toTime": np.datetime64("2021-01-01T17:00:00.000000"),
        "fromWeekDay": 4,
        "toWeekDay": 4,
    },
]


def test_log_name():
    path = Path("./preprocessing/pt_cleaning/inputs/purchase_example_for_test.xes")
    assert get_log_name(path) == "purchase_example_for_test.xes"


def test_contained():
    """Test an activity contained in same day"""

    # make activity timestamps:
    # start - Fri 10:00
    # end - Fri 15:00
    activity_start = np.datetime64("2021-11-12T10:00:00")
    activity_end = np.datetime64("2021-11-12T15:00:00")

    # Including off-timetable hours: 241_200 seconds
    assert get_raw_duration(activity_start, activity_end) == 18_000

    # Excluding off-timetable hours: 10_800 seconds
    processing_time, _, _, _ = get_processing_time(
        activity_start, activity_end, role_timetable
    )
    assert processing_time == 18_000


def test_weekend():
    """Test an activity with a weekend in between"""

    # make activity timestamps:
    # start - Fri 16:00  (weekday 4)
    # end   - Mon 11:00  (weekday 0)
    activity_start = np.datetime64("2021-11-12T16:00:00")
    activity_end = np.datetime64("2021-11-15T11:00:00")

    # Including off-timetable hours: 241_200 seconds
    assert get_raw_duration(activity_start, activity_end) == 241_200

    # Excluding off-timetable hours: 3 h // 10_800 seconds
    processing_time, _, _, _ = get_processing_time(
        activity_start, activity_end, role_timetable
    )
    assert processing_time == 10_800


def test_weekdays():
    """Test an activity worked over several week-days"""

    # make activity timestamps:
    # start - Mon 15:00  (weekday 0)
    # end   - Fri 15:00  (weekday 4)
    activity_start = np.datetime64("2021-11-15T15:00:00")
    activity_end = np.datetime64("2021-11-19T15:00:00")

    # Including off-timetable hours: 96 hours / 345_600 seconds
    assert get_raw_duration(activity_start, activity_end) == 345_600

    # Excluding off-timetable hours:
    # Mon: 7_200 sec
    # Tue, Wed, Thu: 28_800 sec
    # Fri: 21_600 sec
    processing_time, _, _, _ = get_processing_time(
        activity_start, activity_end, role_timetable
    )
    assert processing_time == 115_200


def test_week_wrap():
    """Test an activity started in one week, ending in same day 2 weeks after"""

    # make activity timestamps:
    # start - Mon 15:00  (weekday 0)
    # end   - Mon 15:00  (weekday 0)
    activity_start = np.datetime64("2021-11-15T15:00:00")
    activity_end = np.datetime64("2021-11-29T15:00:00")

    # Including off-timetable hours: 336 hours
    get_raw_duration(activity_start, activity_end) == 1_209_600

    # Excluding off-timetable hours:
    # Mon: 7_200 sec
    # Tue, Wed, Thu, Fri: 28_800 sec
    # Mon: 21_600 sec
    # all this times 2 (2 weeks wrap)
    processing_time, _, _, _ = get_processing_time(
        activity_start, activity_end, role_timetable
    )
    assert processing_time == 288_000


def test_lunch_break():
    """Test an activity contained in same day, with resource that takes 'lunch break'"""

    # make role timetable:
    # Wed 09:00 - 13:00
    # Wed 14:00 - 17:00
    timetable_with_lunchbreak = [
        {
            "fromTime": np.datetime64("2021-01-01T09:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T13:00:00.000000"),
            "fromWeekDay": 2,
            "toWeekDay": 2,
        },
        {
            "fromTime": np.datetime64("2021-01-01T14:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T17:00:00.000000"),
            "fromWeekDay": 2,
            "toWeekDay": 2,
        },
    ]

    # make activity timestamps:
    # start - Wed 10:00
    # end - Wed 16:00
    activity_start = np.datetime64("2021-11-17T10:00:00")
    activity_end = np.datetime64("2021-11-17T15:00:00")

    # Including off-timetable hours: 5 hours / 18_000 seconds
    assert get_raw_duration(activity_start, activity_end) == 18_000

    # Excluding off-timetable hours: 4 hours / 14_400 sec
    processing_time, _, _, _ = get_processing_time(
        activity_start, activity_end, timetable_with_lunchbreak
    )
    assert processing_time == 14_400


def test_consulta_case():
    """Test {caseid: 81802, activity: Homologacion por grupo de cursos}"""

    # make role timetable:
    _timetable = [
        {
            "fromTime": np.datetime64("2021-01-01T00:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T01:59:00.000000"),
            "fromWeekDay": 2,
            "toWeekDay": 2,
        },
        {
            "fromTime": np.datetime64("2021-01-01T03:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T03:59:00.000000"),
            "fromWeekDay": 2,
            "toWeekDay": 2,
        },
        {
            "fromTime": np.datetime64("2021-01-01T13:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T23:59:00.000000"),
            "fromWeekDay": 2,
            "toWeekDay": 2,
        },
        {
            "fromTime": np.datetime64("2021-01-01T00:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T00:59:00.000000"),
            "fromWeekDay": 3,
            "toWeekDay": 3,
        },
        {
            "fromTime": np.datetime64("2021-01-01T02:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T03:59:00.000000"),
            "fromWeekDay": 3,
            "toWeekDay": 3,
        },
        {
            "fromTime": np.datetime64("2021-01-01T13:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T23:59:00.000000"),
            "fromWeekDay": 3,
            "toWeekDay": 3,
        },
    ]

    # make activity timestamps:
    activity_start = np.datetime64("2016-02-03T17:51:35")
    activity_end = np.datetime64("2016-02-04T02:39:09")

    # Including off-timetable hours: 8 hours, 47 minutes and 34 seconds
    assert get_raw_duration(activity_start, activity_end) == 31_654

    # Excluding off-timetable hours: 7 hours, 47 minutes and 34 seconds
    processing_time, _, _, _ = get_processing_time(
        activity_start, activity_end, _timetable
    )
    assert processing_time == 27_934


def test_offtimetable_start():
    """Test an activity starting outside role timetable (same day)"""

    # make role timetable:
    # Wed 09:00 - 13:00
    # Wed 14:00 - 17:00
    # Thu 09:00 - 13:00
    # Thu 14:00 - 17:00
    _timetable = [
        {
            "fromTime": np.datetime64("2021-01-01T09:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T13:00:00.000000"),
            "fromWeekDay": 2,
            "toWeekDay": 2,
        },
        {
            "fromTime": np.datetime64("2021-01-01T14:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T17:00:00.000000"),
            "fromWeekDay": 2,
            "toWeekDay": 2,
        },
        {
            "fromTime": np.datetime64("2021-01-01T09:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T13:00:00.000000"),
            "fromWeekDay": 3,
            "toWeekDay": 3,
        },
        {
            "fromTime": np.datetime64("2021-01-01T14:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T17:00:00.000000"),
            "fromWeekDay": 3,
            "toWeekDay": 3,
        },
    ]

    # make activity timestamps:
    # start - Wed 07:00
    # end - Wed 16:00
    activity_start = np.datetime64("2021-11-17T07:00:00")
    activity_end = np.datetime64("2021-11-17T16:00:00")

    # Including off-timetable hours:
    assert get_raw_duration(activity_start, activity_end) == 32_400

    # Best-possible estimation of effective work:
    # wed 07:00 - wed 13:00 (21_600) + wed 14:00 - wed 16:00 (7_200)
    processing_time, _, started_offtimetable, ended_offtimetable = get_processing_time(
        activity_start, activity_end, _timetable
    )
    assert processing_time == 28_800
    assert started_offtimetable
    assert not ended_offtimetable


def test_offtimetable_start_diff_day():
    """Test an activity starting outside role timetable (different day)"""

    # make role timetable:
    # Wed 09:00 - 13:00
    # Wed 14:00 - 17:00
    # Thu 09:00 - 13:00
    # Thu 14:00 - 17:00
    _timetable = [
        {
            "fromTime": np.datetime64("2021-01-01T09:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T13:00:00.000000"),
            "fromWeekDay": 2,
            "toWeekDay": 2,
        },
        {
            "fromTime": np.datetime64("2021-01-01T14:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T17:00:00.000000"),
            "fromWeekDay": 2,
            "toWeekDay": 2,
        },
        {
            "fromTime": np.datetime64("2021-01-01T09:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T13:00:00.000000"),
            "fromWeekDay": 3,
            "toWeekDay": 3,
        },
        {
            "fromTime": np.datetime64("2021-01-01T14:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T17:00:00.000000"),
            "fromWeekDay": 3,
            "toWeekDay": 3,
        },
    ]

    # make activity timestamps:
    # start - Tue 07:00
    # end - Wed 16:00
    activity_start = np.datetime64("2021-11-16T07:00:00")
    activity_end = np.datetime64("2021-11-17T16:00:00")

    # Including off-timetable hours:
    assert get_raw_duration(activity_start, activity_end) == 118_800

    # Best-possible estimation of effective work:
    # tue 07:00 - wed 13:00 (108_000) + wed 14:00 - wed 16:00 (7_200)
    processing_time, _, started_offtimetable, ended_offtimetable = get_processing_time(
        activity_start, activity_end, _timetable
    )
    assert processing_time == 115_200
    assert started_offtimetable
    assert not ended_offtimetable


def test_offtimetable_end():
    """Test an activity ending outside role timetable (same day)"""

    # make role timetable:
    # Wed 09:00 - 13:00
    # Wed 14:00 - 17:00
    # Thu 09:00 - 13:00
    # Thu 14:00 - 17:00
    _timetable = [
        {
            "fromTime": np.datetime64("2021-01-01T09:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T13:00:00.000000"),
            "fromWeekDay": 2,
            "toWeekDay": 2,
        },
        {
            "fromTime": np.datetime64("2021-01-01T14:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T17:00:00.000000"),
            "fromWeekDay": 2,
            "toWeekDay": 2,
        },
        {
            "fromTime": np.datetime64("2021-01-01T09:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T13:00:00.000000"),
            "fromWeekDay": 3,
            "toWeekDay": 3,
        },
        {
            "fromTime": np.datetime64("2021-01-01T14:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T17:00:00.000000"),
            "fromWeekDay": 3,
            "toWeekDay": 3,
        },
    ]

    # make activity timestamps:
    # start - Wed 11:00
    # end - Wed 20:00
    activity_start = np.datetime64("2021-11-17T11:00:00")
    activity_end = np.datetime64("2021-11-17T20:00:00")

    # Including off-timetable hours:
    assert get_raw_duration(activity_start, activity_end) == 32_400

    # Best-possible estimation of effective work:
    # wed 11:00 - wed 13:00 (7200) + wed 14:00 - wed 20:00 (21_600)
    processing_time, _, started_offtimetable, ended_offtimetable = get_processing_time(
        activity_start, activity_end, _timetable
    )
    assert processing_time == 28_800
    assert not started_offtimetable
    assert ended_offtimetable


def test_offtimetable_end_diff_day():
    """Test an activity ending outside role timetable (different day)"""

    # make role timetable:
    # Wed 09:00 - 13:00
    # Wed 14:00 - 17:00
    # Thu 09:00 - 13:00
    # Thu 14:00 - 17:00
    _timetable = [
        {
            "fromTime": np.datetime64("2021-01-01T09:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T13:00:00.000000"),
            "fromWeekDay": 2,
            "toWeekDay": 2,
        },
        {
            "fromTime": np.datetime64("2021-01-01T14:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T17:00:00.000000"),
            "fromWeekDay": 2,
            "toWeekDay": 2,
        },
        {
            "fromTime": np.datetime64("2021-01-01T09:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T13:00:00.000000"),
            "fromWeekDay": 3,
            "toWeekDay": 3,
        },
        {
            "fromTime": np.datetime64("2021-01-01T14:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T17:00:00.000000"),
            "fromWeekDay": 3,
            "toWeekDay": 3,
        },
    ]

    # make activity timestamps:
    # start - Thu 11:00
    # end - Fri 20:00
    activity_start = np.datetime64("2021-11-18T11:00:00")
    activity_end = np.datetime64("2021-11-19T20:00:00")

    # Including off-timetable hours:
    assert get_raw_duration(activity_start, activity_end) == 118_800

    # Best-possible estimation of effective work:
    # thu 11:00 - thu 13:00 (7200) + thu 14:00 - fri 20:00 (108_000)
    processing_time, _, started_offtimetable, ended_offtimetable = get_processing_time(
        activity_start, activity_end, _timetable
    )
    assert processing_time == 115_200
    assert not started_offtimetable
    assert ended_offtimetable


def test_offtimetable_mixed():
    """Test an activity starting and ending outside role timetable"""

    # make role timetable:
    # Wed 09:00 - 13:00
    # Wed 14:00 - 17:00
    # Thu 09:00 - 13:00
    # Thu 14:00 - 17:00
    _timetable = [
        {
            "fromTime": np.datetime64("2021-01-01T09:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T13:00:00.000000"),
            "fromWeekDay": 2,
            "toWeekDay": 2,
        },
        {
            "fromTime": np.datetime64("2021-01-01T14:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T17:00:00.000000"),
            "fromWeekDay": 2,
            "toWeekDay": 2,
        },
        {
            "fromTime": np.datetime64("2021-01-01T09:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T13:00:00.000000"),
            "fromWeekDay": 3,
            "toWeekDay": 3,
        },
        {
            "fromTime": np.datetime64("2021-01-01T14:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T17:00:00.000000"),
            "fromWeekDay": 3,
            "toWeekDay": 3,
        },
    ]

    # make activity timestamps:
    # start - Tue 11:00
    # end - Fri 20:00
    activity_start = np.datetime64("2021-11-16T11:00:00")
    activity_end = np.datetime64("2021-11-19T20:00:00")

    # Including off-timetable hours:
    assert get_raw_duration(activity_start, activity_end) == 291_600

    # Best-possible estimation of effective work:
    # tue 11:00 - wed 13:00 (93600) +
    # wed 14:00 - wed 17:00 (10800) +
    # thu 09:00 - thu 13:00 (14_400) +
    # thu 14:00 - fri 20:00 (108_000)
    processing_time, _, started_offtimetable, ended_offtimetable = get_processing_time(
        activity_start, activity_end, _timetable
    )
    assert processing_time == 226_800
    assert started_offtimetable
    assert ended_offtimetable


def test_instant_activity():
    """Test an activity starting and ending in same timestamp"""

    # make role timetable:
    # Wed 09:00 - 13:00
    # Wed 14:00 - 17:00
    # Thu 09:00 - 13:00
    # Thu 14:00 - 17:00
    _timetable = [
        {
            "fromTime": np.datetime64("2021-01-01T09:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T13:00:00.000000"),
            "fromWeekDay": 2,
            "toWeekDay": 2,
        },
        {
            "fromTime": np.datetime64("2021-01-01T14:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T17:00:00.000000"),
            "fromWeekDay": 2,
            "toWeekDay": 2,
        },
        {
            "fromTime": np.datetime64("2021-01-01T09:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T13:00:00.000000"),
            "fromWeekDay": 3,
            "toWeekDay": 3,
        },
        {
            "fromTime": np.datetime64("2021-01-01T14:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T17:00:00.000000"),
            "fromWeekDay": 3,
            "toWeekDay": 3,
        },
    ]

    # make activity timestamps:
    # start - Tue 11:00
    # end - Tue 11:00
    activity_start = np.datetime64("2021-11-17T11:00:00")
    activity_end = np.datetime64("2021-11-17T11:00:00")

    assert get_raw_duration(activity_start, activity_end) == 0

    processing_time, _, started_offtimetable, ended_offtimetable = get_processing_time(
        activity_start, activity_end, _timetable
    )
    assert processing_time == 0
    assert not started_offtimetable
    assert not ended_offtimetable


def test_instant_activity_off_timetable():
    """Test an activity starting and ending in same timestamp, offtimetable"""

    # make role timetable:
    # Wed 09:00 - 13:00
    # Wed 14:00 - 17:00
    # Thu 09:00 - 13:00
    # Thu 14:00 - 17:00
    _timetable = [
        {
            "fromTime": np.datetime64("2021-01-01T09:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T13:00:00.000000"),
            "fromWeekDay": 2,
            "toWeekDay": 2,
        },
        {
            "fromTime": np.datetime64("2021-01-01T14:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T17:00:00.000000"),
            "fromWeekDay": 2,
            "toWeekDay": 2,
        },
        {
            "fromTime": np.datetime64("2021-01-01T09:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T13:00:00.000000"),
            "fromWeekDay": 3,
            "toWeekDay": 3,
        },
        {
            "fromTime": np.datetime64("2021-01-01T14:00:00.000000"),
            "toTime": np.datetime64("2021-01-01T17:00:00.000000"),
            "fromWeekDay": 3,
            "toWeekDay": 3,
        },
    ]

    # make activity timestamps:
    # start - Tue 11:00
    # end - Tue 11:00
    activity_start = np.datetime64("2021-11-16T11:00:00")
    activity_end = np.datetime64("2021-11-16T11:00:00")

    assert get_raw_duration(activity_start, activity_end) == 0

    processing_time, _, started_offtimetable, _ = get_processing_time(
        activity_start, activity_end, _timetable
    )
    assert processing_time == 0
    assert started_offtimetable
