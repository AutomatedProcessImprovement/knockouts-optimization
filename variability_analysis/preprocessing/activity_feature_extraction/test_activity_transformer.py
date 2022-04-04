import pytest

import pandas as pd
import numpy as np

from .activity_transformer import ActivityTransformer

traces = [
    {
        "caseid": 0,
        "task": "Activity A",
        "start_timestamp": np.datetime64("2021-01-01T09:00:00.000000"),
        "end_timestamp": np.datetime64("2021-01-01T10:00:00.000000"),
        "processing_time": 3600,
        "rp_start_oc": 0.5,
        "rp_end_oc": 0.3,
    },
    {
        "caseid": 0,
        "task": "Activity A",
        "start_timestamp": np.datetime64("2021-01-01T12:00:00.000000"),
        "end_timestamp": np.datetime64("2021-01-01T14:00:00.000000"),
        "processing_time": 7200,
        "rp_start_oc": 0.2,
        "rp_end_oc": 0.1,
    },
    {
        "caseid": 0,
        "task": "Activity B",
        "start_timestamp": np.datetime64("2021-01-01T10:00:00.000000"),
        "end_timestamp": np.datetime64("2021-01-01T11:00:00.000000"),
        "processing_time": 3600,
        "rp_start_oc": 0.3,
        "rp_end_oc": 0.3,
    },
]

log_df = pd.DataFrame(traces)


def test_repetitions_and_time():

    t = ActivityTransformer(log_df)
    t.add_repetitions_and_time()

    assert t.get_activity_count(case_id=0, activity="Activity A") == 2
    assert t.get_activity_count(case_id=0, activity="Activity B") == 1

    assert t.get_activity_cumulative_time(case_id=0, activity="Activity A") == 10800
    assert t.get_activity_cumulative_time(case_id=0, activity="Activity B") == 3600


def test_position_in_trace():

    t = ActivityTransformer(log_df)
    t.add_activity_positions()

    assert np.array_equal(
        t.get_activity_positions(case_id=0, activity="Activity A"), [0, 2]
    )
    assert np.array_equal(
        t.get_activity_positions(case_id=0, activity="Activity B"), [1]
    )


def test_resource_occupation():

    _traces = [
        {
            "caseid": 0,
            "task": "Activity A",
            "start_timestamp": np.datetime64("2021-01-01T09:00:00.000000"),
            "end_timestamp": np.datetime64("2021-01-01T10:00:00.000000"),
            "processing_time": 3600,
            "user": "001",
        },
        {
            "caseid": 0,
            "task": "Activity B",
            "start_timestamp": np.datetime64("2021-01-01T10:00:00.000000"),
            "end_timestamp": np.datetime64("2021-01-01T11:00:00.000000"),
            "processing_time": 3600,
            "user": "001",
        },
        {
            "caseid": 0,
            "task": "Activity A",
            "start_timestamp": np.datetime64("2021-01-01T12:00:00.000000"),
            "end_timestamp": np.datetime64("2021-01-01T14:00:00.000000"),
            "processing_time": 7200,
            "user": "002",
        },
        {
            "caseid": 1,
            "task": "Activity A",
            "start_timestamp": np.datetime64("2021-01-01T12:00:00.000000"),
            "end_timestamp": np.datetime64("2021-01-01T13:00:00.000000"),
            "processing_time": 3600,
            "user": "001",
        },
        {
            "caseid": 1,
            "task": "Activity B",
            "start_timestamp": np.datetime64("2021-01-01T11:00:00.000000"),
            "end_timestamp": np.datetime64("2021-01-01T14:00:00.000000"),
            "processing_time": 3600,
            "user": "001",
        },
        {
            "caseid": 2,
            "task": "Activity B",
            "start_timestamp": np.datetime64("2021-01-01T09:00:00.000000"),
            "end_timestamp": np.datetime64("2021-01-01T09:30:00.000000"),
            "processing_time": 3600 / 2,
            "user": "001",
        },
    ]

    _log_df = pd.DataFrame(_traces)

    t = ActivityTransformer(_log_df)
    t.add_activity_positions()
    t.add_resource_occupations()

    assert t.get_resource_occupations(
        case_id=0, activity="Activity A", index_in_trace=0
    ) == (2, 1)

    assert t.get_resource_occupations(
        case_id=1, activity="Activity B", index_in_trace=0
    ) == (1, 0)
