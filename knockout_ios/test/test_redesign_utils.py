import pytest

import pandas as pd
from pandas import Timestamp

from collections import Counter

from knockout_ios.utils.constants import *
from knockout_ios.utils.redesign import get_sorted_with_dependencies, find_producers


def test_get_sorted_with_dependencies_1():
    order = ["A", "C", "B"]
    dependencies = {k: [] for k in order}
    dependencies["A"].append(("attr_from_C", "C"))
    dependencies["A"].append(("attr_from_B", "B"))

    optimal_order = get_sorted_with_dependencies(dependencies=dependencies, optimal_order_names=order)

    assert optimal_order == ["C", "B", "A"]


def test_get_sorted_with_dependencies_2():
    order = ["A", "C", "B"]
    dependencies = {k: [] for k in order}
    dependencies["C"].append(("attr_from_B", "B"))
    dependencies["A"].append(("attr_from_C", "C"))
    dependencies["A"].append(("attr_from_B", "B"))

    optimal_order = get_sorted_with_dependencies(dependencies=dependencies, optimal_order_names=order)

    assert optimal_order == ["B", "C", "A"]


def test_find_producer_activity_simple():
    attribute_key = "attr_produced_by_B"
    ko_activity = "D"

    # in this case, the attribute stops being null from activity B, and never changes since then.
    # therefore, B is considered the producer.
    events = [
        {
            SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: 'Start',
            SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 08:00:00"),
            SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 09:00:00"),
            attribute_key: None
        },
        {
            SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: 'A',
            SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 09:00:00"),
            SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 10:00:00"),
            attribute_key: None
        },
        {
            SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: 'B',
            SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 10:00:00"),
            SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 11:00:00"),
            attribute_key: None
        },
        {
            SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: 'C',
            SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 11:00:00"),
            SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 12:00:00"),
            attribute_key: "SOME_VALUE"
        },
        {
            SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: ko_activity,
            SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 12:00:00"),
            SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 13:00:00"),
            attribute_key: "SOME_VALUE"
        },
        {
            SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: 'End',
            SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 13:00:00"),
            SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 14:00:00"),
            attribute_key: "SOME_VALUE"
        }
    ]

    log = pd.DataFrame(events)
    log.set_index(SIMOD_LOG_READER_CASE_ID_COLUMN_NAME, inplace=True)
    log.sort_values(by=SIMOD_END_TIMESTAMP_COLUMN_NAME, inplace=True)

    producers = find_producers(attribute_key, ko_activity, log)

    assert len(producers) > 0
    assert Counter(producers).most_common(1)[0][0] == "B"


def test_find_producer_activity_advanced():
    attribute_key = "attr_produced_by_D"
    ko_activity = "F"

    # in this case, the attribute stops being null from activity B, but keeps changing until activity D.
    # therefore, D is considered the producer.
    events = [
        {
            SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: 'Start',
            SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 08:00:00"),
            SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 09:00:00"),
            attribute_key: None
        },
        {
            SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: 'A',
            SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 09:00:00"),
            SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 10:00:00"),
            attribute_key: None
        },
        {
            SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: 'B',
            SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 10:00:00"),
            SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 11:00:00"),
            attribute_key: None
        },
        {
            SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: 'C',
            SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 11:00:00"),
            SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 12:00:00"),
            attribute_key: "B"
        },
        {
            SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: "D",
            SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 12:00:00"),
            SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 13:00:00"),
            attribute_key: "B,C"
        },
        {
            SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: "E",
            SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 13:00:00"),
            SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 14:00:00"),
            attribute_key: "B,C,D"
        },
        {
            SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: "F",
            SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 13:00:00"),
            SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 14:00:00"),
            attribute_key: "B,C,D"
        },
        {
            SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: 'End',
            SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 13:00:00"),
            SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 14:00:00"),
            attribute_key: "B,C,D"
        }
    ]

    log = pd.DataFrame(events)
    log.set_index(SIMOD_LOG_READER_CASE_ID_COLUMN_NAME, inplace=True)
    log.sort_values(by=SIMOD_END_TIMESTAMP_COLUMN_NAME)

    producers = find_producers(attribute_key, ko_activity, log)

    assert len(producers) > 0
    assert Counter(producers).most_common(1)[0][0] == "D"
