import pytest

import pandas as pd
from pandas import Timestamp

from collections import Counter

from knockout_ios.utils.constants import *
from knockout_ios.utils.redesign import get_sorted_with_dependencies, find_producers, get_relocated_kos


def test_pure_relocation_1():
    activities = ["ko_1", "normal_1", "normal_2", "ko_2", "normal_3", "ko_3"]

    dependencies = {k: [] for k in activities}

    # ko_2 depends on normal_1
    dependencies["ko_2"].append(("attr1", "normal_1"))

    # ko_3 depends on ko_2 and normal_3
    dependencies["ko_3"].append(("attr2", "ko_2"))
    dependencies["ko_3"].append(("attr3", "normal_3"))

    # ko_3 has no dependencies

    proposed_order = get_relocated_kos(current_order_all_activities=activities,
                                       ko_activities=["ko_1", "ko_2", "ko_3"],
                                       dependencies=dependencies)

    assert proposed_order == ["ko_1", "normal_1", "ko_2", "normal_2", "normal_3", "ko_3"]


def test_pure_relocation_2():
    activities = ["ko_1", "normal_1", "normal_2", "normal_3", "ko_2", "ko_3"]

    dependencies = {k: [] for k in activities}

    proposed_order = get_relocated_kos(current_order_all_activities=activities,
                                       ko_activities=["ko_1", "ko_2", "ko_3"],
                                       dependencies=dependencies)

    # we expect to see all the KO ko_activities placed as early as possible because they have no dependencies
    assert proposed_order == ["ko_1", "ko_2", "ko_3", "normal_1", "normal_2", "normal_3"]


def test_get_sorted_with_dependencies_1():
    order = ["A", "C", "B"]
    dependencies = {k: [] for k in order}
    dependencies["A"].append(("attr_from_C", "C"))
    dependencies["A"].append(("attr_from_B", "B"))

    optimal_order = get_sorted_with_dependencies(ko_activities=order, dependencies=dependencies,
                                                 current_activity_order=order)

    assert optimal_order == ["C", "B", "A"]


def test_get_sorted_with_dependencies_2():
    order = ["A", "C", "B"]
    dependencies = {k: [] for k in order}
    dependencies["C"].append(("attr_from_B", "B"))
    dependencies["A"].append(("attr_from_C", "C"))
    dependencies["A"].append(("attr_from_B", "B"))

    optimal_order = get_sorted_with_dependencies(ko_activities=order, dependencies=dependencies,
                                                 current_activity_order=order)

    assert optimal_order == ["B", "C", "A"]


def test_get_sorted_with_dependencies_3():
    order = ["B", "C", "A"]
    dependencies = {k: [] for k in order}
    dependencies["B"].append(("attr_from_A", "A"))
    dependencies["C"].append(("attr_from_A", "A"))

    efforts = [{REPORT_COLUMN_KNOCKOUT_CHECK: "A", REPORT_COLUMN_EFFORT_PER_KO: 10},
               {REPORT_COLUMN_KNOCKOUT_CHECK: "B", REPORT_COLUMN_EFFORT_PER_KO: 0.1},
               {REPORT_COLUMN_KNOCKOUT_CHECK: "C", REPORT_COLUMN_EFFORT_PER_KO: 5}]

    optimal_order = get_sorted_with_dependencies(ko_activities=order, dependencies=dependencies,
                                                 current_activity_order=order,
                                                 efforts=pd.DataFrame(efforts))

    assert optimal_order == ["A", "B", "C"]


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
