import pytest

import pandas as pd
from pandas import Timestamp

from collections import Counter

from knockout_ios.utils.constants import globalColumnNames
from knockout_ios.utils.platform_check import is_windows
from knockout_ios.utils.redesign import get_ko_activities_sorted_with_dependencies, find_producers, get_relocated_kos, \
    find_ko_activity_dependencies, evaluate_knockout_reordering_io, simplify_rule


def test_pure_relocation_1():
    activities = ["ko_1", "normal_1", "normal_2", "ko_2", "normal_3", "ko_3"]

    dependencies = {k: [] for k in ["ko_1", "ko_2", "ko_3"]}

    # ko_2 depends on normal_1
    dependencies["ko_2"].append(("attr1", "normal_1"))

    # ko_3 depends on ko_2 and normal_3
    dependencies["ko_3"].append(("attr2", "ko_2"))
    dependencies["ko_3"].append(("attr3", "normal_2"))

    # ko_3 has no dependencies

    proposed_order = get_relocated_kos(current_order_all_activities=activities,
                                       optimal_ko_order=["ko_1", "ko_2", "ko_3"],
                                       dependencies=dependencies)

    assert proposed_order == ["ko_1", "normal_1", "ko_2", "normal_2", "ko_3", "normal_3"]


def test_pure_relocation_2():
    activities = ["start", "normal_1", "normal_2", "ko_1", "normal_3", "ko_2", "ko_3"]

    dependencies = {k: [] for k in ["ko_1", "ko_2", "ko_3"]}
    dependencies["ko_1"].append(("attr1", "start"))

    proposed_order = get_relocated_kos(current_order_all_activities=activities,
                                       optimal_ko_order=["ko_1", "ko_2", "ko_3"],
                                       dependencies=dependencies)

    # we expect to see all the KO ko_activities placed as early as possible because they have no dependencies, apart from "start"
    assert proposed_order == ["start", "ko_1", "ko_2", "ko_3", "normal_1", "normal_2", "normal_3"]


def test_pure_relocation_3():
    activities = ["start", "normal_1", "normal_2", "ko_1", "normal_3", "ko_2", "ko_3"]

    dependencies = {k: [] for k in ["ko_1", "ko_2", "ko_3"]}

    dependencies["ko_1"].append(("attr1", "start"))

    dependencies["ko_3"].append(("attr3", "normal_3"))

    proposed_order = get_relocated_kos(current_order_all_activities=activities,
                                       optimal_ko_order=["ko_1", "ko_2", "ko_3"],
                                       dependencies=dependencies, )

    # only ko_3 has a dependency on a non-ko activity
    # the rest can be placed as early as possible, given a precomputed optimal order as a constraint
    assert proposed_order == ["start", "ko_1", "ko_2", "normal_1", "normal_2", "normal_3", "ko_3"]


def test_relocation_BPI():
    if not is_windows():
        return

    bpi_knockout_analyzer = pd.read_pickle("test/test_fixtures/bpi_2017_1k_W")

    dependencies = find_ko_activity_dependencies(bpi_knockout_analyzer)
    reordering = evaluate_knockout_reordering_io(bpi_knockout_analyzer,
                                                 dependencies)

    optimal_ko_order = reordering["optimal_ko_order"]

    current_order = ["Start",
                     'A_Create Application',
                     "A_Accepted",
                     "O_Create Offer",
                     'O_Created',
                     'W_Complete application',
                     'W_Call after offers',
                     'O_Accepted',
                     'W_Validate application',
                     'End']

    proposed_order = get_relocated_kos(current_order,
                                       optimal_ko_order,
                                       dependencies
                                       )
    expected_order = ['Start',
                      'A_Create Application',
                      'A_Accepted',
                      'O_Created',
                      'W_Call after offers',
                      'W_Validate application',
                      'W_Complete application',
                      'O_Create Offer',
                      'O_Accepted',
                      'End']

    assert proposed_order == expected_order


def test_relocation_BPI_2():
    if not is_windows():
        return

    bpi_knockout_analyzer = pd.read_pickle("test/test_fixtures/bpi_2017_1k_W_2")

    dependencies = find_ko_activity_dependencies(bpi_knockout_analyzer)
    reordering = evaluate_knockout_reordering_io(bpi_knockout_analyzer,
                                                 dependencies)

    optimal_ko_order = reordering["optimal_ko_order"]

    assert optimal_ko_order == ["W_Call after offers",
                                "O_Created",
                                "W_Assess potential fraud",
                                "W_Call incomplete files",
                                "W_Validate application",
                                "W_Complete application"]

    current_order = ["Start",
                     'A_Create Application',
                     "A_Accepted",
                     "O_Create Offer",
                     'O_Created',
                     'W_Complete application',
                     'W_Call after offers',
                     'O_Accepted',
                     'W_Validate application',
                     'End']

    proposed_order = get_relocated_kos(current_order,
                                       optimal_ko_order,
                                       dependencies
                                       )
    expected_order = ['Start',
                      'A_Create Application',
                      'A_Accepted',
                      'W_Call after offers',
                      'O_Created',
                      'W_Validate application',
                      'W_Complete application',
                      'O_Create Offer',
                      'O_Accepted',
                      'End']

    assert proposed_order == expected_order


def test_get_sorted_with_dependencies_1():
    order = ["A", "C", "B"]
    dependencies = {k: [] for k in order}
    dependencies["A"].append(("attr_from_C", "C"))
    dependencies["A"].append(("attr_from_B", "B"))

    optimal_order = get_ko_activities_sorted_with_dependencies(dependencies=dependencies, current_activity_order=order)

    assert optimal_order == ["C", "B", "A"]


def test_get_sorted_with_dependencies_2():
    order = ["A", "C", "B"]
    dependencies = {k: [] for k in order}
    dependencies["C"].append(("attr_from_B", "B"))
    dependencies["A"].append(("attr_from_C", "C"))
    dependencies["A"].append(("attr_from_B", "B"))

    optimal_order = get_ko_activities_sorted_with_dependencies(dependencies=dependencies, current_activity_order=order)

    assert optimal_order == ["B", "C", "A"]


def test_get_sorted_with_dependencies_3():
    order = ["B", "C", "D", "A"]
    dependencies = {k: [] for k in order}
    dependencies["B"].append(("attr_from_A", "A"))
    dependencies["C"].append(("attr_from_A", "A"))

    efforts = [{globalColumnNames.REPORT_COLUMN_KNOCKOUT_CHECK: "B", globalColumnNames.REPORT_COLUMN_EFFORT_PER_KO: 0.1,
                globalColumnNames.REPORT_COLUMN_REJECTION_RATE: 10},
               {globalColumnNames.REPORT_COLUMN_KNOCKOUT_CHECK: "C", globalColumnNames.REPORT_COLUMN_EFFORT_PER_KO: 5,
                globalColumnNames.REPORT_COLUMN_REJECTION_RATE: 10},
               {globalColumnNames.REPORT_COLUMN_KNOCKOUT_CHECK: "D", globalColumnNames.REPORT_COLUMN_EFFORT_PER_KO: 8,
                globalColumnNames.REPORT_COLUMN_REJECTION_RATE: 10},
               {globalColumnNames.REPORT_COLUMN_KNOCKOUT_CHECK: "A", globalColumnNames.REPORT_COLUMN_EFFORT_PER_KO: 10,
                globalColumnNames.REPORT_COLUMN_REJECTION_RATE: 10}]

    optimal_order = get_ko_activities_sorted_with_dependencies(dependencies=dependencies, current_activity_order=order,
                                                               efforts=pd.DataFrame(efforts))

    assert optimal_order == ["D", "A", "B", "C"]


def test_find_producer_activity_simple():
    attribute_key = "attr_produced_by_B"
    ko_activity = "D"

    # in this case, the attribute stops being null from activity B, and never changes since then.
    # therefore, B is considered the producer.
    events = [
        {
            globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: 'Start',
            globalColumnNames.SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 08:00:00"),
            globalColumnNames.SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 09:00:00"),
            attribute_key: None
        },
        {
            globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: 'A',
            globalColumnNames.SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 09:00:00"),
            globalColumnNames.SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 10:00:00"),
            attribute_key: None
        },
        {
            globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: 'B',
            globalColumnNames.SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 10:00:00"),
            globalColumnNames.SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 11:00:00"),
            attribute_key: None
        },
        {
            globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: 'C',
            globalColumnNames.SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 11:00:00"),
            globalColumnNames.SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 12:00:00"),
            attribute_key: "SOME_VALUE"
        },
        {
            globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: ko_activity,
            globalColumnNames.SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 12:00:00"),
            globalColumnNames.SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 13:00:00"),
            attribute_key: "SOME_VALUE"
        },
        {
            globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: 'End',
            globalColumnNames.SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 13:00:00"),
            globalColumnNames.SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 14:00:00"),
            attribute_key: "SOME_VALUE"
        }
    ]

    log = pd.DataFrame(events)
    log.set_index(globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME, inplace=True)
    log = log.rename_axis('case_id_idx').sort_values(
        by=['case_id_idx', globalColumnNames.SIMOD_END_TIMESTAMP_COLUMN_NAME],
        ascending=[True, True])

    producers = find_producers(attribute_key, log[log["knockout_activity"] == ko_activity])

    assert len(producers) > 0
    assert Counter(producers).most_common(1)[0][0] == "B"


def test_find_producer_activity_advanced():
    attribute_key = "attr_produced_by_D"
    ko_activity = "F"

    # in this case, the attribute stops being null from activity B, but keeps changing until activity D.
    # therefore, D is considered the producer.
    events = [
        {
            globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: 'Start',
            globalColumnNames.SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 08:00:00"),
            globalColumnNames.SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 09:00:00"),
            attribute_key: None
        },
        {
            globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: 'A',
            globalColumnNames.SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 09:00:00"),
            globalColumnNames.SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 10:00:00"),
            attribute_key: None
        },
        {
            globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: 'B',
            globalColumnNames.SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 10:00:00"),
            globalColumnNames.SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 11:00:00"),
            attribute_key: None
        },
        {
            globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: 'C',
            globalColumnNames.SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 11:00:00"),
            globalColumnNames.SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 12:00:00"),
            attribute_key: "B"
        },
        {
            globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: "D",
            globalColumnNames.SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 12:00:00"),
            globalColumnNames.SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 13:00:00"),
            attribute_key: "B,C"
        },
        {
            globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: "E",
            globalColumnNames.SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 13:00:00"),
            globalColumnNames.SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 14:00:00"),
            attribute_key: "B,C,D"
        },
        {
            globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: "F",
            globalColumnNames.SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 13:00:00"),
            globalColumnNames.SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 14:00:00"),
            attribute_key: "B,C,D"
        },
        {
            globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME: 0,
            'knockout_activity': ko_activity,
            'knocked_out_case': True,
            globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME: 'End',
            globalColumnNames.SIMOD_RESOURCE_COLUMN_NAME: 'R1',
            globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 13:00:00"),
            globalColumnNames.SIMOD_END_TIMESTAMP_COLUMN_NAME: Timestamp("2022-02-17 14:00:00"),
            attribute_key: "B,C,D"
        }
    ]

    log = pd.DataFrame(events)
    log.set_index(globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME, inplace=True)
    log = log.rename_axis('case_id_idx').sort_values(
        by=['case_id_idx', globalColumnNames.SIMOD_END_TIMESTAMP_COLUMN_NAME],
        ascending=[True, True])

    producers = find_producers(attribute_key, log[log["knockout_activity"] == ko_activity])

    assert len(producers) > 0
    assert Counter(producers).most_common(1)[0][0] == "D"


@pytest.mark.skip(reason="Not yet implemented")
def test_simplify_rule_1():
    ruleset = "[[Loan_Ammount=11693.71-16840.45] V [Loan_Ammount=>16840.45]]"
    simplified_ruleset = "[[Loan_Ammount=>11693.71]]"
    assert simplify_rule(ruleset) == simplified_ruleset


@pytest.mark.skip(reason="Not yet implemented")
def test_simplify_rule_2():
    ruleset = "[[Monthly_Income=555.77-830.79] V [Monthly_Income=<555.77] V [Monthly_Income=830.79-1019.68]]"
    simplified_ruleset = "[[Monthly_Income=<1019.68]]"
    assert simplify_rule(ruleset) == simplified_ruleset
