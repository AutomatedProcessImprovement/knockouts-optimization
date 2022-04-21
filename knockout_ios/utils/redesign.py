import numpy as np
import pandas as pd
from pm4py.algo.filtering.pandas import ltl

from knockout_ios.knockout_analyzer import KnockoutAnalyzer
from knockout_ios.utils.constants import *


def chained_eventually_follows(log, activities):
    if len(activities) < 2:
        return log

    if len(activities) == 2:
        return ltl.ltl_checker.A_eventually_B(log, activities[0], activities[1])

    for i in range(0, len(activities) - 1):
        log = ltl.ltl_checker.A_eventually_B(log, activities[i], activities[i + 1])

    return log


def get_observed_ko_checks_order(log, ko_activities):
    observed_ko_order = {ko: [] for ko in ko_activities}

    # Idea: for every ko activity, get a list of the index of appearance in every case of the log

    cases = log.groupby(PM4PY_CASE_ID_COLUMN_NAME)
    for ko_activity in observed_ko_order.keys():

        for group in cases.groups.keys():
            case = cases.get_group(group)
            case_ko = case['knockout_activity']

            if not (len(case_ko.values) > 0):
                continue

            if case_ko.values[0] != ko_activity:
                continue

            # Necessary to get the index of the activity in the case (starting from 0 at the beggining of THIS case)
            case.set_index(pd.Index([i for i in range(len(case[PM4PY_ACTIVITY_COLUMN_NAME]))]),
                           PM4PY_ACTIVITY_COLUMN_NAME,
                           inplace=True)

            idx = case[case[PM4PY_ACTIVITY_COLUMN_NAME] == ko_activity].index.item()
            observed_ko_order[ko_activity].append(idx)

        observed_ko_order[ko_activity] = int(np.mean(observed_ko_order[ko_activity]))

    # transform observed_ko_order into a list of its keys, sorted by the value of every key
    observed_ko_order = list(map(lambda x: x[0], sorted(observed_ko_order.items(), key=lambda x: x[1])))

    return observed_ko_order


def evaluate_knockout_relocation_io(analyzer: KnockoutAnalyzer):
    return []


def evaluate_knockout_rule_change_io(analyzer: KnockoutAnalyzer):
    return []


def evaluate_knockout_reordering_io_v1(analyzer: KnockoutAnalyzer):
    '''
    Version 1:
    - Produces suggestion only based on KO efforts.
    - Does not yet take into account availability of attribute values after certain activity.
    - Analyzes only non-knocked out cases to indicate if the observed ko-checks ordering is optimal
    '''

    log = analyzer.discoverer.pm4py_formatted_df

    # TODO: determine optimal order of knockout activities
    sorted_by_effort = analyzer.report_df.sort_values(by=[REPORT_COLUMN_EFFORT_PER_KO], ascending=True, inplace=False)
    optimal_order_names = sorted_by_effort[REPORT_COLUMN_KNOCKOUT_CHECK].values

    # TODO: remove this, just for testing
    # optimal_order_names = ["Check Liability", "Check Risk", "Check Monthly Income", "Assess application"]

    # Determine how many cases respect this order in the log
    # TODO: for the moment, keeping only non knocked out cases to analyze order. Could be useful to see also partial (ko-d) cases
    filtered = log[log['knocked_out_case'] == False]
    total_cases = filtered.groupby([SIMOD_LOG_READER_CASE_ID_COLUMN_NAME]).ngroups

    cases_respecting_order = chained_eventually_follows(filtered, optimal_order_names) \
        .groupby([SIMOD_LOG_READER_CASE_ID_COLUMN_NAME])

    # TODO: show the (most frequently) observed ko order alongside optimal order (include case count?)

    return {"optimal_order_names": list(optimal_order_names), "cases_respecting_order": cases_respecting_order.ngroups,
            "total_cases": total_cases,
            "observed_ko_order": []}


def evaluate_knockout_reordering_io_v2(analyzer: KnockoutAnalyzer):
    '''
    Version 2:
    - Produces suggestion based on both KO efforts and availability of attribute values after certain activity.
    - Returns also the observed ko-checks ordering
    '''

    log = analyzer.discoverer.pm4py_formatted_df

    # TODO: determine optimal order of knockout activities
    sorted_by_effort = analyzer.report_df.sort_values(by=[REPORT_COLUMN_EFFORT_PER_KO], ascending=True, inplace=False)
    optimal_order_names = sorted_by_effort[REPORT_COLUMN_KNOCKOUT_CHECK].values

    # TODO: remove this, just for testing
    # optimal_order_names = ["Check Liability", "Check Risk", "Check Monthly Income", "Assess application"]

    # Determine how many cases respect this order in the log
    # TODO: for the moment, keeping only non knocked out cases to analyze order. Could be useful to see also partial (ko-d) cases
    filtered = log[log['knocked_out_case'] == False]
    total_cases = filtered.groupby([PM4PY_CASE_ID_COLUMN_NAME]).ngroups

    cases_respecting_order = chained_eventually_follows(filtered, optimal_order_names) \
        .groupby([PM4PY_CASE_ID_COLUMN_NAME])

    # TODO: show the (most frequently) observed ko order alongside optimal order (include case count?)
    observed_ko_order = get_observed_ko_checks_order(log, analyzer.discoverer.ko_activities)

    return {"optimal_order_names": list(optimal_order_names), "cases_respecting_order": cases_respecting_order.ngroups,
            "total_cases": total_cases,
            "observed_ko_order": observed_ko_order}
