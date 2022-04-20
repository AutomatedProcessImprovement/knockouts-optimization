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
            "total_cases": total_cases}
