import glob
import itertools
import os
import pprint

from copy import deepcopy
from typing import Callable, Optional

import numpy as np
from tabulate import tabulate

import pandas as pd
from pm4py.statistics.sojourn_time.pandas import get as soj_time_get

from knockout_ios.knockout_discoverer import KnockoutDiscoverer
from knockout_ios.utils.format import seconds_to_hms
from knockout_ios.utils.metrics import find_rejection_rates, calc_available_cases_before_ko, calc_overprocessing_waste, \
    calc_processing_waste, calc_waiting_time_waste_v2

from knockout_ios.utils.constants import *

from knockout_ios.utils.explainer import find_ko_rulesets

from knockout_ios.utils.synthetic_example.preprocessors import *


def clear_cache(cachedir, config_file_name):
    file_list = glob.glob(f'{cachedir}/{config_file_name}*')
    for filePath in file_list:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)


class KnockoutAnalyzer:

    def __init__(self, config_file_name, cache_dir="cache", config_dir="config", always_force_recompute=False,
                 quiet=True,
                 custom_log_preprocessing_function: Callable[
                     ['KnockoutAnalyzer', pd.DataFrame, Optional[str], ...], pd.DataFrame] = None):

        os.makedirs(cache_dir, exist_ok=True)

        self.quiet = quiet
        self.config_file_name = config_file_name
        self.config_dir = config_dir
        self.cache_dir = cache_dir
        self.ko_discovery_metrics = None
        self.rule_discovery_log_df = None
        self.RIPPER_rulesets = None
        self.IREP_rulesets = None
        self.aggregated_by_case_df = None
        self.ko_stats = {}
        self.ruleset_algorithm = None
        self.report_df = None
        self.custom_log_preprocessing_function = custom_log_preprocessing_function

        self.always_force_recompute = always_force_recompute

        if self.always_force_recompute:
            clear_cache(self.cache_dir, config_file_name)

        if not self.quiet:
            print(f"Starting Knockout Analyzer with config file \"{config_file_name}\"\n")

        self.discoverer = KnockoutDiscoverer(config_file_name=config_file_name,
                                             config_dir=self.config_dir,
                                             cache_dir=self.cache_dir,
                                             always_force_recompute=always_force_recompute,
                                             quiet=quiet)

        self.report_file_name = f"{self.discoverer.config.output}/{config_file_name.split('.')[0]}_ko_analysis_report.csv"

    def discover_knockouts(self, expected_kos=None):
        self.discoverer.find_ko_activities()

        if not self.quiet:
            self.discoverer.print_basic_stats()

        # Populate discovery metrics
        if expected_kos is not None:
            self.calc_ko_discovery_metrics(expected_kos)

        # Initialize ko stats dictionary
        for activity in self.discoverer.ko_activities:
            self.ko_stats[activity] = {"rejection_rate": 0}

        # Populate dict with rejection rates and efforts (avg. KO activity durations)
        self.calc_rejection_rates()

        if self.custom_log_preprocessing_function is not None:
            enriched_log_df_cache_path = f'{self.cache_dir}/{self.config_file_name}_enriched_.pkl'
            self.discoverer.log_df = self.custom_log_preprocessing_function(self, self.discoverer.log_df,
                                                                            enriched_log_df_cache_path)

    def calc_rejection_rates(self):
        rejection_rates = find_rejection_rates(self.discoverer.log_df, self.discoverer.ko_activities)

        # Update rejection rates
        for k, v in rejection_rates.items():
            self.ko_stats[k]['rejection_rate'] = v

        if not self.quiet:
            print("\nK.O. rejection rates:\n")
            pprint.pprint(rejection_rates)

    def calc_ko_efforts(self, support_threshold=0.5, confidence_threshold=0.5, algorithm="IREP"):

        if algorithm == "RIPPER":
            rulesets = self.RIPPER_rulesets
        else:
            rulesets = self.IREP_rulesets

        # average processing time of the knock-out check activity
        soj_time = soj_time_get.apply(self.discoverer.log_df,
                                      parameters={
                                          soj_time_get.Parameters.TIMESTAMP_KEY: SIMOD_END_TIMESTAMP_COLUMN_NAME,
                                          soj_time_get.Parameters.START_TIMESTAMP_KEY:
                                              SIMOD_START_TIMESTAMP_COLUMN_NAME}
                                      )

        # Compute KO rejection rates and efforts
        for key in rulesets.keys():
            entry = rulesets[key]
            metrics = entry[2]

            # Mean Processing Time does not depend on rejection rule confidence or support
            self.ko_stats[key]['mean_pt'] = soj_time[key]

            # Effort per rejection = Average PT / Rejection rate
            effort = soj_time[key] / (100 * self.ko_stats[key]['rejection_rate'])

            if (metrics['confidence'] >= confidence_threshold) and (metrics['support'] >= support_threshold):
                # Effort per rejection = (Average PT / Rejection rate) * Confidence
                effort = effort * metrics['confidence']

            # confidence and support are dependent on the rule discovery algorithm used
            self.ko_stats[key][algorithm] = {'effort': 0}
            self.ko_stats[key][algorithm]['effort'] = effort

    def calc_ko_discovery_metrics(self, expected_kos):
        self.ko_discovery_metrics = self.discoverer.get_discovery_metrics(expected_kos)

        if not self.quiet:
            pprint.pprint(self.ko_discovery_metrics)

        return deepcopy(self.ko_discovery_metrics)

    @staticmethod
    def preprocess_for_rule_discovery(log, compute_columns_only=False):

        if log is None:
            raise Exception("log not yet loaded")

        # Pre-processing
        columns_to_ignore = [SIMOD_LOG_READER_CASE_ID_COLUMN_NAME,
                             SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME,
                             SIMOD_RESOURCE_COLUMN_NAME,
                             PM4PY_ACTIVITY_COLUMN_NAME,
                             PM4PY_CASE_ID_COLUMN_NAME,
                             PM4PY_END_TIMESTAMP_COLUMN_NAME,
                             DURATION_COLUMN_NAME, SIMOD_END_TIMESTAMP_COLUMN_NAME,
                             SIMOD_START_TIMESTAMP_COLUMN_NAME,
                             'knockout_activity',
                             'knockout_prefix']

        columns_to_ignore = list(itertools.chain(columns_to_ignore, list(filter(
            lambda c: ('@' in c) | ('id' in c.lower()) | ('daytime' in c) | ('weekday' in c) | ('month' in c) | (
                    'activity' in c.lower()),
            log.columns))))

        # skip the rest of pre-proc in case it has been already done
        if compute_columns_only:
            return None, columns_to_ignore
        else:
            # prepare to edit the log, w/o overwriting original
            log = deepcopy(log)

        # Necessary in case the log contains numerical values as strings
        for attr in log.columns:
            log[attr] = \
                pd.to_numeric(log[attr], errors='ignore')

        # Fill Nan values of non-numerical columns, but drop rows with Nan values in numerical columns
        non_numerical = log.select_dtypes([object]).columns
        log = log.fillna(
            value={c: EMPTY_NON_NUMERICAL_VALUE for c in non_numerical})

        numerical = log.select_dtypes([np.number]).columns
        log = log.fillna(value={c: 0 for c in numerical})

        # group by case id and aggregate grouped_df selecting the most frequent value of each column
        grouped_df = log.groupby(SIMOD_LOG_READER_CASE_ID_COLUMN_NAME)
        log = grouped_df.agg(lambda x: x.value_counts().index[0])

        return log, columns_to_ignore

    def compute_ko_rules(self,
                         algorithm="IREP",
                         max_rules=3,
                         max_rule_conds=2,
                         max_total_conds=None,
                         k=2,
                         n_discretize_bins=10,
                         dl_allowance=16,
                         prune_size=0.33,
                         grid_search=False,
                         param_grid=None,
                         confidence_threshold=0.5,
                         support_threshold=0.5,
                         omit_report=False,
                         print_rule_discovery_stats=False):

        if self.discoverer.log_df is None:
            raise Exception("log not yet loaded")

        if self.discoverer.ko_activities is None:
            raise Exception("ko activities not yet discovered")

        # Discover rules in knockout activities with chosen algorithm
        self.ruleset_algorithm = algorithm

        self.available_cases_before_ko = calc_available_cases_before_ko(self.discoverer.ko_activities,
                                                                        self.discoverer.log_df)

        compute_columns_only = (self.RIPPER_rulesets is not None) or (self.IREP_rulesets is not None)
        preprocessed_df, columns_to_ignore = self.preprocess_for_rule_discovery(self.discoverer.log_df,
                                                                                compute_columns_only=compute_columns_only)
        if preprocessed_df is not None:
            self.rule_discovery_log_df = preprocessed_df

        if not self.quiet:
            print(f"\nDiscovering rulesets of each K.O. activity with {algorithm}")

        if grid_search & (param_grid is None):
            if algorithm == "RIPPER":
                param_grid = {"prune_size": [0.2, 0.33, 0.5], "k": [1, 2, 4], "dl_allowance": [16, 32, 64],
                              "n_discretize_bins": [10, 20, 30]}
            elif algorithm == "IREP":
                param_grid = {"prune_size": [0.33, 0.5], "n_discretize_bins": [10, 20, 30]}

        rulesets = find_ko_rulesets(self.rule_discovery_log_df,
                                    self.discoverer.ko_activities,
                                    self.discoverer.config_file_name,
                                    self.cache_dir,
                                    available_cases_before_ko=self.available_cases_before_ko,
                                    force_recompute=self.always_force_recompute,
                                    columns_to_ignore=columns_to_ignore,
                                    algorithm=algorithm,
                                    max_rules=max_rules,
                                    max_rule_conds=max_rule_conds,
                                    max_total_conds=max_total_conds,
                                    k=k,
                                    n_discretize_bins=n_discretize_bins,
                                    dl_allowance=dl_allowance,
                                    prune_size=prune_size,
                                    grid_search=grid_search,
                                    param_grid=param_grid
                                    )

        if algorithm == "RIPPER":
            self.RIPPER_rulesets = rulesets
        elif algorithm == "IREP":
            self.IREP_rulesets = rulesets

        self.calc_ko_efforts(confidence_threshold=confidence_threshold, support_threshold=support_threshold,
                             algorithm=algorithm)

        self.report_df = self.build_report(omit=omit_report)

        if print_rule_discovery_stats:
            self.print_ko_rulesets_stats(algorithm=algorithm, compact=True)

        return self.report_df, self

    def print_ko_rulesets_stats(self, algorithm, compact=False):

        rulesets = None
        if algorithm == "RIPPER":
            rulesets = self.RIPPER_rulesets
        elif algorithm == "IREP":
            rulesets = self.IREP_rulesets

        if rulesets is None:
            return

        print(f"\n{algorithm}")

        for key in rulesets.keys():
            entry = rulesets[key]
            model = entry[0]
            params = entry[1]
            metrics = entry[2]

            if compact:
                print(f"\n\"{key}\"")
                print(f'# conditions: {metrics["condition_count"]}, # rules: {metrics["rule_count"]}')
                print(
                    f"support: {metrics['support']:.2f}, confidence: {metrics['confidence']:.2f} "
                    f"\nroc_auc score: {metrics['roc_auc_score']:.2f}, f1 score: {metrics['f1_score']:.2f}, accuracy: {metrics['accuracy']:.2f}, precision: {metrics['precision']:.2f}, recall: {metrics['recall']:.2f}"
                )
            else:
                print(f"\n\"{key}\":")
                model.out_model()
                print(f'\n# conditions: {metrics["condition_count"]}, # rules: {metrics["rule_count"]}')
                print(
                    f"support: {metrics['support']:.2f}, confidence: {metrics['confidence']:.2f} "
                    f"\nroc_auc score: {metrics['roc_auc_score']:.2f}, f1 score: {metrics['f1_score']:.2f}, accuracy: {metrics['accuracy']:.2f}, precision: {metrics['precision']:.2f}, recall: {metrics['recall']:.2f}"
                )
                print(f"rule discovery params: {params}")

    def build_report(self, omit=False):
        if self.ruleset_algorithm is None:
            return

        if self.ruleset_algorithm == "RIPPER":
            rulesets = self.RIPPER_rulesets
        else:
            rulesets = self.IREP_rulesets

        _by_case = self.discoverer.log_df.drop_duplicates(subset=[SIMOD_LOG_READER_CASE_ID_COLUMN_NAME])
        freqs = calc_available_cases_before_ko(self.discoverer.ko_activities, self.discoverer.log_df)

        overprocessing_waste = calc_overprocessing_waste(self.discoverer.ko_activities,
                                                         self.discoverer.log_df)
        processing_waste = calc_processing_waste(self.discoverer.ko_activities, self.discoverer.log_df)
        waiting_time_waste = calc_waiting_time_waste_v2(self.discoverer.ko_activities,
                                                        self.discoverer.log_df)

        filtered = self.discoverer.log_df[self.discoverer.log_df['knocked_out_case'] == False]
        total_non_ko_cases = filtered.groupby([PM4PY_CASE_ID_COLUMN_NAME]).ngroups

        entries = []
        for ko in self.discoverer.ko_activities:
            entries.append({("%s" % REPORT_COLUMN_KNOCKOUT_CHECK): ko,
                            REPORT_COLUMN_TOTAL_FREQ:
                                freqs[ko],
                            REPORT_COLUMN_CASE_FREQ:
                                f"{round(100 * freqs[ko] / _by_case.shape[0], ndigits=2)} %",
                            REPORT_COLUMN_MEAN_PT: seconds_to_hms(self.ko_stats[ko]["mean_pt"]),
                            REPORT_COLUMN_REJECTION_RATE: f"{round(100 * self.ko_stats[ko]['rejection_rate'], ndigits=2)} %",
                            f"{REPORT_COLUMN_REJECTION_RULE} ({self.ruleset_algorithm})": rulesets[ko][0].ruleset_,
                            REPORT_COLUMN_EFFORT_PER_KO: round(self.ko_stats[ko][self.ruleset_algorithm]["effort"],
                                                               ndigits=2),
                            REPORT_COLUMN_TOTAL_OVERPROCESSING_WASTE: seconds_to_hms(overprocessing_waste[ko]),
                            REPORT_COLUMN_TOTAL_PT_WASTE: seconds_to_hms(processing_waste[ko]),
                            REPORT_COLUMN_WT_WASTE: seconds_to_hms(waiting_time_waste[ko]),
                            REPORT_COLUMN_MEAN_WT_WASTE: seconds_to_hms(waiting_time_waste[ko] / total_non_ko_cases)
                            }
                           )

        df = pd.DataFrame(entries)

        if not omit:
            df.to_csv(self.report_file_name, index=False)
            print(tabulate(df, headers='keys', tablefmt='psql'))

        return df


if __name__ == "__main__":
    analyzer = KnockoutAnalyzer(config_file_name="synthetic_example.json",
                                config_dir="config",
                                cache_dir="cache/synthetic_example",
                                always_force_recompute=True,
                                quiet=True,
                                custom_log_preprocessing_function=enrich_log_with_fully_known_attributes)

    analyzer.discover_knockouts(
        expected_kos=['Check Liability', 'Check Risk', 'Check Monthly Income', 'Assess application'])

    analyzer.compute_ko_rules(grid_search=True, algorithm="IREP", confidence_threshold=0.1, support_threshold=0.5,
                              print_rule_discovery_stats=True)
