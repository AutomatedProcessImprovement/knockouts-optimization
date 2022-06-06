import glob
import logging
import os
import pprint
import traceback

from copy import deepcopy
from typing import Callable, Optional

from tabulate import tabulate

from pm4py.statistics.sojourn_time.pandas import get as soj_time_get

from knockout_ios.knockout_discoverer import KnockoutDiscoverer
from knockout_ios.utils.custom_exceptions import LogNotLoadedException, EmptyLogException, \
    EmptyKnockoutActivitiesException, KnockoutRuleDiscoveryException
from knockout_ios.utils.formatting import seconds_to_hms, out_pretty
from knockout_ios.utils.metrics import find_rejection_rates, calc_available_cases_before_ko, calc_overprocessing_waste, \
    calc_processing_waste, calc_waiting_time_waste_v2

from knockout_ios.utils.constants import globalColumnNames

from knockout_ios.utils.explainer import find_ko_rulesets
from knockout_ios.utils.preprocessing.configuration import read_log_and_config, Configuration

from knockout_ios.utils.synthetic_example.preprocessors import *


def clear_cache(cachedir, config_file_name):
    file_list = glob.glob(f'{cachedir}/{config_file_name}*')
    for filePath in file_list:
        if "parsed_log" in filePath:
            continue

        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)


class KnockoutAnalyzer:

    def __init__(self, log_df: pd.DataFrame, config: Configuration, config_file_name, cache_dir="cache",
                 config_dir="config", always_force_recompute=False,
                 quiet=True,
                 custom_log_preprocessing_function: Callable[
                     ['KnockoutAnalyzer', pd.DataFrame, Optional[str], ...], pd.DataFrame] = None):

        # TODO: refactor the need for config_dir, cache_dir...

        os.makedirs(cache_dir, exist_ok=True)

        self.quiet = quiet
        self.config = config
        self.one_timestamp = config.read_options.one_timestamp
        self.start_activity = config.start_activity
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
        self.report_file_name = f"{config.output}/{config_file_name.split('.')[0]}_ko_analysis_report.csv"

        if self.always_force_recompute:
            clear_cache(self.cache_dir, config_file_name)

        if not self.quiet:
            print(f"Starting Knockout Analyzer with pipeline_config file \"{config_file_name}\"\n")

        if config.attributes_to_ignore is None:
            self.attributes_to_ignore = []
        else:
            self.attributes_to_ignore = [c.replace(' ', '_') for c in config.attributes_to_ignore]

        log_df = log_df.sort_values(by=[globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME,
                                        globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME])

        self.discoverer = KnockoutDiscoverer(log_df=log_df,
                                             config=config,
                                             config_file_name=config_file_name,
                                             config_dir=self.config_dir,
                                             cache_dir=self.cache_dir,
                                             always_force_recompute=always_force_recompute,
                                             quiet=quiet)

    def get_total_cases(self):
        return self.discoverer.log_df[globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME].nunique()

    def discover_knockouts(self, expected_kos=None):
        if not (self.config.known_ko_activities is None) and (len(self.config.known_ko_activities) > 0):
            self.discoverer.label_cases_with_known_ko_activities(self.config.known_ko_activities)
        else:
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
                                          soj_time_get.Parameters.TIMESTAMP_KEY: globalColumnNames.SIMOD_END_TIMESTAMP_COLUMN_NAME,
                                          soj_time_get.Parameters.START_TIMESTAMP_KEY:
                                              globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME}
                                      )

        # Compute KO rejection rates and efforts
        for key in rulesets.keys():
            entry = rulesets[key]
            metrics = entry[2]

            if self.one_timestamp:
                self.ko_stats[key]['mean_pt'] = 0
                effort = (100 * self.ko_stats[key]['rejection_rate'])
            else:
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

    def preprocess_for_rule_discovery(self, log, compute_columns_only=False):

        if log is None:
            raise LogNotLoadedException

        # Pre-processing
        columns_to_ignore = [globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME,
                             globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME,
                             globalColumnNames.SIMOD_RESOURCE_COLUMN_NAME,
                             globalColumnNames.PM4PY_ACTIVITY_COLUMN_NAME,
                             globalColumnNames.PM4PY_CASE_ID_COLUMN_NAME,
                             globalColumnNames.PM4PY_END_TIMESTAMP_COLUMN_NAME,
                             globalColumnNames.DURATION_COLUMN_NAME,
                             globalColumnNames.SIMOD_END_TIMESTAMP_COLUMN_NAME,
                             globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME,
                             'knockout_activity',
                             'knockout_prefix'
                             ]

        columns_to_ignore.extend(self.attributes_to_ignore)
        logging.info(f"Ignoring columns: {columns_to_ignore}")

        columns_to_ignore.extend(list(filter(
            lambda c: ('@' in c) | ('id' in c.lower()) | ('daytime' in c) | ('weekday' in c) | ('month' in c) | (
                    'activity' in c.lower()) | ('timestamp' in c.lower()),
            log.columns)))
        columns_to_ignore = list(set(columns_to_ignore))

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

        # Aggregate attribute values by case

        # If some column only has nan values, drop it from log dataframe
        log = log.dropna(axis=1, how='all')

        if len(log.columns) == 0:
            raise EmptyLogException("Log empty during pre-processing for rule discovery: all columns have nan values")

        # Select the last non-null value of each column
        log = log.groupby(globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME, as_index=False).last()

        # Drop any remaining rows with nan values
        log = log.dropna(how='any')

        if log.shape[0] == 0:
            raise EmptyLogException("Log is empty after pre-processing for rule discovery: all rows have nan values")

        return log, columns_to_ignore

    def compute_ko_rules(self,
                         algorithm="RIPPER",
                         max_rules=None,
                         max_rule_conds=None,
                         max_total_conds=None,
                         k=2,
                         n_discretize_bins=10,
                         dl_allowance=2,
                         prune_size=0.8,
                         grid_search=False,
                         param_grid=None,
                         confidence_threshold=0.5,
                         support_threshold=0.5,
                         omit_report=False,
                         print_rule_discovery_stats=True):

        if self.discoverer.log_df is None:
            raise LogNotLoadedException

        if self.discoverer.ko_activities is None:
            raise EmptyKnockoutActivitiesException

        # Discover rules in knockout ko_activities with chosen algorithm
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
                param_grid = {"prune_size": [0.5, 0.8, 0.9], "k": [2], "dl_allowance": [1, 2, 4, 8],
                              "n_discretize_bins": [10, 20]}
            elif algorithm == "IREP":
                param_grid = {"prune_size": [0.5, 0.8, 0.9], "n_discretize_bins": [10, 20]}

        try:
            rulesets = find_ko_rulesets(self.rule_discovery_log_df, self.discoverer.ko_activities,
                                        available_cases_before_ko=self.available_cases_before_ko,
                                        columns_to_ignore=columns_to_ignore, algorithm=algorithm, max_rules=max_rules,
                                        max_rule_conds=max_rule_conds, max_total_conds=max_total_conds, k=k,
                                        n_discretize_bins=n_discretize_bins, dl_allowance=dl_allowance,
                                        prune_size=prune_size, grid_search=grid_search, param_grid=param_grid,
                                        skip_temporal_holdout=self.config.skip_temporal_holdout,
                                        balance_classes=self.config.balance_classes)
        except Exception:
            logging.error(traceback.format_exc())
            raise KnockoutRuleDiscoveryException

        if algorithm == "RIPPER":
            self.RIPPER_rulesets = rulesets
        elif algorithm == "IREP":
            self.IREP_rulesets = rulesets

        self.calc_ko_efforts(confidence_threshold=confidence_threshold, support_threshold=support_threshold,
                             algorithm=algorithm)

        self.report_df = self.build_report(omit=omit_report)

        ko_rule_discovery_stats = {}
        if print_rule_discovery_stats:
            ko_rule_discovery_stats = self.get_ko_rulesets_stats(algorithm=algorithm)

        return self.report_df, self, ko_rule_discovery_stats

    def get_ko_rulesets_stats(self, algorithm):

        rulesets = None
        if algorithm == "RIPPER":
            rulesets = self.RIPPER_rulesets
        elif algorithm == "IREP":
            rulesets = self.IREP_rulesets

        if rulesets is None:
            return

        print(f"\n{algorithm}")

        ko_rule_discovery_stats = {}
        for key in rulesets.keys():
            entry = rulesets[key]
            model = entry[0]
            params = entry[1]
            metrics = entry[2]

            if len(model.ruleset_) == 0:
                continue

            print(f"\n\"{key}\"")
            print(f"{algorithm} parameters: ", params)
            pprint.pprint(metrics)
            ko_rule_discovery_stats[key] = metrics

        return ko_rule_discovery_stats

    def build_report(self, omit=False, use_cache=False):
        if (not use_cache) or (self.report_df is None):
            if self.ruleset_algorithm is None:
                return

            if self.ruleset_algorithm == "RIPPER":
                rulesets = self.RIPPER_rulesets
            else:
                rulesets = self.IREP_rulesets

            _by_case = self.discoverer.log_df.drop_duplicates(
                subset=[globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME])

            freqs = calc_available_cases_before_ko(self.discoverer.ko_activities,
                                                   self.discoverer.log_df)
            if not self.one_timestamp:
                overprocessing_waste = calc_overprocessing_waste(self.discoverer.ko_activities, self.discoverer.log_df)
                processing_waste = calc_processing_waste(self.discoverer.ko_activities, self.discoverer.log_df)

                if self.config.skip_slow_time_waste_metrics:
                    waiting_time_waste = {activity: 0 for activity in self.discoverer.ko_activities}
                else:
                    waiting_time_waste = calc_waiting_time_waste_v2(self.discoverer.ko_activities,
                                                                    self.discoverer.log_df)

            filtered = self.discoverer.log_df[self.discoverer.log_df['knocked_out_case'] == False]
            total_non_ko_cases = filtered.groupby([globalColumnNames.PM4PY_CASE_ID_COLUMN_NAME]).ngroups

            entries = []
            for ko in self.discoverer.ko_activities:

                classifier = rulesets[ko][0]
                metrics = rulesets[ko][2]
                # TODO: decide what metric(s) to show in the main table... roc_auc_cv?

                report_entry = {("%s" % globalColumnNames.REPORT_COLUMN_KNOCKOUT_CHECK): ko,
                                globalColumnNames.REPORT_COLUMN_TOTAL_FREQ:
                                    freqs[ko],
                                globalColumnNames.REPORT_COLUMN_CASE_FREQ:
                                    f"{round(100 * freqs[ko] / _by_case.shape[0], ndigits=2)} %",
                                globalColumnNames.REPORT_COLUMN_REJECTION_RATE: f"{round(100 * self.ko_stats[ko]['rejection_rate'], ndigits=2)} %",
                                f"{globalColumnNames.REPORT_COLUMN_REJECTION_RULE} ({self.ruleset_algorithm})":
                                    out_pretty(classifier.ruleset_),
                                # globalColumnNames.REPORT_COLUMN_BALANCED_ACCURACY: f"{round(100 * metrics['balanced_accuracy'], ndigits=2)} %",
                                globalColumnNames.REPORT_COLUMN_EFFORT_PER_KO: round(
                                    self.ko_stats[ko][self.ruleset_algorithm]["effort"],
                                    ndigits=2)
                                }

                if self.one_timestamp:
                    entries.append(report_entry)
                    continue
                else:
                    report_entry[globalColumnNames.REPORT_COLUMN_MEAN_PT] = seconds_to_hms(
                        self.ko_stats[ko]["mean_pt"])
                    report_entry[globalColumnNames.REPORT_COLUMN_TOTAL_OVERPROCESSING_WASTE] = seconds_to_hms(
                        overprocessing_waste[ko])
                    report_entry[globalColumnNames.REPORT_COLUMN_TOTAL_PT_WASTE] = seconds_to_hms(processing_waste[ko])
                    report_entry[globalColumnNames.REPORT_COLUMN_WT_WASTE] = seconds_to_hms(waiting_time_waste[ko])
                    report_entry[globalColumnNames.REPORT_COLUMN_MEAN_WT_WASTE] = seconds_to_hms(
                        waiting_time_waste[ko] / total_non_ko_cases)

                entries.append(report_entry)

            self.report_df = pd.DataFrame(entries)
            self.report_df.sort_values(by=[globalColumnNames.REPORT_COLUMN_KNOCKOUT_CHECK], inplace=True,
                                       ignore_index=True)

        if not omit:
            self.report_df.to_csv(self.report_file_name, index=False)
            print(tabulate(self.report_df, headers='keys', showindex="false", tablefmt="fancy_grid"))

        return self.report_df


if __name__ == "__main__":
    _log, _config = read_log_and_config("test/config", "synthetic_example_enriched.json",
                                        "cache/synthetic_example")
    analyzer = KnockoutAnalyzer(log_df=_log,
                                config=_config,
                                config_file_name="synthetic_example_enriched.json",
                                config_dir="test/config",
                                cache_dir="cache/synthetic_example",
                                always_force_recompute=False,
                                quiet=True)

    analyzer.discover_knockouts()

    analyzer.compute_ko_rules(grid_search=False, algorithm="RIPPER", confidence_threshold=0.1, support_threshold=0.5,
                              print_rule_discovery_stats=True, dl_allowance=_config.dl_allowance,
                              k=_config.k,
                              n_discretize_bins=_config.n_discretize_bins,
                              prune_size=_config.prune_size,
                              max_rules=_config.max_rules,
                              max_rule_conds=_config.max_rule_conds)
