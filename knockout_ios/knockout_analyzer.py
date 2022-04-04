import glob
import itertools
import os
import pprint
from copy import deepcopy

from tabulate import tabulate

import pandas as pd
from pm4py.statistics.sojourn_time.pandas import get as soj_time_get

from knockout_ios.knockout_discoverer import KnockoutDiscoverer
from knockout_ios.utils.analysis import find_rejection_rates
from knockout_ios.utils.constants import *
from knockout_ios.utils.explainer import find_ko_rulesets

# TODO: remove this later
OVERRIDE_FORCE_RECOMPUTE = True


class KnockoutAnalyzer:

    def __init__(self, config_file_name, cache_dir="cache", config_dir="config", always_force_recompute=False,
                 quiet=False):

        self.quiet = quiet
        self.config_dir = config_dir
        self.cache_dir = cache_dir
        self.ko_discovery_metrics = None
        self.RIPPER_rulesets = None
        self.IREP_rulesets = None
        self.aggregated_by_case_df = None
        self.ko_stats = {}

        self.always_force_recompute = always_force_recompute

        if self.always_force_recompute:
            self.clear_cache(self.cache_dir, config_file_name)

        if not self.quiet:
            print(f"Starting Knockout Analyzer with config file \"{config_file_name}\"\n")

        self.discoverer = KnockoutDiscoverer(config_file_name=config_file_name,
                                             config_dir=self.config_dir,
                                             cache_dir=self.cache_dir,
                                             always_force_recompute=always_force_recompute,
                                             quiet=quiet)

    def clear_cache(self, cachedir, config_file_name):
        file_list = glob.glob(f'{cachedir}/{config_file_name}*')
        for filePath in file_list:
            try:
                os.remove(filePath)
            except:
                print("Error while deleting file : ", filePath)

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

    def calc_rejection_rates(self):
        rejection_rates = find_rejection_rates(self.discoverer.log_df, self.discoverer.ko_activities)

        # Update rejection rates
        for k, v in rejection_rates.items():
            self.ko_stats[k]['rejection_rate'] = v

        if not self.quiet:
            print("\nK.O. rejection rates:\n")
            pprint.pprint(rejection_rates)

    def calc_ko_efforts(self, support_threshold=0.5, confidence_threshold=0.5, algorithm="IREP"):

        # TODO: add support for different algorithms (separate ko stats dicts?)
        if algorithm == "RIPPER":
            rulesets = self.RIPPER_rulesets
        else:
            rulesets = self.IREP_rulesets

        # average processing time of the knock-out check activity
        soj_time = soj_time_get.apply(self.discoverer.pm4py_df,
                                      parameters={
                                          soj_time_get.Parameters.TIMESTAMP_KEY: PM4PY_END_TIMESTAMP_COLUMN_NAME,
                                          soj_time_get.Parameters.START_TIMESTAMP_KEY:
                                              PM4PY_START_TIMESTAMP_COLUMN_NAME}
                                      )

        soj_time = {k: v for k, v in soj_time.items() if k in self.discoverer.ko_activities}

        if not self.quiet:
            print("\navg. time spent in each K.O. activity:\n")
            pprint.pprint(soj_time)

        # Update KO rejection rates
        for key in rulesets.keys():
            entry = rulesets[key]
            metrics = entry[2]

            # Effort per rejection = Average PT / Rejection rate
            effort = round(soj_time[key], ndigits=3) / (100 * self.ko_stats[key]['rejection_rate'])

            if (metrics['confidence'] >= confidence_threshold):  # or (metrics['support'] >= support_threshold):
                # Effort per rejection = (Average PT / Rejection rate) * Confidence
                effort = effort * metrics['confidence']

            self.ko_stats[key]['effort'] = effort
            self.ko_stats[key]['mean_pt'] = round(soj_time[key], ndigits=3)

    def calc_ko_discovery_metrics(self, expected_kos):
        self.ko_discovery_metrics = self.discoverer.get_discovery_metrics(expected_kos)

        if not self.quiet:
            pprint.pprint(self.ko_discovery_metrics)

        return deepcopy(self.ko_discovery_metrics)

    def preprocess_for_rule_discovery(self):

        if self.discoverer.log_df is None:
            raise Exception("log not yet loaded")

        # Pre-processing
        columns_to_ignore = [SIMOD_LOG_READER_CASE_ID_COLUMN_NAME,
                             SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME,
                             PM4PY_ACTIVITY_COLUMN_NAME,
                             PM4PY_CASE_ID_COLUMN_NAME,
                             PM4PY_END_TIMESTAMP_COLUMN_NAME,
                             DURATION_COLUMN_NAME,
                             SIMOD_RESOURCE_COLUMN_NAME, SIMOD_END_TIMESTAMP_COLUMN_NAME,
                             SIMOD_START_TIMESTAMP_COLUMN_NAME,
                             'knockout_activity',
                             'knockout_prefix']

        columns_to_ignore = list(itertools.chain(columns_to_ignore, list(filter(
            lambda c: ('@' in c) | ('id' in c.lower()) | ('daytime' in c) | ('weekday' in c) | ('month' in c) | (
                    'activity' in c.lower()),
            self.discoverer.log_df.columns))))

        # skip the rest of pre-proc in case it has been already done
        if (self.RIPPER_rulesets is not None) or (self.IREP_rulesets is not None):
            return columns_to_ignore

        self.discoverer.log_df = self.discoverer.log_df.dropna()

        for attr in self.discoverer.log_df.columns:
            self.discoverer.log_df[attr] = \
                pd.to_numeric(self.discoverer.log_df[attr], errors='ignore')

        return columns_to_ignore

    def get_ko_rules_RIPPER(self,
                            max_rules=None,
                            max_rule_conds=None,
                            max_total_conds=None,
                            k=2,
                            n_discretize_bins=9,
                            dl_allowance=16,
                            prune_size=0.33,
                            grid_search=False,
                            param_grid=None,
                            bucketing_approach="A"
                            ):

        if self.discoverer.log_df is None:
            raise Exception("log not yet loaded")

        if self.discoverer.ko_activities is None:
            raise Exception("ko activities not yet discovered")

        # Discover rules in knockout activities with RIPPER algorithm

        columns_to_ignore = self.preprocess_for_rule_discovery()

        if not self.quiet:
            print("\nDiscovering rulesets of each K.O. activity with RIPPER")

        if grid_search & (param_grid is None):
            param_grid = {"prune_size": [0.33, 0.5, 0.7], "k": [1, 2, 4], "dl_allowance": [16, 32, 64],
                          "n_discretize_bins": [3, 6, 9]}

        self.RIPPER_rulesets = find_ko_rulesets(self.discoverer.log_df,
                                                self.discoverer.ko_activities,
                                                self.discoverer.config_file_name,
                                                self.cache_dir,
                                                force_recompute=OVERRIDE_FORCE_RECOMPUTE,
                                                columns_to_ignore=columns_to_ignore,
                                                algorithm="RIPPER",
                                                max_rules=max_rules,
                                                max_rule_conds=max_rule_conds,
                                                max_total_conds=max_total_conds,
                                                k=k,
                                                n_discretize_bins=n_discretize_bins,
                                                dl_allowance=dl_allowance,
                                                prune_size=prune_size,
                                                grid_search=grid_search,
                                                param_grid=param_grid,
                                                bucketing_approach=bucketing_approach,
                                                )

        if not self.quiet:
            self.print_ko_rulesets(algorithm="RIPPER")

        return self

    def get_ko_rules_IREP(self, max_rules=None,
                          max_rule_conds=None,
                          max_total_conds=None,
                          n_discretize_bins=9,
                          prune_size=0.33,
                          grid_search=True,
                          param_grid=None,
                          bucketing_approach="A"
                          ):

        if self.discoverer.log_df is None:
            raise Exception("log not yet loaded")

        if self.discoverer.ko_activities is None:
            raise Exception("ko activities not yet discovered")

        columns_to_ignore = self.preprocess_for_rule_discovery()

        if not self.quiet:
            print("\nDiscovering rulesets of each K.O. activity with IREP")

        if grid_search & (param_grid is None):
            param_grid = {"prune_size": [0.2, 0.33, 0.5], "n_discretize_bins": [4, 8, 12]}

        self.IREP_rulesets = find_ko_rulesets(self.discoverer.log_df,
                                              self.discoverer.ko_activities,
                                              self.discoverer.config_file_name,
                                              self.cache_dir,
                                              force_recompute=OVERRIDE_FORCE_RECOMPUTE,
                                              algorithm="IREP",
                                              columns_to_ignore=columns_to_ignore,
                                              max_rules=max_rules,
                                              max_rule_conds=max_rule_conds,
                                              max_total_conds=max_total_conds,
                                              n_discretize_bins=n_discretize_bins,
                                              prune_size=prune_size,
                                              grid_search=grid_search,
                                              param_grid=param_grid,
                                              bucketing_approach=bucketing_approach,
                                              )

        if not self.quiet:
            self.print_ko_rulesets(algorithm="IREP")

        return self

    def print_ko_rulesets(self, algorithm="RIPPER", compact=False):

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
                    f"\nf1 score: {metrics['f1_score']:.2f}, accuracy: {metrics['accuracy']:.2f}, precision: {metrics['precision']:.2f}, recall: {metrics['recall']:.2f}"
                )
            else:
                print(f"\n\"{key}\":")
                model.out_model()
                print(f'\n# conditions: {metrics["condition_count"]}, # rules: {metrics["rule_count"]}')
                print(
                    f"support: {metrics['support']:.2f}, confidence: {metrics['confidence']:.2f} "
                    f"\nf1 score: {metrics['f1_score']:.2f}, accuracy: {metrics['accuracy']:.2f}, precision: {metrics['precision']:.2f}, recall: {metrics['recall']:.2f}"
                )
                print(f"rule discovery params: {params}")

    def build_report(self, algorithm="IREP", omit=False):

        if algorithm == "RIPPER":
            rulesets = self.RIPPER_rulesets
        else:
            rulesets = self.IREP_rulesets

        _by_case = self.discoverer.log_df.drop_duplicates(subset=[SIMOD_LOG_READER_CASE_ID_COLUMN_NAME])

        entries = []
        for ko in self.discoverer.ko_activities:
            freq = _by_case[_by_case["knockout_activity"] == ko].shape[0]
            entries.append({"Knock-out check": ko,
                            "Total frequency":
                                freq,
                            "Case frequency":
                                f"{freq / _by_case.shape[0]} %",
                            "Mean PT": self.ko_stats[ko]["mean_pt"],
                            "Rejection rate": self.ko_stats[ko]["rejection_rate"],
                            "Rejection rule": rulesets[ko][0].ruleset_,
                            "Effort per rejection": self.ko_stats[ko]["effort"]
                            })

        df = pd.DataFrame(entries)

        if not omit:
            print(tabulate(df, headers='keys', tablefmt='psql'))

        return df


if __name__ == "__main__":
    analyzer = KnockoutAnalyzer(config_file_name="synthetic_example.json",
                                config_dir="config",
                                cache_dir="cache/synthetic_example",
                                always_force_recompute=True,
                                quiet=True)

    analyzer.discover_knockouts(expected_kos=['Check Liability', 'Check Risk', 'Check Monthly Income'])

    analyzer.get_ko_rules_IREP(grid_search=True)

    analyzer.calc_ko_efforts(support_threshold=0.5, confidence_threshold=0.5, algorithm="IREP")
    analyzer.build_report(algorithm="IREP")

# TODO: work on different pending parts (clean up)

# TODOs - related to KO Rule stage
# TODO: area under curve metric for grid search?
# TODO: fix support calculation: tomar en cuenta no todo N, sino solo los casos que recibe el KO check
# TODO: Manejar nulos, no el valor “N/A”
# TODO: Ver como calcular supp & conf por cada regla del ruleset, limpar segun threshold

# TODOs - related to time waste metrics
# TODO: implement time waste metrics & add columns to report (and test) - first 2, hardest to the last
