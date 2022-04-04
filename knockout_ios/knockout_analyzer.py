import glob
import itertools
import os
import pprint
from copy import deepcopy

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
        self.cache_dir = cache_dir
        self.ko_discovery_metrics = None
        self.RIPPER_rulesets = None
        self.IREP_rulesets = None
        self.aggregated_by_case_df = None
        self.ko_stats = {}

        self.always_force_recompute = always_force_recompute

        if self.always_force_recompute:
            self.clear_cache(cache_dir, config_file_name)

        if not self.quiet:
            print(f"Starting Knockout Analyzer with config file \"{config_file_name}\"\n")

        self.discoverer = KnockoutDiscoverer(config_file_name=config_file_name,
                                             config_dir=config_dir,
                                             cache_dir=cache_dir,
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
            self.ko_stats[activity] = {"rejection_rate": 0, "effort": 0}

        # Populate dict with rejection rates and efforts (avg. KO activity durations)
        self.calc_rejection_rates()
        self.calc_ko_efforts()

        if not self.quiet:
            print("\nK.O. rejection rates & efforts:\n")
            pprint.pprint(self.ko_stats)

    def calc_rejection_rates(self, omit_print=True):
        rejection_rates = find_rejection_rates(self.discoverer.log_df, self.discoverer.ko_activities)
        rejection_rates = find_rejection_rates(self.discoverer.log_df, self.discoverer.ko_activities)

        # Update rejection rates
        for k, v in rejection_rates.items():
            self.ko_stats[k]['rejection_rate'] = v

        if not (self.quiet or omit_print):
            print("\nK.O. rejection rates:\n")
            pprint.pprint(rejection_rates)

    def calc_ko_efforts(self, omit_print=True):
        # average processing time of the knock-out check activity
        soj_time = soj_time_get.apply(self.discoverer.pm4py_df,
                                      parameters={
                                          soj_time_get.Parameters.TIMESTAMP_KEY: PM4PY_END_TIMESTAMP_COLUMN_NAME,
                                          soj_time_get.Parameters.START_TIMESTAMP_KEY:
                                              PM4PY_START_TIMESTAMP_COLUMN_NAME}
                                      )

        soj_time = {k: v for k, v in soj_time.items() if k in self.ko_stats}

        if not (self.quiet or omit_print):
            print("\nK.O. activity efforts (avg. time spent in each activity):\n")
            pprint.pprint(soj_time)

        # Update KO rejection rates
        for k, v in soj_time.items():
            self.ko_stats[k]['effort'] = round(v, ndigits=3)

        pass

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
                            n_discretize_bins=5,
                            dl_allowance=16,
                            prune_size=0.33,
                            grid_search=True,
                            param_grid=None,
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
                                                param_grid=param_grid
                                                )

        if not self.quiet:
            self.print_ko_rulesets(algorithm="RIPPER")

        return deepcopy(self.RIPPER_rulesets)

    def get_ko_rules_IREP(self, max_rules=None,
                          max_rule_conds=None,
                          max_total_conds=None,
                          n_discretize_bins=9,
                          prune_size=0.33,
                          grid_search=True,
                          param_grid=None
                          ):

        if self.discoverer.log_df is None:
            raise Exception("log not yet loaded")

        if self.discoverer.ko_activities is None:
            raise Exception("ko activities not yet discovered")

        columns_to_ignore = self.preprocess_for_rule_discovery()

        if not self.quiet:
            print("\nDiscovering rulesets of each K.O. activity with IREP")

        if grid_search & (param_grid is None):
            param_grid = {"prune_size": [0.33, 0.5, 0.7], "n_discretize_bins": [3, 6, 9]}

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
                                              param_grid=param_grid
                                              )

        if not self.quiet:
            self.print_ko_rulesets(algorithm="IREP")

        return deepcopy(self.IREP_rulesets)

    def print_ko_rulesets(self, algorithm="RIPPER", compact=False):

        rulesets = None
        if algorithm == "RIPPER":
            rulesets = self.RIPPER_rulesets
        elif algorithm == "IREP":
            rulesets = self.IREP_rulesets

        if rulesets is None:
            return

        for key in rulesets.keys():
            entry = rulesets[key]
            model = entry[0]
            params = entry[1]
            metrics = entry[2]

            if compact:
                print(f"\n\"{key}\" ({algorithm}):")
                print(
                    f"f1 score: {metrics['f1_score']:.2f}, # conditions: {metrics['condition_count']}, # rules: {metrics['rule_count']}"
                )
            else:
                print(f"\n{algorithm} Ruleset for\n\"{key}\":\n")
                model.out_model()
                print(f'\n# conditions: {metrics["condition_count"]}, # rules: {metrics["rule_count"]}')
                # TODO: uncomment to print supp and conf
                print(
                    # f"\nsupport: {metrics['support']:.2f}, confidence: {metrics['confidence']:.2f} "
                    f"\nf1 score: {metrics['f1_score']:.2f}, accuracy: {metrics['accuracy']:.2f}, precision: {metrics['precision']:.2f}, recall: {metrics['recall']:.2f}"
                )
                print(f"rule discovery params: {params}")


def compare_approach():
    test_data = ("synthetic_example.json", "cache/synthetic_example_A",
                 ['Check Liability', 'Check Risk', 'Check Monthly Income'])

    analyzerA = KnockoutAnalyzer(config_file_name=test_data[0],
                                 cache_dir=test_data[1],
                                 always_force_recompute=False,
                                 quiet=True)

    analyzerA.discover_knockouts(expected_kos=test_data[2])

    analyzerA.get_ko_rules_RIPPER(max_rules=5,
                                  grid_search=True
                                  )

    analyzerA.get_ko_rules_IREP(max_rules=5,
                                grid_search=True
                                )

    test_data = ("synthetic_example.json", "cache/synthetic_example_B",
                 ['Check Liability', 'Check Risk', 'Check Monthly Income'])

    analyzerB = KnockoutAnalyzer(config_file_name=test_data[0],
                                 cache_dir=test_data[1],
                                 always_force_recompute=False,
                                 quiet=True)

    analyzerB.discover_knockouts(expected_kos=test_data[2])

    analyzerB.get_ko_rules_RIPPER(max_rules=5,
                                  grid_search=True
                                  )

    analyzerB.get_ko_rules_IREP(max_rules=5,
                                grid_search=True
                                )

    print("\nWith approach A:")
    analyzerA.print_ko_rulesets(algorithm="RIPPER", compact=True)
    print("\n\nWith approach B:")
    analyzerB.print_ko_rulesets(algorithm="RIPPER", compact=True)

    print("\nWith approach A:")
    analyzerA.print_ko_rulesets(algorithm="IREP", compact=True)
    print("\n\nWith approach B:")
    analyzerB.print_ko_rulesets(algorithm="IREP", compact=True)


if __name__ == "__main__":
    # OVERRIDE_FORCE_RECOMPUTE = False
    # compare_approach()
    # exit(0)

    # from log_generation.LogWithKnockoutsGenerator import LogWithKnockoutsGenerator
    # gen = LogWithKnockoutsGenerator("../log_generation/outputs/synthetic_example_raw.xes")
    # gen.generate_log(1000)

    test_data = ("synthetic_example.json", "cache/synthetic_example",
                 ['Check Liability', 'Check Risk', 'Check Monthly Income'])

    # Known rules
    # 'Check Liability':        'Total Debt'     > 5000
    # 'Check Risk':             'Loan Ammount'   > 10000
    # 'Check Monthly Income':   'Monthly Income' < 1000

    analyzer = KnockoutAnalyzer(config_file_name=test_data[0],
                                cache_dir=test_data[1],
                                always_force_recompute=False,
                                quiet=True)

    analyzer.discover_knockouts(expected_kos=test_data[2])

    # analyzer.get_ko_rules_RIPPER(max_rules=5,
    #                             grid_search=True
    #                             )

    # analyzer.print_ko_rulesets(algorithm="RIPPER")

    analyzer.get_ko_rules_IREP(max_rules=5,
                               grid_search=True
                               )

    analyzer.print_ko_rulesets(algorithm="IREP")

    # TODO: add KO rule discovery support & confidence metrics
    # TODO: multi-class classification?
