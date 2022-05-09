# Knockout ko_activities source:
#
# - automatically discovered: program considers characteristic ko_activities of top available_cases_before_ko shortest variants as
#                             knock-out-check ko_activities
#
# - semi-automatically discovered: user provides activity name(s) associated to negative case outcome(s),
#                                  program uses that info to refine analysis and find knock-out-check ko_activities.
#
# - manual: user provides names of knock-out-check ko_activities  (utility?)

# Ideas:
#
# - user gives negative outcome; program finds activity where they diverge & rules for that -> suggestion for ko check?
# - bring back dtreeviz: make a DT for every knockout activity; if there is a condition that allows to decide faster,
#   implement it as a separate KO check before!
import os
import pprint

import pandas as pd
import pm4py
from tqdm import tqdm

from knockout_ios.utils.constants import *
from knockout_ios.utils.discovery import config_hash_changed, read_config_cache, dump_config_cache, \
    discover_ko_sequences
from knockout_ios.utils.metrics import get_ko_discovery_metrics
from knockout_ios.utils.postprocessing import format_for_post_proc, plot_cycle_times_per_ko_activity, \
    plot_ko_activities_count
from knockout_ios.utils.preprocessing.configuration import read_log_and_config, Configuration


class KnockoutDiscoverer:

    def __init__(self, log_df: pd.DataFrame, config: Configuration, config_file_name: str,
                 cache_dir="cache",
                 config_dir="config",
                 always_force_recompute=True,
                 quiet=False
                 ):

        # TODO: refactor the need for config_dir, cache_dir...

        os.makedirs(cache_dir, exist_ok=True)

        self.config_dir = config_dir
        self.cache_dir = cache_dir
        self.config_file_name = config_file_name
        self.always_force_recompute = always_force_recompute
        self.quiet = quiet

        self.ko_seqs = None
        self.ko_outcomes = None
        self.ko_activities = None
        self.force_recompute = None
        self.ko_rules_classifiers = None

        self.log_df = log_df
        self.config = config

        self.update_should_recompute()

    def update_should_recompute(self):
        self.force_recompute = True

        # Automatically force recompute if pipeline_config changes
        if self.config is None:
            raise Exception("pipeline_config not yet loaded")

        try:
            if self.always_force_recompute:
                raise FileNotFoundError

            config_cache = read_config_cache(self.config_file_name, cache_dir=self.cache_dir)
            self.force_recompute = config_hash_changed(self.config, config_cache)

        except FileNotFoundError:
            self.force_recompute = True
            dump_config_cache(self.config_file_name, self.config, cache_dir=self.cache_dir)

    def find_ko_activities(self):

        if self.config is None:
            raise Exception("pipeline_config not yet loaded")

        self.update_should_recompute()

        # Idea: iteratively increase the limit until finding a positive outcome in the outcome list;
        #       keep last num before this happened

        if self.config.ko_count_threshold is None:
            # count the unique values in the activity column of log_df
            ko_count_threshold = len(self.log_df[PM4PY_ACTIVITY_COLUMN_NAME].unique())
        else:
            ko_count_threshold = self.config.ko_count_threshold

        self.ko_activities, self.ko_outcomes, self.ko_seqs = discover_ko_sequences(self.log_df,
                                                                                   self.config_file_name,
                                                                                   cache_dir=self.cache_dir,
                                                                                   start_activity_name=self.config.start_activity,
                                                                                   known_ko_activities=self.config.known_ko_activities,
                                                                                   negative_outcomes=self.config.negative_outcomes,
                                                                                   positive_outcomes=self.config.positive_outcomes,
                                                                                   limit=ko_count_threshold,
                                                                                   quiet=self.quiet,
                                                                                   force_recompute=self.force_recompute)

        # Do not consider known negative outcomes as ko ko_activities
        if len(self.config.negative_outcomes) > 0:
            self.ko_activities = list(filter(lambda act: not (act in self.config.negative_outcomes),
                                             self.ko_activities))

        if (len(self.ko_outcomes) == 0) or (len(self.ko_activities) == 0):
            print("Error finding knockouts")
            exit(1)

        if not self.quiet:
            print(f"\nNegative outcomes to search in log: {list(self.ko_outcomes)}"
                  f"\nPotential K.O. ko_activities in log: {list(self.ko_activities)}")

        try:
            if self.force_recompute:
                raise FileNotFoundError

            self.log_df = pd.read_pickle(f"./{self.cache_dir}/{self.config_file_name}_with_knockouts.pkl")
            if not self.quiet:
                print(f"\nFound cache for {self.config_file_name} knockouts\n")

        except FileNotFoundError:

            if len(self.config.negative_outcomes) > 0:
                relations = list(map(lambda ca: (self.config.start_activity, ca), self.config.negative_outcomes))
                rejected = pm4py.filter_eventually_follows_relation(self.log_df, relations)
            else:
                relations = list(map(lambda ca: (self.config.start_activity, ca), self.ko_outcomes))
                rejected = pm4py.filter_eventually_follows_relation(self.log_df, relations)

            rejected = pm4py.convert_to_dataframe(rejected)

            # Mark Knocked-out cases & their knock-out activity
            self.log_df['knocked_out_case'] = False
            self.log_df['knockout_activity'] = False

            def find_ko_activity(_ko_activities, _sorted_case):
                case_activities = list(_sorted_case[PM4PY_ACTIVITY_COLUMN_NAME].values)

                idxs = []
                for ko_act in _ko_activities:
                    for i, act in enumerate(case_activities):
                        if ko_act == act:
                            idxs.append((i, ko_act))
                            break

                if len(idxs) > 0:
                    idxs = sorted(idxs, key=lambda e: e[0], reverse=True)
                    last = idxs[0]
                    return last[1]

                return False

            gr = rejected.groupby(PM4PY_CASE_ID_COLUMN_NAME)

            self.log_df.set_index(SIMOD_LOG_READER_CASE_ID_COLUMN_NAME, inplace=True)

            for group in tqdm(gr.groups.keys(), desc="Marking knocked-out cases in log"):
                case_df = gr.get_group(group)
                sorted_case = case_df.sort_values("start_timestamp")
                knockout_activity = find_ko_activity(self.ko_activities, sorted_case)

                self.log_df.at[group, 'knocked_out_case'] = True
                self.log_df.at[group, 'knockout_activity'] = knockout_activity

            self.log_df.reset_index(inplace=True)

            self.log_df.to_pickle(f"./{self.cache_dir}/{self.config_file_name}_with_knockouts.pkl")

        self.ko_activities = list(filter(lambda act: act, set(self.log_df['knockout_activity'])))

        # Throw error when no KOs are distinguished (all cases are considered 'knocked out')
        # Ask user for more info
        if self.log_df['knocked_out_case'].all():
            raise Exception("No K.O. ko_activities could be distinguished."
                            "\n\nSuggestions:"
                            "\n- Reduce the ko_count_threshold"
                            "\n- Provide negative outcome activity name(s)"
                            "\n- Provide positive outcome activity name(s)")
        elif not self.quiet:
            print(f"\nNegative outcomes found in log: {list(self.ko_outcomes)}"
                  f"\nK.O. ko_activities found in log: {list(self.ko_activities)}")

    def print_basic_stats(self):
        # Basic impact assessment
        # See processing time distributions for cases that have the knockout and end in negative end
        # vs. cases that don't get knockout out but have negative end

        if self.log_df is None:
            raise Exception("log not yet processed")

        aggregated_df = format_for_post_proc(self.log_df)

        plot_cycle_times_per_ko_activity(aggregated_df, self.ko_activities)
        plot_ko_activities_count(aggregated_df)

    def get_activities(self):
        return list(set(self.log_df[PM4PY_ACTIVITY_COLUMN_NAME]))

    def get_discovery_metrics(self, expected_kos):

        if self.ko_activities is None:
            raise Exception("ko ko_activities not yet computed")

        return get_ko_discovery_metrics(self.get_activities(), expected_kos, self.ko_activities)


if __name__ == "__main__":
    test_data = ("credit_app_simple.json", ['Assess application', 'Check credit history', 'Check income sources'])

    log, configuration = read_log_and_config("config", "credit_app_simple.json", "cache/credit_app")

    analyzer = KnockoutDiscoverer(log_df=log, config=configuration, config_file_name=test_data[0],
                                  cache_dir="cache/credit_app",
                                  always_force_recompute=True, quiet=False)

    analyzer.find_ko_activities()
    analyzer.print_basic_stats()
    pprint.pprint(analyzer.get_discovery_metrics(test_data[1]))
