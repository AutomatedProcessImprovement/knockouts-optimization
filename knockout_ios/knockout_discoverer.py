# Knockout activities source:
#
# - automatically discovered: program considers characteristic activities of top available_cases_before_ko shortest variants as
#                             knock-out-check activities
#
# - semi-automatically discovered: user provides activity name(s) associated to negative case outcome(s),
#                                  program uses that info to refine analysis and find knock-out-check activities.
#
# - manual: user provides names of knock-out-check activities  (utility?)

# Ideas:
#
# - user gives negative outcome; program finds activity where they diverge & rules for that -> suggestion for ko check?
# - bring back dtreeviz: make a DT for every knockout activity; if there is a condition that allows to decide faster,
#   implement it as a separate KO check before!
import os
import pprint

import numpy as np
import pm4py
from tqdm import tqdm

from knockout_ios.utils.discovery import *

from knockout_ios.utils.discovery import config_hash_changed, read_config_cache, dump_config_cache
from knockout_ios.utils.metrics import get_ko_discovery_metrics
from knockout_ios.utils.post_proc import format_for_post_proc, plot_cycle_times_per_ko_activity, \
    plot_ko_activities_count
from knockout_ios.utils.pre_proc import preprocess


class KnockoutDiscoverer:

    def __init__(self, config_file_name, cache_dir="cache", config_dir="config", always_force_recompute=True,
                 quiet=False):

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
        self.log_df = None
        self.pm4py_formatted_df = None
        self.config = None

        self.read_log_and_config()

        self.force_recompute = True
        self.update_should_recompute()

    def read_log_and_config(self):
        # TODO: document that add_intercase_and_context and add_only_context are False
        self.log_df, self.config = preprocess(config_file=f"./{self.config_dir}/{self.config_file_name}",
                                              config_dir=self.config_dir,
                                              cache_dir=self.cache_dir,
                                              add_intercase_and_context=False,
                                              clean_processing_times=False, add_only_context=False)

        # TODO: document why this is needed - for the moment, it's just to workaround the fact that None cannot be
        #  serialized, so the best the Log Generation module can do is write an empty string...
        # replace empty strings in log_df with NaN
        self.log_df = self.log_df.replace("", np.nan)

        self.pm4py_formatted_df = pm4py.format_dataframe(self.log_df, case_id='caseid', activity_key='task',
                                                         timestamp_key=SIMOD_END_TIMESTAMP_COLUMN_NAME,
                                                         start_timestamp_key=SIMOD_START_TIMESTAMP_COLUMN_NAME)

    def update_should_recompute(self):
        # Automatically force recompute if config changes
        if self.config is None:
            raise Exception("config not yet loaded")

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
            raise Exception("config not yet loaded")

        self.update_should_recompute()

        # Idea: iteratively increase the limit until finding a positive outcome in the outcome list;
        #       keep last num before this happened

        if self.config.ko_count_threshold is None:
            # count the unique values in the activity column of pm4py_formatted_df
            ko_count_threshold = len(self.pm4py_formatted_df[PM4PY_ACTIVITY_COLUMN_NAME].unique())
        else:
            ko_count_threshold = self.config.ko_count_threshold

        self.ko_activities, self.ko_outcomes, self.ko_seqs = discover_ko_sequences(self.pm4py_formatted_df,
                                                                                   self.config_file_name,
                                                                                   cache_dir=self.cache_dir,
                                                                                   start_activity_name=self.config.start_activity,
                                                                                   known_ko_activities=self.config.known_ko_activities,
                                                                                   negative_outcomes=self.config.negative_outcomes,
                                                                                   positive_outcomes=self.config.positive_outcomes,
                                                                                   limit=ko_count_threshold,
                                                                                   quiet=self.quiet,
                                                                                   force_recompute=self.force_recompute)

        # Do not consider known negative outcomes as ko activities
        if len(self.config.negative_outcomes) > 0:
            self.ko_activities = list(filter(lambda act: not (act in self.config.negative_outcomes),
                                             self.ko_activities))

        if (len(self.ko_outcomes) == 0) or (len(self.ko_activities) == 0):
            print("Error finding knockouts")
            exit(1)

        if not self.quiet:
            print(f"\nNegative outcomes to search in log: {list(self.ko_outcomes)}"
                  f"\nPotential K.O. activities in log: {list(self.ko_activities)}")

        try:
            if self.force_recompute:
                raise FileNotFoundError

            self.log_df = pd.read_pickle(f"./{self.cache_dir}/{self.config_file_name}_with_knockouts.pkl")
            self.pm4py_formatted_df = pd.read_pickle(
                f"./{self.cache_dir}/{self.config_file_name}_pm4pyf_with_knockouts.pkl")
            if not self.quiet:
                print(f"\nFound cache for {self.config_file_name} knockouts\n")

        except FileNotFoundError:

            if len(self.config.negative_outcomes) > 0:
                relations = list(map(lambda ca: (self.config.start_activity, ca), self.config.negative_outcomes))
                rejected = pm4py.filter_eventually_follows_relation(self.pm4py_formatted_df, relations)
            else:
                relations = list(map(lambda ca: (self.config.start_activity, ca), self.ko_outcomes))
                rejected = pm4py.filter_eventually_follows_relation(self.pm4py_formatted_df, relations)

            rejected = pm4py.convert_to_dataframe(rejected)

            # Mark Knocked-out cases & their knock-out activity
            self.log_df['knocked_out_case'] = False
            self.log_df['knockout_activity'] = False

            self.pm4py_formatted_df['knocked_out_case'] = False
            self.pm4py_formatted_df['knockout_activity'] = False

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
            self.pm4py_formatted_df.set_index(PM4PY_CASE_ID_COLUMN_NAME, inplace=True)

            for group in tqdm(gr.groups.keys(), desc="Marking knocked-out cases in log"):
                case_df = gr.get_group(group)
                sorted_case = case_df.sort_values("start_timestamp")
                knockout_activity = find_ko_activity(self.ko_activities, sorted_case)

                self.log_df.at[group, 'knocked_out_case'] = True
                self.log_df.at[group, 'knockout_activity'] = knockout_activity

                self.pm4py_formatted_df.at[group, 'knocked_out_case'] = True
                self.pm4py_formatted_df.at[group, 'knockout_activity'] = knockout_activity

            self.log_df.reset_index(inplace=True)
            self.pm4py_formatted_df.reset_index(inplace=True)

            self.log_df.to_pickle(f"./{self.cache_dir}/{self.config_file_name}_with_knockouts.pkl")
            self.pm4py_formatted_df.to_pickle(f"./{self.cache_dir}/{self.config_file_name}_pm4pyf_with_knockouts.pkl")

        self.ko_activities = list(filter(lambda act: act, set(self.log_df['knockout_activity'])))

        # Throw error when no KOs are distinguished (all cases are considered 'knocked out')
        # Ask user for more info
        if self.log_df['knocked_out_case'].all():
            raise Exception("No K.O. activities could be distinguished."
                            "\n\nSuggestions:"
                            "\n- Reduce the ko_count_threshold"
                            "\n- Provide negative outcome activity name(s)"
                            "\n- Provide positive outcome activity name(s)")
        elif not self.quiet:
            print(f"\nNegative outcomes found in log: {list(self.ko_outcomes)}"
                  f"\nK.O. activities found in log: {list(self.ko_activities)}")

    def print_basic_stats(self):
        # Basic impact assessment
        # See processing time distributions for cases that have the knockout and end in negative end
        # vs. cases that don't get knockout out but have negative end

        if self.pm4py_formatted_df is None:
            raise Exception("pm4py-formatted log missing")

        aggregated_df = format_for_post_proc(self.pm4py_formatted_df)

        plot_cycle_times_per_ko_activity(aggregated_df, self.ko_activities)
        plot_ko_activities_count(aggregated_df)

    def get_activities(self):
        return list(set(self.pm4py_formatted_df[PM4PY_ACTIVITY_COLUMN_NAME]))

    def get_discovery_metrics(self, expected_kos):

        if self.ko_activities is None:
            raise Exception("ko activities not yet computed")

        return get_ko_discovery_metrics(self.get_activities(), expected_kos, self.ko_activities)


if __name__ == "__main__":
    test_data = ("credit_app_simple.json", ['Assess application', 'Check credit history', 'Check income sources'])
    analyzer = KnockoutDiscoverer(config_file_name=test_data[0], cache_dir="cache/credit_app",
                                  always_force_recompute=True)

    analyzer.find_ko_activities()
    analyzer.print_basic_stats()
    pprint.pprint(analyzer.get_discovery_metrics(test_data[1]))
