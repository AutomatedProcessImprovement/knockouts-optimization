import pickle

import pandas as pd
from tabulate import tabulate

from knockout_ios.knockout_analyzer import KnockoutAnalyzer
from knockout_ios.utils.configuration import read_log_and_config
from knockout_ios.utils.constants import REPORT_COLUMN_KNOCKOUT_CHECK

from knockout_ios.utils.redesign import evaluate_knockout_relocation_io, \
    evaluate_knockout_rule_change_io, evaluate_knockout_reordering_io
from knockout_ios.utils.synthetic_example.preprocessors import enrich_log_with_fully_known_attributes

'''
KO redesign strategies

1. Relocate a knock-out
   place the knock-out check as early in the process as the attribute value (based on which the knock-out is performed)
   is available.

2. Reorder knock-outs
   according to the knock-out principle:
   checks are ordered according to the principle of “least effort to reject”– checks that require less effort and are more
   likely to reject the case come first. In addition, consider the knock-out rule so that the attribute value (based on which
   the knock-out is performed) is available at that point in the process.

3. Change the knock-out rule
   change the attribute value (or its range) based on which the knock-out is performed;

NOT IN CURRENT SCOPE:

4. Remove the knock-out

5. Add a new knock-out
'''


def read_analyzer_cache(cache_dir, cache_name) -> KnockoutAnalyzer:
    binary_file = open(f'{cache_dir}/{cache_name}', 'rb')
    ko_analyzer = pickle.load(binary_file)
    binary_file.close()
    return ko_analyzer


def dump_analyzer_cache(ko_analyzer, cache_dir, cache_name):
    binary_file = open(f'{cache_dir}/{cache_name}', 'wb')
    pickle.dump(ko_analyzer, binary_file)
    binary_file.close()


class KnockoutRedesignAdviser(object):
    def __init__(self, knockout_analyzer: KnockoutAnalyzer, quiet=False, attribute_range_confidence_interval=0.99):
        self.knockout_analyzer = knockout_analyzer
        self.quiet = quiet
        self.attribute_range_confidence_interval = attribute_range_confidence_interval
        self.redesign_options = {}

    def compute_redesign_options(self):
        self.redesign_options["relocation"] = evaluate_knockout_relocation_io(self.knockout_analyzer)
        self.redesign_options["reordering"] = evaluate_knockout_reordering_io(self.knockout_analyzer,
                                                                              dependencies=self.redesign_options[
                                                                                  "relocation"])
        self.redesign_options["rule_change"], raw_rulesets = evaluate_knockout_rule_change_io(self.knockout_analyzer,
                                                                                              self.attribute_range_confidence_interval)

        if not self.quiet:
            print(f"\n** Redesign options **\n")

            # TODO: cleaner printing/reporting method...
            if "relocation" in self.redesign_options:
                print("> Knock-out Re-location")
                entries = []
                attribute_dependencies_dict = self.redesign_options["relocation"]
                for activity in attribute_dependencies_dict.keys():
                    if not (len(attribute_dependencies_dict[activity]) > 0):
                        entries.append(
                            {REPORT_COLUMN_KNOCKOUT_CHECK: activity,
                             "Dependencies": "required attributes are available from the start."})
                        continue

                    dependencies_str = ""
                    for pair in attribute_dependencies_dict[activity]:
                        dependencies_str += f"'{pair[0]}', available after activity '{pair[1]}'"

                    entries.append(
                        {REPORT_COLUMN_KNOCKOUT_CHECK: activity,
                         "Dependencies": dependencies_str})

                df = pd.DataFrame(entries)
                df.sort_values(by="Dependencies", inplace=True)
                print(tabulate(df, headers='keys', showindex="false", tablefmt="fancy_grid"))

            if "reordering" in self.redesign_options:
                print("\n\n> Knock-out Re-ordering\n")
                optimal_order = [f'{i + 1}. {ko}' + '\n' for i, ko in
                                 enumerate(self.redesign_options['reordering']['optimal_order_names'])]
                observed_order = [f'{i + 1}. {ko}' + '\n' for i, ko in
                                  enumerate(self.redesign_options['reordering']['observed_ko_order'])]
                cases_respecting_order = self.redesign_options['reordering']['cases_respecting_order']
                total_cases = self.redesign_options['reordering']['total_cases']

                print(
                    "Observed Order of Knock-out checks:\n" + f"{''.join(observed_order)}")
                print(
                    "Optimal Order of Knock-out checks (taking into account attribute dependencies):\n" + f"{''.join(optimal_order)}\n{cases_respecting_order}/{total_cases} non-knocked-out case(s) follow it.")

            if "rule_change" in self.redesign_options:
                print("\n\n> Knock-out rule value ranges\n")

                entries = []

                rule_attribute_ranges_dict = self.redesign_options["rule_change"]
                for activity in rule_attribute_ranges_dict.keys():
                    confidence_intervals_string = f"Rule:\n{raw_rulesets[activity]}"
                    if not (len(raw_rulesets[activity]) > 0):
                        continue
                    confidence_intervals_string += f"\n\nfor {self.attribute_range_confidence_interval * 100:.0f}% of knocked out cases, these attributes fall within the ranges:"
                    for attribute in rule_attribute_ranges_dict[activity]:
                        confidence_intervals_string += f"\n- {attribute}: {rule_attribute_ranges_dict[activity][attribute][0]:.2f} - {rule_attribute_ranges_dict[activity][attribute][1]:.2f}"

                    confidence_intervals_string += "\n"
                    entries.append(
                        {REPORT_COLUMN_KNOCKOUT_CHECK: activity, "Observation": confidence_intervals_string})

                df = pd.DataFrame(entries)
                print(tabulate(df, headers='keys', showindex="false", tablefmt="fancy_grid"))

        return self.redesign_options


if __name__ == "__main__":
    log, configuration = read_log_and_config("config", "synthetic_example_ko_order_io.json",
                                             "cache/synthetic_example")

    analyzer = KnockoutAnalyzer(log_df=log,
                                config=configuration,
                                config_file_name="synthetic_example_ko_order_io.json",
                                config_dir="config",
                                cache_dir="test/knockout_ios/cache/synthetic_example",
                                always_force_recompute=True,
                                quiet=True,
                                custom_log_preprocessing_function=enrich_log_with_fully_known_attributes)

    analyzer.discover_knockouts()

    analyzer.compute_ko_rules(grid_search=False, algorithm="IREP", confidence_threshold=0.5, support_threshold=0.1,
                              print_rule_discovery_stats=False, omit_report=False)

    adviser = KnockoutRedesignAdviser(analyzer)
    adviser.compute_redesign_options()
