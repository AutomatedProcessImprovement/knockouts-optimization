import pickle
import pprint

from knockout_ios.knockout_analyzer import KnockoutAnalyzer

from knockout_ios.utils.redesign import evaluate_knockout_relocation_io, \
    evaluate_knockout_rule_change_io, evaluate_knockout_reordering_io_v2
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
    def __init__(self, knockout_analyzer: KnockoutAnalyzer, quiet=True):
        self.knockout_analyzer = knockout_analyzer
        self.quiet = quiet

    def get_redesign_options(self):
        self.redesign_options = {}
        self.redesign_options["reordering"] = evaluate_knockout_reordering_io_v2(self.knockout_analyzer)
        self.redesign_options["relocation"] = evaluate_knockout_relocation_io(self.knockout_analyzer)
        self.redesign_options["rule_change"] = evaluate_knockout_rule_change_io(self.knockout_analyzer)

        if not self.quiet:
            print(f"\n** Redesign options **\n")

            # TODO: cleaner printing/reporting method...
            print("> Knock-out Re-location\n")
            attribute_dependencies_dict = self.redesign_options["relocation"]
            for activity in attribute_dependencies_dict.keys():
                if not (len(attribute_dependencies_dict[activity]) > 0):
                    print(f"- '{activity}': required attributes are available from the start.")
                    continue

                print(f"- '{activity}' requires attribute(s):")
                for pair in attribute_dependencies_dict[activity]:
                    print(f"  '{pair[0]}', available after activity '{pair[1]}'")

            print("\n> Knock-out Re-ordering\n")
            optimal_order = [f'{i + 1}. {ko}' + '\n' for i, ko in
                             enumerate(self.redesign_options['reordering']['optimal_order_names'])]
            observed_order = [f'{i + 1}. {ko}' + '\n' for i, ko in
                              enumerate(self.redesign_options['reordering']['observed_ko_order'])]
            cases_respecting_order = self.redesign_options['reordering']['cases_respecting_order']
            total_cases = self.redesign_options['reordering']['total_cases']

            print(
                "Observed Order of Knock-out checks:\n" + f"{''.join(observed_order)}")
            print(
                "Optimal Order of Knock-out checks:\n" + f"{''.join(optimal_order)}\n{cases_respecting_order}/{total_cases} case(s) follow it.")

        return self.redesign_options


if __name__ == "__main__":
    analyzer = KnockoutAnalyzer(config_file_name="synthetic_example_ko_order_io.json",
                                config_dir="config",
                                cache_dir="test/knockout_ios/cache/synthetic_example",
                                always_force_recompute=True,
                                quiet=True,
                                custom_log_preprocessing_function=enrich_log_with_fully_known_attributes)

    analyzer.discover_knockouts()

    analyzer.get_ko_rules(grid_search=True, algorithm="IREP", confidence_threshold=0.5, support_threshold=0.1,
                          print_rule_discovery_stats=False, omit_report=False)

    adviser = KnockoutRedesignAdviser(analyzer, quiet=True)
    adviser.get_redesign_options()
