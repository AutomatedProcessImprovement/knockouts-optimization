import pprint

from knockout_ios.knockout_analyzer import KnockoutAnalyzer

from knockout_ios.utils.redesign import evaluate_knockout_reordering_io, evaluate_knockout_relocation_io, \
    evaluate_knockout_rule_change_io


# KO redesign strategies

# 1. Relocate a knock-out
#    place the knock-out check as early in the process as the attribute value (based on which the knock-out is performed)
#    is available.
#
# 2. Reorder knock-outs
#    according to the knock-out principle:
#    checks are ordered according to the principle of “least effort to reject”– checks that require less effort and are more
#    likely to reject the case come first. In addition, consider the knock-out rule so that the attribute value (based on which
#    the knock-out is performed) is available at that point in the process.
#
# 3. Change the knock-out rule
#    change the attribute value (or its range) based on which the knock-out is performed;
#
# NOT IN CURRENT SCOPE:
#
# 4. Remove the knock-out
#
# 5. Add a new knock-out


class KnockoutRedesignAdviser(object):
    def __init__(self, knockout_analyzer, quiet=True):
        self.knockout_analyzer = knockout_analyzer
        self.quiet = quiet

    def get_redesign_options(self):
        self.redesign_options = {}
        self.redesign_options["reordering"] = evaluate_knockout_reordering_io(self.knockout_analyzer)
        self.redesign_options["relocation"] = evaluate_knockout_relocation_io(self.knockout_analyzer)
        self.redesign_options["rule_change"] = evaluate_knockout_rule_change_io(self.knockout_analyzer)

        if not self.quiet:
            pprint.pprint(self.redesign_options)

        return self.redesign_options


if __name__ == "__main__":
    analyzer = KnockoutAnalyzer(config_file_name="synthetic_example_ko_order_io.json",
                                config_dir="config",
                                cache_dir="knockout_ios/cache/synthetic_example",
                                always_force_recompute=False,
                                quiet=True)

    analyzer.discover_knockouts()

    analyzer.get_ko_rules(grid_search=True, algorithm="IREP", confidence_threshold=0.5, support_threshold=0.1,
                          print_rule_discovery_stats=False, omit_report=False)

    adviser = KnockoutRedesignAdviser(analyzer, quiet=False)
    adviser.get_redesign_options()

# TODO: [X] modify synthetic example log/simulation parameters to test ko order io
# TODO: [ ] modify synthetic example log/simulation parameters to test ko relocation io
# TODO: [ ] modify synthetic example log/simulation parameters to test ko rule change io
# TODO: [ ] modify synthetic example log/simulation parameters to test redesign strategies combined
