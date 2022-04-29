from knockout_ios.knockout_analyzer import KnockoutAnalyzer
from knockout_ios.knockout_redesign_adviser import KnockoutRedesignAdviser
from knockout_ios.utils.synthetic_example.preprocessors import *

if __name__ == "__main__":
    analyzer = KnockoutAnalyzer(config_file_name="synthetic_example_ko_order_io_advanced.json",
                                config_dir="config",
                                cache_dir="test/knockout_ios/cache/synthetic_example",
                                always_force_recompute=True,
                                quiet=True,
                                custom_log_preprocessing_function=enrich_log_for_ko_order_advanced_test_fixed_values_wrapper)

    analyzer.discover_knockouts()

    analyzer.compute_ko_rules(algorithm="IREP", confidence_threshold=0.5, support_threshold=0.1,
                              print_rule_discovery_stats=False, omit_report=False,
                              max_rules=3, grid_search=True
                              )

    adviser = KnockoutRedesignAdviser(analyzer)

    adviser.compute_redesign_options()
