from knockout_ios.knockout_analyzer import KnockoutAnalyzer
from knockout_ios.knockout_redesign_adviser import KnockoutRedesignAdviser
from knockout_ios.utils.preprocessing.configuration import read_log_and_config
from knockout_ios.utils.synthetic_example.preprocessors import *

if __name__ == "__main__":
    log, configuration = read_log_and_config("config_examples", "synthetic_example_ko_order_io.json",
                                             "cache/synthetic_example")

    analyzer = KnockoutAnalyzer(log_df=log,
                                config=configuration,
                                config_file_name="synthetic_example_ko_order_io.json",
                                config_dir="config_examples",
                                cache_dir="test/knockout_ios/cache/synthetic_example",
                                always_force_recompute=False,
                                quiet=True,
                                custom_log_preprocessing_function=enrich_log_for_synthetic_example_validation)

    analyzer.discover_knockouts()

    analyzer.compute_ko_rules(grid_search=False, algorithm="IREP", confidence_threshold=0.5, support_threshold=0.1,
                              print_rule_discovery_stats=False, omit_report=False)

    adviser = KnockoutRedesignAdviser(analyzer)
    adviser.compute_redesign_options()
