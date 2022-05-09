from knockout_ios.knockout_analyzer import KnockoutAnalyzer
from knockout_ios.knockout_redesign_adviser import KnockoutRedesignAdviser
from knockout_ios.utils.preprocessing.configuration import read_log_and_config
from knockout_ios.utils.synthetic_example.preprocessors import *


def test_ko_reorder_io_simple():
    log, configuration = read_log_and_config("config", "synthetic_example_ko_order_io.json",
                                             "cache/synthetic_example")

    analyzer = KnockoutAnalyzer(log_df=log, config=configuration,
                                config_file_name="synthetic_example_ko_order_io.json",
                                config_dir="./config",
                                cache_dir="./cache/synthetic_example",
                                always_force_recompute=True,
                                quiet=True,
                                custom_log_preprocessing_function=enrich_log_with_fully_known_attributes)

    analyzer.discover_knockouts()

    analyzer.compute_ko_rules(grid_search=False, algorithm="IREP", confidence_threshold=0.5, support_threshold=0.1,
                              print_rule_discovery_stats=False, omit_report=True)

    adviser = KnockoutRedesignAdviser(analyzer, quiet=True)
    adviser.compute_redesign_options()

    assert adviser.redesign_options['reordering']['current_activity_order'] == ["Check Monthly Income", "Check Risk",
                                                                                "Check Liability", "Assess application"]


def test_ko_reorder_io_advanced():
    log, configuration = read_log_and_config("config", "synthetic_example_ko_order_io_advanced.json",
                                             "cache/synthetic_example")

    analyzer = KnockoutAnalyzer(log_df=log, config=configuration,
                                config_file_name="synthetic_example_ko_order_io_advanced.json",
                                config_dir="./config",
                                cache_dir="./cache/synthetic_example",
                                always_force_recompute=True,
                                quiet=True,
                                custom_log_preprocessing_function=enrich_log_for_ko_order_advanced_test_fixed_values_wrapper)

    analyzer.discover_knockouts()

    analyzer.compute_ko_rules(grid_search=False, algorithm="IREP", confidence_threshold=0.5, support_threshold=0.1,
                              print_rule_discovery_stats=False, omit_report=False)

    adviser = KnockoutRedesignAdviser(analyzer, quiet=True)
    adviser.compute_redesign_options()

    # "Aggregated Risk Score Check" has the lowest KO effort but requires an attribute that is available after "Check Risk"
    assert adviser.redesign_options['reordering']['current_activity_order'] == ["Check Liability", "Check Risk",
                                                                                "Aggregated Risk Score Check",
                                                                                "Check Monthly Income"]
