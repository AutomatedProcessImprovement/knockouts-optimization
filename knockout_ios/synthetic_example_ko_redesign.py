from knockout_ios.knockout_analyzer import KnockoutAnalyzer
from knockout_ios.knockout_redesign_adviser import KnockoutRedesignAdviser, read_analyzer_cache, dump_analyzer_cache
from knockout_ios.utils.synthetic_example.preprocessors import *

ignore_caches = False


def test_ko_reorder_io_simple():
    print("\n\nLog: Synthetic Example (KO Order IO)\n")

    try:
        if ignore_caches:
            raise FileNotFoundError

        analyzer = read_analyzer_cache('./test/test_fixtures', 'synthetic_example_ko_order_io')
        analyzer.build_report()

    except FileNotFoundError:
        analyzer = KnockoutAnalyzer(config_file_name="synthetic_example_ko_order_io.json",
                                    config_dir="config",
                                    cache_dir="test/knockout_ios/cache/synthetic_example",
                                    always_force_recompute=True,
                                    quiet=True,
                                    custom_log_preprocessing_function=enrich_log_with_fully_known_attributes)

        analyzer.discover_knockouts()

        analyzer.compute_ko_rules(grid_search=True, algorithm="IREP", confidence_threshold=0.5, support_threshold=0.1,
                                  print_rule_discovery_stats=False, omit_report=False)

        dump_analyzer_cache(cache_dir="./test/test_fixtures", cache_name="synthetic_example_ko_order_io",
                            ko_analyzer=analyzer)

    adviser = KnockoutRedesignAdviser(analyzer)
    adviser.compute_redesign_options()

    assert adviser.redesign_options['reordering']['optimal_order_names'] == ["Check Monthly Income", "Check Risk",
                                                                             "Check Liability", "Assess application"]


def test_ko_reorder_io_advanced():
    print("\n\nLog: Synthetic Example (KO Order IO advanced)\n")

    try:
        if ignore_caches:
            raise FileNotFoundError

        analyzer = read_analyzer_cache('./test/test_fixtures', 'synthetic_example_ko_order_io_advanced')
        analyzer.build_report()

    except FileNotFoundError:
        analyzer = KnockoutAnalyzer(config_file_name="synthetic_example_ko_order_io_advanced.json",
                                    config_dir="config",
                                    cache_dir="test/knockout_ios/cache/synthetic_example",
                                    always_force_recompute=True,
                                    quiet=True,
                                    custom_log_preprocessing_function=enrich_log_for_ko_order_advanced_test_fixed_values_wrapper)

        analyzer.discover_knockouts()

        analyzer.compute_ko_rules(grid_search=True, algorithm="IREP", confidence_threshold=0.5, support_threshold=0.1,
                                  print_rule_discovery_stats=False, omit_report=False)

        dump_analyzer_cache(cache_dir="./test/test_fixtures", cache_name="synthetic_example_ko_order_io_advanced",
                            ko_analyzer=analyzer)

    adviser = KnockoutRedesignAdviser(analyzer)
    adviser.compute_redesign_options()

    # "Aggregated Risk Score Check" has the lowest KO effort but requires an attribute that is available after "Check Risk"
    assert adviser.redesign_options['reordering']['optimal_order_names'] == ["Check Liability", "Check Risk",
                                                                             "Aggregated Risk Score Check",
                                                                             "Check Monthly Income"]


if __name__ == "__main__":
    test_ko_reorder_io_simple()
    # test_ko_reorder_io_advanced()
