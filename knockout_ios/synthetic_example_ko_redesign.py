from knockout_ios.knockout_analyzer import KnockoutAnalyzer
from knockout_ios.knockout_redesign_adviser import KnockoutRedesignAdviser, read_analyzer_cache, dump_analyzer_cache
from knockout_ios.utils.synthetic_example.preprocessors import *

ignore_caches = False


def test_ko_reorder_io():
    print("\n\nLog: Synthetic Example (KO Order IO 2)\n")

    try:
        if ignore_caches:
            raise FileNotFoundError

        analyzer = read_analyzer_cache('test/test_fixtures', 'synthetic_example_ko_order_io_2')
        analyzer.build_report()

    except FileNotFoundError:
        analyzer = KnockoutAnalyzer(config_file_name="synthetic_example_ko_order_io_2.json",
                                    config_dir="config",
                                    cache_dir="cache/synthetic_example",
                                    always_force_recompute=True,
                                    quiet=True,
                                    custom_log_preprocessing_function=enrich_log_for_synthetic_example_validation)

        analyzer.discover_knockouts()

        analyzer.compute_ko_rules(algorithm="RIPPER", confidence_threshold=0.5, support_threshold=0.1,
                                  print_rule_discovery_stats=True, max_rules=3)

        dump_analyzer_cache(cache_dir="test/test_fixtures", cache_name="synthetic_example_ko_order_io_2",
                            ko_analyzer=analyzer)

    adviser = KnockoutRedesignAdviser(analyzer)
    adviser.compute_redesign_options()

    # assert adviser.redesign_options['reordering']['optimal_order_names'] == ["Check Monthly Income", "Check Risk",
    #                                                                          "Assess application", "Check Liability"]


def test_ko_reorder_io_improved():
    print("\n\nLog: Improved Synthetic Example (KO Order IO 2)\n")

    try:
        if ignore_caches:
            raise FileNotFoundError

        analyzer = read_analyzer_cache('test/test_fixtures', 'synthetic_example_ko_order_io_2_improved')
        analyzer.build_report()

    except FileNotFoundError:
        analyzer = KnockoutAnalyzer(config_file_name="synthetic_example_ko_order_io_2_improved.json",
                                    config_dir="config",
                                    cache_dir="test/knockout_ios/cache/synthetic_example",
                                    always_force_recompute=True,
                                    quiet=True,
                                    custom_log_preprocessing_function=enrich_log_for_synthetic_example_validation)

        analyzer.discover_knockouts()

        analyzer.compute_ko_rules(algorithm="RIPPER", confidence_threshold=0.5, support_threshold=0.1,
                                  print_rule_discovery_stats=True, max_rules=3)

        dump_analyzer_cache(cache_dir="test/test_fixtures", cache_name="synthetic_example_ko_order_io_2_improved",
                            ko_analyzer=analyzer)

    adviser = KnockoutRedesignAdviser(analyzer)
    adviser.compute_redesign_options()

    # assert adviser.redesign_options['reordering']['optimal_order_names'] == ["Check Monthly Income", "Check Risk",
    #                                                                          "Assess application", "Check Liability"]


if __name__ == "__main__":
    ignore_caches = True
    test_ko_reorder_io()
    test_ko_reorder_io_improved()
