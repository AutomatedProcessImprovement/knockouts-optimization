import pytest

from knockout_ios.knockout_analyzer import KnockoutAnalyzer
from knockout_ios.knockout_redesign_adviser import KnockoutRedesignAdviser
from knockout_ios.utils.synthetic_example.preprocessors import enrich_log_for_synthetic_example_validation


@pytest.mark.skip(reason="Not robust enough yet")
def test_ko_reorder_io():
    analyzer = KnockoutAnalyzer(config_file_name="synthetic_example_ko_order_io_2.json",
                                config_dir="./config",
                                cache_dir="./cache/synthetic_example",
                                always_force_recompute=True,
                                quiet=True,
                                custom_log_preprocessing_function=enrich_log_for_synthetic_example_validation)

    analyzer.discover_knockouts()

    analyzer.compute_ko_rules(algorithm="IREP", confidence_threshold=0.5, support_threshold=0.1,
                              print_rule_discovery_stats=False, omit_report=False, grid_search=False, max_rules=2,
                              prune_size=0.95)

    adviser = KnockoutRedesignAdviser(analyzer)
    adviser.compute_redesign_options()

    assert adviser.redesign_options['reordering']['optimal_order_names'] == ["Check Monthly Income", "Check Risk",
                                                                             "Assess application", "Check Liability"]
