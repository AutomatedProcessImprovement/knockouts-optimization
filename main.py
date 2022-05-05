from knockout_ios.knockout_analyzer import KnockoutAnalyzer
from knockout_ios.knockout_redesign_adviser import KnockoutRedesignAdviser
from knockout_ios.utils.synthetic_example.preprocessors import *

if __name__ == "__main__":
    analyzer = KnockoutAnalyzer(config_file_name="synthetic_example_ko_order_io.json",
                                config_dir="config_examples",
                                cache_dir="cache/synthetic_example",
                                always_force_recompute=True,
                                custom_log_preprocessing_function=enrich_log_for_synthetic_example_validation)

    analyzer.discover_knockouts()
    analyzer.compute_ko_rules(algorithm="RIPPER", confidence_threshold=0.5, support_threshold=0.1)

    adviser = KnockoutRedesignAdviser(analyzer)
    adviser.compute_redesign_options()
