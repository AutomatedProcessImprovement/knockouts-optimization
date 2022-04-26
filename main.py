from knockout_ios.knockout_analyzer import KnockoutAnalyzer
from knockout_ios.utils.synthetic_example.preprocessors import enrich_log_with_fully_known_attributes

if __name__ == "__main__":
    #  Synthetic Example Ground Truth
    #  (K.O. checks and their rejection rules):
    #
    # 'Check Liability':        'Total Debt'     > 5000 ||  'Owns Vehicle' = False
    # 'Check Risk':             'Loan Ammount'   > 10000
    # 'Check Monthly Income':   'Monthly Income' < 1000
    # 'Assess application':     'External Risk Score' > 0.3

    analyzer = KnockoutAnalyzer(config_file_name="synthetic_example.json",
                                config_dir="config",
                                cache_dir="knockout_ios/cache/synthetic_example",
                                always_force_recompute=True,
                                custom_log_preprocessing_function=enrich_log_with_fully_known_attributes)

    analyzer.discover_knockouts()

    analyzer.compute_ko_rules(grid_search=True, algorithm="IREP", confidence_threshold=0.5, support_threshold=0.1,
                              print_rule_discovery_stats=True)
