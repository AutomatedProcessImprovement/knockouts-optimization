from knockout_ios.knockout_analyzer import KnockoutAnalyzer

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
                                always_force_recompute=True)

    analyzer.discover_knockouts()

    analyzer.get_ko_rules(grid_search=True, algorithm="IREP", confidence_threshold=0.5, support_threshold=0.1,
                          print_rule_discovery_stats=True)

# TODOs - time waste metrics
# TODO: implement Mean WT waste v2

# TODOs - validation & wrap up
# TODO: clean dependencies on variability analysis module, exclude it from repo hereafter
# TODO: add minimum column names to config file, refactor files & parsers, and document in readme

# TODOs - other improvements
# TODO: improve/clarify ko discovery parameters in config file
# TODO: try to improve execution times in bottlenecks
# TODO: fix exporting attribute-enriched event log to .xes
