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

# TODOs - ko redesign

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
