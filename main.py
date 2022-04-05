from knockout_ios.knockout_analyzer import KnockoutAnalyzer

if __name__ == "__main__":
    # Known rules
    # 'Check Liability':        'Total Debt'     > 5000 ||  'Vehicle Owned' = "N/A"
    # 'Check Risk':             'Loan Ammount'   > 10000
    # 'Check Monthly Income':   'Monthly Income' < 1000

    analyzer = KnockoutAnalyzer(config_file_name="synthetic_example.json",
                                config_dir="config",
                                cache_dir="knockout_ios/cache/synthetic_example",
                                always_force_recompute=True,
                                quiet=True)

    analyzer.discover_knockouts()

    analyzer.get_ko_rules_IREP()

    analyzer.calc_ko_efforts(support_threshold=0.5, confidence_threshold=0.5, algorithm="IREP")

    analyzer.build_report(algorithm="IREP")
