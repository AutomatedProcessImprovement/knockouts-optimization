from knockout_ios.knockout_analyzer import KnockoutAnalyzer


def test_report_creation():

    analyzer = KnockoutAnalyzer(config_file_name="synthetic_example.json",
                                config_dir="config",
                                cache_dir="cache/synthetic_example",
                                always_force_recompute=True,
                                quiet=True)

    analyzer.discover_knockouts(expected_kos=['Check Liability', 'Check Risk', 'Check Monthly Income'])

    analyzer.get_ko_rules_IREP(grid_search=False, bucketing_approach="B")

    analyzer.calc_ko_efforts(support_threshold=0.5, confidence_threshold=0.5, algorithm="IREP")
    df = analyzer.build_report(algorithm="IREP", omit=True)

    # assert all 3 knockouts are in the report
    assert df.shape[0] == 3
    assert sorted(analyzer.discoverer.ko_activities) == sorted(['Check Liability', 'Check Risk', 'Check Monthly Income'])

    # assert all columns are in the report
    assert sorted(['Knock-out check', 'Total frequency', 'Case frequency', 'Mean PT',
                   'Rejection rate', 'Rejection rule', 'Effort per rejection']) == sorted(df.columns.tolist())

    # assert that there are no rows with NaN values
    assert df.isnull().sum().sum() == 0

    # assert that there are no rows with negative values
    assert df.loc[df['Effort per rejection'] < 0].shape[0] == 0

    # assert that there are no rows where 'Rejection rule' is equal to "[]"
    assert df.loc[df['Rejection rule'] == "[]", 'Rejection rule'].shape[0] == 0
