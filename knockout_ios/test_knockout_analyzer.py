import pytest

from knockout_ios.knockout_analyzer import KnockoutAnalyzer


@pytest.mark.parametrize("algorithm", ["RIPPER", "IREP"])
def test_report_creation(algorithm):
    analyzer = KnockoutAnalyzer(config_file_name="synthetic_example.json",
                                config_dir="config",
                                cache_dir="cache/synthetic_example",
                                always_force_recompute=True,
                                quiet=True)

    analyzer.discover_knockouts(expected_kos=['Check Liability', 'Check Risk', 'Check Monthly Income'])

    if algorithm == "RIPPER":
        analyzer.get_ko_rules_RIPPER(grid_search=False)
    elif algorithm == "IREP":
        analyzer.get_ko_rules_IREP(grid_search=False)

    analyzer.calc_ko_efforts(support_threshold=0.5, confidence_threshold=0.5, algorithm=algorithm)
    df = analyzer.build_report(algorithm=algorithm, omit=True)

    # assert all 3 knockouts are in the report
    assert df.shape[0] == 3
    assert sorted(analyzer.discoverer.ko_activities) == sorted(
        ['Check Liability', 'Check Risk', 'Check Monthly Income'])

    # assert all columns are in the report
    assert sorted(['Knock-out check', 'Total frequency', 'Case frequency', 'Mean PT',
                   'Rejection rate', f'Rejection rule ({algorithm})', 'Effort per rejection']) == sorted(
        df.columns.tolist())

    # assert that there are no rows with NaN values
    assert df.isnull().sum().sum() == 0

    # assert that there are no rows with negative values
    assert df.loc[df['Effort per rejection'] < 0].shape[0] == 0

    # assert that there are no rows where 'Rejection rule' is empty
    assert df.loc[df[f'Rejection rule ({algorithm})'] == "[]", f'Rejection rule ({algorithm})'].shape[0] == 0
