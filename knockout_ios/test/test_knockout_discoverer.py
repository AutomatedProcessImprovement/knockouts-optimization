import pytest
from knockout_ios.knockout_discoverer import KnockoutDiscoverer

# Credit Application
from knockout_ios.utils.preprocessing.configuration import read_log_and_config

credit_app_kos = ['Assess application', 'Check credit history', 'Check income sources']
credit_app_scenarios = [
    ("credit_app_simple.json", ['Notify rejection'], credit_app_kos),
    ("credit_app_simple_known_ko.json", ['Notify rejection'], credit_app_kos),
    ("credit_app_nested_1.json", ['Reject application'], credit_app_kos),
    ("credit_app_nested_2.json", ['Reject application'], credit_app_kos),
    ("credit_app_multiple_negative.json", ["Reject application", "Ban client forever"], credit_app_kos),
    ("credit_app_nested_rework.json", ['Reject application'], ['Check income sources', 'Assess application']),
    ("credit_app_nested_rework_2.json", ['Reject application'], credit_app_kos),
]


@pytest.mark.parametrize("config_file, expected_outcomes, expected_kos", credit_app_scenarios)
def test_credit_app(config_file, expected_outcomes, expected_kos):
    log, configuration = read_log_and_config("config", config_file, "./cache/credit_app")

    analyzer = KnockoutDiscoverer(log_df=log, config=configuration, config_file_name=config_file,
                                  cache_dir="./cache/credit_app",
                                  always_force_recompute=True, quiet=True)
    analyzer.find_ko_activities()

    assert len(analyzer.ko_outcomes) == len(expected_outcomes)
    for outcome in expected_outcomes:
        assert outcome in analyzer.ko_outcomes

    assert len(analyzer.ko_activities) == len(expected_kos)
    for ko in expected_kos:
        assert ko in analyzer.ko_activities

    metrics = analyzer.get_discovery_metrics(expected_kos)
    assert metrics['accuracy'] == pytest.approx(1.0)
    assert metrics['f1_score'] == pytest.approx(1.0)
    assert metrics['precision'] == pytest.approx(1.0)
    assert metrics['recall'] == pytest.approx(1.0)


# Loan Application

loan_app_kos = ['Assess eligibility']
loan_app_post_knockout_activities = ['Loan application rejected']

loan_app_scenarios = [
    ("loan_app.json", loan_app_post_knockout_activities, loan_app_kos),
    ("loan_app_w_positive.json", loan_app_post_knockout_activities, loan_app_kos),
    ("loan_app_w_negative.json", loan_app_post_knockout_activities, loan_app_kos),
]


# TODO: This test is slow. Skip depending on slow/no-slow flag like Simod.
@pytest.mark.skip(reason="Slow test")
@pytest.mark.parametrize("config_file, expected_outcomes, expected_kos", loan_app_scenarios)
def test_loan_app(config_file, expected_outcomes, expected_kos):
    log, configuration = read_log_and_config("config", config_file, "./cache/loan_app")

    analyzer = KnockoutDiscoverer(log_df=log, config=configuration, config_file_name=config_file,
                                  cache_dir="./cache/loan_app",
                                  always_force_recompute=True, quiet=True)
    analyzer.find_ko_activities()

    assert len(analyzer.ko_outcomes) == len(expected_outcomes)
    for outcome in expected_outcomes:
        assert outcome in analyzer.ko_outcomes

    assert len(analyzer.ko_activities) == len(expected_kos)
    for ko in expected_kos:
        assert ko in analyzer.ko_activities

    metrics = analyzer.get_discovery_metrics(expected_kos)
    assert metrics['accuracy'] == pytest.approx(1.0)
    assert metrics['f1_score'] == pytest.approx(1.0)
    assert metrics['precision'] == pytest.approx(1.0)
    assert metrics['recall'] == pytest.approx(1.0)
