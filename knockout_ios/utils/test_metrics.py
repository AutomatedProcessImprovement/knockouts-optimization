from knockout_ios.utils.metrics import get_categorical_evaluation_metrics


def test_correct_kos():
    activities = ['Start', 'ko_1', 'not_ko_1', 'not_ko_2', 'ko_2']
    expected_kos = ['ko_1', 'ko_2']
    computed_kos = ['ko_1', 'ko_2']
    metrics = get_categorical_evaluation_metrics(activities, expected_kos, computed_kos)

    conf_matrix = metrics['confusion_matrix']

    assert conf_matrix['true_positives'] == 2
    assert conf_matrix['false_positives'] == 0
    assert conf_matrix['true_negatives'] == 3
    assert conf_matrix['false_negatives'] == 0


def test_wrong_kos():
    activities = ['Start', 'ko_1', 'not_ko_1', 'not_ko_2', 'ko_2']
    expected_kos = ['ko_1', 'ko_2']
    computed_kos = ['Start', 'not_ko_1', 'not_ko_2']
    metrics = get_categorical_evaluation_metrics(activities, expected_kos, computed_kos)

    conf_matrix = metrics['confusion_matrix']

    assert conf_matrix['true_positives'] == 0
    assert conf_matrix['false_positives'] == 3
    assert conf_matrix['true_negatives'] == 0
    assert conf_matrix['false_negatives'] == 2


def test_partially_correct_kos():
    activities = ['Start', 'ko_1', 'not_ko_1', 'not_ko_2', 'ko_2']
    expected_kos = ['ko_1', 'ko_2']
    computed_kos = ['ko_1', 'ko_2', 'not_ko_1']
    metrics = get_categorical_evaluation_metrics(activities, expected_kos, computed_kos)

    conf_matrix = metrics['confusion_matrix']

    assert conf_matrix['true_positives'] == 2
    assert conf_matrix['false_positives'] == 1
    assert conf_matrix['true_negatives'] == 2
    assert conf_matrix['false_negatives'] == 0




