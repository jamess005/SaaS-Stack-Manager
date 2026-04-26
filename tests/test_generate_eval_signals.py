from training.generate_eval_signals import CASES


def test_holdout_eval_set_has_60_balanced_cases():
    assert len(CASES) == 60
    verdict_counts = {}
    for case in CASES:
        verdict_counts[case["expected_verdict"]] = verdict_counts.get(case["expected_verdict"], 0) + 1
    assert verdict_counts == {"SWITCH": 20, "STAY": 20, "HOLD": 20}


def test_holdout_case_ids_are_unique():
    case_ids = [case["case_id"] for case in CASES]
    assert len(case_ids) == len(set(case_ids))
