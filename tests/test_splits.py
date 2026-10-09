from abmil_oct.create_splits import create_folds


def test_subject_splits_are_stable_and_disjoint():
    subjects = [
        "AD_1",
        "AD_2",
        "AD_3",
        "AD_4",
        "NC_1",
        "NC_2",
        "NC_3",
        "NC_4",
    ]
    first = create_folds(subjects, "NC", "AD", 10, 2, 0)
    second = create_folds(subjects, "NC", "AD", 10, 2, 0)

    assert first == second
    for fold in first:
        assert not set(fold["train"]) & set(fold["test"])
        assert sum(subject.startswith("AD_") for subject in fold["test"]) == 2
        assert sum(subject.startswith("NC_") for subject in fold["test"]) == 2
