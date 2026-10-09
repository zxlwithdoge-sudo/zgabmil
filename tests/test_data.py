from abmil_oct.data import subject_from_stem


def test_subject_from_stem():
    cases = [
        ("NC_6839_1", "NC_6839"),
        ("AD_20832_104", "AD_20832"),
        ("CTE_8572_scan_5", "CTE_8572"),
    ]
    for stem, expected in cases:
        assert subject_from_stem(stem) == expected
