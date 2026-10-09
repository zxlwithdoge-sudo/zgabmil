import argparse
import hashlib
from pathlib import Path

from .data import discover_subjects
from .provenance import sha256_file, write_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create deterministic repeated subject-level splits"
    )
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--label0", required=True)
    parser.add_argument("--label1", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--test-per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def stable_rank(subject, seed, fold_index, label):
    value = f"{seed}:{fold_index}:{label}:{subject}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def create_folds(subjects, label0, label1, fold_count, test_per_class, seed):
    by_label = {
        label0: [
            subject
            for subject in subjects
            if subject.startswith(f"{label0}_")
        ],
        label1: [
            subject
            for subject in subjects
            if subject.startswith(f"{label1}_")
        ],
    }
    for label, members in by_label.items():
        if len(members) <= test_per_class:
            raise ValueError(
                f"Class {label} has {len(members)} subjects; need more than "
                f"{test_per_class}"
            )

    folds = []
    for fold_index in range(fold_count):
        test = []
        for label in (label1, label0):
            ranked_subjects = sorted(
                by_label[label],
                key=lambda subject: stable_rank(
                    subject, seed, fold_index, label
                ),
            )
            test.extend(ranked_subjects[:test_per_class])
        test_set = set(test)
        train = [subject for subject in subjects if subject not in test_set]
        folds.append({"train": train, "test": test})
    return folds


def main():
    args = parse_args()
    subjects = discover_subjects(
        args.features_dir, labels=(args.label0, args.label1)
    )
    folds = create_folds(
        subjects,
        args.label0,
        args.label1,
        args.folds,
        args.test_per_class,
        args.seed,
    )

    feature_manifest = args.features_dir / "feature_manifest.json"
    payload = {
        "design": "repeated_stratified_subject_holdout",
        "selection_algorithm": "sha256_rank_v1",
        "seed": args.seed,
        "test_subjects_per_class": args.test_per_class,
        "label0": args.label0,
        "label1": args.label1,
        "features_directory": str(args.features_dir.resolve()),
        "feature_manifest_sha256": (
            sha256_file(feature_manifest)
            if feature_manifest.exists()
            else None
        ),
        "folds": folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload)
    print(
        f"Saved {len(folds)} splits for {len(subjects)} subjects to "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
