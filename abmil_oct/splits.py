import json
import pickle
from pathlib import Path


def load_splits(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    if path.suffix.lower() == ".json":
        with path.open() as handle:
            payload = json.load(handle)
    elif path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    else:
        raise ValueError("Split file must be JSON or pickle")

    if isinstance(payload, dict):
        payload = payload.get("folds")
    if not isinstance(payload, list):
        raise ValueError("Split file must contain a list of folds")

    folds = []
    for index, fold in enumerate(payload, start=1):
        if isinstance(fold, dict):
            train, test = fold.get("train"), fold.get("test")
        elif isinstance(fold, (list, tuple)) and len(fold) == 2:
            train, test = fold
        else:
            raise ValueError(f"Invalid fold {index}")
        train, test = list(train), list(test)
        overlap = set(train) & set(test)
        if overlap:
            raise ValueError(f"Fold {index} has train/test overlap: {overlap}")
        folds.append((train, test))
    return folds
