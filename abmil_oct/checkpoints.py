from pathlib import Path

import torch

from .model import build_model


def load_checkpoint(path, device):
    """Rebuild the model recorded in a fold checkpoint and load its weights.

    ``args["head"]`` selects the predictor. It is absent from checkpoints
    written before the additive head existed, which is why the default is
    ``"linear"``: every old checkpoint keeps loading as the manuscript model.
    """
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    config = checkpoint.get("args", {})
    if config.get("head", "linear") == "additive":
        from .additive import build_additive_model

        model = build_additive_model(config, device=device)
    else:
        model = build_model(config, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model
