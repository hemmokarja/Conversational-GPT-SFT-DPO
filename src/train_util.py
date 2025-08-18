import torch


def check_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"[NaN/Inf DETECTED] in {name} "
            f"(shape={tuple(tensor.shape)})\n"
            f"Sample values: {tensor.flatten()[:10]}"
        )
