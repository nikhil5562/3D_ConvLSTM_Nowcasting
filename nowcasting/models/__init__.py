def build_model(*args, **kwargs):
    from .model_tf2 import build_model as _build_model

    return _build_model(*args, **kwargs)

__all__ = ["build_model"]
