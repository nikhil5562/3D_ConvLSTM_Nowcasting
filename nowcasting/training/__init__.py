def train(*args, **kwargs):
    from .train import train as _train

    return _train(*args, **kwargs)

__all__ = ["train"]
