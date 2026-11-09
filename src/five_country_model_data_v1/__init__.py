"""Read-only model observations; no role allocation, fitting or training."""
from .reader import Reader, build_sample

__all__ = ['Reader', 'build_sample']
