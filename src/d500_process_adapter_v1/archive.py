"""Archiving is owned by the parent via existing src.research_trace.collaboration.

This implementation emits immutable run records, contracts, configuration,
metrics, logs and delivery references; it does not add a separate uploader or
archiving framework. Actual source and cumulative/incremental diffs are collected
by the established project entry point after parent acceptance.
"""
