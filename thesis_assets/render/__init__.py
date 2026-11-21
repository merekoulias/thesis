"""Render functions for thesis-only figures and tables.

Each function takes ``out_stem`` (``thesis_assets/<kind dir>/<id>``, no suffix),
writes one or more files ``<out_stem>.<ext>`` and returns their paths. It reads
pipeline artifacts only; it never recomputes a metric.
"""
