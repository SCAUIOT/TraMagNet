"""
Shared data layer: txt parsing and (clean, noisy) pairing by matching ``sample{i}.txt`` names.

- ``txt_io``: 3- or 4-column flat ``sample{i}.txt`` parsing
- ``pair_specs``: enumerate pairs from matching filenames in reference_signal/ and noise_signal/
- ``viz_export``: triple-curve figure export
"""

from . import pair_specs, txt_io, viz_export  # noqa: F401

__all__ = ["txt_io", "pair_specs", "viz_export"]
