"""
Repo-root shared data layer: data1/data2/data3 read_official and training code can read txt and enumerate pairs here.

- ``txt_io``: line-by-line parsing matching legacy ``2/data/our_data_dataset`` + dual-channel subway files
- ``pair_specs``: auto-detect band layout (data1/2) or subway layout (data3) pairs by directory
- ``viz_export``: triple-curve export (forwarded from each data*/viz_export.py)
"""

from . import pair_specs, txt_io, viz_export  # noqa: F401

__all__ = ["txt_io", "pair_specs", "viz_export"]
