"""
Single-image forensics: checks that work on ONE photo, no reference image needed.

- ``metadata``    — EXIF / XMP / PNG-text checks (editing software, AI markers, dates)
- ``fingerprint`` — perceptual hash to find reused / duplicate photos
"""

from .fingerprint import FingerprintIndex, Match, hamming_distance, perceptual_hash
from .metadata import Flag, MetadataReport, analyze_metadata

__all__ = [
    "FingerprintIndex",
    "Flag",
    "Match",
    "MetadataReport",
    "analyze_metadata",
    "hamming_distance",
    "perceptual_hash",
]
