"""Data engineering module for protein structure curation and analysis.

This module provides tools for offline data curation, dataset preparation,
and analysis of protein structures. It complements the runtime data
pipeline which handles training/inference featurization.

Submodules:
    - metrics: Structural and quality metrics for protein structures
    - processors: Parallel processing of protein databases (FoldComp, PDB)
    - filters: Filtering logic and visualization for dataset curation
    - cli: Command-line interfaces for data engineering tasks
"""

__version__ = "0.1.0"
