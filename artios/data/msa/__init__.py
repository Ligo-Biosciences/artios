"""v1 MSA pipeline. See artios/data/msa/CLAUDE.md for the design contract."""

from artios.data.msa.a3m import encode_rows, parse_a3m

__all__ = ["encode_rows", "parse_a3m"]
