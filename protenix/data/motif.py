"""
Houses all the logic for generating motifs during model training.
"""

from biotite.structure import AtomArray

from protenix.data.tokenizer import TokenArray


class MotifMaker:
    def __init__(self, **kwargs):
        pass

    def make_motif(self, atom_array: AtomArray, token_array: TokenArray) -> AtomArray:
        """
        Makes the motif for the given atom and token arrays. This is the main function generating
        the motif. It returns the motif atom array rather than changing the original atom array.
        """
        pass

    def add_motif_attributes_to_atom_array(
        self, atom_array: AtomArray, token_array: TokenArray
    ) -> AtomArray:
        """
        Adds the motif attribute to the atom array.
        """
        pass
