# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class Token(object):
    """
    Used to store information related to Tokens.

    Example:
    >>> token = Token(1)
    >>> token.value
    1
    >>> token.atom_indices = [1, 2, 3]
    """

    def __init__(self, value, **kwargs):
        self.value = value
        self._annot = {}
        for name, annotation in kwargs.items():
            self._annot[name] = annotation

    def __getattr__(self, attr):
        if attr in super().__getattribute__("_annot"):
            return self._annot[attr]
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{attr}'"
            )

    def __len__(self):
        return len(self.atom_indices)

    def __repr__(self):
        annot_lst = []
        for k, v in self._annot.items():
            annot_lst.append(f"{k}={v}")
        return f'Token({self.value}, {",".join(annot_lst)})'

    def __setattr__(self, attr, value):
        if attr == "_annot" or attr == "value":
            super().__setattr__(attr, value)
        else:
            self._annot[attr] = value


class TokenArray(object):
    """
    A group of Token objects used for batch operations.
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens

    def __repr__(self):
        repr_str = "TokenArray(\n"
        for token in self.tokens:
            repr_str += f"\t{token}\n"
        repr_str += ")"
        return repr_str

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        for token in self.tokens:
            yield token

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.tokens[index]
        else:
            return TokenArray([self.tokens[i] for i in index])

    def get_annotation(self, category):
        return [token._annot[category] for token in self.tokens]

    def set_annotation(self, category, values):
        assert len(values) == len(
            self.tokens
        ), "Length of values must match the number of tokens"
        for token, value in zip(self.tokens, values):
            token._annot[category] = value


# AtomArrayTokenizer was deleted in PR C — tokenization now lives on
# ``AnnotatedStructure.grouping`` via ``compute_token_grouping_vectorized``.
# ``Token`` and ``TokenArray`` remain because other callers (collate,
# BasicContext, test helpers) still use them as plain containers.
