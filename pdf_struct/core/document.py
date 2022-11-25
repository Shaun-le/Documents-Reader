from typing import List, Tuple, Optional, Dict
from itertools import chain

from pdf_struct.core.transition_labels import ListAction


class TextBlock(object):
    def __init__(self, text: str):
        self.text: str = text


class Document(object):
    def __init__(self,
                 path: str,
                 texts: List[str],
                 text_blocks: List[TextBlock],
                 labels: Optional[List[ListAction]],
                 pointers: Optional[List[Optional[int]]],
                 cv_key: str):
        assert labels is None or len(texts) == len(labels)
        self.path: str = path
        self.texts: List[str] = texts
        self.text_blocks: List[TextBlock] = text_blocks
        # Ground-truth/predicted labels
        self.labels: Optional[List[ListAction]] = labels
        # Ground-truth/predicted pointer labels
        self.pointers: Optional[List[Optional[int]]] = pointers
        # Key to use for CV partitioning
        self.cv_key: str = cv_key

    @property
    def n_blocks(self):
        return len(self.texts)

    @property
    def n_features(self):
        assert self.feats is not None and 'self.feats accessed before set'
        return sum(map(len, self.feats.values()))

    @property
    def n_pointer_features(self):
        return None if self.pointer_feats is None else sum(map(len, self.pointer_feats.values()))

    def get_feature_names(self):
        return [k for k, _ in self._unpack_features(self.feats_test)]

    @staticmethod
    def _get_feature_matrix(feats) -> Optional[List[List[float]]]:
        if feats is None:
            return None
        n_blocks = len(list(list(feats.values())[0].values())[0])
        # List of list of size (n_blocks, n_feats)
        features = [[] for _ in range(n_blocks)]
        for _, feature in Document._unpack_features(feats):
            for j, f in enumerate(feature):
                features[j].append(float(f))
        return features

    @staticmethod
    def get_feature_array(features):
        return [v for _, v in Document._unpack_features(features)]

    @staticmethod
    def _unpack_features(features):
        return [
            (k, v) for k, v in sorted(
                chain(*[fg.items() for fg in features.values()]),
                key=lambda k_v: k_v[0])
        ]
