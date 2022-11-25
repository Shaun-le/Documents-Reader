from collections import Counter
from typing import List, Type
import numpy as np
from sklearn.metrics import accuracy_score
from pdf_struct.core import predictor, CRFsuite, CRFsuite4Ja
from pdf_struct.core.document import Document
from pdf_struct.core.export import to_paragraphs, to_tree
from pdf_struct.core.feature_extractor import BaseFeatureExtractor
from pdf_struct.core.structure_evaluation import evaluate_structure, \
    evaluate_labels
import tqdm


def _create_prediction_jsons(documents, documents_pred) -> List[dict]:
    predictions = []
    for d, d_p in zip(documents, documents_pred):
        assert d.path == d_p.path
        transition_prediction_accuracy = accuracy_score(
            np.array([l.value for l in d.labels]),
            np.array([l.value for l in d_p.labels])
        )
        feature_array = []
        for d in documents:
            feature_array.append(Document._get_feature_matrix(d[5]))
        predictions.append({
            'path': d[0],
            'texts': d[1],
            'features': feature_array,
            'transition_prediction_accuracy': transition_prediction_accuracy,
            'ground_truth': {
                'labels': [l.name for l in d[3]],
                'pointers': d[4]
            },
            'prediction': {
                'labels': [l.name for l in d_p[3]],
                'pointers': d_p[4]
            }
        })
    return predictions


def evaluate(documents,
             feature_extractor_cls: Type[BaseFeatureExtractor],
             k_folds: int,
             prediction: bool=False):

    documents = [feature_extractor_cls.append_features_to_document(document)
                 for document in tqdm.tqdm(documents)]
    print('Extracting features from documents')
    print(f'Extracted {sum(map(lambda d: len(d[1]), documents))} lines from '
          f'{len(documents)} documents with label distribution: '
          f'{Counter(sum(map(lambda d: d[3], documents), []))} for evaluation.')
    for d in documents:
        print(f'Extracted {sum(map(len, d[5].values()))}-dim features and '
              f'{sum(map(len, d[7].values()))}-dim pointer features.')
        documents_pred = predictor.k_fold_train_predict(
            documents, n_splits = k_folds)

        #if you want to use CRF for trainning model.
        '''documents_pred = CRFsuite.k_fold_train_predict(
            documents, n_splits=k_folds)'''
        '''documents_pred = CRFsuite4Ja.k_fold_train_predict(
            documents, n_splits=k_folds)'''
        break
    metrics = {
        'structure': evaluate_structure(documents, documents_pred),
        'labels': evaluate_labels(documents, documents_pred)
    }
    if prediction:
        prediction_jsons = _create_prediction_jsons(documents, documents_pred)
        return metrics, prediction_jsons
    else:
        return metrics
