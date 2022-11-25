import copy
import random
from typing import List, Optional
from collections import defaultdict
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV
from pdf_struct.core.transition_labels import ListAction
from pdf_struct.core.document import Document

def train_classifiers(documents, used_features: Optional[List[int]]=None):
    used_features = None if used_features is None else np.array(used_features)

    # First, classify transition between consecutive lines
    feature_array = []
    for d in documents:
        feature_array.append(Document._get_feature_matrix(d[5]))
    X_train = np.array(sum(feature_array, []),dtype=np.float64)
    y_train = np.array([l.value for d in documents for l in d[3]], dtype=int)

    if used_features is not None:
        X_train = X_train[:, used_features]

    clf = RandomForestClassifier().fit(X_train,y_train)

    # Next, predict pointers
    pointer_feats_array = []
    for d in documents:
        if d[7] is not None and len(d[7]) > 0:
            pointer_feats_array.append(Document._get_feature_matrix(d[7]))
    X_train = np.array(
        [pointer_feats_array[i][j] for i in range(len(pointer_feats_array)) for j in
         range(len(pointer_feats_array[i]))],dtype=np.float64)
    y_train = np.array([p == d[4][c] for d in documents for p, c in d[8]],dtype=int)

    clf_ptr = RandomForestClassifier().fit(X_train,y_train)

    return clf, clf_ptr


def predict_with_classifiers(clf, clf_ptr, documents, used_features: Optional[List[int]]=None):
    used_features = None if used_features is None else np.array(used_features)

    feature_test_array = []
    for d in documents:
        feature_test_array.append(Document._get_feature_matrix(d[6]))
    X_test = np.array(sum(feature_test_array, []),dtype=np.float64)
    if used_features is not None:
        X_test = X_test[:, used_features]
    y_pred = clf.predict(X_test)
    predicted_documents = []
    cum_j = 0
    for document in documents:
        d = copy.deepcopy(document)
        d[3] = [ListAction(yi) for yi in y_pred[cum_j:cum_j + len(document[1])]]
        states = dict()
        for i in range(len(d[2])):
            tb1 = d[2][i - 1] if i != 0 else None
            tb2 = d[2][i]
            if d[3][i] == ListAction.ELIMINATE:
                tb3 = d[2][i + 1] if i + 1 < len(
                    d[2]) else None
                tb4 = d[2][i + 2] if i + 2 < len(
                    d[2]) else None
            else:
                tb3 = None
                for j in range(i + 1, len(d[2])):
                    if d[3][j] != ListAction.ELIMINATE:
                        tb3 = d[2][j]
                        break
                tb4 = d[2][j + 1] if j + 1 < len(
                    d[2]) else None
            # still execute extract_features even if d.labels[i] != ListAction.ELIMINATE
            # to make the state consistent

            feat, states = d[9].extract_features(tb1, tb2, tb3, tb4, states)
            feat = np.array([Document.get_feature_array(feat)])
            if used_features is not None:
                feat = feat[:, used_features]
            if d[3][i] != ListAction.ELIMINATE:
                d[3][i] = ListAction(clf.predict(feat)[0])

        pointers = []
        for j in range(len(d[3])):
            X_test_ptr = []
            ptr_candidates = []
            if d[3][j] == ListAction.UP:
                for i in range(j):
                    if d[3][i] == ListAction.DOWN:
                        feat = d[9].extract_pointer_features(
                            d[2], d[3][:j], i, j)
                        X_test_ptr.append(Document.get_feature_array(feat))
                        ptr_candidates.append(i)
                if len(X_test_ptr) > 0:
                    pointers.append(ptr_candidates[np.argmax(
                        clf_ptr.predict_proba(np.array(X_test_ptr))[:, 1])])
                else:
                    # When it is UP but there exists no DOWN to point to
                    d[3][j] = ListAction.SAME_LEVEL
                    pointers.append(-1)
            else:
                pointers.append(-1)
        d[4] = pointers
        predicted_documents.append(d)
        cum_j += len(document[1])
    return predicted_documents


def k_fold_train_predict(documents, n_splits: int=5, used_features: Optional[List[int]]=None):
    test_indices = []
    predicted_documents = []
    cv_documents = defaultdict(list)
    for i, document in enumerate(documents):
        cv_documents[document[0]].append((document, i))
    cv_documents = list(cv_documents.values())

    random.seed(123)
    np.random.seed(123)
    for train_index, test_index in KFold(n_splits=n_splits, shuffle=True).split(X=cv_documents):
        test_indices.append([ind for j in test_index for _, ind in cv_documents[j]])
        documents_train = [document for j in train_index for document, ind in cv_documents[j]]
        documents_test = [document for j in test_index for document, ind in cv_documents[j]]
        clf, clf_ptr = train_classifiers(documents_train, used_features)
        predicted_documents.extend(
            predict_with_classifiers(clf, clf_ptr, documents_test, used_features))
    predicted_documents = [
        predicted_documents[j] for j in np.argsort(np.concatenate(test_indices))]
    return predicted_documents
