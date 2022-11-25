import copy
import random
from typing import List, Optional
from collections import defaultdict
import numpy as np
import sklearn_crfsuite
from sklearn.model_selection import KFold

from pdf_struct.core.transition_labels import ListAction
from pdf_struct.core.document import Document

def document2features(sent, t:int):
    feats = []
    for i in range(len(sent)):
        d = sent[i]
        for y in range(len(d[2])):
            features = {
                ''''Fontname_1_2': str(list(list(d[t].values())[9].values())[0][y]),
                'Fontname_2_3': str(list(list(d[t].values())[9].values())[1][y]),
                'Fontsize_0_1_2': str(list(list(d[t].values())[8].values())[0][y]),
                'Fontsize_0_2_3': str(list(list(d[t].values())[8].values())[1][y]),
                'Fontsize_1_1_2': str(list(list(d[t].values())[8].values())[2][y]),'''
                'Fontsize': str(list(list(d[t].values())[8].values())[0][y]),
                'all_capital_2': str(list(list(d[t].values())[19].values())[0][y]),
                'all_capital_3': str(list(list(d[t].values())[19].values())[1][y]),
                'centered_2': str(list(list(d[t].values())[5].values())[0][y]),
                'centered_3': str(list(list(d[t].values())[5].values())[1][y]),
                'colon_ish_1': str(list(list(d[t].values())[15].values())[0][y]),
                'colon_ish_2': str(list(list(d[t].values())[15].values())[1][y]),
                'dict_like_2': str(list(list(d[t].values())[9].values())[0][y]),
                'dict_like_3': str(list(list(d[t].values())[9].values())[1][y]),
                'extra_line_space_1_2': str(list(list(d[t].values())[6].values())[0][y]),
                'extra_line_space_2_3': str(list(list(d[t].values())[6].values())[1][y]),
                'footer_region_2': str(list(list(d[t].values())[2].values())[0][y]),
                'header_region_2': str(list(list(d[t].values())[1].values())[0][y]),
                'indent_1_2': str(list(list(d[t].values())[4].values())[0][y]),
                'indent_2_3': str(list(list(d[t].values())[4].values())[1][y]),
                'line_break_1_2': str(list(list(d[t].values())[3].values())[0][y]),
                'line_break_2_3': str(list(list(d[t].values())[3].values())[0][y]),
                'list_ish_2': str(list(list(d[t].values())[17].values())[0][y]),
                'mask_continuation_1_2': str(list(list(d[t].values())[18].values())[0][y]),
                'mask_continuation_2_3': str(list(list(d[t].values())[18].values())[1][y]),
                'numbered_list_state_value': str(list(list(d[t].values())[12].values())[0][y]),
                'page_change_1_2': str(list(list(d[t].values())[7].values())[0][y]),
                'page_change_2_3': str(list(list(d[t].values())[7].values())[1][y]),
                'page_like_1': str(list(list(d[t].values())[10].values())[0][y]),
                'page_like_2': str(list(list(d[t].values())[10].values())[1][y]),
                'page_like_3': str(list(list(d[t].values())[10].values())[2][y]),
                'page_like2_1': str(list(list(d[t].values())[11].values())[0][y]),
                'page_like2_2': str(list(list(d[t].values())[11].values())[1][y]),
                'page_like2_3': str(list(list(d[t].values())[11].values())[2][y]),
                'punctuated_1': str(list(list(d[t].values())[16].values())[0][y]),
                'punctuated_2': str(list(list(d[t].values())[16].values())[1][y]),
                'similar_position_similar_text_2': str(list(list(d[t].values())[0].values())[0][y]),
                'space_separated_2': str(list(list(d[t].values())[20].values())[0][y]),
                'space_separated_3': str(list(list(d[t].values())[20].values())[1][y]),
                'whereas_3': str(list(list(d[t].values())[13].values())[0][y]),
                'therefore_3': str(list(list(d[t].values())[14].values())[0][y]),
            }
            feats.append([features])
    return feats

def document2features4pointer(sent):
    feats = []
    for i in range(len(sent)):
        d = sent[i]
        pointer_feats_array = []
        if d[7] is not None and len(d[7]) > 0:
            pointer_feats_array.append(Document._get_feature_matrix(d[7]))
        for i in range(len(pointer_feats_array)):
            for y in range(len(pointer_feats_array[i])):
                features = {
                    'n_downs': str(list(list(d[7].values())[0].values())[0][y]),
                    'n_ups': str(list(list(d[7].values())[0].values())[1][y]),
                    'n_ups_downs_difference': str(list(list(d[7].values())[0].values())[2][y]),
                    'pointer_indent_1_2': str(list(list(d[7].values())[1].values())[0][y]),
                    'pointer_indent_1_3': str(list(list(d[7].values())[1].values())[1][y]),
                    'pointer_indent_head_2': str(list(list(d[7].values())[1].values())[2][y]),
                    'pointer_indent_head_3': str(list(list(d[7].values())[1].values())[3][y]),
                    'pointer_left_aligned_1': str(list(list(d[7].values())[2].values())[0][y]),
                    'pointer_left_aligned_3': str(list(list(d[7].values())[2].values())[1][y]),
                    'pointer_left_aligned_head': str(list(list(d[7].values())[2].values())[2][y]),
                    'pointer_section_number_3_next_of_1': str(list(list(d[7].values())[3].values())[0][y]),
                    'pointer_section_number_3_next_of_head': str(list(list(d[7].values())[3].values())[1][y])
                }
                feats.append([features])
    return feats

def feat2testptr(feat):
    feats = []
    features = {
        'n_downs': str(feat['transition']['n_downs']),
        'n_ups': str(feat['transition']['n_ups']),
        'n_ups_downs_difference': str(feat['transition']['n_ups_downs_difference']),
        'pointer_indent_1_2': str(feat['pointer_indent']['pointer_indent_1_2']),
        'pointer_indent_1_3': str(feat['pointer_indent']['pointer_indent_1_3']),
        'pointer_indent_head_2': str(feat['pointer_indent']['pointer_indent_head_2']),
        'pointer_indent_head_3': str(feat['pointer_indent']['pointer_indent_head_3']),
        'pointer_left_aligned_1': str(feat['pointer_left_aligned']['pointer_left_aligned_1']),
        'pointer_left_aligned_3': str(feat['pointer_left_aligned']['pointer_left_aligned_3']),
        'pointer_left_aligned_head': str(feat['pointer_left_aligned']['pointer_left_aligned_head']),
        'pointer_section_number_3_next_of_1': str(feat['pointer_section_number']['pointer_section_number_3_next_of_1']),
        'pointer_section_number_3_next_of_head': str(feat['pointer_section_number']['pointer_section_number_3_next_of_head'])
        }
    feats.append(features)
    return feats

def feat2lb(feat):
    feats = []
    features = {
        ''''Fontname_1_2': str(feat['Fontname']['Fontname_1_2']),
        'Fontname_2_3': str(feat['Fontname']['Fontname_2_3']),
        'Fontsize_0_1_2': str(feat['Fontsize']['Fontsize_0_1_2']),
        'Fontsize_0_2_3': str(feat['Fontsize']['Fontsize_0_2_3']),
        'Fontsize_1_1_2': str(feat['Fontsize']['Fontsize_1_1_2']),'''
        'Fontsize': str(feat['Fontsize']['Fontsize_2']),
        'all_capital_2': str(feat['all_capital']['all_capital_2']),
        'all_capital_3': str(feat['all_capital']['all_capital_3']),
        'centered_2': str(feat['centered']['centered_2']),
        'centered_3': str(feat['centered']['centered_3']),
        'colon_ish_1': str(feat['colon_ish']['colon_ish_1']),
        'colon_ish_2': str(feat['colon_ish']['colon_ish_2']),
        'dict_like_2': str(feat['dict_like']['dict_like_2']),
        'dict_like_3': str(feat['dict_like']['dict_like_3']),
        'extra_line_space_1_2': str(feat['extra_line_space']['extra_line_space_1_2']),
        'extra_line_space_2_3': str(feat['extra_line_space']['extra_line_space_2_3']),
        'footer_region_2': str(feat['footer_region']['footer_region_2']),
        'header_region_2': str(feat['header_region']['header_region_2']),
        'indent_1_2': str(feat['indent']['indent_1_2']),
        'indent_2_3': str(feat['indent']['indent_2_3']),
        'line_break_1_2': str(feat['line_break']['line_break_1_2']),
        'line_break_2_3': str(feat['line_break']['line_break_2_3']),
        'list_ish_2': str(feat['list_ish']['list_ish_2']),
        'mask_continuation_1_2': str(feat['mask_continuation']['mask_continuation_1_2']),
        'mask_continuation_2_3': str(feat['mask_continuation']['mask_continuation_2_3']),
        'numbered_list_state_value': str(feat['numbered_list_state']['numbered_list_state_value']),
        'page_change_1_2': str(feat['page_change']['page_change_1_2']),
        'page_change_2_3': str(feat['page_change']['page_change_2_3']),
        'page_like_1': str(feat['page_like']['page_like_1']),
        'page_like_2': str(feat['page_like']['page_like_2']),
        'page_like_3': str(feat['page_like']['page_like_3']),
        'page_like2_1': str(feat['page_like2']['page_like2_1']),
        'page_like2_2': str(feat['page_like2']['page_like2_2']),
        'page_like2_3': str(feat['page_like2']['page_like2_3']),
        'punctuated_1': str(feat['punctuated']['punctuated_1']),
        'punctuated_2': str(feat['punctuated']['punctuated_2']),
        'similar_position_similar_text_2': str(feat['similar_position_similar_text']['similar_position_similar_text_2']),
        'space_separated_2': str(feat['space_separated']['space_separated_2']),
        'space_separated_3': str(feat['space_separated']['space_separated_3']),
        'whereas_3': str(feat['whereas']['whereas_3']),
        'therefore_3': str(feat['therefore']['therefore_3']),
    }
    feats.append(features)
    return features

def train_classifiers(documents, used_features: Optional[List[int]]=None):
    used_features = None if used_features is None else np.array(used_features)

    X_train = document2features(documents,5)
    y_train = np.array([l.value for d in documents for l in d[3]], dtype=int)
    y_train = y_train.astype(str)
    if used_features is not None:
        X_train = X_train[:, used_features]

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=300,
        all_possible_transitions=True
    )
    clf = crf.fit(X_train, y_train)

    # Next, predict pointers
    X_train = document2features4pointer(documents)
    y_train = np.array([p == d[4][c] for d in documents for p, c in d[8]],dtype=int)
    y_train = y_train.astype(str)
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=300,
        all_possible_transitions=True
    )
    clf_ptr = crf.fit(X_train, y_train)
    return clf, clf_ptr

def predict_with_classifiers(clf, clf_ptr, documents, used_features: Optional[List[int]]=None):
    used_features = None if used_features is None else np.array(used_features)
    X_test = document2features(documents,6)
    if used_features is not None:
        X_test = X_test[:, used_features]
    y_pred = clf.predict(X_test)
    y_pred = np.array(sum(y_pred,[]),dtype=int)
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
            feats = []
            feat, states = d[9].extract_features(tb1, tb2, tb3, tb4, states)
            feats.append([feat2lb(feat)])
            if used_features is not None:
                feats = feats[:, used_features]
            if d[3][i] != ListAction.ELIMINATE:
                d[3][i] = ListAction(int(clf.predict(feats)[0][0]))


        pointers = []
        for j in range(len(d[3])):
            X_test_ptr = []
            ptr_candidates = []
            if d[3][j] == ListAction.UP:
                for i in range(j):
                    if d[3][i] == ListAction.DOWN:
                        feat = d[9].extract_pointer_features(
                            d[2], d[3][:j], i, j)
                        X_test_ptr.append(feat2testptr(feat))
                        ptr_candidates.append(i)
                if len(X_test_ptr) > 0:
                    pointers.append(ptr_candidates[np.argmax(
                        clf_ptr.predict(X_test_ptr))])
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