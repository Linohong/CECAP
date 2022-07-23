from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

def record_recall_result(recall_result, indices):
    for N, cur_score in recall_result['score'].items():
        # 0 is the history index
        if indices.index(1) <= N-1:  # found at 0 position --> plus 1 point to the recall @ 1
            recall_result['score'][N] += 1.0

    recall_result['total_num'] += 1

def sort_candidates_by(method, dialogue_history, candidates, pred_record=None, recall_result=None):
    '''
        this module is currently (2022.01.28) necessary
        only at the inference state.
        including the sorting objective to the training phase
        will be considered as the future work.
    '''
    if method == 'tf-idf':
        # corpus = [
        #             'This is the first document.',
        #             'This document is the second document.',
        #             'And this is the third one.',
        #             'Is this the first document?',
        #         ]
        corpus = [dialogue_history] + candidates

        # Initialize an instance of tf-idf Vectorizer
        tfidf_vectorizer = TfidfVectorizer()

        # Generate the tf-idf vectors for the corpus
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

        # compute and print the cosine similarity matrix
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        sorted_indices = [i[0] for i in sorted(enumerate(cosine_sim[0,:]), key=lambda k: k[1], reverse=True) if i[0] != 0] # index 0 is a dialogue history itself.
        # sorted_indices has only the candidate indices from here on.

        # 2022.04.20 Lino
        record_recall_result(recall_result, sorted_indices)

        sorted_candidates = [candidates[i-1] for i in sorted_indices]  # exclude 0 (=history itself), thus -1 in "candidates[i-1]".
        sorted_candidates = sorted_candidates[:5]
        print('to 5th candidates')

    elif method == 'ks_pred_record':
        try:
            sorted_dict_list = pred_record['<s>' + dialogue_history.replace(' ', '')]
            sorted_candidates = [item['response'] for item in sorted_dict_list]
        except KeyError:
            print('Key Error occured in the pred_record')
            sorted_candidates = candidates
        # sorted_candidates = sorted_candidates[:min(10, len(sorted_candidates))]

    elif method == 'answer_kno':
        # Lino's setting (2022.04.22)
        neg_samples = []
        # if len(candidates) > 1:
        #     neg_samples = random.sample(candidates[1:], k=min(4, len(candidates[1:])))
        sorted_candidates = [candidates[0]] + neg_samples

    elif method == 'wo_answer_kno':
        # Lino's setting (2022.04.26)
        neg_samples = []
        if len(candidates) > 1:
            neg_samples = random.sample(candidates[1:], k=min(5, len(candidates[1:])))
        else:
            neg_samples = candidates[0]

        sorted_candidates = neg_samples

    return sorted_candidates