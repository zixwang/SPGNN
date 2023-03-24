import logging
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import torch
import torch.nn.functional as F


def k_mers(k, seq):
    if k > len(seq):
        return []

    num = len(seq)-k+1
    split = []
    for i in range(num):
        split.append(seq[i:i+k])

    return split


def create_tagged_documents(mers, name):
    # Define a function to create tagged documents from a list of strings
    tagged_docs = [TaggedDocument(mers[i], name[i])
                   for i in range(len(name))]

    return tagged_docs


def train_doc2vec_model(mers, name):
    # Define a function to train a Doc2Vec model on a list of strings
    tagged_docs = create_tagged_documents(mers, name)
    model = Doc2Vec(vector_size=100, min_count=1, epochs=100)
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count,
                epochs=model.epochs)

    return model


def get_vector_embeddings(all_mers, all_name, model):
    # Define a function to get vector embeddings of strings using a Doc2Vec model
    tagged_docs = create_tagged_documents(all_mers, all_name)
    vectors = np.array([])
    vectors = {}
    for doc in tagged_docs:
        vectors[doc.tags] = model.infer_vector(doc.words)

    return vectors


def log_and_print(message):
    logging.info(message)
    print(message)


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]),
                        torch.zeros(neg_score.shape[0])]).cuda()
    return F.binary_cross_entropy_with_logits(scores, labels)


def get_dcg(y_pred, y_true, k):
    df = pd.DataFrame({"y_pred": y_pred, "y_true": y_true})
    df = df.sort_values(by="y_pred", ascending=False)
    df = df.iloc[0:k, :]
    dcg = (2 ** df["y_true"] - 1) / \
        np.log2(np.arange(1, df["y_true"].count() + 1) + 1)
    dcg = np.sum(dcg)
    return dcg


def NDCG(y_true, y_pred):
    k = len(y_pred)
    dcg = get_dcg(y_pred, y_true, k)
    idcg = get_dcg(y_true, y_true, k)
    ndcg = dcg / idcg
    return ndcg


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores), average_precision_score(labels, scores), NDCG(labels, scores)
