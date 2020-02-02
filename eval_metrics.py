#import sklearn.metrics as metrics
import logging

import numpy as np

import utils


def precision_at_k(relevance_score, K):
  assert K >= 1
  #print ("P@k here", relevance_score)
  relevance_score = np.asarray(relevance_score)[:K] != 0
  if relevance_score.size != K:
    raise ValueError('Relevance score length < K')
  return np.mean(relevance_score)


def mean_precision_at_k(relevance_scores, K):
  mean_precision_at_k = np.mean(
      [precision_at_k(r, K) for r in relevance_scores]).astype(np.float32)
  return mean_precision_at_k


def jaccard(relevance_scores, K, M_list):
  relevance_scores = np.asarray(relevance_scores)[:, :K]
  M_list = np.asarray(M_list)
  tp = np.sum(relevance_scores, 1).astype(np.float32)
  jaccard = tp / (M_list + K - tp)
  return np.mean(jaccard)


def recall_at_k(relevance_score, K, M):
  ''' M is # relevant outputs'''
  assert K >= 1
  relevance_score = np.asarray(relevance_score)[:K] != 0
  if relevance_score.size != K:
    raise ValueError('Relevance score length < K')
  return np.sum(relevance_score) / float(M)


def mean_recall_at_k(relevance_scores, K, M_list):
  mean_recall_at_k = np.mean([
      recall_at_k(r, K, M) for r, M in zip(relevance_scores, M_list)
  ]).astype(np.float32)
  return mean_recall_at_k


def average_precision(relevance_score, K, M):
  ''' For average precision, we use K as input since the number of predictions is not fixed unlike standard IR evaluation.'''
  r = np.asarray(relevance_score) != 0
  out = [precision_at_k(r, k + 1) for k in range(0, K) if r[k]]
  if not out:
    return 0.
  return np.sum(out) / float(min(K, M))


def MAP(relevance_scores, K, M_list):
  map_val = np.mean([
      average_precision(r, K, M) for r, M in zip(relevance_scores, M_list)
  ]).astype(np.float32)
  #print ("MAP here ", map_val)
  return map_val


# Mean reciprocal rank -- MRR
def MRR(relevance_scores):
  rs = (np.asarray(r).nonzero()[0] for r in relevance_scores)
  mrr_val = np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs
                    ]).astype(np.float32)
  #logging.info("MRR here %f",mrr_val)
  return mrr_val


def DCG_at_k(r, k):
  r = np.asfarray(r)[:k]
  if r.size:
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))
  return 0.


def NDCG_at_k(r, k):
  dcg_max = DCG_at_k(sorted(r, reverse=True), k)
  if not dcg_max:
    return 0.
  return DCG_at_k(r, k) / dcg_max


def mean_NDCG_at_k(relevance_scores, k):
  mean_ndcg = np.mean([NDCG_at_k(r, k) for r in relevance_scores
                      ]).astype(np.float32)
  return mean_ndcg


def get_masks(topk, decoder_inputs):
  masks = []
  for i in range(0, topk.shape[0]):
    seeds = set(decoder_inputs[i])
    if len(seeds) == 1 and list(seeds)[0] == utils.start_token:
      masks.append(0)
    else:
      masks.append(1)
  return np.array(masks).astype(np.int32)


# Also, remove the padded data in test set.
def remove_seeds(topk, decoder_inputs):
  result = []
  for i in range(0, topk.shape[0]):
    seeds = set(decoder_inputs[i])
    l = list(topk[i])
    for s in seeds:
      if s in l:
        l.remove(s)
    for k in range(len(topk[i]) - len(l)):
      l.append(-1)
    result.append(l)
  return np.array(result).astype(np.int32)


def get_relevance_scores(topk_filter, decoder_targets):
  # topk_filter - [batch_size, 200]
  # decoder_targets - [batch_size, _]
  output = []
  for i in range(0, topk_filter.shape[0]):
    z = np.isin(
        topk_filter[i], decoder_targets[i]
    )  # decoder_targets may have start_token but topk_filter doesn't.
    output.append(z)
  return np.array(output)
