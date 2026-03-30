import re
import math
import sys
from base_index import BaseIndex
from my_compression import StandardPostings, VBEPostings, EliasGammaPostings

######## >>>>> IR metric: RBP p = 0.8

def rbp(ranking, p = 0.8):
  """ Computes the Rank Biased Precision (RBP) search effectiveness
      metric score.

      Parameters
      ----------
      ranking: List[int]
         Binary relevance vector, e.g. [1, 0, 1, 1, 1, 0].
         Gold standard relevance of documents at rank 1, 2, 3, etc.
         Example: [1, 0, 1, 1, 1, 0] means the document at rank-1 is
                  relevant, rank-2 is not, ranks 3-5 are relevant,
                  and rank-6 is not relevant.

      Returns
      -------
      float
        RBP score.
  """
  score = 0.
  for i in range(1, len(ranking)):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score

def ap(ranking):
  """
  Computes Average Precision (AP) for a single query.

  Parameters
  ----------
  ranking : List[int]
      Binary relevance vector (1 = relevant, 0 = not relevant).

  Returns
  -------
  float
      Average Precision score.
  """
  score = 0
  num_rel = 0
  approximated_R = sum(ranking)

  for i in range(len(ranking)):
    if ranking[i] == 1:
      num_rel += 1
      score += num_rel / (i + 1)
  return score / approximated_R

def dcg(ranking):
  """
  Computes Discounted Cumulative Gain (DCG) for a single query.

  Parameters
  ----------
  ranking : List[int]
      Binary relevance vector (1 = relevant, 0 = not relevant).

  Returns
  -------
  float
      DCG score.
  """
  score = 0
  for i in range(len(ranking)):
    if ranking[i] == 1:
      score += 1 /math.log2(i + 2)
  return score

def ndcg(ranking):
  """
  Computes Normalized Discounted Cumulative Gain (nDCG) for a single
  query. nDCG = DCG / ideal DCG.

  Parameters
  ----------
  ranking : List[int]
      Binary relevance vector (1 = relevant, 0 = not relevant).

  Returns
  -------
  float
      nDCG score in [0, 1].
  """
  return dcg(ranking) / dcg(sorted(ranking, reverse = True))


######## >>>>> loading qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ Loads query relevance judgments (qrels) into a dictionary of
      dictionaries: qrels[query_id][document_id].

      For example, qrels["Q3"][12] = 1 means Doc 12 is relevant to Q3;
      qrels["Q3"][10] = 0 means Doc 10 is not relevant to Q3.
  """
  qrels = {"Q" + str(i) : {i:0 for i in range(1, max_doc_id + 1)} \
                 for i in range(1, max_q_id + 1)}
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1
  return qrels

######## >>>>> EVALUATION !

def eval(qrels, query_file = "queries.txt", k = 1000, postings_encoding=None):
  """ 
    Loops over all 30 queries, computes scores for each query,
    then computes the MEAN SCORE across those 30 queries.
    For each query, returns top-1000 documents.
  """
  searcher = BaseIndex(data_dir = 'collection', \
                       postings_encoding = postings_encoding, \
                       output_dir = 'index')

  methods = {
      "TF-IDF": searcher.retrieve_tfidf,
      "BM25": searcher.retrieve_bm25,
      "BM25 + WAND": searcher.retrieve_bm25_wand
  }

  scores = {name: {"RBP": [], "AP": [], "DCG": [], "nDCG": []} for name in methods}

  with open(query_file) as file:
    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])

      for name, retrieve_func in methods.items():
        ranking = []
        
        for (score, doc) in retrieve_func(query, k = k):
            did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
            ranking.append(qrels[qid][did])

        scores[name]["RBP"].append(rbp(ranking))
        scores[name]["AP"].append(ap(ranking))
        scores[name]["DCG"].append(dcg(ranking))
        scores[name]["nDCG"].append(ndcg(ranking))

  for name in methods:
      print(f"{name} evaluation results over 30 queries")
      for metric in ["RBP", "AP", "DCG", "nDCG"]:
          score_list = scores[name][metric]
          avg_score = sum(score_list) / len(score_list) if score_list else 0.0
          print(f"{metric} score = {avg_score}")
      print()

if __name__ == '__main__':
  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "qrels incorrect"
  assert qrels["Q1"][300] == 0, "qrels incorrect"

  eval(qrels, postings_encoding=None)