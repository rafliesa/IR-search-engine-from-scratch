import os
import pickle
import contextlib
import heapq
from bisect import bisect_left
import math
from core import my_compression
from core.index import InvertedIndexReader, InvertedIndexWriter
from core.util import IdMap, sorted_merge_posts_and_tfs, preprocess_text
import sys
sys.modules['util'] = sys.modules['core.util']
sys.modules['my_compression'] = sys.modules['core.my_compression']
sys.modules['index'] = sys.modules['core.index']


class BaseIndex:
    """
    Abstract base class for inverted index construction and retrieval.

    Provides shared infrastructure used by both BSBI and SPIMI indexing
    schemes: ID mappings, serialization, merging of intermediate indices,
    and all retrieval methods (TF-IDF, BM25, BM25 + WAND).

    Subclasses must implement the `index()` method with their specific
    indexing strategy.

    Attributes
    ----------
    term_id_map(IdMap): Maps terms to termIDs.
    doc_id_map(IdMap): Maps relative document paths (e.g.,
                    /collection/0/gamma.txt) to docIDs.
    data_dir(str): Path to the data directory.
    output_dir(str): Path to the output index files directory.
    postings_encoding: Postings encoding scheme (e.g., StandardPostings,
                    VBEPostings, EliasGammaPostings).
    index_name(str): Name of the file containing the inverted index.
    """
    def __init__(self, data_dir, output_dir, postings_encoding=None, index_name = "main_index"):
        """
        Initializes the index.

        Parameters
        ----------
        data_dir : str
            Path to the directory containing the document collection.
        output_dir : str
            Path to the directory where index files will be stored.
        postings_encoding : class
            Encoding scheme for postings lists (e.g., StandardPostings,
            VBEPostings, EliasGammaPostings).
        index_name : str
            Base name for the merged index file (default: "main_index").
        """
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Stores filenames of all intermediate inverted indices
        self.intermediate_indices = []

    def save(self):
        """Saves doc_id_map and term_id_map to the output directory via pickle."""

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)
        with open(os.path.join(self.output_dir, 'encoding.dict'), 'wb') as f:
            pickle.dump(self.postings_encoding.__name__, f)

    def load(self):
        """Loads doc_id_map and term_id_map from the output directory."""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

        if os.path.exists(os.path.join(self.output_dir, 'encoding.dict')):
            with open(os.path.join(self.output_dir, 'encoding.dict'), 'rb') as f:
                enc_str = pickle.load(f)
                self.postings_encoding = getattr(my_compression, enc_str)

    def merge(self, indices, merged_index):
        """
        Merges all intermediate inverted indices into a single index
        using external merge sort.

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, each
            representing an iterable intermediate inverted index for
            one block.

        merged_index: InvertedIndexWriter
            The InvertedIndexWriter that will hold the result of merging
            all intermediate indices.
        """
        # The following code assumes at least 1 term exists
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def _merge_intermediate_indices(self):
        """
        Saves ID mappings and merges all intermediate indices into a
        single final index. Called at the end of index().
        """
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)

    def retrieve_tfidf(self, query, k = 10):
        """
        Performs Ranked Retrieval using Term-at-a-Time (TaaT) with TF-IDF.
        Returns top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       if tf(t, D) > 0
                = 0                        otherwise

        w(t, Q) = IDF = log (N / df(t))

        Score = for each query term, accumulate w(t, Q) * w(t, D).
                (no document length normalization)

        Parameters
        ----------
        query: str
            Space-separated query tokens.

        Result
        ------
        List[(float, str)]
            List of (score, doc_name) tuples, top-K sorted in descending
            order by score.

        Does NOT raise exceptions for terms not found in the collection.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in preprocess_text(query) if word in self.term_id_map]
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    N = len(merged_index.doc_length)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        if tf > 0:
                            scores[doc_id] += math.log(N / df) * (1 + math.log(tf))

            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def retrieve_bm25(self, query, k = 10, k1 = 1.2, b = 0.75):
        """
        Performs Ranked Retrieval using Term-at-a-Time (TaaT) with BM25 scoring.

        BM25 formula:
            Score(D, Q) = Σ IDF(t) * (tf(t,D) * (k1 + 1)) / (tf(t,D) + k1 * (1 - b + b * dl/avdl))

        where:
            IDF(t) = log(N / df(t))
            dl     = document length (number of tokens)
            avdl   = average document length across the collection

        Document lengths (dl) are pre-computed during indexing and stored
        in doc_length. avdl is derived from doc_length.

        Parameters
        ----------
        query: str
            Space-separated query tokens.

        k: int
            Number of top-K documents to return.

        k1: float
            BM25 term frequency scaling parameter (default: 1.2).

        b: float
            BM25 document length normalization parameter (default: 0.75).

        Result
        ------
        List[(float, str)]
            List of (score, doc_name) tuples, top-K sorted in descending
            order by score.

        Does NOT raise exceptions for terms not found in the collection.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in preprocess_text(query) if word in self.term_id_map]
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            N = len(merged_index.doc_length)
            avdl = sum(merged_index.doc_length.values()) / N

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    idf = math.log(N / df)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        dl = merged_index.doc_length[doc_id]

                        tf_normalized = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avdl))
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        scores[doc_id] += idf * tf_normalized

            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def retrieve_bm25_wand(self, query, k = 10, k1 = 1.2, b = 0.75):
        """
        Performs Ranked Retrieval using the WAND (Weak AND) algorithm
        with BM25 scoring. Unlike exhaustive BM25, this method skips
        documents that cannot possibly enter the top-K by maintaining
        per-term upper-bound BM25 scores computed from max TF values
        stored in the inverted index.

        The algorithm:
        1. Computes an upper-bound BM25 contribution for each query term
           using the max TF from the index.
        2. Maintains pointers into each term's postings list.
        3. Sorts terms by their current docID pointer.
        4. Accumulates upper-bounds until exceeding the current top-K
           threshold (pivot selection). If the pivot document is
           reachable, computes its exact BM25 score; otherwise, advances
           pointers using bisect for efficient skipping.
        5. Repeats until all postings lists are exhausted.

        Parameters
        ----------
        query : str
            Space-separated query tokens.
        k : int
            Number of top-K documents to return (default: 10).
        k1 : float
            BM25 term frequency scaling parameter (default: 1.2).
        b : float
            BM25 document length normalization parameter (default: 0.75).

        Returns
        -------
        List[(float, str)]
            List of (score, doc_name) tuples, top-K sorted in descending
            order by score.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in preprocess_text(query) if word in self.term_id_map]
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            n = len(merged_index.doc_length)
            avdl = sum(merged_index.doc_length.values()) / n

            postings_data = []
            for term in terms:
                if term not in merged_index.postings_dict:
                    continue
                meta = merged_index.postings_dict[term]
                df = meta[1]
                max_tf = meta[4] if len(meta) > 4 else 1
                idf = math.log(n / df)

                postings, tf_list = merged_index.get_postings_list(term)

                ub_tf = (max_tf * (k1 + 1)) / (max_tf + k1 * (1 - b))
                ub = idf * ub_tf

                postings_data.append([postings, tf_list, 0, idf, ub])

            if not postings_data:
                return []

            n_terms = len(postings_data)

            top_k_heap = []
            threshold = 0.0

            while True:
                postings_data = [pd for pd in postings_data if pd[2] < len(pd[0])]
                n_terms = len(postings_data)
                if n_terms == 0:
                    break

                postings_data.sort(key=lambda pd: pd[0][pd[2]])

                pivot_idx = -1
                ub_sum = 0.0
                for i in range(n_terms):
                    ub_sum += postings_data[i][4]
                    if ub_sum >= threshold:
                        pivot_idx = i
                        break

                if pivot_idx == -1:
                    break

                pivot_doc = postings_data[pivot_idx][0][postings_data[pivot_idx][2]]

                if postings_data[0][0][postings_data[0][2]] == pivot_doc:
                    score = 0.0
                    dl = merged_index.doc_length[pivot_doc]
                    denom_base = k1 * (1 - b + b * dl / avdl)
                    for pd in postings_data:
                        ptr = pd[2]
                        if ptr < len(pd[0]) and pd[0][ptr] == pivot_doc:
                            tf = pd[1][ptr]
                            tf_normalized = (tf * (k1 + 1)) / (tf + denom_base)
                            score += pd[3] * tf_normalized
                            pd[2] = ptr + 1

                    if len(top_k_heap) < k:
                        heapq.heappush(top_k_heap, (score, pivot_doc))
                        if len(top_k_heap) == k:
                            threshold = top_k_heap[0][0]
                    elif score > threshold:
                        heapq.heapreplace(top_k_heap, (score, pivot_doc))
                        threshold = top_k_heap[0][0]
                else:
                    for pd in postings_data:
                        ptr = pd[2]
                        if ptr < len(pd[0]) and pd[0][ptr] < pivot_doc:
                            new_ptr = bisect_left(pd[0], pivot_doc, ptr)
                            pd[2] = new_ptr

            results = [(score, self.doc_id_map[doc_id]) for (score, doc_id) in top_k_heap]
            return sorted(results, key=lambda x: x[0], reverse=True)

    def index(self):
        """
        Builds the inverted index. Must be implemented by subclasses
        (BSBIIndex or SPIMIIndex).
        """
        raise NotImplementedError("Subclasses must implement index()")
