import os

from index import InvertedIndexWriter
from base_index import BaseIndex, preprocess_text
from my_compression import StandardPostings, VBEPostings, EliasGammaPostings
from tqdm import tqdm
import sys


class SPIMIIndex(BaseIndex):
    """
    Implements Single-Pass In-Memory Indexing (SPIMI).

    Unlike BSBI which first collects all (termID, docID) pairs and then
    sorts them, SPIMI builds the inverted index dictionary directly
    while parsing each document — no intermediate pair list is created.
    Terms and their postings lists are inserted into the dictionary
    on-the-fly, and the dictionary is written to disk when the block is
    fully processed.
    """

    def spimi_invert(self, block_dir_relative, index):
        """
        Parses documents in a block directory and directly builds the
        inverted index in memory (single-pass), then writes it to disk.

        Unlike BSBI's two-step approach (parse_block → invert_write),
        SPIMI constructs the term → postings dictionary on-the-fly
        as each document is read. No intermediate list of (termID, docID)
        pairs is ever created.

        Parameters
        ----------
        block_dir_relative : str
            Relative path to the directory containing text files for a
            single block.
        index : InvertedIndexWriter
            On-disk inverted index to write the block's data to.
        """
        term_postings = {}

        dir_path = "./" + self.data_dir + "/" + block_dir_relative
        for filename in next(os.walk(dir_path))[2]:
            docname = dir_path + "/" + filename
            doc_id = self.doc_id_map[docname]

            with open(docname, "r", encoding="utf8", errors="surrogateescape") as f:
                content = f.read()
                for token in preprocess_text(content):
                    term_id = self.term_id_map[token]

                    if term_id not in term_postings:
                        term_postings[term_id] = {}
                    if doc_id not in term_postings[term_id]:
                        term_postings[term_id][doc_id] = 0
                    term_postings[term_id][doc_id] += 1

        for term_id in sorted(term_postings.keys()):
            sorted_doc_ids = sorted(term_postings[term_id].keys())
            tf_list = [term_postings[term_id][doc_id] for doc_id in sorted_doc_ids]
            index.append(term_id, sorted_doc_ids, tf_list)

    def index(self):
        """
        Main indexing method using the SPIMI (Single-Pass In-Memory
        Indexing) scheme.

        For each block, directly builds an in-memory inverted index
        while parsing documents (no intermediate pair list), writes
        intermediate indices to disk, then merges them into a single
        final index.
        """
        
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            index_id = 'intermediate_index_' + block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
                self.spimi_invert(block_dir_relative, index)

        self._merge_intermediate_indices()


if __name__ == "__main__":
    postings_encoding = VBEPostings
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "standard":
            postings_encoding = StandardPostings
        elif arg == "elias":
            postings_encoding = EliasGammaPostings
        elif arg == "vbe":
            postings_encoding = VBEPostings
        else:
            print("Unknown encoding. Defaulting to VBE. Options: standard, vbe, elias.")

    SPIMI_instance = SPIMIIndex(data_dir = 'collection',
                                postings_encoding = postings_encoding,
                                output_dir = 'index')
    SPIMI_instance.index()
