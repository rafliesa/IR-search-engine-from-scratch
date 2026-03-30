import os

from index import InvertedIndexWriter
from base_index import BaseIndex, preprocess_text
from my_compression import StandardPostings, VBEPostings, EliasGammaPostings
from tqdm import tqdm
import sys


class BSBIIndex(BaseIndex):
    """
    Implements Blocked Sort-Based Indexing (BSBI).

    For each block, collects all (termID, docID) pairs into an
    in-memory list, sorts them, builds the inverted index, and writes
    it to an intermediate index file. After all blocks are processed,
    merges the intermediate indices into a single final index.
    """

    def parse_block(self, block_dir_relative):
        """
        Parses text files in a block directory into a sequence of
        <termID, docID> pairs. Applies tokenization, stopword removal,
        and stemming.

        Parameters
        ----------
        block_dir_relative : str
            Relative path to the directory containing text files for a
            single block. Each folder in the collection represents one block.

        Returns
        -------
        List[Tuple[int, int]]
            All <termID, docID> pairs extracted from the block.

        Uses self.term_id_map and self.doc_id_map to obtain termIDs and
        docIDs. These mappings persist across all calls to parse_block().
        """
        dir = "./" + self.data_dir + "/" + block_dir_relative
        td_pairs = []
        for filename in next(os.walk(dir))[2]:
            docname = dir + "/" + filename
            with open(docname, "r", encoding = "utf8", errors = "surrogateescape") as f:
                content = f.read()
                for token in preprocess_text(content):
                    td_pairs.append((self.term_id_map[token], self.doc_id_map[docname]))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Inverts td_pairs (list of <termID, docID> pairs) and writes them
        to the given index.

        Assumption: td_pairs fits in memory.

        Parameters
        ----------
        td_pairs: List[Tuple[int, int]]
            List of termID-docID pairs.
        index: InvertedIndexWriter
            On-disk inverted index associated with a block.
        """
        term_dict = {}
        term_tf = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
                term_tf[term_id] = {}
            term_dict[term_id].add(doc_id)
            if doc_id not in term_tf[term_id]:
                term_tf[term_id][doc_id] = 0
            term_tf[term_id][doc_id] += 1
        for term_id in sorted(term_dict.keys()):
            sorted_doc_id = sorted(list(term_dict[term_id]))
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_id]
            index.append(term_id, sorted_doc_id, assoc_tf)

    def index(self):
        """
        Main indexing method using the BSBI (Blocked Sort-Based Indexing)
        scheme.

        Scans all data in the collection, calls parse_block to parse
        documents, and calls invert_write to invert and store each block.
        Finally merges all intermediate indices into a single index.
        """
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None

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

    BSBI_instance = BSBIIndex(data_dir = 'collection',
                              postings_encoding = postings_encoding,
                              output_dir = 'index')
    BSBI_instance.index()
