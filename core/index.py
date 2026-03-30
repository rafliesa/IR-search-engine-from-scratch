import pickle
import os

class InvertedIndex:
    """
    Implements efficient reading and writing of an Inverted Index stored
    on disk.

    Attributes
    ----------
    postings_dict: Dictionary mapping:

            termID -> (start_position_in_index_file,
                       number_of_postings_in_list,
                       length_in_bytes_of_postings_list,
                       length_in_bytes_of_tf_list,
                       max_tf_in_postings_list)

        The postings_dict is the "Dictionary" component of the Inverted
        Index. It is assumed to fit entirely in memory.

        Implemented as a Python dictionary that maps a term ID (integer)
        to a 5-tuple:
           1. start_position_in_index_file : byte offset where the
              corresponding postings list begins in the index file.
           2. number_of_postings_in_list : number of docIDs in the
              postings list (i.e., Document Frequency).
           3. length_in_bytes_of_postings_list : length of the encoded
              postings list in bytes.
           4. length_in_bytes_of_tf_list : length of the encoded term
              frequency list in bytes.
           5. max_tf_in_postings_list : maximum term frequency across
              all documents in this term's postings list. Required for
              computing upper-bound BM25 scores in the WAND algorithm.

    terms: List[int]
        Ordered list of term IDs, preserving the insertion order during
        index construction.

    """
    def __init__(self, index_name, postings_encoding, directory=''):
        """
        Parameters
        ----------
        index_name (str): Base name used for storing the index files.
        postings_encoding : Encoding scheme for postings lists (e.g.,
                        StandardPostings, VBEPostings, EliasGammaPostings).
        directory (str): Directory where the index files are stored.
        """

        self.index_file_path = os.path.join(directory, index_name+'.index')
        self.metadata_file_path = os.path.join(directory, index_name+'.dict')

        self.postings_encoding = postings_encoding
        self.directory = directory

        self.postings_dict = {}
        self.terms = []         # Tracks the insertion order of terms into the index
        self.doc_length = {}    # key: doc ID (int), value: document length (number of tokens)
                                # Used for document length normalization when computing
                                # TF-IDF or BM25 scores

    def __enter__(self):
        """
        Loads all metadata when entering the context.
        Metadata includes:
            1. Dictionary ---> postings_dict
            2. An iterator over the ordered list of terms inserted during
               index construction ---> term_iter
            3. doc_length, a dictionary mapping doc ID to the number of
               tokens (document length). Used for length normalization
               in TF-IDF or BM25 scoring, and to obtain N (total number
               of documents) for IDF computation.

        Metadata is persisted to disk using Python's pickle library.

        See also: Python Context Managers
        https://docs.python.org/3/reference/datamodel.html#object.__enter__
        """
        # Open the index file
        self.index_file = open(self.index_file_path, 'rb+')

        # Load postings dict and terms iterator from the metadata file
        with open(self.metadata_file_path, 'rb') as f:
            self.postings_dict, self.terms, self.doc_length = pickle.load(f)
            self.term_iter = self.terms.__iter__()

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Closes the index file and saves postings_dict and terms when exiting the context."""
        # Close the index file
        self.index_file.close()

        # Save metadata (postings dict and terms) to the metadata file using pickle
        with open(self.metadata_file_path, 'wb') as f:
            pickle.dump([self.postings_dict, self.terms, self.doc_length], f)


class InvertedIndexReader(InvertedIndex):
    """
    Implements efficient sequential scanning and random access reading
    of an Inverted Index stored on disk.
    """
    def __iter__(self):
        return self

    def reset(self):
        """
        Resets the file pointer to the beginning and resets the term
        iterator to the start.
        """
        self.index_file.seek(0)
        self.term_iter = self.terms.__iter__()  # reset term iterator

    def __next__(self):
        """
        InvertedIndexReader is iterable. When used as an iterator in a
        loop, __next__() returns the next (term, postings_list, tf_list)
        triple from the inverted index.

        IMPORTANT: This method must return only a small portion of data
        from the (potentially large) index file to keep memory usage low.
        DO NOT load the entire index into memory.
        """
        curr_term = next(self.term_iter)
        pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf, *_ = self.postings_dict[curr_term]
        postings_list = self.postings_encoding.decode(self.index_file.read(len_in_bytes_of_postings))
        tf_list = self.postings_encoding.decode_tf(self.index_file.read(len_in_bytes_of_tf))
        return (curr_term, postings_list, tf_list)

    def get_postings_list(self, term):
        """
        Returns the postings list (list of docIDs) and the associated
        term frequency list for a given term, as a tuple
        (postings_list, tf_list).

        IMPORTANT: This method must NOT iterate over the entire index.
        It must seek directly to the byte position in the index file
        where the postings list (and TF list) for the term is stored.
        """
        pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf, *_ = self.postings_dict[term]
        self.index_file.seek(pos)
        postings_list = self.postings_encoding.decode(self.index_file.read(len_in_bytes_of_postings))
        tf_list = self.postings_encoding.decode_tf(self.index_file.read(len_in_bytes_of_tf))
        return (postings_list, tf_list)


class InvertedIndexWriter(InvertedIndex):
    """
    Implements efficient writing of an Inverted Index to a file on disk.
    """
    def __enter__(self):
        """Opens the index file for writing and returns the writer instance."""
        if self.directory:
            os.makedirs(self.directory, exist_ok=True)
        self.index_file = open(self.index_file_path, 'wb+')
        return self

    def append(self, term, postings_list, tf_list):
        """
        Appends a term, its postings_list, and the associated TF list
        to the end of the index file.

        This method performs the following steps:
        1. Encodes postings_list using self.postings_encoding.encode(),
        2. Encodes tf_list using self.postings_encoding.encode_tf(),
        3. Updates metadata: self.terms, self.postings_dict, and self.doc_length.
        4. Writes the encoded bytestreams to the end of the index file on disk.

        Parameters
        ----------
        term:
            The term or termID serving as a unique identifier.
        postings_list: List[int]
            List of docIDs where the term appears.
        tf_list: List[int]
            List of term frequencies.
        """
        self.terms.append(term) 

        # update self.doc_length
        for i in range(len(postings_list)):
            doc_id, freq = postings_list[i], tf_list[i]
            if doc_id not in self.doc_length:
                self.doc_length[doc_id] = 0
            self.doc_length[doc_id] += freq

        # Compute max TF for this term's postings list needed by WAND
        max_tf = max(tf_list) if tf_list else 0

        self.index_file.seek(0, os.SEEK_END)
        curr_position_in_byte = self.index_file.tell()
        compressed_postings = self.postings_encoding.encode(postings_list)
        compressed_tf_list = self.postings_encoding.encode_tf(tf_list)
        self.index_file.write(compressed_postings)
        self.index_file.write(compressed_tf_list)
        self.postings_dict[term] = (curr_position_in_byte, len(postings_list), \
                                    len(compressed_postings), len(compressed_tf_list), \
                                    max_tf)


if __name__ == "__main__":

    from my_compression import VBEPostings

    with InvertedIndexWriter('test', postings_encoding=VBEPostings, directory='./tmp/') as index:
        index.append(1, [2, 3, 4, 8, 10], [2, 4, 2, 3, 30])
        index.append(2, [3, 4, 5], [34, 23, 56])
        index.index_file.seek(0)
        assert index.terms == [1,2], "terms incorrect"
        assert index.doc_length == {2:2, 3:38, 4:25, 5:56, 8:3, 10:30}, "doc_length incorrect"
        assert index.postings_dict == {1: (0, \
                                           5, \
                                           len(VBEPostings.encode([2,3,4,8,10])), \
                                           len(VBEPostings.encode_tf([2,4,2,3,30])), \
                                           30),
                                       2: (len(VBEPostings.encode([2,3,4,8,10])) + len(VBEPostings.encode_tf([2,4,2,3,30])), \
                                           3, \
                                           len(VBEPostings.encode([3,4,5])), \
                                           len(VBEPostings.encode_tf([34,23,56])), \
                                           56)}, "postings dictionary incorrect"
        
        index.index_file.seek(index.postings_dict[2][0])
        assert VBEPostings.decode(index.index_file.read(len(VBEPostings.encode([3,4,5])))) == [3,4,5], "decoding error"
        assert VBEPostings.decode_tf(index.index_file.read(len(VBEPostings.encode_tf([34,23,56])))) == [34,23,56], "decoding error"
