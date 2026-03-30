import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

stemmer = SnowballStemmer("english")

def preprocess_text(text):
    """
    Cleans text (lowercases), removes stopwords, and performs stemming
    using NLTK's SnowballStemmer.
    """
    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)
    stems = [stemmer.stem(t) for t in tokens if t not in stop_words]
    return stems

class TrieNode:
    """
    A node within the Trie data structure.

    Attributes
    ----------
    children : dict
        A dictionary mapping a single character to its corresponding TrieNode child.
    value : int or None
        The integer value associated with the string that ends at this node.
        If None, the node simply represents an intermediate prefix character.
    """
    def __init__(self):
        self.children = {}
        self.value = None

class Trie:
    """
    A Prefix Tree (Trie) used to efficiently map string terms to integers.
    Supports dictionary-like operations: __contains__, __setitem__, __getitem__.
    """
    def __init__(self):
        """Initializes the Trie with an empty root node."""
        self.root = TrieNode()

    def __setitem__(self, key, value):
        """
        Inserts a new string key and maps it to an integer value.

        Parameters
        ----------
        key : str
            The term or word being inserted.
        value : int
            The unique identifier corresponding to the term.
        """
        node = self.root
        for char in key:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.value = value

    def __getitem__(self, key):
        """
        Retrieves the integer value mapped to a given string key.

        Parameters
        ----------
        key : str
            The term whose mapped integer value is required.

        Returns
        -------
        int
            The integer identifier mapped to the term.

        Raises
        ------
        KeyError
            If the key does not exist within the Trie.
        """
        node = self.root
        for char in key:
            if char not in node.children:
                raise KeyError(key)
            node = node.children[char]
        if node.value is None:
            raise KeyError(key)
        return node.value

    def __contains__(self, key):
        """
        Checks whether a given key currently exists within the Trie.

        Parameters
        ----------
        key : str
            The term to dynamically check.

        Returns
        -------
        bool
            True if the term exists and holds a value, False otherwise.
        """
        node = self.root
        for char in key:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.value is not None


class IdMap:
    """
    Maintains a bidirectional mapping between string terms (or document
    names) and their corresponding integer IDs. In practice, both
    documents and terms are represented as integers for efficiency.
    """

    def __init__(self):
        """
        The forward mapping (string -> id) is stored in a Python dictionary
        for efficient lookup. The reverse mapping (id -> string) is stored
        in a Python list.

        Example:
            str_to_id["hello"] ---> 8
            str_to_id["/collection/dir0/gamma.txt"] ---> 54

            id_to_str[8] ---> "hello"
            id_to_str[54] ---> "/collection/dir0/gamma.txt"
        """
        self.str_to_id = Trie()
        self.id_to_str = []

    def __len__(self):
        """Returns the number of terms (or documents) stored in the IdMap."""
        return len(self.id_to_str)

    def __get_str(self, i):
        """Returns the string associated with the given index i."""
        return self.id_to_str[i]

    def __get_id(self, s):
        """
        Returns the integer id corresponding to the given string s.
        If s is not yet in the IdMap, assigns a new integer id and returns it.
        """
        if s not in self.str_to_id:
            self.id_to_str.append(s)
            self.str_to_id[s] = len(self.id_to_str) - 1
        return self.str_to_id[s]

    def __getitem__(self, key):
        """
        __getitem__(...) is a Python special method that allows a collection
        class (like IdMap) to support element access using the [..] syntax,
        similar to lists and dictionaries.

        See: https://stackoverflow.com/questions/43627405/understanding-getitem-method

        If key is an integer, delegates to __get_str;
        if key is a string, delegates to __get_id.
        """
        if type(key) is int:
            return self.__get_str(key)
        elif type(key) is str:
            return self.__get_id(key)
        else:
            raise TypeError

def sorted_merge_posts_and_tfs(posts_tfs1, posts_tfs2):
    """
    Merges two sorted lists of (doc_id, tf) tuples and returns the merged
    result. Term frequencies are accumulated for tuples sharing the same
    doc_id.

    Example:
        posts_tfs1 = [(1, 34), (3, 2), (4, 23)]
        posts_tfs2 = [(1, 11), (2, 4), (4, 3 ), (6, 13)]

        return   [(1, 34+11), (2, 4), (3, 2), (4, 23+3), (6, 13)]
               = [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)]

    Parameters
    ----------
    list1: List[(Comparable, int)]
    list2: List[(Comparable, int)]
        Two sorted lists of tuples to merge.

    Returns
    -------
    List[(Comparable, int)]
        The merged sorted list of tuples.
    """
    i, j = 0, 0
    merge = []
    while (i < len(posts_tfs1)) and (j < len(posts_tfs2)):
        if posts_tfs1[i][0] == posts_tfs2[j][0]:
            freq = posts_tfs1[i][1] + posts_tfs2[j][1]
            merge.append((posts_tfs1[i][0], freq))
            i += 1
            j += 1
        elif posts_tfs1[i][0] < posts_tfs2[j][0]:
            merge.append(posts_tfs1[i])
            i += 1
        else:
            merge.append(posts_tfs2[j])
            j += 1
    while i < len(posts_tfs1):
        merge.append(posts_tfs1[i])
        i += 1
    while j < len(posts_tfs2):
        merge.append(posts_tfs2[j])
        j += 1
    return merge

def test(output, expected):
    """ simple function for testing """
    return "PASSED" if output == expected else "FAILED"

if __name__ == '__main__':

    doc = ["halo", "semua", "selamat", "pagi", "semua"]
    term_id_map = IdMap()
    assert [term_id_map[term] for term in doc] == [0, 1, 2, 3, 1], "term_id incorrect"
    assert term_id_map[1] == "seimport sysmua", "term_id incorrect"
    assert term_id_map[0] == "halo", "term_id incorrect"
    assert term_id_map["selamat"] == 2, "term_id incorrect"
    assert term_id_map["pagi"] == 3, "term_id incorrect"

    docs = ["/collection/0/data0.txt",
            "/collection/0/data10.txt",
            "/collection/1/data53.txt"]
    doc_id_map = IdMap()
    assert [doc_id_map[docname] for docname in docs] == [0, 1, 2], "docs_id incorrect"

    assert sorted_merge_posts_and_tfs([(1, 34), (3, 2), (4, 23)], \
                                      [(1, 11), (2, 4), (4, 3 ), (6, 13)]) == [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)], "sorted_merge_posts_and_tfs incorrect"
