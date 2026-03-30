# Information Retrieval System

An Information Retrieval (IR) system implementing various indexing, compression, and retrieval algorithms for document search.

## Key Features

### 1. Indexing Algorithms

**BSBI (Blocked Sort-Based Indexing)**

- Implements block-based indexing for large document collections
- Divides document collection into smaller blocks
- Performs sorting and merging to produce the final inverted index

**SPIMI (Single-Pass In-Memory Indexing)**

- Single-pass indexing algorithm with more efficient memory usage
- Does not require global sorting, directly writes intermediate index to disk

### 2. Data Structures

**Trie (Prefix Tree) Data Structure**

- Custom Trie implementation using `TrieNode` class for efficient string-to-integer mapping
- Each `TrieNode` contains:
  - `children`: Dictionary mapping characters to child TrieNodes
  - `value`: Integer ID associated with the string ending at this node
- Used in `IdMap` class for bidirectional mapping between terms/documents and their integer IDs
- Provides O(m) lookup time where m is the length of the string
- Supports dictionary-like operations: `__setitem__`, `__getitem__`, `__contains__`

**IdMap Class**

- Maintains bidirectional mapping between strings and integer IDs
- Forward mapping (string → id) uses Trie for memory efficiency
- Reverse mapping (id → string) uses Python list for O(1) access
- Used for both term-to-termID and document-to-docID mappings

### 3. Compression Algorithms

The system supports 3 compression algorithms for postings lists and term frequencies:

**StandardPostings**

- Simple encoding using Python array library
- No compression, stores integers directly as bytes
- Suitable for baseline comparison

**VBEPostings (Variable-Byte Encoding)**

- Uses gap encoding (delta encoding) before compression
- Variable-byte encoding for more efficient representation
- Uses high-bit flag to mark the last byte of each number
- **Example:** `[34, 67, 89, 454]` → gaps `[34, 33, 22, 365]` → compressed bytes

**EliasGammaPostings**

- Elias Gamma coding for more aggressive compression
- Uses gap encoding before compression
- Adds 3-bit header for tracking padding bits
- More efficient for postings lists with zipfian distribution

### 4. Retrieval Methods

**TF-IDF Scoring**

- Term Frequency - Inverse Document Frequency
- Measures term importance in a document relative to the collection

**BM25 Scoring**

- Okapi BM25 ranking function
- State-of-the-art probabilistic retrieval model
- Uses k1 and b parameters for tuning

**WAND (Weak AND) Algorithm**

- Retrieval optimization for BM25
- Dynamically prunes irrelevant documents
- Significantly faster for queries with many terms

### 5. Evaluation Metrics

The system provides evaluation using the following metrics:

- **RBP (Rank-Biased Precision)**: Measures precision with exponential decay for lower ranks (persistence parameter p = 0.8)
- **AP (Average Precision)**: Average of precision values at each rank where a relevant document is retrieved
- **DCG (Discounted Cumulative Gain)**: Measures gain with logarithmic discount based on rank position
- **nDCG (normalized DCG)**: DCG normalized by the ideal DCG, providing a score between 0 and 1

## Project Structure

```
.
├── main.py              # CLI entry point
├── core/
│   ├── base_index.py    # Base class for indexer
│   ├── bsbi.py          # BSBI indexing implementation
│   ├── spimi.py         # SPIMI indexing implementation
│   ├── index.py         # Inverted index writer/reader
│   ├── my_compression.py # Compression algorithms
│   ├── search.py        # Search functionality
│   └── evaluation.py    # Evaluation metrics (RBP, AP, DCG, nDCG)
├── collection/          # Document collection (folder containing subfolders with .txt files)
└── data/
    ├── queries.txt      # Queries for evaluation
    └── qrels.txt        # Relevance judgments
```

## How to Run

### Prerequisites

Ensure Python 3.7+ is installed and install dependencies:

```bash
pip install tqdm nltk
```

### 1. Build Index

**Using BSBI with VBE compression:**

```bash
python main.py index --method bsbi --compression vbe --clean
```

**Using SPIMI with Elias Gamma compression:**

```bash
python main.py index --method spimi --compression elias --clean
```

**Parameters:**

- `--method` (Optional): Determines the indexing algorithm to use (`bsbi` or `spimi`).
  - **Default:** `bsbi`
- `-c` / `--compression` (Optional): Defines the compression algorithm for postings lists and term frequencies (`standard`, `vbe`, or `elias`).
  - **Default:** `vbe`
- `--clean` (Optional Flag): If provided, automatically deletes the old index and temporary directories before building the new one. If omitted, new indexing data might append to existing structures.
- `--data-dir` (Optional): Path to the folder containing the document collection.
  - **Default:** `collection`
- `--output-dir` (Optional): Path to the directory where the resulting index files will be saved.
  - **Default:** `index`

### 2. Search

**Search with interactive query:**

```bash
python main.py search "the crystalline lens in vertebrates"
```

**Search with BM25+WAND (default):**

```bash
python main.py search "the crystalline lens" -k 10
```

**Search with different modes:**

```bash
python main.py search "the crystalline lens" -m tfidf -k 10
python main.py search "the crystalline lens" -m bm25 -k 10
python main.py search "the crystalline lens" -m wand -k 10
python main.py search "the crystalline lens" -m all -k 10
```

**Parameters:**

- `query` (Optional): The search query text (wrap in quotes if it contains spaces). If omitted, the system will execute 3 predefined default queries.
- `-k` (Optional): The total number of Top-K relevant documents to retrieve.
  - **Default:** `10`
- `-m` / `--mode` (Optional): Selects the scoring/retrieval algorithm. Options include `tfidf`, `bm25`, `wand` (which uses the WAND algorithm with BM25 scoring), or `all` to run all three sequentially and compare their results.
  - **Default:** `wand`
- `--data-dir` (Optional): Path to the dataset directory. Needed to map the exact document names when displaying search results.
  - **Default:** `collection`
- `--output-dir` (Optional): Path to the index files directory previously generated by the `index` command.
  - **Default:** `index`

### 3. Evaluate

**Evaluate IR system performance:**

```bash
python main.py evaluate
```

Or:

```bash
python main.py eval
```

This command will evaluate the system against queries in `data/queries.txt` and relevance judgments in `data/qrels.txt`. The evaluation reports RBP, AP, DCG, and nDCG scores for all three retrieval methods (TF-IDF, BM25, and BM25+WAND).

## Complete Usage Example

```bash
# 1. Build index with BSBI + Elias Gamma compression
python main.py index --method bsbi --compression elias --clean

# 2. Search with query
python main.py search "alkylated with radioactive iodoacetate" -k 5 -m wand

# 3. Evaluate results
python main.py evaluate
```

## Testing Compression

To test compression algorithms separately:

```bash
python core/my_compression.py
```

Output will display size comparison between Standard, VBE, and Elias Gamma encoding.

## Data Format

**Collection Directory:**

- Main folder containing subfolders (each subfolder = 1 block)
- Each subfolder contains `.txt` files (each file = 1 document)

**Queries File (`data/queries.txt`):**

```
Q1 query text here
Q2 another query text
...
```

**Qrels File (`data/qrels.txt`):**

```
Q1 doc1.txt doc2.txt doc3.txt
Q2 doc4.txt doc5.txt
...
```

## Compression Performance

Based on testing with sample postings list `[34, 67, 89, 454, 2345738]` and frequencies `[1, 2, 3, 4, 5]`:

| Algorithm   | Postings Size | TF List Size |
| ----------- | ------------- | ------------ |
| Standard    | 40 bytes      | 40 bytes     |
| VBE         | 9 bytes       | 5 bytes      |
| Elias Gamma | 12 bytes      | 3 bytes      |

**Conclusion:** Elias Gamma and VBE provide significantly better compression ratios compared to Standard encoding.

## Evaluation Output Example

When running evaluation, the system outputs metrics for each retrieval method:

```
TF-IDF evaluation results over 30 queries
RBP score = 0.6512393244792557
AP score = 0.5460607844389903
DCG score = 5.83686170539342
nDCG score = 0.815524318839492

BM25 evaluation results over 30 queries
RBP score = 0.6782936738676539
AP score = 0.5787318563254875
DCG score = 5.958719619318515
nDCG score = 0.8331549788470489

BM25 + WAND evaluation results over 30 queries
RBP score = 0.6782936743457858
AP score = 0.5787373448926625
DCG score = 5.958733210332523
nDCG score = 0.8331578604823037
```

## Important Notes

1. Ensure the `collection/` folder contains the correct data structure (subfolders with .txt files)
2. Indexing will create `index/` and `tmp/` folders in the working directory
3. Use `--clean` to start fresh indexing (deletes old index)
4. WAND mode requires an index built with max_tf metadata
