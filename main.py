import argparse
import os
import sys
from base_index import BaseIndex
from bsbi import BSBIIndex
from spimi import SPIMIIndex
from my_compression import StandardPostings, VBEPostings, EliasGammaPostings
from evaluation import eval as eval_metrics, load_qrels

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

def clear_index_directory(output_dir="index", tmp_dir="tmp"):
    """Clean up existing directories if the user requests a fresh indexing process"""
    import shutil
    for d in [output_dir, tmp_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

def cmd_index(args):
    encoding_map = {
        "standard": StandardPostings,
        "vbe": VBEPostings,
        "elias": EliasGammaPostings
    }
    postings_encoding = encoding_map[args.compression]

    if args.clean:
        clear_index_directory(args.output_dir)

    print(f"[!] Info: Initialize indexing with {args.method.upper()} and {args.compression.upper()} compression...")

    if args.method == "bsbi":
        indexer = BSBIIndex(data_dir=args.data_dir, postings_encoding=postings_encoding, output_dir=args.output_dir)
    elif args.method == "spimi":
        indexer = SPIMIIndex(data_dir=args.data_dir, postings_encoding=postings_encoding, output_dir=args.output_dir)
    
    indexer.index()
    print("[✓] Indexing completed successfully.")

def cmd_search(args):
    searcher = BaseIndex(data_dir=args.data_dir, postings_encoding=None, output_dir=args.output_dir)

    if args.query:
        queries = [args.query]
    else:
        queries = [
            "alkylated with radioactive iodoacetate",
            "psychodrama for disturbed children",
            "lipid metabolism in toxemia and normal pregnancy"
        ]

    for query in queries:
        print(f"\nQuery  :  {query}")
        
        if args.mode in ["tfidf", "all"]:
            print("Results (TF-IDF):")
            for (score, doc) in searcher.retrieve_tfidf(query, k=args.k):
                print(f"{doc:30} {score:>.3f}")
            print()
            
        if args.mode in ["bm25", "all"]:
            print("Results (BM25):")
            for (score, doc) in searcher.retrieve_bm25(query, k=args.k):
                print(f"{doc:30} {score:>.3f}")
            print()
            
        if args.mode in ["wand", "all"]:
            print("Results (BM25 + WAND):")
            for (score, doc) in searcher.retrieve_bm25_wand(query, k=args.k):
                print(f"{doc:30} {score:>.3f}")
            print()

def cmd_evaluate(args):
    qrels = load_qrels("data/qrels.txt")
    eval_metrics(qrels=qrels, query_file="data/queries.txt", k=1000, postings_encoding=None)

def main():
    parser = argparse.ArgumentParser(
        description="Information Retrieval System CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Select action to perform", required=True)

    # 1. COMMAND: index
    parser_index = subparsers.add_parser("index", help="Build an inverted-index structure from document collection")
    parser_index.add_argument("--method", type=str, choices=["bsbi", "spimi"], default="bsbi",
                              help="Indexing methodology (BSBI or SPIMI)")
    parser_index.add_argument("-c", "--compression", type=str, choices=["standard", "vbe", "elias"], default="vbe",
                              help="Compression algorithm for postings and tf lists")
    parser_index.add_argument("--data-dir", type=str, default="collection", help="Folder containing the text collection")
    parser_index.add_argument("--output-dir", type=str, default="index", help="Folder to output the index files")
    parser_index.add_argument("--clean", action="store_true", help="Automatically delete old index folder before indexing")

    # 2. COMMAND: search
    parser_search = subparsers.add_parser("search", help="Find Top-K relevant documents for a query")
    parser_search.add_argument("query", type=str, nargs="?", default="", help="Interactive search query text")
    parser_search.add_argument("-k", type=int, default=10, help="Retrieve Top-K highest rank values")
    parser_search.add_argument("-m", "--mode", type=str, choices=["tfidf", "bm25", "wand", "all"], default="wand", 
                               help="Search scoring mode. Defaults to BM25+WAND")
    parser_search.add_argument("--data-dir", type=str, default="collection", help="Collection dataset folder")
    parser_search.add_argument("--output-dir", type=str, default="index", help="Index output folder location")

    # 3. COMMAND: evaluate
    subparsers.add_parser("evaluate", aliases=['eval'], help="Evaluate IR model precision and rank metrics against qrels")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command in ("evaluate", "eval"):
        cmd_evaluate(args)

if __name__ == "__main__":
    main()
