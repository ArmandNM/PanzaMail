import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List, Text

import torch
from langchain_core.documents import Document
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from panza.utils import prompting, rag

sys.path.pop(0)

RAG_RELEVANCE_THRESHOLD = 0.2


SYSTEM_PREAMBLE_PATH = "/nfs/scistore19/alistgrp/anicolic/repos/PanzaMailFork/prompt_preambles/system_preamble.txt"
USER_PREAMBLE_PATH = "/nfs/scistore19/alistgrp/anicolic/repos/PanzaMailFork/prompt_preambles/user_preamble.txt"
RAG_PREAMBLE_PATH = "/nfs/scistore19/alistgrp/anicolic/repos/PanzaMailFork/prompt_preambles/rag_preamble.txt"

SYSTEM_PREAMBLE = prompting.load_preamble(SYSTEM_PREAMBLE_PATH)
USER_PREAMBLE = prompting.load_user_preamble(USER_PREAMBLE_PATH)
RAG_PREAMBLE = prompting.load_preamble(RAG_PREAMBLE_PATH)


def determine_context_length(batch, db, num_emails, tokenizer):
    lengths = []
    for email in batch:
        try:
            relevant_emails = db._similarity_search_with_relevance_scores(
                email["email"], k=num_emails
            )
        except Exception as e:
            print(f"Error in RAG search: {e}")
            relevant_emails = []
            return relevant_emails

        relevant_emails = [
            {"email": r[0].page_content, "score": r[1]}
            for r in relevant_emails
            if r[0].page_content not in email["email"]
        ]

        relevant_emails = [
            r["email"] for r in relevant_emails if r["score"] >= RAG_RELEVANCE_THRESHOLD
        ]
        relevant_emails = [
            Document(page_content=email, metadata={}) for email in relevant_emails
        ]
        relevant_emails = relevant_emails[:num_emails]

        prompt_with_rag = prompting.create_prompt(
            email["email"],
            SYSTEM_PREAMBLE,
            USER_PREAMBLE,
            RAG_PREAMBLE,
            relevant_emails,
        )
        prompt_without_rag = prompting.create_prompt(
            email["email"], SYSTEM_PREAMBLE, USER_PREAMBLE
        )
        current_lengths = []
        current_lengths.append(len(prompt_without_rag))
        current_lengths.append(len(prompt_with_rag))
        print("Characters length without RAG:", len(prompt_without_rag))
        print("Characters length with RAG:", len(prompt_with_rag))

        prompt_with_rag = tokenizer.encode(prompt_with_rag, return_tensors="pt")
        prompt_without_rag = tokenizer.encode(prompt_without_rag, return_tensors="pt")
        current_lengths.append(prompt_without_rag.size()[1])
        current_lengths.append(prompt_with_rag.size()[1])
        print("Tokens length without RAG:", prompt_without_rag.size()[1])
        print("Tokens length with RAG:", prompt_with_rag.size()[1])
        print()

        lengths.append(current_lengths)

    return lengths


def main():
    parser = argparse.ArgumentParser(
        description="Get similar emails for Retrieval Augmented Fine Tuning (RAFT)"
    )
    parser.add_argument("--path-to-emails", help="Path to the cleaned emails")
    parser.add_argument(
        "--embedding-model", type=str, default="sentence-transformers/all-mpnet-base-v2"
    )
    parser.add_argument(
        "--generative-model",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
    )
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--index-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rag-num-emails", type=int, default=7)
    args = parser.parse_args()

    assert args.path_to_emails.endswith(
        ".jsonl"
    ), f"Expecting a .jsonl file, but given = {args.path_to_emails}"

    print(f"--> Reading emails from: {args.path_to_emails}")

    # Read emails
    with open(args.path_to_emails, "r") as f:
        lines = f.readlines()
        json_lines = [json.loads(line.strip(",")) for line in lines]
        print(f"--> # emails = {len(json_lines)}")

    embeddings_model = rag.get_embeddings_model(args.embedding_model)
    db = rag.load_vector_db_from_disk(args.db_path, args.index_name, embeddings_model)

    tokenizer = AutoTokenizer.from_pretrained(args.generative_model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    start_time = time.time()

    lengths = []
    for i in tqdm(range(0, len(json_lines), args.batch_size)):
        # TODO(armand): Fix this print for batched inference
        print(f"--> Processing batch {i}/{len(json_lines)}")
        batch = json_lines[i : i + args.batch_size]
        lengths += determine_context_length(batch, db, args.rag_num_emails, tokenizer)

    print(lengths)

    elapsed_time = time.time() - start_time
    print(f"{elapsed_time:.2f} seconds to process {len(json_lines)} emails.")


if __name__ == "__main__":
    main()
