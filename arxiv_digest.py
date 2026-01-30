#!/usr/bin/env python3
"""
ArXiv Daily Digest Fetcher and Ranker

Fetches RSS feeds for specified arXiv categories, ranks papers by semantic
similarity to a standing query using Ollama embeddings, and saves results
to a JSONL file.

Usage:
    python arxiv_digest.py --categories cs.CL cs.IR --output ./digests \
        --model nomic-embed-text --query "retrieval augmented generation for question answering"
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import feedparser
import requests
import numpy as np


def get_embedding(text: str, model: str, ollama_url: str = "http://localhost:11434") -> Optional[list[float]]:
    """Get embedding for text using Ollama API."""
    try:
        response = requests.post(
            f"{ollama_url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=60
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except requests.exceptions.RequestException as e:
        print(f"Error getting embedding: {e}", file=sys.stderr)
        return None


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def fetch_arxiv_rss(category: str) -> list[dict]:
    """Fetch and parse arXiv RSS feed for a category."""
    url = f"https://rss.arxiv.org/atom/{category}"
    print(f"Fetching {url}...")
    
    feed = feedparser.parse(url)
    
    if feed.bozo and feed.bozo_exception:
        print(f"Warning: Feed parsing issue for {category}: {feed.bozo_exception}", file=sys.stderr)
    
    papers = []
    for entry in feed.entries:
        # Extract arXiv ID from the link
        arxiv_id = entry.link.split("/abs/")[-1] if "/abs/" in entry.link else entry.id
        
        # Parse authors - feedparser puts them in 'authors' list
        authors = []
        if hasattr(entry, 'authors'):
            authors = [author.get('name', '') for author in entry.authors]
        elif hasattr(entry, 'author'):
            authors = [entry.author]
        
        # Get categories/tags
        categories = []
        if hasattr(entry, 'tags'):
            categories = [tag.term for tag in entry.tags]
        
        # Clean up summary (remove HTML tags if present)
        summary = entry.summary if hasattr(entry, 'summary') else ""
        # Basic HTML cleanup
        summary = summary.replace("<p>", "").replace("</p>", "\n").strip()
        
        paper = {
            "arxiv_id": arxiv_id,
            "title": entry.title.replace("\n", " ").strip(),
            "authors": authors,
            "summary": summary,
            "link": entry.link,
            "published": entry.published if hasattr(entry, 'published') else "",
            "updated": entry.updated if hasattr(entry, 'updated') else "",
            "categories": categories,
            "primary_category": category,  # The category we fetched from
        }
        papers.append(paper)
    
    print(f"  Found {len(papers)} papers in {category}")
    return papers


def deduplicate_papers(papers: list[dict]) -> list[dict]:
    """Remove duplicate papers (same arxiv_id), keeping first occurrence."""
    seen = set()
    unique = []
    for paper in papers:
        if paper["arxiv_id"] not in seen:
            seen.add(paper["arxiv_id"])
            unique.append(paper)
    return unique


def rank_papers(
    papers: list[dict],
    query: str,
    model: str,
    ollama_url: str = "http://localhost:11434"
) -> list[dict]:
    """Rank papers by similarity to query using embeddings."""
    
    print(f"\nComputing embeddings using model '{model}'...")
    
    # Get query embedding
    print("  Embedding query...")
    query_embedding = get_embedding(query, model, ollama_url)
    if query_embedding is None:
        print("Error: Could not get query embedding. Returning unranked papers.", file=sys.stderr)
        for paper in papers:
            paper["similarity_score"] = 0.0
        return papers
    
    # Get embeddings for each paper (title + summary)
    print(f"  Embedding {len(papers)} papers...")
    for i, paper in enumerate(papers):
        # Combine title and summary for embedding
        text = f"{paper['title']}\n\n{paper['summary']}"
        paper_embedding = get_embedding(text, model, ollama_url)
        
        if paper_embedding is not None:
            paper["similarity_score"] = cosine_similarity(query_embedding, paper_embedding)
        else:
            paper["similarity_score"] = 0.0
        
        # Progress indicator
        if (i + 1) % 10 == 0 or i == len(papers) - 1:
            print(f"    Processed {i + 1}/{len(papers)} papers")
    
    # Sort by similarity score (descending)
    papers.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    return papers


def save_papers(
    papers: list[dict],
    output_dir: Path,
    categories: list[str],
    query: str
) -> Path:
    """Save papers to a JSONL file."""
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with categories and date
    date_str = datetime.now().strftime("%Y-%m-%d")
    categories_str = "_".join(sorted(categories))
    filename = f"arxiv_digest_{categories_str}_{date_str}.jsonl"
    filepath = output_dir / filename
    
    # Create metadata header
    metadata = {
        "_metadata": True,
        "fetch_date": datetime.now().isoformat(),
        "categories": categories,
        "standing_query": query,
        "total_papers": len(papers)
    }
    
    # Write to JSONL
    with open(filepath, 'w', encoding='utf-8') as f:
        # Write metadata as first line
        f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        
        # Write each paper
        for paper in papers:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")
    
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Fetch arXiv RSS feeds, rank papers by query similarity, and save to JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --categories cs.CL cs.IR --output ./digests --model nomic-embed-text --query "large language models"
  %(prog)s -c cs.AI -o ./output -m mxbai-embed-large -q "reinforcement learning from human feedback"
        """
    )
    
    parser.add_argument(
        "-c", "--categories",
        nargs="+",
        required=True,
        help="List of arXiv categories (e.g., cs.CL cs.IR cs.AI)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Directory to save the digest JSONL file"
    )
    
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Ollama embedding model name (e.g., nomic-embed-text, mxbai-embed-large)"
    )
    
    parser.add_argument(
        "-q", "--query",
        required=True,
        help="Standing query to rank papers against"
    )
    
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)"
    )
    
    parser.add_argument(
        "--no-rank",
        action="store_true",
        help="Skip ranking (just fetch and save)"
    )
    
    args = parser.parse_args()
    
    # Fetch papers from all categories
    print(f"Fetching arXiv RSS feeds for categories: {', '.join(args.categories)}\n")
    all_papers = []
    for category in args.categories:
        papers = fetch_arxiv_rss(category)
        all_papers.extend(papers)
    
    if not all_papers:
        print("No papers found. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    # Deduplicate (papers can appear in multiple categories)
    all_papers = deduplicate_papers(all_papers)
    print(f"\nTotal unique papers: {len(all_papers)}")
    
    # Rank papers by query similarity
    if not args.no_rank:
        all_papers = rank_papers(all_papers, args.query, args.model, args.ollama_url)
    else:
        print("\nSkipping ranking (--no-rank specified)")
        for paper in all_papers:
            paper["similarity_score"] = 0.0
    
    # Save to JSONL
    filepath = save_papers(all_papers, args.output, args.categories, args.query)
    print(f"\nSaved {len(all_papers)} papers to: {filepath}")
    
    # Print top 5 papers
    print("\n" + "="*60)
    print("TOP 5 PAPERS BY RELEVANCE")
    print("="*60)
    for i, paper in enumerate(all_papers[:5], 1):
        print(f"\n{i}. [{paper['similarity_score']:.4f}] {paper['title']}")
        print(f"   arXiv: {paper['arxiv_id']}")
        print(f"   Authors: {', '.join(paper['authors'][:3])}" + 
              (" et al." if len(paper['authors']) > 3 else ""))


if __name__ == "__main__":
    main()
