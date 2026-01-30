# ArXiv Daily Digest

A tool for fetching, ranking, and browsing arXiv papers with semantic search.
This simple project is fully written by Claude.

## Components

1. **`arxiv_digest.py`** - Fetches RSS feeds, ranks papers by embedding similarity
2. **`arxiv_viewer.py`** - Streamlit web app for browsing digests

## Installation

```bash
pip install -r requirements.txt
```

Make sure you have [Ollama](https://ollama.ai) installed and running for embeddings and TLDR generation.

```bash
# Pull embedding model
ollama pull nomic-embed-text

# Pull a model for TLDR generation
ollama pull llama3.2
```

## Usage

### 1. Fetch Daily Digest

```bash
python arxiv_digest.py \
    --categories cs.CL cs.IR \
    --output ./digests \
    --model nomic-embed-text \
    --query "retrieval augmented generation for question answering"
```

**Arguments:**
| Argument | Short | Description |
|----------|-------|-------------|
| `--categories` | `-c` | arXiv categories to fetch (space-separated) |
| `--output` | `-o` | Directory to save JSONL digest files |
| `--model` | `-m` | Ollama embedding model for ranking |
| `--query` | `-q` | Standing query to rank papers against |
| `--ollama-url` | | Ollama API URL (default: http://localhost:11434) |
| `--no-rank` | | Skip ranking, just fetch papers |

### 2. Browse Digests

```bash
streamlit run arxiv_viewer.py -- --digest-dir ./digests
```

**Arguments (pass after `--`):**
| Argument | Description |
|----------|-------------|
| `--digest-dir` | Directory containing digest JSONL files |
| `--ollama-model` | Model for TLDR generation (default: llama3.2) |
| `--ollama-url` | Ollama API URL (default: http://localhost:11434) |
| `--log-file` | Custom path for reading progress log |
| `--cache-file` | Custom path for TLDR cache |

## Features

### Digest Fetcher
- Fetches from arXiv RSS/Atom feeds (daily new papers)
- Ranks papers by cosine similarity to your standing query
- Deduplicates papers appearing in multiple categories
- Outputs sorted JSONL with metadata

### Viewer App
- 📅 Browse digests organized by date
- 📊 Track reading progress per digest
- 🤖 Generate one-liner TLDRs on-demand via Ollama
- 💾 Caches TLDRs to avoid regeneration
- 📄 Expandable abstracts
- 🔗 Direct links to arXiv

## Output Format

### Digest JSONL Structure
```jsonl
{"_metadata": true, "fetch_date": "...", "categories": [...], "standing_query": "...", "total_papers": N}
{"arxiv_id": "2501.12345", "title": "...", "authors": [...], "summary": "...", "similarity_score": 0.85, ...}
{"arxiv_id": "2501.12346", "title": "...", "authors": [...], "summary": "...", "similarity_score": 0.72, ...}
```

### Progress Log (reading_progress.json)
```json
{
  "arxiv_digest_cs.CL_cs.IR_2025-01-30.jsonl": {
    "last_page": 2,
    "last_viewed": "2025-01-30 14:30",
    "papers_seen": 30
  }
}
```

### TLDR Cache (tldr_cache.json)
```json
{
  "2501.12345_tldr": "This paper improves RAG by combining multiple retrieval strategies with rank fusion.",
  "2501.12346_tldr": "A new dense retrieval method achieves SOTA on QA benchmarks."
}
```

## Automation (Optional)

Set up a daily cron job to fetch new papers:

```bash
# Edit crontab
crontab -e

# Add line to fetch at 8 AM daily
0 8 * * * cd /path/to/project && python arxiv_digest.py -c cs.CL cs.IR -o ./digests -m nomic-embed-text -q "your query"
```

## Tips

- Run the fetcher after arXiv's daily update (4AM ET)
- Use specific standing queries for better ranking
- Popular embedding models: `qwen3-embedding-0.6b`
- For TLDRs: `llama3.2`, `mistral`, `phi3` work well
