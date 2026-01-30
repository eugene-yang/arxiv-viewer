#!/usr/bin/env python3
"""
ArXiv Digest Viewer - Two Panel Layout with Keyboard Navigation

Usage:
    streamlit run arxiv_viewer.py -- --digest-dir ./digests
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import streamlit as st
from streamlit_shortcuts import add_shortcuts

# --- Configuration ---
DEFAULT_DIGEST_DIR = "./digests"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"
PAPERS_PER_PAGE = 10


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--digest-dir", type=Path, default=DEFAULT_DIGEST_DIR)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--cache-file", type=Path, default=None)
    
    args, _ = parser.parse_known_args()
    
    if args.log_file is None:
        args.log_file = args.digest_dir / "reading_progress.json"
    if args.cache_file is None:
        args.cache_file = args.digest_dir / "tldr_cache.json"
    
    return args


def load_progress(log_file: Path) -> dict:
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_progress(log_file: Path, progress: dict):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, 'w') as f:
        json.dump(progress, f, indent=2)


def load_tldr_cache(cache_file: Path) -> dict:
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_tldr_cache(cache_file: Path, cache: dict):
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def generate_tldr(title: str, summary: str, model: str, ollama_url: str) -> Optional[str]:
    prompt = f"""Write a single concise sentence (max 30 words) summarizing this paper. Just output the sentence, nothing else.

Title: {title}

Abstract: {summary}

One-sentence TLDR:"""

    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 100}
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()["response"].strip()
        for prefix in ["TLDR:", "TL;DR:", "Summary:", "Here's", "Here is"]:
            if result.lower().startswith(prefix.lower()):
                result = result[len(prefix):].strip()
        return result
    except requests.exceptions.RequestException as e:
        return f"Error generating TLDR: {e}"


def list_digest_files(digest_dir: Path) -> list[dict]:
    digests = []
    if not digest_dir.exists():
        return digests
    
    for filepath in sorted(digest_dir.glob("arxiv_digest_*.jsonl"), reverse=True):
        parts = filepath.stem.replace("arxiv_digest_", "").rsplit("_", 3)
        if len(parts) >= 3:
            date_str = f"{parts[-3]}-{parts[-2]}-{parts[-1]}"
            categories = "_".join(parts[:-3]) if len(parts) > 3 else parts[0]
        else:
            date_str = "unknown"
            categories = filepath.stem
        
        try:
            with open(filepath, 'r') as f:
                first_line = f.readline()
                metadata = json.loads(first_line)
                if metadata.get("_metadata"):
                    total_papers = metadata.get("total_papers", 0)
                    query = metadata.get("standing_query", "")
                    categories = ", ".join(metadata.get("categories", [categories]))
                else:
                    total_papers = sum(1 for _ in f) + 1
                    query = ""
        except (json.JSONDecodeError, IOError):
            total_papers = 0
            query = ""
        
        digests.append({
            "filepath": filepath,
            "filename": filepath.name,
            "date": date_str,
            "categories": categories,
            "total_papers": total_papers,
            "query": query,
        })
    
    return digests


def load_digest(filepath: Path) -> tuple[dict, list[dict]]:
    metadata = {}
    papers = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data.get("_metadata"):
                metadata = data
            else:
                papers.append(data)
    return metadata, papers


def nav_previous_paper():
    """Navigate to previous paper."""
    if st.session_state.selected_paper_idx is None:
        st.session_state.selected_paper_idx = st.session_state.current_page * PAPERS_PER_PAGE
    elif st.session_state.selected_paper_idx > 0:
        st.session_state.selected_paper_idx -= 1
        if st.session_state.selected_paper_idx < st.session_state.current_page * PAPERS_PER_PAGE:
            st.session_state.current_page -= 1


def nav_next_paper(total_papers: int):
    """Navigate to next paper."""
    if st.session_state.selected_paper_idx is None:
        st.session_state.selected_paper_idx = st.session_state.current_page * PAPERS_PER_PAGE
    elif st.session_state.selected_paper_idx < total_papers - 1:
        st.session_state.selected_paper_idx += 1
        if st.session_state.selected_paper_idx >= (st.session_state.current_page + 1) * PAPERS_PER_PAGE:
            st.session_state.current_page += 1


def nav_previous_page():
    """Navigate to previous page."""
    if st.session_state.current_page > 0:
        st.session_state.current_page -= 1
        st.session_state.selected_paper_idx = None


def nav_next_page(total_pages: int):
    """Navigate to next page."""
    if st.session_state.current_page < total_pages - 1:
        st.session_state.current_page += 1
        st.session_state.selected_paper_idx = None


def main():
    args = get_args()
    
    st.set_page_config(
        page_title="ArXiv Digest Viewer",
        page_icon="📚",
        layout="wide",
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .block-container {
        padding-top: 3rem;
        padding-bottom: 1rem;
    }
    
    div[data-testid="stButton"] > button {
        text-align: left !important;
        padding: 8px 12px !important;
        font-size: 13px !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        height: auto !important;
        min-height: 40px !important;
        line-height: 1.4 !important;
    }
    
    div[data-testid="stButton"] > button > div {
        text-align: left !important;
        justify-content: flex-start !important;
    }
    
    div[data-testid="stButton"] > button > div > p {
        text-align: left !important;
    }
    
    /* Style for navigation buttons row */
    .nav-buttons {
        display: flex;
        gap: 5px;
        margin-bottom: 10px;
    }
    .nav-buttons button {
        font-size: 12px !important;
        padding: 4px 8px !important;
        min-height: 30px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "tldr_cache" not in st.session_state:
        st.session_state.tldr_cache = load_tldr_cache(args.cache_file)
    if "progress" not in st.session_state:
        st.session_state.progress = load_progress(args.log_file)
    if "selected_paper_idx" not in st.session_state:
        st.session_state.selected_paper_idx = None
    if "current_page" not in st.session_state:
        st.session_state.current_page = 0
    
    # We need to load digest info first to know total papers/pages for navigation
    # So we'll add shortcuts after loading the digest
    
    # Sidebar
    with st.sidebar:
        st.header("📚 ArXiv Digest")
        
        digests = list_digest_files(args.digest_dir)
        
        if not digests:
            st.warning(f"No digests in `{args.digest_dir}`")
            st.stop()
        
        digest_options = {f"{d['date']} ({d['total_papers']})": d for d in digests}
        selected_label = st.selectbox("📅 Select Date:", options=list(digest_options.keys()))
        
        selected_digest = digest_options[selected_label]
        digest_key = selected_digest["filename"]
        
        metadata, papers = load_digest(selected_digest["filepath"])
        
        current_progress = st.session_state.progress.get(digest_key, {
            "last_page": 0,
            "last_viewed": None,
            "papers_seen": 0,
        })
        
        if "current_digest" not in st.session_state or st.session_state.current_digest != digest_key:
            st.session_state.current_digest = digest_key
            st.session_state.current_page = current_progress["last_page"]
            st.session_state.selected_paper_idx = None
        
        total_pages = (len(papers) + PAPERS_PER_PAGE - 1) // PAPERS_PER_PAGE
        
        # Register keyboard shortcuts for navigation buttons
        # add_shortcuts maps widget_key=keyboard_shortcut
        add_shortcuts(
            nav_up=["arrowup", "k"],
            nav_down=["arrowdown", "j"],
            nav_left=["arrowleft", "h"],
            nav_right=["arrowright", "l"],
        )
        
        st.markdown("---")
        st.subheader("📊 Progress")
        progress_pct = min(100, (current_progress["papers_seen"] / len(papers) * 100)) if papers else 0
        st.progress(progress_pct / 100)
        st.caption(f"{current_progress['papers_seen']}/{len(papers)} papers seen")
        
        st.markdown("---")
        st.subheader("⚙️ Settings")
        ollama_model = st.text_input("TLDR Model", value=args.ollama_model)
        st.caption(f"Categories: {selected_digest['categories']}")
        
        # Keyboard navigation buttons (compact) + hints
        st.markdown("---")
        st.subheader("⌨️ Navigation")
        
        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            if st.button("⬆️ Prev Paper", key="nav_up", use_container_width=True):
                nav_previous_paper()
                st.rerun()
            if st.button("⬅️ Prev Page", key="nav_left", use_container_width=True):
                nav_previous_page()
                st.rerun()
        with nav_col2:
            if st.button("⬇️ Next Paper", key="nav_down", use_container_width=True):
                nav_next_paper(len(papers))
                st.rerun()
            if st.button("➡️ Next Page", key="nav_right", use_container_width=True):
                nav_next_page(total_pages)
                st.rerun()
        
        st.caption("**Keyboard:** ↑↓ or j/k = papers, ←→ or h/l = pages")
    
    # Main content
    col_list, col_detail = st.columns([2, 3])
    
    with col_list:
        if metadata.get("standing_query"):
            st.info(f"🔍 **Query:** {metadata['standing_query']}")
        
        st.markdown(f"**{len(papers)} papers** • Page {st.session_state.current_page + 1}/{total_pages}")
        
        start_idx = st.session_state.current_page * PAPERS_PER_PAGE
        end_idx = min(start_idx + PAPERS_PER_PAGE, len(papers))
        page_papers = papers[start_idx:end_idx]
        
        for i, paper in enumerate(page_papers):
            global_idx = start_idx + i
            rank = global_idx + 1
            score = paper.get("similarity_score", 0)
            title = paper["title"]
            
            is_selected = st.session_state.selected_paper_idx == global_idx
            btn_label = f"#{rank} [{score:.2f}] {title}"
            
            if st.button(
                btn_label,
                key=f"paper_{global_idx}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.selected_paper_idx = global_idx
                st.rerun()
        
        # Pagination
        st.markdown("---")
        col_prev, col_page_info, col_next = st.columns([1, 2, 1])
        
        with col_prev:
            if st.button("⬅️ Prev", disabled=st.session_state.current_page == 0, use_container_width=True):
                nav_previous_page()
                st.rerun()
        
        with col_page_info:
            st.markdown(f"<center>Page {st.session_state.current_page + 1} / {total_pages}</center>", unsafe_allow_html=True)
        
        with col_next:
            if st.button("Next ➡️", disabled=st.session_state.current_page >= total_pages - 1, use_container_width=True):
                nav_next_page(total_pages)
                st.rerun()
    
    with col_detail:
        if st.session_state.selected_paper_idx is not None:
            paper = papers[st.session_state.selected_paper_idx]
            rank = st.session_state.selected_paper_idx + 1
            score = paper.get("similarity_score", 0)
            
            st.markdown(f"### #{rank} — Score: {score:.3f}")
            st.markdown(f"## [{paper['title']}]({paper['link']})")
            
            col_id, col_cats = st.columns(2)
            with col_id:
                st.markdown(f"**arXiv:** `{paper.get('arxiv_id', 'N/A')}`")
            with col_cats:
                cats = ", ".join(paper.get("categories", []))
                st.markdown(f"**Categories:** {cats}")
            
            authors = paper.get("authors", [])
            if authors:
                if len(authors) > 8:
                    author_str = ", ".join(authors[:8]) + f" *et al.* ({len(authors)} total)"
                else:
                    author_str = ", ".join(authors)
                st.markdown(f"**Authors:** {author_str}")
            
            st.markdown("---")
            
            arxiv_id = paper.get("arxiv_id", "")
            tldr_key = f"{arxiv_id}_tldr"
            
            st.markdown("### 💡 TLDR")
            
            if tldr_key in st.session_state.tldr_cache:
                st.success(st.session_state.tldr_cache[tldr_key])
            else:
                with st.spinner("🤖 Generating TLDR..."):
                    tldr = generate_tldr(
                        paper["title"],
                        paper.get("summary", ""),
                        ollama_model,
                        args.ollama_url
                    )
                    if tldr:
                        st.session_state.tldr_cache[tldr_key] = tldr
                        save_tldr_cache(args.cache_file, st.session_state.tldr_cache)
                        st.success(tldr)
                    else:
                        st.warning("Could not generate TLDR. Is Ollama running?")
            
            st.markdown("---")
            st.markdown("### 📄 Abstract")
            st.markdown(paper.get("summary", "No abstract available."))
            
            st.markdown("---")
            col_link1, col_link2, col_link3 = st.columns(3)
            with col_link1:
                st.link_button("📄 Abstract Page", paper["link"])
            with col_link2:
                pdf_link = paper["link"].replace("/abs/", "/pdf/") + ".pdf"
                st.link_button("📕 PDF", pdf_link)
            with col_link3:
                html_link = paper["link"].replace("/abs/", "/html/")
                st.link_button("🌐 HTML", html_link)
        
        else:
            st.markdown("### 👈 Select a paper")
            st.markdown("Click a paper from the list or use navigation buttons in the sidebar.")
            
            st.markdown("---")
            st.markdown("#### 📊 Digest Stats")
            
            if papers:
                scores = [p.get("similarity_score", 0) for p in papers]
                avg_score = sum(scores) / len(scores)
                max_score = max(scores)
                min_score = min(scores)
                
                col_s1, col_s2, col_s3 = st.columns(3)
                col_s1.metric("Avg Score", f"{avg_score:.3f}")
                col_s2.metric("Max Score", f"{max_score:.3f}")
                col_s3.metric("Min Score", f"{min_score:.3f}")
                
                high_count = sum(1 for s in scores if s >= 0.8)
                med_count = sum(1 for s in scores if 0.5 <= s < 0.8)
                low_count = sum(1 for s in scores if s < 0.5)
                
                st.markdown("**Score Distribution:**")
                st.markdown(f"- 🟢 High (≥0.8): {high_count} papers")
                st.markdown(f"- 🟡 Medium (0.5-0.8): {med_count} papers")
                st.markdown(f"- ⚪ Low (<0.5): {low_count} papers")
    
    # Save progress
    current_progress["last_page"] = st.session_state.current_page
    current_progress["last_viewed"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    papers_on_current_page = min((st.session_state.current_page + 1) * PAPERS_PER_PAGE, len(papers))
    current_progress["papers_seen"] = max(current_progress["papers_seen"], papers_on_current_page)
    
    st.session_state.progress[digest_key] = current_progress
    save_progress(args.log_file, st.session_state.progress)


if __name__ == "__main__":
    main()
