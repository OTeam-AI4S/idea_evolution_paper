#!/usr/bin/env python3
"""
Bulk paper download — ~10,000 papers from top venues in CS + Biology.
Tries arXiv source / PMC XML → PDF → Gateway, writes compact JSON, and
deletes temporary full-text artifacts.

Usage:  python3 bulk_download.py
Resume: re-run the same command — skips already-searched/downloaded papers.
"""

import argparse, json, logging, sys, time, os, re
from pathlib import Path
from datetime import datetime
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from paper_mining.pipeline import PaperMiningPipeline
from paper_mining.local_pipeline import LocalPDFPipeline
from paper_mining.downloader.base import PaperInfo
from paper_mining.downloader.direct_downloader import (
    is_arxiv_doi,
)
from paper_mining.downloader.publication_metadata import (
    PublicationMetadataResolver,
)
from paper_mining.utils import ensure_dir

# ── Config ─────────────────────────────────────────────────────────────
OUTPUT_DIR    = Path(
    os.environ.get("PAPER_MINING_OUTPUT_DIR", "data")
).expanduser()
PAPERS_DIR    = ensure_dir(str(OUTPUT_DIR / "papers"))
PARSED_DIR    = ensure_dir(str(OUTPUT_DIR / "parsed"))
PROGRESS_FILE = OUTPUT_DIR / "bulk_progress.json"
CACHE_FILE    = OUTPUT_DIR / "all_papers_cache.json"
CONFIG_PATH   = str(Path(__file__).parent / "config.yaml")
PIPELINE_VERSION = 7
PUBLICATION_CACHE_FILE = OUTPUT_DIR / "publication_metadata_cache.jsonl"
TRIAL_CANDIDATE_MULTIPLIER = 10

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(OUTPUT_DIR / "bulk_download.log")),
    ],
)
logger = logging.getLogger("bulk")

# ── arXiv Categories ───────────────────────────────────────────────────
CS_CATS = [
    "cs.AI","cs.CL","cs.CV","cs.LG","cs.NE","cs.IR","cs.RO",
    "cs.CR","cs.DS","cs.DB","cs.HC","cs.SE","cs.GT","cs.IT",
    "cs.MA","cs.MM","cs.SI","cs.CC","cs.CG","cs.DC","cs.DM",
    "cs.ET","cs.FL","cs.GR","cs.MS","cs.NA","cs.OH","cs.OS",
    "cs.PF","cs.PL","cs.SC","cs.SY","cs.SD",
]

BIO_CATS = [
    "q-bio.BM","q-bio.CB","q-bio.GN","q-bio.MN","q-bio.NC",
    "q-bio.OT","q-bio.PE","q-bio.QM","q-bio.SC","q-bio.TO",
]

ALL_CATS = CS_CATS + BIO_CATS

# ── Queries ────────────────────────────────────────────────────────────
CS_QUERIES = [
    # Transformer & LLM
    "transformer neural network attention",
    "self-attention multi-head attention mechanism",
    "large language model pretraining scaling",
    "LLM instruction tuning alignment RLHF",
    "chain-of-thought reasoning in-context learning",
    "retrieval augmented generation RAG",
    "mixture of experts MoE transformer",
    "language model quantization pruning efficient",
    "multimodal large language model vision language",
    "LLM agent tool use planning reasoning",
    # Vision
    "vision transformer ViT DeiT Swin",
    "object detection segmentation DETR mask R-CNN",
    "image generation diffusion model stable",
    "neural radiance field NeRF 3D reconstruction",
    "video understanding action recognition transformer",
    "image super-resolution restoration deep learning",
    "contrastive learning self-supervised vision",
    "masked autoencoder MAE pretraining",
    # NLP
    "text summarization abstractive extractive",
    "machine translation neural multilingual",
    "question answering reading comprehension QA",
    "named entity recognition relation extraction",
    "sentiment analysis opinion mining deep",
    "dialogue system conversational chatbot",
    "code generation program synthesis LLM",
    "semantic parsing text-to-SQL knowledge base",
    "speech recognition ASR transformer conformer",
    # Deep Learning
    "deep neural network optimization SGD Adam",
    "batch normalization layer normalization transformer",
    "residual network ResNet DenseNet EfficientNet",
    "generative adversarial network GAN StyleGAN",
    "variational autoencoder VAE normalizing flow",
    "diffusion probabilistic model score-based",
    "neural ordinary differential equation ODE",
    "knowledge distillation model compression teacher student",
    "neural architecture search NAS AutoML",
    "meta-learning few-shot learning MAML",
    "federated learning privacy preserving FL",
    "dropout regularization data augmentation mixup",
    # Reinforcement Learning
    "deep reinforcement learning DQN PPO SAC",
    "multi-agent reinforcement learning MARL",
    "model-based reinforcement learning planning",
    "offline reinforcement learning batch RL",
    "inverse reinforcement learning IRL",
    # Graph & Structured Data
    "graph neural network GNN message passing",
    "graph attention network GAT GraphSAGE",
    "molecular graph drug discovery GNN",
    "knowledge graph embedding reasoning transE",
    "graph contrastive learning self-supervised",
    # Information Retrieval & RecSys
    "collaborative filtering matrix factorization deep",
    "sequential recommendation transformer SASRec",
    "dense retrieval DPR information retrieval",
    "learning to rank neural ranking",
    # Systems & Theory
    "distributed training data parallelism model",
    "adversarial attack defense robustness deep learning",
    "differential privacy machine learning",
    "generalization theory deep learning overparameterization",
    "neural tangent kernel infinite width",
    "implicit bias gradient descent",
    "representation learning theory identifiability",
]

BIO_QUERIES = [
    # Genomics & Genetics
    "genome sequencing assembly deep learning",
    "single-cell RNA sequencing scRNA-seq analysis",
    "CRISPR gene editing computational prediction",
    "epigenomics chromatin accessibility deep learning",
    "gene regulatory network inference GRN",
    "genome-wide association study GWAS machine learning",
    "population genetics deep learning selection",
    "metagenomics microbiome deep learning",
    "structural variant detection long-read sequencing",
    "multi-omics integration machine learning",
    # Protein & Structural Biology
    "protein structure prediction AlphaFold deep",
    "protein folding molecular dynamics simulation",
    "protein-protein interaction prediction docking",
    "protein design engineering deep learning",
    "protein function prediction GO annotation",
    "antibody design computational deep learning",
    "enzyme engineering directed evolution machine",
    # Drug Discovery
    "drug-target interaction prediction deep",
    "virtual screening molecular docking deep learning",
    "molecular property prediction QSAR deep",
    "de novo drug design generative model",
    "ADMET prediction absorption distribution metabolism",
    "drug repurposing computational systems biology",
    "cheminformatics deep learning representation",
    # Neuroscience
    "neural coding information theory brain",
    "connectome analysis network neuroscience",
    "fMRI functional MRI analysis deep learning",
    "brain-computer interface decoding neural",
    "spiking neural network computational model",
    "neuroimaging segmentation MRI CT deep",
    # Systems Biology
    "metabolic network modeling constraint-based FBA",
    "signaling pathway reconstruction inference",
    "biological network analysis graph theory",
    "systems biology dynamical modeling ODE",
    "synthetic biology circuit design genetic",
    # Biomedical
    "medical image segmentation deep learning U-Net",
    "pathology image analysis computational digital",
    "electronic health record EHR prediction ML",
    "biomarker discovery machine learning omics",
    "cancer genomics tumor heterogeneity evolution",
    "immunoinformatics epitope prediction T-cell",
    "precision medicine pharmacogenomics personalized",
    # Evolution & Ecology
    "phylogenetics maximum likelihood Bayesian inference",
    "molecular evolution positive selection dN/dS",
    "comparative genomics evolution conservation",
    "epidemiological modeling SIR COVID infectious",
    # Bioinformatics Methods
    "sequence alignment algorithm BLAST HMM",
    "RNA-seq differential expression DESeq2 edgeR",
    "Hi-C chromatin 3D genome structure",
    "spatial transcriptomics analysis deep learning",
    "single-cell ATAC-seq multiome integration",
]

# ── Top Venues ─────────────────────────────────────────────────────────
TOP_VENUES_CS = [
    "NeurIPS","ICML","ICLR","AAAI","IJCAI",
    "CVPR","ICCV","ECCV",
    "ACL","EMNLP","NAACL",
    "KDD","WWW","SIGIR","WSDM","CIKM",
    "OSDI","SOSP","ASPLOS","ISCA",
    "IEEE S&P","CCS","USENIX Security",
    "STOC","FOCS","SODA","COLT",
    "ICRA","IROS","RSS","CoRL",
    "JMLR","TPAMI","TACL",
]

TOP_VENUES_BIO = [
    "Nature","Science","PNAS","Nature Communications","Science Advances",
    "eLife","PLOS Biology","BMC Biology",
    "Nature Biotechnology","Nature Methods","Nature Genetics",
    "Nature Neuroscience","Nature Medicine","Nature Chemical Biology",
    "Cell","Molecular Cell","Developmental Cell","Cancer Cell",
    "Cell Systems","Cell Reports",
    "Genome Biology","Genome Research",
    "Nucleic Acids Research","Bioinformatics",
    "PLOS Computational Biology","Briefings in Bioinformatics",
    "Molecular Systems Biology",
    "Neuron","Journal of Neuroscience",
    "Nature Structural Molecular Biology",
    "GigaScience","Bioinformatics Advances",
]
TOP_VENUES = TOP_VENUES_CS + TOP_VENUES_BIO

# ── Helpers ────────────────────────────────────────────────────────────

def load_json(path: Path, default=None):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default if default is not None else {}

def save_json(data, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    path.chmod(0o644)

def load_progress() -> dict:
    p = load_json(PROGRESS_FILE, {})
    p.setdefault("searches_done", [])
    p.setdefault("papers_processed", 0)
    p.setdefault("total_collected", 0)
    return p

def save_progress(p: dict):
    save_json(p, PROGRESS_FILE)

def load_papers_cache() -> List[dict]:
    return load_json(CACHE_FILE, [])

def save_papers_cache(papers: List[dict]):
    save_json(papers, CACHE_FILE)


# ── Search Phase ───────────────────────────────────────────────────────

def build_search_tasks() -> list:
    """Build diverse search tasks to maximize unique papers."""
    tasks = []
    MAX = 300  # max results per search

    # Journal references are the strongest venue evidence, so collect these
    # before the weaker comment-based candidates.
    for venue in TOP_VENUES:
        for year in range(2015, 2027):
            date_range = (
                f"submittedDate:[{year}01010000 TO {year}12312359]"
            )
            tasks.append({
                "query": f'jr:"{venue}" AND {date_range}',
                "cats": None,
                "max": MAX,
                "tag": "journal_venue",
            })

    # Conference publication is commonly stated only in the arXiv comment.
    # The parser still requires explicit "accepted/published/to appear" text
    # before treating a comment match as a verified venue.
    for venue in TOP_VENUES:
        for year in range(2015, 2027):
            date_range = (
                f"submittedDate:[{year}01010000 TO {year}12312359]"
            )
            tasks.append({
                "query": f'co:"{venue}" AND {date_range}',
                "cats": None,
                "max": MAX,
                "tag": "comment_venue",
            })

    # Topic × category group searches (CS)
    for q in CS_QUERIES:
        for i in range(0, len(CS_CATS), 4):
            tasks.append({"query": q, "cats": CS_CATS[i:i+4], "max": MAX, "tag": "cs_topic"})

    # Topic × category group searches (Bio)
    for q in BIO_QUERIES:
        for i in range(0, len(BIO_CATS), 3):
            tasks.append({"query": q, "cats": BIO_CATS[i:i+3], "max": MAX, "tag": "bio_topic"})

    # Broad year sweeps (2015–2026) to fill gaps
    for term in ["deep learning", "neural network", "machine learning",
                 "computational biology", "genomics", "protein structure"]:
        for year in range(2015, 2026):
            tasks.append({
                "query": (
                    f"({term}) AND "
                    f"submittedDate:[{year}01010000 TO {year}12312359]"
                ),
                "cats": None,
                "max": MAX,
                "tag": "broad",
            })

    logger.info(f"Total search tasks: {len(tasks)}")
    return tasks


def run_search_phase(
    pipeline: PaperMiningPipeline,
    stop_after: Optional[int] = None,
):
    """Phase 1: Search all sources, collect metadata, deduplicate, cache."""
    candidate_target = (
        stop_after * TRIAL_CANDIDATE_MULTIPLIER
        if stop_after else None
    )
    progress = load_progress()
    done = (
        set(progress["searches_done"])
        if progress.get("pipeline_version") == PIPELINE_VERSION
        else set()
    )
    all_papers: List[dict] = load_papers_cache()

    if candidate_target and len(all_papers) >= candidate_target:
        logger.info(
            f"Cache already has {len(all_papers)} papers; "
            f"candidate target is {candidate_target}"
        )
        return _enrich_and_rank_papers(all_papers)

    tasks = build_search_tasks()
    arxiv_dl = pipeline.downloaders["arxiv"]
    successful_searches = 0
    consecutive_failures = 0

    for idx, t in enumerate(tasks):
        category_key = ",".join(t["cats"] or [])
        key = f"{t['tag']}:{category_key}:{t['query']}"
        if key in done:
            continue

        if idx % 50 == 0 or idx == len(tasks) - 1:
            logger.info(f"[{idx+1}/{len(tasks)}] {key}  (collected: {len(all_papers)})")

        try:
            results = arxiv_dl.search(
                query=t["query"],
                max_results=t["max"],
                categories=t["cats"],
            )
            successful_searches += 1
            consecutive_failures = 0

            for r in results:
                all_papers.append({
                    "title": r.title,
                    "authors": r.authors,
                    "year": r.year,
                    "doi": r.doi,
                    "pmid": r.pmid,
                    "pmcid": r.pmcid,
                    "arxiv_id": r.arxiv_id,
                    "field": r.field,
                    "venue": r.venue,
                    "abstract": r.abstract,
                    "url": r.url,
                    "publisher_url": r.publisher_url,
                    "pdf_url": r.pdf_url,
                    "source": r.source,
                    "keywords": r.keywords,
                    "citation_count": r.citation_count,
                })

            done.add(key)
            progress["searches_done"] = list(done)
            progress["total_collected"] = len(all_papers)

        except Exception as e:
            consecutive_failures += 1
            logger.warning(f"Search failed '{key}': {e}")
            if (
                not all_papers
                and successful_searches == 0
                and consecutive_failures >= 3
            ):
                raise RuntimeError(
                    "Discovery aborted after three consecutive arXiv failures; "
                    "check compute-node proxy/DNS connectivity"
                ) from e
            continue

        # Dedup & save periodically
        if idx % 30 == 0 and idx > 0:
            all_papers = _dedup_dicts(all_papers)
            save_papers_cache(all_papers)
            save_progress(progress)
            logger.info(f"  Unique papers: {len(all_papers)}")

        if candidate_target and len(all_papers) >= candidate_target:
            logger.info(
                "Trial discovery candidate target reached: "
                f"{len(all_papers)}/{candidate_target}"
            )
            break

        time.sleep(2.5)

    all_papers = _dedup_dicts(all_papers)
    all_papers = _enrich_and_rank_papers(all_papers)
    save_papers_cache(all_papers)
    save_progress(progress)
    logger.info(f"Search done: {len(all_papers)} unique papers")
    return all_papers


def _dedup_dicts(papers: List[dict]) -> List[dict]:
    """Deduplicate and merge newer, richer metadata into cached records."""
    arxiv_index = {}
    doi_index = {}
    title_index = {}
    unique = []

    def keys(paper):
        arxiv_id = re.sub(
            r"(?i)v\d+$", "", paper.get("arxiv_id") or ""
        ).lower()
        doi = (paper.get("doi") or "").lower().strip()
        title = re.sub(
            r"[^a-z0-9]+", " ", paper.get("title", "").lower()
        ).strip()
        return arxiv_id, doi, title

    for paper in papers:
        arxiv_id, doi, title = keys(paper)
        indexes = [
            mapping[key]
            for mapping, key in (
                (arxiv_index, arxiv_id),
                (doi_index, doi),
                (title_index, title),
            )
            if key and key in mapping
        ]
        if indexes:
            index = indexes[0]
            existing = unique[index]
            for key, value in paper.items():
                if value in (None, "", [], {}):
                    continue
                if key == "venue":
                    old = existing.get("venue") or ""
                    if old.startswith("arXiv (") or old in {
                        "", "Unknown", "arXiv preprint"
                    }:
                        existing[key] = value
                elif key == "field":
                    if existing.get(key) in (None, "", "Unknown"):
                        existing[key] = value
                elif not existing.get(key):
                    existing[key] = value
            paper = existing
        else:
            index = len(unique)
            unique.append(dict(paper))

        arxiv_id, doi, title = keys(paper)
        if arxiv_id:
            arxiv_index[arxiv_id] = index
        if doi:
            doi_index[doi] = index
        if title:
            title_index[title] = index
    return unique


def _enrich_and_rank_papers(papers: List[dict]) -> List[dict]:
    """Normalize legacy cache rows, enrich DOI venues, and rank quality."""
    for paper in papers:
        keywords = paper.get("keywords") or []
        primary = keywords[0] if keywords else ""
        old_venue = (paper.get("venue") or "").strip()
        if old_venue.startswith("arXiv ("):
            label = old_venue[7:-1]
            if not paper.get("field"):
                if primary.startswith("cs."):
                    paper["field"] = f"Computer Science / {label}"
                elif primary.startswith("q-bio."):
                    paper["field"] = f"Quantitative Biology / {label}"
                else:
                    paper["field"] = label
            paper["venue"] = "arXiv preprint"
        paper["field"] = paper.get("field") or "Unknown"
        paper["venue"] = paper.get("venue") or "arXiv preprint"

    resolver = PublicationMetadataResolver(
        cache_path=str(PUBLICATION_CACHE_FILE),
        allow_remote=os.environ.get(
            "PAPER_MINING_PUBLICATION_LOOKUP", "1"
        ).lower() not in {"0", "false", "no"},
    )
    papers = resolver.enrich(papers)
    papers.sort(key=_paper_priority, reverse=True)
    return papers


def _paper_priority(paper: dict) -> tuple:
    venue = re.sub(
        r"[^a-z0-9]+", " ", paper.get("venue", "").lower()
    ).strip()
    top = any(
        re.search(
            rf"(?:^|\s){re.escape(re.sub(r'[^a-z0-9]+', ' ', candidate.lower()).strip())}(?:\s|$)",
            venue,
        )
        for candidate in TOP_VENUES
    )
    formal = venue not in {"", "unknown", "arxiv preprint"}
    return (
        3 if top else 2 if formal else 1 if paper.get("doi") else 0,
        int(paper.get("citation_count") or 0),
        int(paper.get("year") or 0),
    )


# ── Download + Parse Phase ─────────────────────────────────────────────

def _download_known_pdf(info, arxiv_dl):
    """Download a known open PDF URL without invoking Gateway."""
    if info.pdf_url:
        try:
            pdf_path = arxiv_dl.download_pdf(info)
            if pdf_path:
                source = "arxiv" if info.arxiv_id else "open_access"
                return (pdf_path, source)
        except Exception as e:
            logger.debug(f"Known PDF URL failed for {info.title[:60]}: {e}")
    return (None, None)


def _failure_category(errors: List[str]) -> str:
    message = " ".join(errors).lower()
    if any(
        marker in message
        for marker in (
            "too short", "no cited-paper abstracts resolved",
            "too few cited-paper abstracts resolved",
            "invalid schema", "contains no tex", "too little article text",
            "lack a complete cited-paper abstract",
            "lack a complete abstract or citation context",
        )
    ):
        return "quality_validation"
    return "source_unavailable"


def _safe_output_stem(paper: dict) -> str:
    value = (
        paper.get("arxiv_id") or paper.get("doi") or paper.get("pmcid")
        or paper.get("title") or "paper"
    )
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return (value or "paper")[:120]


def run_download_phase(
    papers: List[dict],
    pipeline: PaperMiningPipeline,
    limit: Optional[int] = None,
    shard_index: int = 0,
    shard_count: int = 1,
    reference_metadata_path: Optional[str] = None,
):
    """Phase 2: structured full text → PDF → Gateway → compact JSON."""
    progress = load_progress()
    existing_outputs = list(PARSED_DIR.glob("*.json"))
    if progress.get("pipeline_version") == PIPELINE_VERSION:
        processed_ids = set(progress.get("processed_ids", []))
    else:
        # A parser/source upgrade should retry old failures while preserving
        # already validated outputs.
        processed_ids = set()
    current_version = progress.get("pipeline_version") == PIPELINE_VERSION
    existing_stems = (
        {path.stem for path in existing_outputs}
        if current_version else set()
    )
    if current_version:
        for paper in papers:
            if _safe_output_stem(paper) in existing_stems:
                processed_ids.add(
                    paper.get("arxiv_id")
                    or paper.get("title", "unknown")
                )
    stats = {
        # The directory is authoritative. Outputs may have been removed by a
        # later quality audit, in which case the trial must replenish them.
        "ok": len(existing_outputs) if current_version else 0,
        "fail": (
            progress.get("fail", 0)
            if progress.get("pipeline_version") == PIPELINE_VERSION else 0
        ),
        "arxiv_source_ok": progress.get("arxiv_source_ok", 0),
        "pmc_xml_ok": progress.get("pmc_xml_ok", 0),
        "gateway_ok": progress.get("gateway_ok", 0),
        "arxiv_ok": progress.get("arxiv_ok", 0),
        "open_access_ok": progress.get("open_access_ok", 0),
        "gateway_fail": progress.get("gateway_fail", 0),
        "failure_reasons": progress.get("failure_reasons", {}),
    }
    if progress.get("pipeline_version") != PIPELINE_VERSION:
        for key in (
            "arxiv_source_ok", "pmc_xml_ok", "gateway_ok", "arxiv_ok",
            "open_access_ok", "gateway_fail",
        ):
            stats[key] = 0
        stats["failure_reasons"] = {}
    # Skip already-processed
    remaining = [p for p in papers if p.get("arxiv_id", "") not in processed_ids
                 and p.get("title", "") not in processed_ids]
    selected = sorted(remaining, key=_paper_priority, reverse=True)
    selected = selected[shard_index::shard_count]
    logger.info(
        "Priority pool: %d top-venue, %d other formal venue, %d preprint",
        sum(_paper_priority(p)[0] == 3 for p in selected),
        sum(_paper_priority(p)[0] == 2 for p in selected),
        sum(_paper_priority(p)[0] < 2 for p in selected),
    )
    logger.info(
        "Download shard: %d/%d (%d candidates)",
        shard_index + 1,
        shard_count,
        len(selected),
    )
    logger.info(
        f"Trial target: {limit or 'unlimited'}, "
        f"Already done: {len(processed_ids)}, Remaining: {len(remaining)}"
    )

    arxiv_dl = pipeline.downloaders["arxiv"]
    structured_dl = pipeline.downloaders["structured"]
    gateway_dl = pipeline.downloaders["direct"]
    compact_pipeline = LocalPDFPipeline(
        output_dir=str(PARSED_DIR),
        engine=pipeline.config.get("parsing", {}).get("engine", "hybrid"),
        reference_metadata_paths=[
            reference_metadata_path or str(CACHE_FILE),
            *[
                path for path in os.environ.get(
                    "PAPER_MINING_REFERENCE_METADATA_PATHS", ""
                ).split(os.pathsep)
                if path
            ],
        ],
        reference_cache_path=str(
            OUTPUT_DIR / "reference_abstract_cache.jsonl"
        ),
        resolve_reference_abstracts_remotely=True,
        min_references=int(os.environ.get(
            "PAPER_MINING_MIN_REFERENCES", "5"
        )),
        max_reference_title_lookups=int(os.environ.get(
            "PAPER_MINING_MAX_TITLE_LOOKUPS", "12"
        )),
        allow_crossref_title_search=os.environ.get(
            "PAPER_MINING_CROSSREF_TITLE_SEARCH", "0"
        ).lower() not in {"0", "false", "no"},
        max_citation_contexts_per_reference=int(os.environ.get(
            "PAPER_MINING_MAX_CITATION_CONTEXTS", "2"
        )),
        citation_context_sentences_before=int(os.environ.get(
            "PAPER_MINING_CONTEXT_SENTENCES_BEFORE", "1"
        )),
        citation_context_sentences_after=int(os.environ.get(
            "PAPER_MINING_CONTEXT_SENTENCES_AFTER", "1"
        )),
    )

    gateway_enabled = os.environ.get(
        "PAPER_MINING_DISABLE_GATEWAY", "0"
    ).lower() in {"0", "false", "no"}
    gateway_candidate_count = sum(
        1 for p in remaining
        if gateway_enabled and (
            (p.get("doi") and not is_arxiv_doi(p["doi"]))
            or p.get("publisher_url")
            or p.get("pmid")
        )
    )
    logger.info(f"Papers with Gateway identifiers: {gateway_candidate_count}")
    if gateway_candidate_count:
        if gateway_dl.is_available:
            logger.info("Gateway reachable — available as a DOI candidate")
        else:
            logger.warning("Gateway unavailable — using known PDF URLs only")

    for i, paper in enumerate(selected):
        if limit and stats["ok"] >= limit:
            break

        pid = paper.get("arxiv_id") or paper.get("title", "unknown")
        ptitle = paper.get("title", "unknown")[:60]
        artifact_paths = []
        gateway_attempted = False
        stage_errors: List[str] = []

        try:
            # Reconstruct PaperInfo
            info = PaperInfo(
                title=paper.get("title", ""),
                authors=paper.get("authors", []),
                year=paper.get("year"),
                doi=paper.get("doi"),
                pmid=paper.get("pmid"),
                pmcid=paper.get("pmcid"),
                arxiv_id=paper.get("arxiv_id"),
                field=paper.get("field"),
                venue=paper.get("venue"),
                abstract=paper.get("abstract"),
                url=paper.get("url"),
                publisher_url=paper.get("publisher_url"),
                pdf_url=paper.get("pdf_url"),
                source=paper.get("source"),
                keywords=paper.get("keywords", []),
                citation_count=paper.get("citation_count"),
            )

            output_id = info.arxiv_id or info.doi or info.pmcid or info.title
            download_source = None

            # 1. arXiv TeX source.
            if info.arxiv_id:
                source_path = structured_dl.download_arxiv_source(info)
                if source_path:
                    artifact_paths.append(source_path)
                    try:
                        compact_pipeline.process_arxiv_source(
                            str(source_path),
                            title=info.title,
                            abstract=info.abstract,
                            field=info.field,
                            venue=info.venue,
                            year=info.year,
                            paper_id=output_id,
                        )
                        download_source = "arxiv_source"
                    except Exception as exc:
                        stage_errors.append(f"arxiv_source: {exc}")

            # 2. PMC JATS XML.
            if not download_source and (info.pmcid or info.pmid):
                xml_path = structured_dl.download_pmc_xml(info)
                if xml_path:
                    artifact_paths.append(xml_path)
                    try:
                        compact_pipeline.process_pmc_xml(
                            str(xml_path),
                            title=info.title,
                            abstract=info.abstract,
                            field=info.field,
                            venue=info.venue,
                            year=info.year,
                            paper_id=output_id,
                        )
                        download_source = "pmc_xml"
                    except Exception as exc:
                        stage_errors.append(f"pmc_xml: {exc}")

            # 3. Known open PDF.
            if not download_source:
                pdf_path, pdf_source = _download_known_pdf(info, arxiv_dl)
                if pdf_path:
                    artifact_paths.append(pdf_path)
                    try:
                        compact_pipeline.process_pdf(
                            str(pdf_path),
                            title=info.title,
                            abstract=info.abstract,
                            field=info.field,
                            venue=info.venue,
                            year=info.year,
                            paper_id=output_id,
                        )
                        download_source = pdf_source
                    except Exception as exc:
                        stage_errors.append(f"pdf: {exc}")
                else:
                    stage_errors.append("pdf: unavailable")

            # 4. Gateway is the last full-paper fallback.
            if (
                not download_source
                and gateway_enabled
                and gateway_dl.candidate_identifiers(info)
            ):
                gateway_attempted = True
                alternate = gateway_dl.download_paper(info)
                if alternate:
                    alternate_path = Path(alternate)
                    artifact_paths.append(alternate_path)
                    try:
                        compact_pipeline.process_pdf(
                            str(alternate_path),
                            title=info.title,
                            abstract=info.abstract,
                            field=info.field,
                            venue=info.venue,
                            year=info.year,
                            paper_id=output_id,
                        )
                        download_source = "gateway"
                    except Exception as exc:
                        stage_errors.append(f"gateway: {exc}")
                else:
                    stage_errors.append("gateway: unavailable")

            if not download_source:
                raise RuntimeError("; ".join(stage_errors) or "all sources failed")

            if download_source == "arxiv_source":
                stats["arxiv_source_ok"] += 1
            elif download_source == "pmc_xml":
                stats["pmc_xml_ok"] += 1
            elif download_source == "gateway":
                stats["gateway_ok"] += 1
            elif download_source == "open_access":
                stats["open_access_ok"] += 1
            else:
                stats["arxiv_ok"] += 1
            stats["ok"] += 1

        except Exception as e:
            logger.info(f"Rejected {ptitle}: {e}")
            if gateway_attempted:
                stats["gateway_fail"] += 1
            category = _failure_category(stage_errors or [str(e)])
            stats["failure_reasons"][category] = (
                stats["failure_reasons"].get(category, 0) + 1
            )
            stats["fail"] += 1
        finally:
            for artifact_path in artifact_paths:
                try:
                    os.remove(str(artifact_path))
                except OSError:
                    pass

        processed_ids.add(pid)

        # Progress update every N papers
        if (i + 1) % 50 == 0:
            total = i + 1
            logger.info(
                f"Progress: {total} ok={stats['ok']} "
                f"(SRC:{stats['arxiv_source_ok']} PMC:{stats['pmc_xml_ok']} "
                f"GW:{stats['gateway_ok']} AX:{stats['arxiv_ok']} "
                f"OA:{stats['open_access_ok']}) fail={stats['fail']}"
            )
            progress["pipeline_version"] = PIPELINE_VERSION
            progress["processed_ids"] = list(processed_ids)
            for key, value in stats.items():
                progress[key] = value
            save_progress(progress)

        # Rate limit
        time.sleep(2.0)

    progress["pipeline_version"] = PIPELINE_VERSION
    progress["processed_ids"] = list(processed_ids)
    for key, value in stats.items():
        progress[key] = value
    progress["completed_at"] = datetime.now().isoformat()
    save_progress(progress)

    logger.info(
        f"\nDownload+Parse complete: {stats['ok']} ok "
        f"(arXiv source: {stats['arxiv_source_ok']}, "
        f"PMC XML: {stats['pmc_xml_ok']}, Gateway: {stats['gateway_ok']}, "
        f"PDF/arXiv: {stats['arxiv_ok']}, "
        f"OA: {stats['open_access_ok']}), {stats['fail']} fail"
    )
    return stats


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional trial limit; omit for no paper-count limit.",
    )
    parser.add_argument(
        "--max-pdf-mb",
        type=float,
        default=None,
        help="Optional PDF size limit, useful for a quick trial.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--search-only",
        action="store_true",
        help="Build the shared candidate metadata cache and stop.",
    )
    mode.add_argument(
        "--download-only",
        action="store_true",
        help="Read an existing candidate cache and only download/parse.",
    )
    parser.add_argument(
        "--source-cache",
        default=None,
        help="Candidate JSON cache used by --download-only.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Number of independent download shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based index of this download shard.",
    )
    args = parser.parse_args()
    if args.shard_count < 1:
        parser.error("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, shard-count)")
    if args.search_only and args.shard_count != 1:
        parser.error("--search-only does not use download shards")

    logger.info("=" * 60)
    logger.info("  BULK PAPER MINING — CS + Bio, Top Venues")
    logger.info(f"  Output JSON: {PARSED_DIR}")
    logger.info(f"  Started:     {datetime.now().isoformat()}")
    logger.info("=" * 60)

    pipeline = PaperMiningPipeline(
        CONFIG_PATH,
        enabled_downloaders={"structured", "direct", "arxiv"},
    )
    if args.max_pdf_mb:
        max_pdf_bytes = int(args.max_pdf_mb * 1024 * 1024)
        pipeline.downloaders["arxiv"].max_pdf_bytes = max_pdf_bytes
        pipeline.downloaders["direct"].max_pdf_bytes = max_pdf_bytes

    reference_metadata_path = args.source_cache or str(CACHE_FILE)
    if args.download_only:
        logger.info("\n=== PHASE 1: Loading shared candidate cache ===")
        all_papers = load_json(Path(reference_metadata_path), [])
        all_papers = sorted(all_papers, key=_paper_priority, reverse=True)
    else:
        # Phase 1: Search & collect metadata
        logger.info("\n=== PHASE 1: Searching ===")
        all_papers = run_search_phase(pipeline, stop_after=args.limit)
    if not all_papers:
        raise RuntimeError(
            "Discovery returned zero papers; refusing to report a successful run"
        )
    logger.info(f"Collected {len(all_papers)} unique papers")

    # Show stats
    years = [p.get("year") for p in all_papers if p.get("year")]
    sources = {}
    for p in all_papers:
        s = p.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1
    with_doi = sum(1 for p in all_papers if p.get("doi"))
    logger.info(f"Year range: {min(years) if years else '?'} – {max(years) if years else '?'}")
    logger.info(f"Sources: {sources}")
    logger.info(f"With DOI: {with_doi}/{len(all_papers)}")
    gateway_candidates = sum(
        1 for p in all_papers
        if (
            (p.get("doi") and not is_arxiv_doi(p["doi"]))
            or p.get("publisher_url")
            or p.get("pmid")
        )
    )
    logger.info(
        f"With Gateway identifiers: {gateway_candidates}/{len(all_papers)}"
    )
    if args.search_only:
        logger.info(
            "Search-only complete: shared cache is %s",
            CACHE_FILE,
        )
        return

    # Phase 2: Download → Parse → Delete PDF
    logger.info("\n=== PHASE 2: Download → Parse → Save JSON ===")
    stats = run_download_phase(
        all_papers,
        pipeline,
        limit=args.limit,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        reference_metadata_path=reference_metadata_path,
    )

    json_count = len(list(PARSED_DIR.glob("*.json")))
    logger.info(f"\n{'='*60}")
    logger.info(f"  DONE")
    logger.info(f"  Searched:      {len(all_papers)} papers")
    logger.info(f"  Parsed JSON:   {json_count} files in {PARSED_DIR}")
    logger.info(f"  Success:       {stats['ok']}")
    logger.info(f"    arXiv source:{stats.get('arxiv_source_ok', 0)}")
    logger.info(f"    PMC XML:     {stats.get('pmc_xml_ok', 0)}")
    logger.info(f"    via Gateway: {stats.get('gateway_ok', 0)}")
    logger.info(f"    via arXiv:   {stats.get('arxiv_ok', 0)}")
    logger.info(f"    via OA URL:  {stats.get('open_access_ok', 0)}")
    logger.info(f"  Failed:        {stats['fail']}")
    logger.info(f"  Failure types: {stats.get('failure_reasons', {})}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
