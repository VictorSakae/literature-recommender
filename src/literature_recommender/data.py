"""Load article metadata and directed citations without filtering coverage."""

from pathlib import Path

import pandas as pd


def _missing_text(values: pd.Series) -> pd.Series:
    """Recognize null, empty and whitespace-only text without changing it."""
    return values.isna() | values.str.strip().eq("").fillna(False)


def load_papers(path: str | Path) -> pd.DataFrame:
    """Read the official XLSX schema, preserving all articles and duplicate IDs.

    Return nullable string columns paper_id, title, abstract. Blank text and
    the exact abstract marker 'noabstract' become pd.NA. Missing IDs are errors.
    """
    papers = pd.read_excel(
        path,
        sheet_name="Sheet 1",
        engine="openpyxl",
        usecols=["paper_ID", "title", "abstract"],
        dtype="string",
        keep_default_na=False,
    ).rename(columns={"paper_ID": "paper_id"})
    papers = papers[["paper_id", "title", "abstract"]]
    if _missing_text(papers["paper_id"]).any():
        raise ValueError("Article IDs must not be missing or blank")
    for column in ("title", "abstract"):
        papers[column] = papers[column].mask(_missing_text(papers[column]), pd.NA)
    papers["abstract"] = papers["abstract"].mask(
        papers["abstract"].eq("noabstract").fillna(False), pd.NA
    )
    return papers


def load_citations(path: str | Path, *, deduplicate: bool = False) -> pd.DataFrame:
    """Read source (citing article) -> target (cited article) from CSV.

    Preserve duplicate rows by default, external IDs and self-loops. Explicit
    deduplication keeps the first occurrence of each directed pair.
    """
    citations = pd.read_csv(
        path, usecols=["source", "target"], dtype="string", keep_default_na=False
    ).rename(columns={"source": "source_id", "target": "target_id"})
    citations = citations[["source_id", "target_id"]]
    if any(_missing_text(citations[column]).any() for column in citations):
        raise ValueError("Citation IDs must not be missing or blank")
    if deduplicate:
        citations = citations.drop_duplicates().reset_index(drop=True)
    return citations


def get_article(papers: pd.DataFrame, paper_id: str) -> pd.Series:
    """Return an independent row; raise KeyError if absent, ValueError if ambiguous."""
    matches = papers.loc[papers["paper_id"].eq(paper_id)]
    if matches.empty:
        raise KeyError(paper_id)
    if len(matches) > 1:
        raise ValueError(f"Duplicate article ID: {paper_id!r}")
    return matches.iloc[0].copy()


def validate_data(papers: pd.DataFrame, citations: pd.DataFrame) -> dict[str, int]:
    """Report counts without mutating either input (internal schemas required).

    duplicate_*_rows counts excess occurrences after the first; duplicate_*_ids
    counts distinct IDs that repeat. Self-loops and missing-metadata rows count
    every occurrence, including duplicates. Missing-metadata IDs are distinct.
    Abstract absence includes the raw 'noabstract' marker for in-memory inputs.
    """
    ids = papers["paper_id"]
    source = citations["source_id"]
    target = citations["target_id"]
    missing_source = ~source.isin(ids)
    missing_target = ~target.isin(ids)
    network_ids = pd.concat([source, target], ignore_index=True).unique()
    return {
        "articles": len(papers),
        "unique_article_ids": int(ids.nunique()),
        "duplicate_article_ids": int(ids[ids.duplicated(keep=False)].nunique()),
        "duplicate_article_rows": int(ids.duplicated().sum()),
        "missing_titles": int(_missing_text(papers["title"]).sum()),
        "missing_abstracts": int((
            _missing_text(papers["abstract"])
            | papers["abstract"].eq("noabstract").fillna(False)
        ).sum()),
        "citation_rows": len(citations),
        "unique_citation_pairs": len(citations.drop_duplicates(["source_id", "target_id"])),
        "duplicate_citation_rows": int(citations.duplicated(["source_id", "target_id"]).sum()),
        "self_loop_rows": int(source.eq(target).sum()),
        "network_ids": len(network_ids),
        "source_ids_without_metadata": int(source[missing_source].nunique()),
        "source_rows_without_metadata": int(missing_source.sum()),
        "target_ids_without_metadata": int(target[missing_target].nunique()),
        "target_rows_without_metadata": int(missing_target.sum()),
        "article_ids_outside_network": int(ids[~ids.isin(network_ids)].nunique()),
    }
