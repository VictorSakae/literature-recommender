"""Small synthetic fixtures; no dependency on data/raw or a test framework."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from literature_recommender.data import (
    get_article,
    load_citations,
    load_papers,
    validate_data,
)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def papers_file(self, **overrides):
        data = {
            "paper_ID": ["paper_1", "paper_2", "paper_3", "paper_4"],
            "title": ["Título válido", "", "  ", "NA"],
            "abstract": ["noabstract", "", " \t\n", " Texto válido "],
        }
        data.update(overrides)
        path = self.root / "articles.xlsx"
        pd.DataFrame(data).to_excel(path, sheet_name="Sheet 1", index=False)
        return path

    def citations_file(self):
        path = self.root / "citations.csv"
        path.write_text(
            "source,target\npaper_1,paper_2\npaper_2,paper_1\n"
            "paper_1,paper_2\npaper_1,outside\npaper_1,paper_1\n",
            encoding="utf-8",
        )
        return path

    def test_paper_mapping_and_missing_text(self):
        papers = load_papers(self.papers_file())
        self.assertEqual(list(papers), ["paper_id", "title", "abstract"])
        self.assertEqual(len(papers), 4)
        self.assertTrue(all(isinstance(dtype, pd.StringDtype) for dtype in papers.dtypes))
        self.assertEqual(papers.paper_id.tolist(), [f"paper_{i}" for i in range(1, 5)])
        self.assertTrue(papers.abstract.iloc[:3].isna().all())
        self.assertEqual(papers.abstract.iloc[3], " Texto válido ")
        self.assertEqual(papers.title.iloc[0], "Título válido")
        self.assertTrue(papers.title.iloc[1:3].isna().all())
        self.assertEqual(papers.title.iloc[3], "NA")

    def test_duplicate_article_ids_are_preserved_and_reported(self):
        papers = load_papers(self.papers_file(paper_ID=["a", "a", "a", "b"]))
        report = validate_data(papers, load_citations(self.citations_file()))
        self.assertEqual(report["articles"], 4)
        self.assertEqual(report["duplicate_article_ids"], 1)
        self.assertEqual(report["duplicate_article_rows"], 2)
        with self.assertRaisesRegex(ValueError, "Duplicate article ID"):
            get_article(papers, "a")

    def test_get_article_and_missing_id(self):
        papers = load_papers(self.papers_file())
        article = get_article(papers, "paper_1")
        self.assertEqual(article.title, "Título válido")
        self.assertTrue(pd.isna(article.abstract))
        article["title"] = "Changed"
        self.assertEqual(papers.title.iloc[0], "Título válido")
        with self.assertRaises(KeyError) as error:
            get_article(papers, "missing")
        self.assertEqual(error.exception.args, ("missing",))

    def test_citation_mapping_direction_and_preservation(self):
        citations = load_citations(self.citations_file())
        self.assertEqual(list(citations), ["source_id", "target_id"])
        self.assertTrue(all(isinstance(dtype, pd.StringDtype) for dtype in citations.dtypes))
        self.assertEqual(list(citations.itertuples(index=False, name=None)), [
            ("paper_1", "paper_2"), ("paper_2", "paper_1"),
            ("paper_1", "paper_2"), ("paper_1", "outside"),
            ("paper_1", "paper_1"),
        ])

    def test_explicit_deduplication_preserves_reverse_pair(self):
        path = self.citations_file()
        expected = load_citations(path).iloc[[0, 1, 3, 4]].reset_index(drop=True)
        assert_frame_equal(load_citations(path, deduplicate=True), expected)

    def test_validation_counts_and_no_mutation(self):
        papers = load_papers(self.papers_file())
        citations = load_citations(self.citations_file())
        citations.loc[len(citations)] = ["external_source", "outside"]
        before_papers, before_citations = papers.copy(deep=True), citations.copy(deep=True)
        self.assertEqual(validate_data(papers, citations), {
            "articles": 4, "unique_article_ids": 4,
            "duplicate_article_ids": 0, "duplicate_article_rows": 0,
            "missing_titles": 2, "missing_abstracts": 3,
            "citation_rows": 6, "unique_citation_pairs": 5,
            "duplicate_citation_rows": 1, "self_loop_rows": 1,
            "network_ids": 4, "source_ids_without_metadata": 1,
            "source_rows_without_metadata": 1, "target_ids_without_metadata": 1,
            "target_rows_without_metadata": 2, "article_ids_outside_network": 2,
        })
        assert_frame_equal(papers, before_papers)
        assert_frame_equal(citations, before_citations)

    def test_missing_required_columns(self):
        path = self.root / "invalid.xlsx"
        pd.DataFrame({"id": ["a"]}).to_excel(path, sheet_name="Sheet 1", index=False)
        with self.assertRaises(ValueError):
            load_papers(path)
        path = self.root / "invalid.csv"
        path.write_text("source\na\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_citations(path)

    def test_blank_ids_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "Article IDs"):
            load_papers(self.papers_file(paper_ID=["", "b", "c", "d"]))
        path = self.root / "blank.csv"
        path.write_text("source,target\na, \n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Citation IDs"):
            load_citations(path)

    def test_empty_datasets(self):
        path = self.papers_file(paper_ID=[], title=[], abstract=[])
        papers = load_papers(path)
        csv_path = self.root / "empty.csv"
        csv_path.write_text("source,target\n", encoding="utf-8")
        citations = load_citations(csv_path)
        self.assertTrue(all(value == 0 for value in validate_data(papers, citations).values()))
        with self.assertRaises(KeyError):
            get_article(papers, "missing")

    def test_input_files_unchanged(self):
        xlsx, csv_path = self.papers_file(), self.citations_file()
        before = (xlsx.read_bytes(), csv_path.read_bytes())
        load_papers(xlsx)
        load_citations(csv_path, deduplicate=True)
        self.assertEqual(before, (xlsx.read_bytes(), csv_path.read_bytes()))


if __name__ == "__main__":
    unittest.main()
