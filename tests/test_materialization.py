"""Synthetic 7B evidence materialization. No external requests."""
from copy import deepcopy
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from gensim.models import KeyedVectors
import networkx as nx
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from literature_recommender.experiment import prepare_reference_query
from literature_recommender.materialization import (
    COMMON, TEXTUAL, common_scores, textual_scores, validate_tables,
    needed_texts, embedding_inventory, ensure_embeddings, persist_scores,
    frozen_plans, verify_hashes, file_hash,
)
from literature_recommender.similarity import abstract_similarities
from literature_recommender.tfidf import fit_tfidf, tfidf_abstract_similarities
from literature_recommender.word2vec import prepare_word2vec_corpus, word2vec_abstract_similarities


class MaterializationTests(unittest.TestCase):
    def setUp(self):
        ids = ['Q', 'R'] + [f'C{i}' for i in range(22)]
        self.papers = pd.DataFrame(dict(paper_id=ids, title=['alpha beta'] * len(ids),
                                        abstract=['alpha beta'] * len(ids)), dtype='string')
        self.papers.loc[self.papers.paper_id.eq('C0'), 'abstract'] = pd.NA
        self.graph = nx.DiGraph([('Q', 'C0'), ('Q', 'C1'), ('R', 'C1'), ('R', 'C21')])
        self.graph.add_edges_from(('C1', f'C{i}') for i in range(2, 21))
        self.plans = [(s, prepare_reference_query(self.papers, self.graph, q))
                      for s, q in [('tuning', 'Q'), ('evaluation', 'R')]]
        self.common = common_scores(self.papers, self.graph, self.plans)
        tfidf = fit_tfidf(self.papers)
        self.tfidf = textual_scores(self.plans, lambda q, ids: tfidf_abstract_similarities(tfidf, q, ids))

    def fake(self):
        create = Mock(side_effect=lambda **kwargs: SimpleNamespace(data=[
            SimpleNamespace(index=i, embedding=[1.] + [0.] * 1535)
            for i in reversed(range(len(kwargs['input'])))]))
        return SimpleNamespace(embeddings=SimpleNamespace(create=create))

    def test_integrated_four_tables_and_nonmutation(self):
        before = self.papers.copy(deep=True)
        graph = deepcopy(self.graph)
        plans = deepcopy(self.plans)
        corpus = prepare_word2vec_corpus(self.papers)
        wv = KeyedVectors(vector_size=2)
        wv.add_vectors(['alpha', 'beta'], np.array([[1., 0.], [0., 1.]]))
        word2vec = textual_scores(self.plans, lambda q, ids: word2vec_abstract_similarities(corpus, wv, q, ids))
        with TemporaryDirectory() as directory:
            client = self.fake()
            vectors, stats = ensure_embeddings(needed_texts(self.plans), directory, client=client, batch_size=7)
            embedding = textual_scores(self.plans, lambda q, ids: abstract_similarities(
                vectors[q], {p: None if p in corpus.missing_ids else vectors[p] for p in ids}))
        result = validate_tables(self.common, dict(tfidf=self.tfidf, word2vec=word2vec, embedding=embedding))
        self.assertEqual(result['n_queries'], {'evaluation': 1, 'tuning': 1})
        self.assertEqual(result['n_pairs'], 42)
        self.assertEqual(result['n_imputed'], 1)
        self.assertEqual(list(self.common), COMMON)
        self.assertEqual(list(embedding), TEXTUAL)
        self.assertEqual(self.common.relevant.dtype, bool)
        self.assertEqual(embedding.abstract_similarity.dtype, np.dtype('float64'))
        q = self.common[self.common.query_id.eq('Q')].set_index('candidate_id')
        self.assertEqual(set(q.index[q.relevant]), {'C0', 'C1'})
        self.assertAlmostEqual(q.loc['C1', 'title_similarity'], 1)
        self.assertTrue(q.citation_similarity.eq(0).all())
        self.assertNotEqual(self.plans[0][1]['eligible_ids'], self.plans[1][1]['eligible_ids'])
        assert_frame_equal(before, self.papers)
        self.assertTrue(nx.utils.graphs_equal(graph, self.graph))
        self.assertEqual(plans, self.plans)

    def test_citation_full_pool_extrema(self):
        self.graph.add_edge('C2', 'C1')
        common = common_scores(self.papers, self.graph, self.plans)
        q = common[common.query_id.eq('Q')].set_index('candidate_id')
        self.assertEqual(q.loc['C2', 'citation_similarity'], 1)
        self.assertEqual(q.loc['C3', 'citation_similarity'], 0)

    def test_key_mismatches(self):
        for kind in ['absent', 'extra', 'different', 'duplicate', 'self']:
            changed = self.tfidf.copy()
            if kind == 'absent':
                changed = changed.iloc[1:]
            elif kind == 'extra':
                row = changed.iloc[:1].copy()
                row.candidate_id = 'EXTRA'
                changed = pd.concat([changed, row])
            elif kind == 'different':
                changed.loc[0, 'candidate_id'] = 'OTHER'
            elif kind == 'duplicate':
                changed = pd.concat([changed, changed.iloc[:1]])
            else:
                changed.loc[0, 'candidate_id'] = changed.loc[0, 'query_id']
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_tables(self.common, {'tfidf': changed})

    def test_flags_schema_bounds_means(self):
        changes = [lambda f: f.drop(columns='abstract_imputed'),
                   lambda f: f.assign(abstract_imputed=~f.abstract_imputed),
                   lambda f: f.assign(abstract_similarity=np.nan),
                   lambda f: f.assign(abstract_similarity=1.1),
                   lambda f: f.assign(abstract_similarity='0.5')]
        for change in changes:
            with self.assertRaises(ValueError):
                validate_tables(self.common, {'tfidf': change(self.tfidf.copy())})
        changed = self.tfidf.copy()
        changed.loc[changed.abstract_imputed, 'abstract_similarity'] = .2
        with self.assertRaisesRegex(ValueError, 'mean'):
            validate_tables(self.common, {'tfidf': changed})

    def test_inventory_resume_and_cache_identity(self):
        texts = needed_texts(self.plans)
        self.assertEqual(len(texts), 23)  # shared candidate IDs deduplicated, C0 absent
        with TemporaryDirectory() as directory:
            self.assertEqual(embedding_inventory(texts, directory)['n_new_embeddings_required'], 23)
            client = self.fake()
            ensure_embeddings(texts, directory, client=client, batch_size=8)
            self.assertEqual(client.embeddings.create.call_count, 3)
            client.embeddings.create.reset_mock()
            _, stats = ensure_embeddings(texts, directory, client=client)
            client.embeddings.create.assert_not_called()
            self.assertEqual(stats['new_embeddings_generated'], 0)
            changed = dict(texts)
            changed['Q'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'identity'):
                embedding_inventory(changed, directory)
            path = next(Path(directory).glob('*.json'))
            record = json.loads(path.read_text())
            record['vector'] = [0.] * 1536
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'Invalid cached'):
                embedding_inventory(texts, directory)

    def test_provider_failure_preserves_completed_batches(self):
        with TemporaryDirectory() as directory:
            client = self.fake()
            successful = client.embeddings.create.side_effect
            calls = 0
            def fail(**kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise TimeoutError('provider')
                return successful(**kwargs)
            client.embeddings.create.side_effect = fail
            texts = needed_texts(self.plans)
            with self.assertRaises(TimeoutError):
                ensure_embeddings(texts, directory, client=client, batch_size=5)
            self.assertEqual(embedding_inventory(texts, directory)['n_already_available'], 5)
            client = self.fake()
            _, stats = ensure_embeddings(texts, directory, client=client)
            self.assertEqual(stats['new_embeddings_generated'], 18)

    def test_unrepresentable_valid_text_raises(self):
        corpus = prepare_word2vec_corpus(self.papers)
        wv = KeyedVectors(vector_size=2)
        wv.add_vector('unrelated', np.array([1., 0.]))
        with self.assertRaisesRegex(ValueError, 'no tokens'):
            textual_scores(self.plans, lambda q, ids: word2vec_abstract_similarities(corpus, wv, q, ids))

    def test_persist_and_input_hash_protection(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'scores.csv'
            persist_scores(self.common, path)
            original = path.read_bytes()
            persist_scores(self.common, path)
            self.assertEqual(original, path.read_bytes())
            changed = self.common.copy()
            changed.loc[0, 'relevant'] = ~changed.loc[0, 'relevant']
            with self.assertRaises(AssertionError):
                persist_scores(changed, path)
            verify_hashes({str(path): file_hash(path)})
            with self.assertRaises(ValueError):
                verify_hashes({str(path): 'wrong'})
            self.assertEqual(original, path.read_bytes())

    def test_protocol_inconsistency_raises_without_mutation(self):
        protocol = dict(seed=41)
        before = deepcopy(protocol)
        with self.assertRaises(ValueError):
            frozen_plans(self.papers, self.graph, protocol, pd.DataFrame(), {})
        self.assertEqual(protocol, before)

    def test_embeddings_only_entry_preserves_local_tables_without_fitting(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/materialize_experiment_scores.py'
        spec = importlib.util.spec_from_file_location('materialize_script', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = dict(sg=0, vector_size=100, window=5, min_count=5, epochs=5,
                      negative=5, hs=0, sample=.001, seed=42, wv=None)
        client = self.fake()
        with TemporaryDirectory() as directory, ExitStack() as stack:
            previous = Path.cwd()
            os.chdir(directory)
            stack.callback(os.chdir, previous)
            output = Path('data/processed')
            output.mkdir(parents=True)
            protocol = dict(source_sha256={}, artifact_sha256={},
                            diagnostics=dict(combined=dict(n_unique_papers_with_abstract=23)))
            (output / 'experiment_protocol.json').write_text(json.dumps(protocol))
            for name in ('experiment_population', 'tuning_queries', 'evaluation_queries'):
                pd.DataFrame(dict(query_id=[])).to_csv(output / f'{name}.csv', index=False)
            tables = dict(common=self.common, tfidf=self.tfidf, word2vec=self.tfidf)
            paths = [output / f'experiment_scores_{name}.csv' for name in tables]
            for (name, frame), path in zip(tables.items(), paths):
                frame.to_csv(path, index=False)
            hashes = {str(path): file_hash(path) for path in paths}
            stack.enter_context(patch('sys.argv', ['materialize', '--embeddings-only']))
            stack.enter_context(patch.object(module.Word2Vec, 'load', return_value=SimpleNamespace(**config)))
            stack.enter_context(patch.object(module, 'load_papers', return_value=self.papers))
            stack.enter_context(patch.object(module, 'load_citations', return_value=None))
            stack.enter_context(patch.object(module, 'build_citation_graph', return_value=self.graph))
            stack.enter_context(patch.object(module, 'frozen_plans', return_value=self.plans))
            stack.enter_context(patch('literature_recommender.experiment.prepare_reference_query',
                                      return_value=self.plans[0][1]))
            fake_context = Mock()
            fake_context.__enter__ = Mock(return_value=client)
            fake_context.__exit__ = Mock(return_value=False)
            stack.enter_context(patch.object(module, 'OpenAI', return_value=fake_context))
            for name in ('common_scores', 'fit_tfidf', 'tfidf_abstract_similarities',
                         'word2vec_abstract_similarities'):
                stack.enter_context(patch.object(module, name, side_effect=AssertionError('Local regeneration')))
            # Two synthetic queries intentionally fail only the final 100+100 guard.
            with self.assertRaisesRegex(ValueError, 'Queries disappeared'):
                module.main()
            self.assertEqual(embedding_inventory(needed_texts(self.plans), output / 'embeddings')['n_already_available'], 23)
            verify_hashes(hashes)

    def test_frozen_reconstruction_and_diagnostics_without_mutation(self):
        ids = [f'Q{i}' for i in range(200)]
        papers = pd.concat([self.papers, pd.DataFrame(dict(
            paper_id=ids, title=['alpha beta'] * 200, abstract=['alpha beta'] * 200),
            dtype='string')], ignore_index=True)
        graph = self.graph.copy()
        graph.add_edges_from((q, 'C1') for q in ids)
        rows = [dict(query_id=q, eligible=True, n_ground_truth=1,
                     n_structural_candidates=20, n_eligible_candidates=20,
                     n_candidates_with_abstract=20, n_candidates_without_abstract=0,
                     missing_abstract_fraction=0.) for q in ids]
        population = pd.DataFrame(rows)
        splits = dict(tuning=population.iloc[:100].copy(), evaluation=population.iloc[100:].copy())
        protocol = dict(seed=42, n_tuning=100, n_evaluation=100,
                        candidate_min=20, candidate_max=None, eligible_population_size=200,
                        tuning_query_ids=ids[:100], evaluation_query_ids=ids[100:],
                        population_diagnostics=dict(n_articles=len(papers), n_graph_nodes=len(graph)),
                        diagnostics={s: dict(sum_eligible_candidates=2000,
                                             n_candidates_without_abstract=0)
                                     for s in splits})
        original = deepcopy(protocol)
        before = population.copy(deep=True)
        plans = frozen_plans(papers, graph, protocol, population, splits)
        self.assertEqual(len(plans), 200)
        self.assertEqual(protocol, original)
        assert_frame_equal(before, population)
        splits['tuning'].loc[0, 'n_ground_truth'] = 2
        with self.assertRaisesRegex(ValueError, 'diagnostic'):
            frozen_plans(papers, graph, protocol, population, splits)


if __name__ == '__main__':
    unittest.main()
