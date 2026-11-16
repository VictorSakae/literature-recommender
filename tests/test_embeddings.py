"""Provider boundary tested with a fake client; no credentials or network."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from openai.types import CreateEmbeddingResponse

from literature_recommender.embeddings import embed_texts


def item(index, value=1.):
    return SimpleNamespace(index=index, embedding=[value] + [0.] * 1535)


class EmbedTextsTests(unittest.TestCase):
    def setUp(self):
        self.client = SimpleNamespace(embeddings=SimpleNamespace(create=Mock()))
        self.client.embeddings.create.return_value = SimpleNamespace(data=[item(0)])

    def test_model_batch_order_and_extraction_with_sdk_response(self):
        response = CreateEmbeddingResponse(
            data=[{"object": "embedding", "index": i, "embedding": item(i, v).embedding}
                  for i, v in ((1, 2.), (0, 1.))],
            model="text-embedding-3-small", object="list",
            usage={"prompt_tokens": 4, "total_tokens": 4},
        )
        self.client.embeddings.create.return_value = response
        texts = [" First abstract ", "Second abstract"]
        vectors = embed_texts(texts, client=self.client)
        self.client.embeddings.create.assert_called_once_with(
            input=texts, model="text-embedding-3-small", encoding_format="float",
        )
        self.assertEqual(vectors, [item(0).embedding, item(1, 2.).embedding])
        self.assertTrue(all(type(value) is float for vector in vectors for value in vector))
        vectors[0][0] = 99.
        self.assertEqual(response.data[1].embedding[0], 1.)
        self.assertEqual(texts, [" First abstract ", "Second abstract"])

    def test_reuses_supplied_client(self):
        embed_texts(["One"], client=self.client)
        embed_texts(["Two"], client=self.client)
        self.assertEqual(self.client.embeddings.create.call_count, 2)

    def test_empty_input_does_not_call_provider(self):
        self.assertEqual(embed_texts([], client=self.client), [])
        self.client.embeddings.create.assert_not_called()

    def test_invalid_inputs_do_not_call_provider(self):
        for texts, error in (("text", TypeError), ([None], TypeError),
                             (["ok", 1], TypeError), ([""], ValueError), ([" \n"], ValueError)):
            with self.subTest(texts=texts), self.assertRaises(error):
                embed_texts(texts, client=self.client)
        self.client.embeddings.create.assert_not_called()

    def test_provider_error_propagates_unchanged(self):
        error = RuntimeError("provider failure")
        self.client.embeddings.create.side_effect = error
        with self.assertRaises(RuntimeError) as raised:
            embed_texts(["One"], client=self.client)
        self.assertIs(raised.exception, error)

    def test_missing_data_and_wrong_cardinality(self):
        for response in (None, SimpleNamespace(), SimpleNamespace(data=None),
                         SimpleNamespace(data=[]), SimpleNamespace(data=[item(0), item(1)])):
            self.client.embeddings.create.return_value = response
            with self.subTest(response=response), self.assertRaises(ValueError):
                embed_texts(["One"], client=self.client)

    def test_bad_indices(self):
        for indices in ((0, 0), (-1, 1), (0, 2), (0, "1"), (0, True), (0, None)):
            self.client.embeddings.create.return_value = SimpleNamespace(
                data=[item(index) for index in indices],
            )
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                embed_texts(["One", "Two"], client=self.client)

    def test_missing_item_fields(self):
        for entry in (SimpleNamespace(embedding=item(0).embedding), SimpleNamespace(index=0)):
            self.client.embeddings.create.return_value = SimpleNamespace(data=[entry])
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                embed_texts(["One"], client=self.client)

    def test_invalid_vectors(self):
        invalid = [None, [], [1.], [0.] * 1536, [[1.]] * 1536]
        invalid += [[value] + [0.] * 1535 for value in (float("nan"), float("inf"), "1", 1j, True)]
        for vector in invalid:
            self.client.embeddings.create.return_value = SimpleNamespace(
                data=[SimpleNamespace(index=0, embedding=vector)],
            )
            with self.subTest(first=str(vector)[:20]), self.assertRaises(ValueError):
                embed_texts(["One"], client=self.client)


if __name__ == "__main__":
    unittest.main()
