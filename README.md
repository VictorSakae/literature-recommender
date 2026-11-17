# literature-recommender

Projeto acadêmico de recomendação de artigos científicos, baseado no trabalho de
[Liu et al. (2025)](https://doi.org/10.1007/s11192-025-05420-0). O método combina
similaridades de títulos, resumos e citações para gerar rankings de artigos
relacionados.

A implementação reproduz parcialmente o método original. Os experimentos
comparam três representações dos resumos — embeddings, TF-IDF e Word2Vec —
mantendo os componentes de similaridade de títulos e citações no score híbrido.
A substituição da representação dos resumos é uma adaptação experimental do
projeto e difere dos baselines textuais apresentados no artigo.

## Instalação

Instale Python 3.11 e [uv](https://docs.astral.sh/uv/). Na raiz do projeto, execute:

```sh
uv sync --locked
```

Todos os comandos devem ser executados na raiz do repositório.

## Testes

```sh
uv run python -m unittest discover -s tests -v
```

Os testes utilizam dados sintéticos e simulam as chamadas à API. Após a
instalação das dependências, a execução não requer o conjunto de dados original,
credenciais ou acesso à internet.

## Dados e exploração

Obtenha o conjunto de dados no [repositório dos autores](https://github.com/Gaotianchen97/Academic-Literature-Recommendation-in-Large-scale-Citation-Networks)
e armazene os arquivos na seguinte estrutura de diretórios:

```text
data/raw/Academic-Literature-Recommendation-in-Large-scale-Citation-Networks-main/
├── article_info.xlsx
└── citation_network.csv
```

Os arquivos em `data/raw/` devem permanecer inalterados. Na rede de citações,
uma aresta direcionada de A para B indica que A cita B. Dados, modelos e
resultados experimentais devem ser obtidos ou gerados localmente.

Para explorar os dados, abra `notebooks/preliminary_evaluation.ipynb` em uma IDE
com suporte a Jupyter, selecione o interpretador Python de `.venv` como kernel
e execute as células em sequência. O notebook analisa a elegibilidade dos
artigos e os conjuntos de candidatos, registrando contagens em
`data/processed/preliminary_diagnostic.csv`. Essa execução não realiza chamadas
à API nem calcula métricas de ranking.

## Experimentos

**A execução dos experimentos requer artefatos locais não distribuídos com o
código.** O conjunto de dados original permite executar a exploração descrita
acima. A reprodução completa depende também de resultados preliminares, modelos
e arquivos de configuração do experimento. Não há um comando único para gerar
todos esses pré-requisitos a partir de uma instalação inicial.

Os scripts de avaliação preliminar requerem o ranking e o arquivo de metadados
de uma execução anterior com embeddings:
`data/processed/preliminary_hybrid_paper_84898.csv` e
`data/processed/preliminary_hybrid_paper_84898_summary.json`. Com esses arquivos
disponíveis, execute as avaliações com TF-IDF e Word2Vec, nessa ordem:

```sh
uv run python scripts/preliminary_tfidf.py
PYTHONHASHSEED=42 uv run python scripts/preliminary_word2vec.py
```

A execução com Word2Vec também persiste o modelo treinado em
`data/processed/models/word2vec_cbow_100_seed42.model`.

Para definir os conjuntos de artigos utilizados no ajuste de pesos e na
avaliação, a execução requer o modelo Word2Vec e uma cópia do artigo publicado em
`docs/papers/2025_Liu_AcademicLiteratureRecommendation.pdf`.
O primeiro comando apresenta o diagnóstico de elegibilidade e amostragem.
A opção `--freeze` persiste o protocolo e fixa os conjuntos para as etapas
seguintes:

```sh
uv run python scripts/freeze_experiment_protocol.py
uv run python scripts/freeze_experiment_protocol.py --freeze
```

Com o protocolo, o modelo e os resultados preliminares disponíveis, calcule as
similaridades de títulos, citações e resumos com TF-IDF e Word2Vec:

```sh
uv run python scripts/materialize_experiment_scores.py --local-only
```

Para calcular as similaridades de resumos com embeddings, configure a variável
de ambiente `OPENAI_API_KEY` e execute:

```sh
uv run python scripts/materialize_experiment_scores.py --embeddings-only
```

Essa etapa envia resumos à API da OpenAI para gerar vetores com o modelo
`text-embedding-3-small`, com custos de utilização da API. Mantenha a credencial
fora do código e do controle de versão. O script lê a variável de ambiente;
arquivos `.env` não são carregados automaticamente.

Após concluir o cálculo das similaridades, execute a busca dos pesos, a
avaliação com os pesos fixados e a análise dos resultados, nessa ordem:

```sh
uv run python scripts/grid_search_weights.py
uv run python scripts/evaluate_frozen_experiment.py
uv run python scripts/analyze_final_results.py
```

Os resultados e gráficos são gravados em `data/processed/`. As três últimas
etapas são executadas localmente, sem chamadas à API. Os scripts validam os
hashes dos artefatos em relação ao experimento original; arquivos de uma nova
execução podem ser incompatíveis com essas verificações. Vários scripts impedem
a sobrescrita de resultados existentes. Preserve os artefatos anteriores antes
de repetir uma etapa.

## Referência

Liu, K., Zhang, Y., Pan, R., Gao, T., & Wang, H. (2025).
*Academic literature recommendation in large-scale citation networks enhanced by
large language models*. Scientometrics, 130, 5143–5169.
[Artigo publicado](https://doi.org/10.1007/s11192-025-05420-0) ·
[Preprint de acesso aberto](https://arxiv.org/abs/2503.01189).

Ao utilizar o método ou os dados dos autores, cite o artigo acima.

## Licença

O código original deste projeto é distribuído sob a [licença MIT](LICENSE).
O artigo, os dados dos autores e as dependências de terceiros não são
relicenciados por este projeto; seu uso deve respeitar as respectivas licenças
e condições de utilização. A citação do artigo não substitui essas condições.
