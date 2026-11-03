# Tekstanalyse med Nasjonalbibliotekets modeller og dhlab

Installér avhengigheter med pdm: 
```bash
pdm install
```

## Skript og notebooks
Skriptene er nummerert i kjøre-rekkefølge.  
Disse må kjøres før man tar i bruk notebookene. 
Notebookene kjøres uavhengig av hverandre.

### Skript
- `01_last_ned_data.py`: Laster ned nynorsk-delen av [Målfrid 2026](https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-107/) fra Språkbanken og pakker den ut til `data/maalfrid/`.
- `02_pakk_ut_data.py`: Skriver hvert dokument til en egen tekstfil, `data/maalfrid/textfiles/<domene>/<type>_<uuid>.txt`.
- `03_lag_subsett.py`: Trekker 100 tilfeldige html- og doc-filer fra hvert av fem håndplukkede domener (fiskeridir.no, mattilsynet.no, uio.no, uit.no og varsom.no) til `data/maalfrid/subset/`. Mappen tømmes før subsettet skrives.

### Notebooks
- `dhlab_embeddings.ipynb`: Bag-of-words-embeddinger av subsettet, tokenisert med `nb_tokenizer` og vektet med ordfrekvenser fra dhlab (`totals`).
- `borealis_embeddings.ipynb`: Embeddinger av subsettet med [NbAiLab/borealis-embed-212m](https://huggingface.co/NbAiLab/borealis-embed-212m) via sentence-transformers.

Begge notebookene viser embeddingene i et interaktivt spredningsplott per domene og finner det mest og minst like domenet for hvert domene.

### Hjelpemoduler
- `visualize.py`: Projiserer embeddinger til 2D (truncated SVD + t-SNE) og lager det interaktive plottet.
- `embedding_similarity.py`: Regner ut domene-sentroider og cosinuslikhet mellom dem.
