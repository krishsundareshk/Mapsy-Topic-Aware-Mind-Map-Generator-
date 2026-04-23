# Mapsy — Topic-Aware Mind Map Generator

Mapsy is an NLP pipeline that automatically generates structured, topic-aware mind maps from raw text documents. Given an input document, it extracts key themes via topic modelling, identifies concept hierarchies, and renders an interactive HTML mind map.

## How It Works


```text
Input Text
    │
    ▼
Phase 1 — Preprocessing
    │  Tokenisation, lemmatisation, stopword removal
    ▼
Phase 2 — Thematic Clustering
    │  LDA topic modelling + BERTopic for semantic grouping
    ▼
Phase 3 — Concept Extraction
    │  BERT embeddings + SPO (Subject-Predicate-Object) parsing
    ▼
Phase 4 — Hierarchy Building
    │  K-Means / BIRCH / DBSCAN clustering for node grouping
    ▼
Phase 5 — Visualisation
    │  Interactive HTML mind map output
    ▼
mindmap.html
```

## Project Structure
```text
Mapsy-Topic-Aware-Mind-Map-Generator/
│
├── pipeline/                        # Core NLP pipeline modules
│   │
│   ├── phase1_preprocessing.py      # Text cleaning, tokenisation, lemmatisation
│   │
│   ├── phase2_thematic_clustering.py       # Main thematic clustering entry point
│   ├── phase2_thematic_clustering_lda.py   # LDA-specific clustering logic
│   ├── lda_topic_modelling.py              # LDA model training and inference
│   ├── coherence_optimisation.py           # LDA coherence score tuning
│   ├── optimal_k_selection.py              # Automatic topic count selection
│   │
│   ├── phase3_bert.py               # BERT sentence embeddings
│   ├── phase3_concept_extraction.py # SPO triple extraction and concept scoring
│   ├── node_lemmatizer.py           # Node-level lemmatisation for mind map nodes
│   ├── tfidf_cosine_scoring.py      # TF-IDF relevance scoring
│   │
│   ├── phase4_kmeans.py             # K-Means clustering for node grouping
│   ├── phase4_birch.py              # BIRCH clustering (scalable alternative)
│   ├── phase4_dbscan.py             # DBSCAN clustering (density-based)
│   ├── phase4_hierarchy.py          # Hierarchical tree construction
│   │
│   ├── phase5_visualisation.py      # Mind map rendering to HTML
│   │
│   ├── bertopic_modelling.py        # BERTopic integration
│   ├── dominance_threshold.py       # Topic dominance filtering
│   └── evaluate_gt.py               # Ground truth evaluation (ROUGE metrics)
│
├── main.py                          # Pipeline entry point
│
├── input.txt                        # Sample input document
├── document.txt                     # Alternate input document
├── gt_sentences.txt                 # Ground truth sentences for evaluation
├── gt_mindmap.json                  # Ground truth mind map structure (JSON)
│
├── mindmap.html                     # Generated mind map output
├── mindmap_sample.html              # Sample output for reference
│
└── requirements.txt                 # Python dependencies
```

## Quickstart
1. Clone the repository
bash
git clone https://github.com/your-username/Mapsy-Topic-Aware-Mind-Map-Generator.git
cd Mapsy-Topic-Aware-Mind-Map-Generator
2. Install dependencies
bash
pip install -r requirements.txt
3. Run the pipeline
bash
python main.py --input input.txt
The output mind map will be saved as mindmap.html. Open it in any browser.

### Optional flags
bash
python main.py --input input.txt --method lda      # Use LDA for topic modelling (default)
python main.py --input input.txt --method bertopic  # Use BERTopic instead
python main.py --input input.txt --cluster kmeans   # Clustering algorithm (kmeans/birch/dbscan)

## Pipeline Phases

### Phase 1 — Preprocessing
Cleans the raw input text: lowercasing, punctuation removal, tokenisation, stopword filtering, and lemmatisation using node_lemmatizer.py.

### Phase 2 — Thematic Clustering
Extracts high-level themes from the document using LDA (Latent Dirichlet Allocation). Coherence scores are optimised automatically to find the ideal number of topics. Alternatively, BERTopic can be used for transformer-based topic discovery.

### Phase 3 — Concept Extraction
Uses BERT sentence embeddings to represent each sentence semantically. SPO (Subject-Predicate-Object) triples are extracted to identify key relationships. TF-IDF cosine scoring ranks the most relevant concepts per theme.

### Phase 4 — Hierarchy Building
Extracted concepts are grouped into a tree structure using clustering algorithms:

K-Means — fast, works well for balanced clusters

BIRCH — scalable, suitable for larger documents

DBSCAN — density-based, handles irregular cluster shapes

The phase4_hierarchy.py module assembles the final parent-child node tree.

### Phase 5 — Visualisation
The hierarchy is rendered as an interactive HTML mind map using a force-directed or radial layout. The output (mindmap.html) is self-contained and opens in any browser without additional dependencies.

### Evaluation
Ground truth evaluation is supported via evaluate_gt.py. Place your reference mind map in gt_mindmap.json and reference sentences in gt_sentences.txt, then run:

```bash
python pipeline/evaluate_gt.py
```
Evaluation uses ROUGE-1, ROUGE-2, and ROUGE-L scores to compare generated output against the ground truth.

Dependencies
Key libraries used:
```text
Library	Purpose
nltk / spaCy	Tokenisation, lemmatisation, POS tagging
gensim	LDA topic modelling
bertopic	Transformer-based topic modelling
sentence-transformers	BERT sentence embeddings
scikit-learn	K-Means, BIRCH, DBSCAN clustering
rouge-score	ROUGE evaluation metrics
Install all dependencies via:
```
```bash
pip install -r requirements.txt
```