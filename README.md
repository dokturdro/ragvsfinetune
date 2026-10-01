# RAG vs Fine-Tuning

A side-by-side experiment that teaches **Qwen2.5-7B-Instruct** a niche knowledge domain in two ways: by **retrieving** context at query time (RAG), or by **training** the knowledge into the weights (LoRA / DoRA fine-tuning). The outputs are then scored against a ground-truth answer.

<a href="https://colab.research.google.com/github/dokturdro/ragvsfinetune/blob/main/ragvsfinetune.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

---

## Pipeline

```mermaid
flowchart LR
    KB["supplements_data.txt<br/>(supplement guide)"]
    QA["4 instruction / output<br/>Q&A pairs"]
    BASE["Qwen2.5-7B-Instruct<br/>4-bit (Unsloth)"]

    subgraph RAG["Retrieval-Augmented Generation"]
        direction TB
        SPLIT["Markdown header split +<br/>RecursiveCharacterTextSplitter<br/>chunk 600 / overlap 100"]
        EMB["all-MiniLM-L6-v2<br/>embeddings"]
        FAISS[("FAISS index")]
        BASIC["Basic RAG<br/>top-3 similarity"]
        TOP10["top-10 similarity"]
        RERANK["CrossEncoder rerank<br/>ms-marco-MiniLM-L-6-v2<br/>→ top-3"]
        SPLIT --> EMB --> FAISS
        FAISS --> BASIC
        FAISS --> TOP10 --> RERANK
    end

    subgraph FT["Parameter-Efficient Fine-Tuning"]
        direction TB
        SFT["TRL SFTTrainer<br/>60 steps · lr 2e-4"]
        LORA["LoRA adapter<br/>r=16 · attention only"]
        DORA["DoRA adapter<br/>r=16 · attention + MLP"]
        SFT --> LORA
        SFT --> DORA
    end

    KB --> SPLIT
    QA --> SFT
    BASE --> SFT

    BASIC -- "context + query" --> GEN["Generation"]
    RERANK -- "context + query" --> GEN
    LORA -- "query only" --> GEN
    DORA -- "query only" --> GEN
    BASE -.->|same base model| GEN

    GEN --> EVAL["Evaluation<br/>fact recall, ROUGE-L, semantic similarity<br/>+ Claude LLM judge"]
```

## Approaches compared

| Method | How knowledge reaches the model | Key settings |
|---|---|---|
| **Basic RAG** | Top-3 chunks from FAISS go into the prompt | MiniLM embeddings, `k=3` |
| **Advanced RAG** | Two-stage retrieval: top-10 vector hits, reranked by a cross-encoder, top-3 kept | `ms-marco-MiniLM-L-6-v2` reranker |
| **LoRA FT** | Low-rank adapters trained on Q&A pairs | `r=16`, `q/k/v/o_proj`, ~10.1M trainable params (0.13%) |
| **DoRA FT** | Weight-decomposed LoRA (magnitude + direction) | `r=16`, attention + `gate/up/down_proj`, ~41.8M trainable params (0.55%) |

The notebook also demonstrates **multi-query retrieval**: it searches several rephrasings of the query and de-duplicates the results. This retriever isn't used in the evaluation.

## Evaluation

Each method answers a held-out set of paraphrased and unseen-supplement questions. Answers are scored with key-fact recall, ROUGE-L F1, and embedding similarity. ROUGE-L rewards copying the reference wording, which favours the fine-tuned models, so Claude also grades every answer blind (it never sees the method name) for correctness, completeness, and, for RAG, faithfulness to the retrieved context. The judge cell needs an `ANTHROPIC_API_KEY` in Colab Secrets and is skipped without one.

## Results

> These numbers come from an earlier single-query version of the notebook. They predate the held-out eval set, the new chunking, and the LLM judge, so re-run the notebook for current figures.

Query: *"What are the primary benefits and forms of Magnesium?"*

| Method | ROUGE-L F1 | Semantic Similarity |
|---|---:|---:|
| Basic RAG | 0.3307 | 0.7691 |
| Advanced RAG | 0.2313 | 0.8313 |
| LoRA FT | 0.6341 | 0.9196 |
| DoRA FT | **0.9041** | **0.9562** |

Training summary (Tesla T4):

| Run | Final train loss | Runtime |
|---|---:|---:|
| LoRA | 0.020 | ~2 min |
| DoRA | 0.193 | ~10 min |
