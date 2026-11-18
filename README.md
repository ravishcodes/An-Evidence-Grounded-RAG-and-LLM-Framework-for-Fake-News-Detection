# An Evidence-Grounded RAG and LLM Framework for Fake News Detection

**An AI-driven research project for explainable misinformation detection using Natural Language Processing (NLP), Retrieval-Augmented Generation (RAG), Large Language Models (LLMs), and evidence-based fact verification.**

## Overview

The rapid spread of misinformation across digital platforms presents significant challenges to information reliability and public trust. Traditional machine learning models often classify news based only on linguistic patterns, without verifying factual claims against external evidence.

This project investigates an **Evidence-Grounded and Temporally Aware RAG Framework** that combines machine learning, information retrieval, and large language models to support explainable fake news detection.

The proposed framework aims to retrieve relevant evidence, assess source credibility and temporal validity, and generate interpretable verdicts supported by verifiable sources.

**Current status:** Research prototype under active development. The deployed Streamlit application currently supports claim-only classification and lookup of previously retrieved evidence. Full live RAG verification is not yet integrated.

## Research Objectives

1. Develop a fake news detection system using NLP and machine learning.
2. Implement sparse, dense, and hybrid evidence retrieval.
3. Incorporate source credibility and temporal relevance into evidence ranking.
4. Integrate LLMs for evidence-grounded claim verification.
5. Generate explainable predictions with source references.
6. Investigate uncertainty handling and abstention for insufficient evidence.
7. Compare traditional machine learning, Transformer, and RAG-based approaches.

## System Architecture

The intended end-to-end architecture is:

```text
                 User News Claim
                        |
                        v
            Claim Preprocessing
                        |
                        v
           Evidence Retrieval
           /                \
       Sparse BM25       Dense Embeddings
           \                /
            Hybrid Retrieval
                        |
                        v
          Cross-Encoder Reranking
                        |
                        v
        Source Credibility Assessment
                        |
                        v
         Temporal Relevance Analysis
                        |
                        v
             RAG + LLM Verification
                        |
                        v
          Explainable Verdict Output
                        |
                        v
           Streamlit Web Interface
```

This diagram represents the target architecture. Individual retrieval and RAG components have been explored experimentally, but the complete pipeline is not yet connected to the deployed application.

## Technologies Used

| Component | Technology |
|---|---|
| Programming language | Python |
| Dataset | LIAR |
| Traditional ML | Logistic Regression, Linear SVM |
| Text features | TF-IDF |
| Transformer | DeBERTa-v3-small |
| Sparse retrieval | BM25 |
| Dense retrieval | Sentence Transformers, all-MiniLM-L6-v2 |
| Reranking | Cross-Encoder |
| Language model | Qwen2.5-1.5B-Instruct |
| Evidence sources | Wikipedia, Tavily search experiments |
| Frontend | Streamlit |
| Development | Kaggle Notebooks |
| Version control | Git and GitHub |

## Dataset

The research uses the **LIAR dataset**, which contains political statements with six truthfulness categories.

The current experimental subset focuses on health-related claims.

| Dataset split | Samples |
|---|---:|
| Training | 1,256 |
| Validation | 182 |
| Testing | 143 |
| **Total** | **1,581** |

### Six-class labels

- Pants on Fire
- False
- Barely True
- Half True
- Mostly True
- True

For binary classification, the first three categories are mapped to **False**, while the remaining three
