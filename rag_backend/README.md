# Evidence-Grounded RAG Fact Verification

## Components

- Tavily live web retrieval
- BM25 sparse retrieval
- MiniLM dense embeddings
- Cross-encoder reranking
- Source credibility and temporal scoring
- Qwen2.5-1.5B-Instruct verification
- Five-class fact-checking verdicts

## Files

- `rag_pipeline.py`: Exported RAG implementation
- `rag_pipeline_snapshot.py`: Original notebook function snapshot
- `dependency_report.json`: Dependency audit
- `example_result.json`: Example live verification result
- `module_integration_test.json`: Module execution test
- `requirements-rag.txt`: Python dependencies

## Initialization

```python
import os
import rag_pipeline

rag_pipeline.initialize_pipeline(
    tavily_api_key=os.environ["TAVILY_API_KEY"]
)

result = rag_pipeline.verify_claim_live(
    "The Eiffel Tower is located in Berlin, Germany."
)

print(result["verdict"])
```

## Important deployment limitations

The Qwen model currently runs locally using PyTorch.
This may exceed the memory available on standard
Streamlit Community Cloud instances.

The module has been tested using models already loaded
in Kaggle. Fresh initialization in a separate environment
has not yet been tested.

Citation IDs are checked against retrieved passages,
but source relevance is not fully validated.

Do not interpret pilot accuracy as independent
production performance.

## Integration

Add this module to the existing GitHub repository.
Do not replace the existing Streamlit application
until the live inference backend is configured.
