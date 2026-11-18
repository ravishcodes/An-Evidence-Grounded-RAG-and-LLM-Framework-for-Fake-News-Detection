"""Honest reproducible evaluation helpers. No fabricated performance metrics."""
import json
import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LABELS = ['TRUE','FALSE','PARTIALLY_TRUE','MISLEADING','UNVERIFIED']

def evaluate(df):
    required={'actual_label','predicted_label'}
    if not required.issubset(df):
        raise ValueError('Required columns: actual_label, predicted_label')
    if df.empty:
        raise ValueError('Evaluation data is empty')
    actual=df.actual_label.astype(str).str.strip().str.upper()
    predicted=df.predicted_label.astype(str).str.strip().str.upper()
    if not (actual.isin(LABELS)&predicted.isin(LABELS)).all():
        raise ValueError('Invalid or missing labels in evaluation CSV')
    report=classification_report(actual,predicted,labels=LABELS,output_dict=True,zero_division=0)
    return {'n':len(df),'accuracy':float(accuracy_score(actual,predicted)),
            'macro_f1':report['macro avg']['f1-score'],
            'macro_precision':report['macro avg']['precision'],
            'macro_recall':report['macro avg']['recall'],
            'abstention_rate':float((predicted=='UNVERIFIED').mean()),
            'confusion_matrix':confusion_matrix(actual,predicted,labels=LABELS).tolist(),
            'labels':LABELS,'classification_report':report}

def citation_checks(result):
    evidence={e.get('id'):e for e in result.get('evidence',[]) if isinstance(e,dict)}
    audit={a.get('id'):a for a in result.get('audit',[]) if isinstance(a,dict)}
    citations=result.get('citations',[])
    valid_ids=[c for c in citations if c in evidence]
    exact_quotes=[c for c in valid_ids if c in audit and len(audit[c].get('quote',''))>=18 and
                  audit[c]['quote'].casefold() in evidence[c].get('snippet','').casefold()]
    stance={'TRUE':'SUPPORTS','FALSE':'REFUTES'}.get(result.get('verdict'))
    aligned=[c for c in exact_quotes if stance and audit[c].get('stance')==stance]
    return {'citations':len(citations),'valid_id_count':len(valid_ids),
            'verbatim_quote_count':len(exact_quotes),'aligned_quote_count':len(aligned),
            'citation_id_validity':len(valid_ids)/len(citations) if citations else None,
            'verbatim_quote_rate':len(exact_quotes)/len(citations) if citations else None,
            'automated_stance_alignment':len(aligned)/len(citations) if citations and stance else None,
            'note':'Automated checks are not human-assessed factual faithfulness.'}

def summarize_runs(rows):
    df=pd.DataFrame(rows)
    metrics=evaluate(df) if len(df) else None
    checks=[citation_checks(json.loads(x)) for x in df.result_json] if len(df) else []
    for key in ('citation_id_validity','verbatim_quote_rate','automated_stance_alignment'):
        vals=[x[key] for x in checks if x[key] is not None]
        if metrics is not None:
            metrics[key+'_mean']=float(np.mean(vals)) if vals else None
    return df,metrics
