import json
import os

import pandas as pd

RUNS = {"Dense": "baseline_dense", "+ Hybrid": "hybrid", "+ Reranker": "rerank"}

print("| Setup | hit@5 | Faithfulness | Answer relevancy | Context recall | Refusals |")
print("|---|---|---|---|---|---|")

for label, run in RUNS.items():
    path = f"eval/results_{run}.json"
    if not os.path.exists(path):
        print(f"| {label} | (not run) | | | | |")
        continue
    rows = json.load(open(path))

    ans = [r for r in rows if r["answerable"]]
    hit = sum(r["gold_url"] in r["retrieved_urls"] for r in ans) / len(ans)

    unans = [r for r in rows if not r["answerable"]]
    refused = sum("couldn't find" in r["answer"].lower() for r in unans)

    ragas = {}
    csv = f"eval/ragas_{run}.csv"
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        ragas = {c: df[c].mean() for c in df.columns if df[c].dtype.kind == "f"}

    def g(*names):
        for n in names:
            if n in ragas:
                return f"{ragas[n]:.2f}"
        return "-"

    print(f"| {label} | {hit:.2f} | {g('faithfulness')} | "
          f"{g('answer_relevancy', 'response_relevancy')} | "
          f"{g('context_recall', 'llm_context_recall')} | {refused}/{len(unans)} |")