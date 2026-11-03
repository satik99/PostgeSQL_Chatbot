import json
import sys

from generate import answer

name = sys.argv[1] if len(sys.argv) > 1 else "dense"
mode = name if name in ("dense", "hybrid", "rerank") else "dense"

rows = [json.loads(l) for l in open("eval/evalset.jsonl")]

results, hits, n_ans = [], 0, 0
for i, row in enumerate(rows, 1):
    r = answer(row["question"], mode=mode)
    urls = [s["url"] for s in r["sources"]]
    if row["answerable"]:
        n_ans += 1
        hits += row["gold_url"] in urls
    results.append({
        **row,
        "answer": r["answer"],
        "contexts": r["contexts"],
        "retrieved_urls": urls,
    })
    print(f"{i}/{len(rows)} done")

json.dump(results, open(f"eval/results_{name}.json", "w"), indent=1)
print(f"\n{name}: hit@5 = {hits}/{n_ans} = {hits / n_ans:.2f}")