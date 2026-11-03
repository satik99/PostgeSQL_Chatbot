import sys
import types

# Workaround: ragas imports a module that newer langchain-community removed
_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_stub.ChatVertexAI = object
sys.modules["langchain_community.chat_models.vertexai"] = _stub

import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy
from ragas.run_config import RunConfig


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy
from ragas.run_config import RunConfig

JUDGE_MODEL = "gpt-4o-mini"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

name = sys.argv[1] if len(sys.argv) > 1 else "baseline_dense"
limit = int(sys.argv[2]) if len(sys.argv) > 2 else None   # optional: score only N questions

all_rows = json.load(open(f"eval/results_{name}.json"))

rows = [r for r in all_rows if r["answerable"]]
if limit:
    rows = rows[:limit]

unans = [r for r in all_rows if not r["answerable"]]
if unans:
    refused = sum("couldn't find" in r["answer"].lower() for r in unans)
    print(f"Refusal rate on unanswerable questions: {refused}/{len(unans)}")

dataset = EvaluationDataset.from_list([
    {
        "user_input": r["question"],
        "response": r["answer"],
        "retrieved_contexts": r["contexts"],
        "reference": r["ground_truth"],
    }
    for r in rows
])

llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0))
emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBED_MODEL))

result = evaluate(
    dataset,
    metrics=[Faithfulness(), ResponseRelevancy(), LLMContextRecall()],
    llm=llm,
    embeddings=emb,
    run_config=RunConfig(timeout=120, max_workers=8),
)

df = result.to_pandas()
df.to_csv(f"eval/ragas_{name}.csv", index=False)

print(f"\n=== {name} ({len(rows)} questions) ===")
for col in df.columns:
    if df[col].dtype.kind in "fi":
        print(f"{col}: {df[col].mean():.3f}  (NaN rows: {int(df[col].isna().sum())})")
print(f"\nPer-question scores saved to eval/ragas_{name}.csv")