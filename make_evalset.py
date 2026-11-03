import json
import random

from llm import chat

random.seed(42)

chunks = [json.loads(l) for l in open("data/chunks_pg.jsonl")]
chunks = [c for c in chunks if len(c["text"]) > 600]
sample = random.sample(chunks, 70)   # extra, since you'll delete weak ones

SYSTEM = "You write evaluation questions for a documentation search system. Return JSON only."
PROMPT = """Below is a passage from the PostgreSQL docs.
Write ONE realistic question a developer might ask that this passage answers.
Do not mention "the passage" or "the text" in the question.
Also write a short, correct answer using only the passage.
Return JSON: {{"question": "...", "ground_truth": "..."}}

Passage:
{text}"""

out = []
for c in sample:
    try:
        qa = json.loads(chat(SYSTEM, PROMPT.format(text=c["text"]), json_mode=True))
        assert qa["question"] and qa["ground_truth"]
    except Exception:
        continue
    out.append({**qa, "gold_url": c["url"], "answerable": True})

# Unanswerable questions, used to test refusals
REFUSE = "I couldn't find this in the documentation."
for q in [
    "How do I configure Redis replication?",
    "What is the best pizza topping?",
    "How do I set up MongoDB sharding?",
    "Who won the 2018 FIFA World Cup?",
    "How do I deploy a Kubernetes cluster?",
    "What is the capital of Australia?",
]:
    out.append({"question": q, "ground_truth": REFUSE, "gold_url": None, "answerable": False})

with open("eval/evalset.jsonl", "w") as f:
    for row in out:
        f.write(json.dumps(row) + "\n")
print(len(out), "questions written to eval/evalset.jsonl")