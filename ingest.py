import json
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

RAW = Path("data/raw_pg")
OUT = Path("data/chunks_pg.jsonl")
BASE_URL = "https://www.postgresql.org/docs/18/"

splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

n = 0
with OUT.open("w") as f:
    for page in sorted(RAW.glob("*.html")):
        soup = BeautifulSoup(page.read_text(errors="ignore"), "lxml")
        title = soup.title.get_text(strip=True) if soup.title else page.stem
        body = soup.select_one("#docContent") or soup.body
        if body is None:
            continue
        text = body.get_text("\n", strip=True)
        for piece in splitter.split_text(text):
            f.write(json.dumps({
                "text": f"{title}\n\n{piece}",
                "title": title,
                "url": BASE_URL + page.name,
            }) + "\n")
            n += 1
print(n, "chunks written to", OUT)