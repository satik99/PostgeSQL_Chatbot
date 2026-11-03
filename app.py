import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("PostgreSQL Docs Assistant")
mode = st.sidebar.selectbox("Retrieval mode", ["rerank", "hybrid", "dense"])

q = st.text_input("Ask a question about PostgreSQL")
if q:
    with st.spinner("Searching..."):
        r = requests.post(
            f"{API_URL}/ask",
            json={"question": q, "mode": mode},
            timeout=180,
        ).json()
    st.write(r["answer"])
    st.subheader("Sources")
    for s in r["sources"]:
        st.markdown(f"[{s['n']}] [{s['url']}]({s['url']})")