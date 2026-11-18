import json
from html import escape

import streamlit as st

from workspace_api import ApiError, api, diagnostics, health, pages, saved_question_count
from workspace_ui import (
    METRICS, MODES, comparison_chart, conversation_message, empty, entry, error, heading,
    job_progress, load_style, metrics, pill, service_list, service_overview, service_states, size_label,
    stamp, status_pill,
)

NAVIGATION = {
    "Workspace": {
        "dashboard": ("Dashboard", "space_dashboard"),
        "documents": ("Documents", "folder_open"),
        "upload": ("Upload Documents", "upload_file"),
        "chat": ("AI Chat", "forum"),
        "conversations": ("Conversations", "history"),
    },
    "Intelligence": {"analytics": ("Retrieval Analytics", "monitoring")},
    "System": {"settings": ("Settings", "settings")},
}
PROMPTS = ("Summarize the key points in my documents.",
           "What deadlines or requirements are mentioned?",
           "Which passages support the main conclusions?")


def navigate(page):
    st.query_params["page"] = page


def start_chat():
    st.session_state.pop("conversation_id", None)
    navigate("chat")


def reopen_chat(identifier):
    st.session_state["conversation_id"] = identifier
    navigate("chat")


def sidebar(page):
    live, ready = health()
    with st.sidebar:
        st.html('<div class="dm-brand"><span class="dm-mark" aria-hidden="true"></span><span class="dm-brand-name">DocuMind AI</span></div>')
        for group, items in NAVIGATION.items():
            st.html(f'<div class="dm-nav-label">{group.upper()}</div>')
            for key, (label, icon) in items.items():
                st.button(label, icon=f":material/{icon}:", key=f"nav_{key}", width="stretch",
                          type="primary" if key == page else "tertiary", on_click=navigate, args=(key,))
        label, tone = ("System operational", "success") if ready else (
            ("Database unavailable", "error") if live else ("Backend unavailable", "error")
        )
        st.html(f'<div class="dm-sidebar-status">{pill(label, tone)}</div>')


def latest_retrieval(report):
    if report:
        mode, data = (
            ("hybrid_rerank", report["modes"]["hybrid_rerank"])
            if "hybrid_rerank" in report["modes"] else next(iter(report["modes"].items()))
        )
        score = data["averages"].get("recall_at_k")
        if score is not None:
            return f"{score:.0%}", f"K={report['k']} · {MODES.get(mode, mode)}"
    return "—", "Run an evaluation to measure"


def dashboard():
    with st.container(key="dashboard_hero"):
        st.html(
            '<div class="dm-hero"><h1>Document intelligence</h1>'
            '<p>Search your library, ask questions grounded in your documents, and review retrieval quality.</p></div>'
        )
        left, right, _ = st.columns([1, 1, 2])
        left.button("Upload documents", type="primary", icon=":material/upload:", width="stretch",
                    on_click=navigate, args=("upload",))
        right.button("Start a chat", icon=":material/add_comment:", width="stretch", on_click=start_chat)

    live, ready = health()
    if not ready:
        st.subheader("System health", anchor=False)
        service_list(service_states(live, ready, {}))
        return
    result = api("GET", "/dashboard")
    conversations = list(pages("/conversations"))
    recent_runs = api("GET", "/evaluations", params={"limit": 1})
    recall, recall_note = latest_retrieval(recent_runs[0]["report"] if recent_runs else None)
    questions = saved_question_count(tuple(row["id"] for row in conversations))
    metrics([
        ("Indexed documents", f"{result['ready_documents']:,}", f"{result['total_documents']:,} in library"),
        ("Indexed chunks", f"{result['indexed_chunks']:,}", "Searchable passages"),
        ("Questions asked", f"{questions:,}", "Across saved conversations"),
        ("Latest Recall@K", recall, recall_note),
    ])
    left, right = st.columns([1.15, 1])
    with left.container(border=True):
        st.subheader("Recent documents", anchor=False)
        if not result["recent"]:
            empty("The library is empty", "Upload a document to start indexing your workspace.")
        for row in result["recent"][:4]:
            entry(row["extension"].lstrip(".").upper(), row["filename"],
                  f"{stamp(row['updated_at'], compact=True)} · {row['chunk_count']:,} chunks", status_pill(row["status"]))
        st.button("Open library", icon=":material/arrow_forward:", on_click=navigate, args=("documents",))
    with right.container(border=True):
        st.subheader("Recent conversations", anchor=False)
        if not conversations:
            empty("No conversations yet", "Ask a question to start a saved conversation.")
        for row in conversations[:4]:
            entry("CHAT", row["title"], stamp(row["updated_at"]))
        st.button("View conversations", icon=":material/arrow_forward:", on_click=navigate, args=("conversations",))
    with st.container(border=True):
        st.subheader("System health", anchor=False)
        values = st.session_state.get("external_diagnostics") or diagnostics(False)
        service_overview(service_states(live, ready, values))
        st.button("Open system settings", icon=":material/arrow_forward:", on_click=navigate, args=("settings",))


def window(items, key, per_page=15):
    if len(items) <= per_page:
        return items
    count = (len(items) + per_page - 1) // per_page
    selected = st.number_input(f"Page (of {count})", min_value=1, max_value=count,
                               value=1, step=1, key=key)
    start = (selected - 1) * per_page
    return items[start:start + per_page]


def document_entry(row):
    cols = st.columns([5, 1], vertical_alignment="center")
    with cols[0]:
        entry(row["extension"].lstrip(".").upper(), row["filename"],
              f"{stamp(row['created_at'])} · {size_label(row['size_bytes'])} · {row['chunk_count']:,} chunks",
              status_pill(row["status"]))
        if row["status"] in {"queued", "processing"}:
            st.caption("Waiting to start" if row["status"] == "queued" else "Extracting text and building the search index")
        if row["error"]:
            st.text(row["error"])
    if row["status"] != "deleting":
        with cols[1].popover("Manage", icon=":material/more_horiz:", width="stretch"):
            if row["status"] == "failed" and st.button("Retry indexing", key=f"retry_{row['id']}"):
                api("POST", f"/documents/{row['id']}/retry")
                st.rerun()
            if st.checkbox("Confirm removal", key=f"confirm_{row['id']}") and st.button(
                "Remove document", key=f"remove_{row['id']}", type="primary", icon=":material/delete:"
            ):
                api("DELETE", f"/documents/{row['id']}")
                st.rerun()


@st.fragment(run_every=6)
def library_content():
    search, filters = st.columns([3, 2])
    term = search.text_input("Search documents", placeholder="Search by filename")
    kinds = filters.multiselect("File type", ["PDF", "DOCX", "TXT"], placeholder="All file types")
    try:
        rows = [row for row in pages("/documents")
                if term.strip().casefold() in row["filename"].casefold()
                and (not kinds or row["extension"].lstrip(".").upper() in kinds)]
        st.caption(f"{len(rows):,} matching documents")
        if not rows:
            empty("No matching documents", "Try a different search or upload a document.")
        else:
            with st.container(border=True):
                for row in window(rows, "library_page"):
                    document_entry(row)
    except ApiError as exc:
        error(exc)


def library():
    heading("Document Library", "Find, track, and manage the documents in your search index.")
    st.button("Upload documents", icon=":material/upload:", type="primary", on_click=navigate, args=("upload",))
    library_content()


def upload_batch(files, settings):
    limit = settings["max_upload_size_mb"] * 1024**2
    valid = []
    for file in files:
        if file.size > limit:
            error(f"{file.name} exceeds the per-file size limit.")
        else:
            valid.append(file)
    groups, batch, bytes_in_batch = [], [], 0
    for file in valid:
        if len(batch) == settings["max_upload_files"] or bytes_in_batch + file.size > 2 * limit:
            groups.append(batch)
            batch, bytes_in_batch = [], 0
        batch.append(file)
        bytes_in_batch += file.size
    if batch:
        groups.append(batch)
    if not groups:
        return
    progress = st.progress(0.0, text="Sending files to the upload queue…")
    tracking = st.session_state.setdefault("upload_jobs", {})
    for index, group in enumerate(groups, 1):
        try:
            results = api("POST", "/documents", timeout=120, files=[
                ("files", (file.name, file.getvalue(), file.type or "application/octet-stream")) for file in group
            ])
            for result in results:
                if result["error"]:
                    error(f"{result['filename']} · {result['error']}")
                else:
                    row = result["document"]
                    tracking[row["id"]] = result["duplicate"]
        except ApiError as exc:
            error(exc)
        progress.progress(index / len(groups), text=f"Upload batches handled: {index} of {len(groups)}")


@st.fragment(run_every=4)
def track_uploads():
    jobs = st.session_state.get("upload_jobs", {})
    if not jobs:
        return
    with st.container(border=True):
        st.subheader("Upload activity", anchor=False)
        for identifier, duplicate in reversed(list(jobs.items())):
            try:
                row = api("GET", f"/documents/{identifier}")
            except ApiError as exc:
                if exc.status != 404:
                    error(exc)
                continue
            status = row["status"]
            stage = {
                "queued": "Queued · waiting for the document worker",
                "processing": "Extracting text and indexing passages",
                "ready": f"Complete · {row['chunk_count']:,} chunks indexed",
            }.get(status, row["error"] or "Document is being removed")
            if duplicate:
                stage = f"Already in library · {stage}"
            entry(row["extension"].lstrip(".").upper(), row["filename"], stage, status_pill(status))
            job_progress(status)


def upload_documents():
    heading("Upload Documents", "Turn source files into searchable passages with document and page references.")
    settings = diagnostics(False)
    with st.container(border=True):
        st.subheader("Add documents", anchor=False)
        st.caption(f"PDF, DOCX, and TXT · up to {settings['max_upload_size_mb']} MiB per file")
        files = st.file_uploader("Choose documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        if st.button("Upload and index", type="primary", icon=":material/upload:", disabled=not files):
            upload_batch(files, settings)
    track_uploads()
    st.button("Go to document library", icon=":material/arrow_forward:", on_click=navigate, args=("documents",))


def document_scope(key):
    rows = list(pages("/documents", ready_only="true"))
    labels = {row["id"]: f"{row['filename']} · {row['id'][:8]}" for row in rows}
    return st.multiselect("Search within", list(labels), format_func=labels.__getitem__, key=key,
                          placeholder="All ready documents") or None


def history_view(identifier, key, history=None):
    history = history or api("GET", f"/conversations/{identifier}")
    pages_loaded = st.session_state.setdefault("history_pages", {})
    state_key = (key, identifier)
    older, has_more = pages_loaded.get(state_key, ([], history["has_older"]))
    messages = {message["id"]: message for message in [*older, *history["messages"]]}
    if messages and has_more and st.button("Load older messages", key=f"history_older_{key}", icon=":material/history:"):
        batch = api("GET", f"/conversations/{identifier}", params={"before": min(messages)})
        pages_loaded[state_key] = ([*batch["messages"], *older], batch["has_older"])
        st.rerun()
    for identifier in sorted(messages):
        conversation_message(messages[identifier], key)
    return bool(messages)


def chat():
    heading("AI Chat", "Ask questions about your documents and inspect the evidence behind each answer.")
    identifier = st.session_state.get("conversation_id")
    with st.container(border=True):
        scope, action = st.columns([4, 1], vertical_alignment="bottom")
        with scope:
            filter_ids = document_scope("chat_scope")
        action.button("New chat", icon=":material/edit_square:", width="stretch", on_click=start_chat)
    suggested = None
    if not identifier or not history_view(identifier, "chat"):
        st.html('<div class="dm-hero"><h2>What would you like to find?</h2>'
                '<p>Ask a question, then open the numbered citations to see the original passages.</p></div>')
        for column, prompt in zip(st.columns(3), PROMPTS):
            if column.button(prompt, key=f"suggest_{prompt}", icon=":material/arrow_outward:", width="stretch"):
                suggested = prompt
    question = st.chat_input("Ask about your documents", max_chars=1500) or suggested
    if question:
        with st.chat_message("user"):
            st.text(question)
        with st.spinner("Finding relevant passages and writing an answer…"):
            result = api("POST", "/chat", timeout=200,
                         json={"question": question, "conversation_id": identifier, "document_ids": filter_ids})
        st.session_state["conversation_id"] = result["conversation_id"]
        saved_question_count.clear()
        st.rerun()


def conversation_history():
    heading("Conversation History", "Find an earlier answer, revisit its sources, or continue the conversation.")
    term = st.text_input("Search conversations", placeholder="Search by conversation title")
    rows = [row for row in pages("/conversations") if term.strip().casefold() in row["title"].casefold()]
    if not rows:
        empty("No matching conversations", "Start a chat or try a different search.")
        st.button("Start a chat", icon=":material/add_comment:", type="primary", on_click=start_chat)
        return
    options = {row["id"]: f"{row['title']} · {stamp(row['updated_at'])}" for row in window(rows, "conversations_page")}
    identifier = st.selectbox("Saved conversation", list(options), format_func=options.__getitem__)
    history = api("GET", f"/conversations/{identifier}")
    left, right, _ = st.columns([1.35, 1, 2])
    left.button("Continue in AI Chat", type="primary", icon=":material/arrow_forward:", on_click=reopen_chat,
                args=(identifier,))
    with right.popover("Delete conversation", icon=":material/delete:"):
        st.caption("This removes the saved conversation and its messages.")
        if st.button("Confirm deletion", type="primary"):
            api("DELETE", f"/conversations/{identifier}")
            if st.session_state.get("conversation_id") == identifier:
                st.session_state.pop("conversation_id")
            for key in ("chat", "history"):
                st.session_state.get("history_pages", {}).pop((key, identifier), None)
            saved_question_count.clear()
            st.rerun()
    if not history_view(identifier, "history", history):
        empty("This conversation is empty", "Continue in AI Chat to add a message.")


def show_retrieval(result):
    times = result["latency_ms"]
    metrics([(label, f"{times[key]:,.1f} ms", note) for key, label, note in (
        ("retrieval", "Retrieval", "Candidate search"),
        ("reranking", "Reranking", "Cross-encoder scoring"),
        ("total", "Total latency", "Retrieval and ranking"),
    )])
    candidates = result["candidates"]
    st.caption("Query embedding reused from cache" if result["embedding_cache_hit"] else "No cached query embedding used")
    if not candidates:
        empty("No matching passages", "Adjust the query, document filter, or minimum dense score.")
        return
    final = {hit["chunk_id"]: (index, hit["rerank_score"]) for index, hit in enumerate(result["hits"], 1)}
    with st.container(border=True):
        st.subheader("Candidate ranking", anchor=False)
        st.dataframe([{
            "Candidate": index, "Final rank": final.get(hit["chunk_id"], (None, None))[0],
            "Document": hit["filename"], "Location": hit["location"], "Retrieval score": hit["retrieval_score"],
            "Reranker score": final.get(hit["chunk_id"], (None, None))[1],
        } for index, hit in enumerate(candidates, 1)], hide_index=True, width="stretch")
    for index, hit in enumerate(result["hits"], 1):
        with st.expander(f"Final passage {index}", icon=":material/description:"):
            st.text(f"{hit['filename']} · {hit['location']}")
            st.text(hit["text"])


def retrieval_inspector():
    filter_ids = document_scope("analytics_scope")
    settings = diagnostics(False)
    with st.form("retrieval_search", border=True):
        query = st.text_input("Query", max_chars=1500, placeholder="Enter a question or keyword")
        left, middle, right = st.columns(3)
        strategy = left.selectbox("Retrieval mode", ["hybrid", "dense", "sparse"], format_func=MODES.__getitem__)
        limit = middle.number_input("Top-K candidates", min_value=1, max_value=50, value=10)
        threshold = right.slider("Minimum dense score", min_value=-1.0, max_value=1.0, value=0.35)
        enabled = settings["reranker"] != "disabled"
        rerank = st.checkbox("Apply cross-encoder reranker", value=enabled, disabled=not enabled)
        submit = st.form_submit_button("Run retrieval", type="primary", icon=":material/search:")
    if submit:
        with st.spinner("Searching and scoring passages…"):
            st.session_state["retrieval_result"] = api("POST", "/retrieval", timeout=180, json={
                "question": query, "mode": strategy, "top_k": limit, "dense_min_score": threshold,
                "rerank": rerank, "document_ids": filter_ids,
            })
    if result := st.session_state.get("retrieval_result"):
        show_retrieval(result)


def evaluations():
    settings = diagnostics(False)
    runs = list(pages("/evaluations"))
    with st.expander("Evaluation controls", expanded=not runs, icon=":material/tune:"):
        st.caption("Upload the demo handbook and wait until it is ready to compare all four retrieval modes.")
        demo = api("GET", "/evaluations/demo", raw=True)
        st.download_button("Download demo handbook", demo, "handbook.txt", "text/plain", icon=":material/download:")
        k = st.number_input("Evaluation top-K", min_value=1, max_value=settings["rerank_top_k"],
                            value=min(settings["rerank_top_k"], 5))
        judge = st.checkbox("Score generated answers with DeepSeek and RAGAS")
        if judge:
            st.caption("Answer scoring makes multiple paid DeepSeek requests.")
        if st.button("Run evaluation", type="primary", icon=":material/play_arrow:"):
            with st.spinner("Comparing four retrieval modes…"):
                api("POST", "/evaluations", json={"top_k": k, "judge_answers": judge}, timeout=900)
            st.rerun()
    if not runs:
        empty("No saved evaluations", "Evaluation charts appear after the first completed run.")
        return
    st.subheader("Evaluation history", anchor=False)
    reports = {f"{stamp(row['created_at'])} · {row['id'][:8]}": row["report"] for row in runs}
    report = reports[st.selectbox("Saved run", list(reports))]
    averages = {mode: data["averages"] for mode, data in report["modes"].items()}
    mode = st.selectbox("Metric cards for", list(averages), index=list(averages).index("hybrid_rerank")
                        if "hybrid_rerank" in averages else 0, format_func=lambda key: MODES.get(key, key))
    metrics([(label, f"{averages[mode][key]:.1%}" if key in averages[mode] else "—", f"K={report['k']}")
             for key, label in METRICS.items()])
    with st.container(border=True):
        st.subheader("Retrieval mode comparison", anchor=False)
        comparison_chart(averages)
        st.dataframe([{"Mode": MODES.get(mode, mode), **values} for mode, values in averages.items()],
                     hide_index=True, width="stretch")
        st.download_button("Export this run", json.dumps(report, indent=2), "retrieval-evaluation.json",
                           "application/json", icon=":material/download:")


def analytics():
    heading("Retrieval Analytics", "Inspect search results and compare retrieval quality across evaluation runs.")
    selection = st.segmented_control("Analytics view", ["Inspect retrieval", "Evaluation runs"],
                                     default="Inspect retrieval", selection_mode="single")
    if selection == "Evaluation runs":
        evaluations()
    else:
        retrieval_inspector()


def system_settings():
    heading("System Settings", "Review service health, the model configuration, and application details.")
    if st.button("Check hosted services", type="primary", icon=":material/refresh:"):
        diagnostics.clear()
        health.clear()
        with st.spinner("Checking Qdrant Cloud and DeepSeek…"):
            st.session_state["external_diagnostics"] = diagnostics(True)
    live, ready = health()
    values = st.session_state.get("external_diagnostics") or (diagnostics(False) if live else {})
    with st.container(border=True):
        st.subheader("Service health", anchor=False)
        service_list(service_states(live, ready, values), {
            "Backend": "FastAPI liveness endpoint",
            "PostgreSQL": "Database and migration readiness",
            "Redis": values.get("redis", ""),
            "Qdrant Cloud": values.get("qdrant", ""),
            "DeepSeek": values.get("deepseek", ""),
        })
    if values:
        with st.container(border=True):
            st.subheader("Models and application", anchor=False)
            for label, key, kind in (
                ("Embedding model", "embedding_model", "EMB"),
                ("Reranker", "reranker", "RANK"),
                ("Answer model", "answer_model", "LLM"),
                ("Application", "application", "APP"),
                ("Environment", "environment", "ENV"),
            ):
                entry(kind, label, str(values[key]))
            st.caption(f"Chunk size {values['chunk_size']:,} · overlap {values['chunk_overlap']:,}"
                       f" · upload limit {values['max_upload_size_mb']} MiB per file")


VIEWS = {"dashboard": dashboard, "documents": library, "upload": upload_documents, "chat": chat,
         "conversations": conversation_history, "analytics": analytics, "settings": system_settings}


def main():
    st.set_page_config(page_title="DocuMind AI", page_icon=":material/description:", layout="wide",
                       initial_sidebar_state="auto")
    load_style()
    page = st.query_params.get("page", "dashboard")
    if page not in VIEWS:
        page = "dashboard"
    sidebar(page)
    try:
        VIEWS[page]()
    except ApiError as exc:
        error(exc)


main()
