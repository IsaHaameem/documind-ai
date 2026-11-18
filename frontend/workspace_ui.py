import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from markdown_it import MarkdownIt

STATUS = {
    "queued": ("Queued", "muted"), "processing": ("Processing", "info"),
    "ready": ("Ready", "success"), "failed": ("Failed", "error"),
    "deleting": ("Removing", "info"), "deletion_failed": ("Removal failed", "error"),
}
MODES = {"dense": "Dense", "sparse": "Sparse", "hybrid": "Hybrid", "hybrid_rerank": "Hybrid + rerank"}
METRICS = {"recall_at_k": "Recall@K", "mrr": "MRR", "context_precision": "Context precision",
           "context_recall": "Context recall"}


def load_style():
    st.html(Path(__file__).with_name("workspace.css"))


def heading(title, description):
    st.title(title, anchor=False)
    st.caption(description)


def stamp(value, *, compact=False):
    time = datetime.fromisoformat(value).astimezone(timezone.utc)
    return time.strftime("%d %b %Y" if compact else "%d %b %Y · %H:%M UTC")


def size_label(value):
    for scale, unit in ((1024**2, "MiB"), (1024, "KiB")):
        if value >= scale:
            return f"{value / scale:,.1f} {unit}"
    return f"{value:,} B"


def pill(label, tone="muted"):
    return f'<span class="dm-pill dm-{tone}"><span></span>{escape(str(label))}</span>'


def status_pill(status):
    return pill(*STATUS.get(status, (status, "muted")))


def error(message):
    st.error(re.sub(r"([\\\[\]()*_~`>#])", r"\\\1", str(message)))


def metrics(cards):
    elements = []
    for label, value, note in cards:
        meter = ""
        if isinstance(value, str) and re.fullmatch(r"\d{1,3}(?:\.\d+)?%", value) and float(value[:-1]) <= 100:
            meter = f'<meter min="0" max="100" value="{value[:-1]}" aria-label="{escape(label)}"></meter>'
        elements.append(f'<div class="dm-metric"><div class="dm-metric-label">{escape(label)}</div>'
                        f'<div class="dm-metric-value">{escape(str(value))}</div>{meter}'
                        f'<div class="dm-metric-caption">{escape(note)}</div></div>')
    st.html(f'<div class="dm-metrics">{"".join(elements)}</div>')


def entry(category, name, subtitle, status=""):
    st.html(
        f'<div class="dm-entry"><span class="dm-entry-icon">{escape(category)}</span>'
        f'<div class="dm-entry-main"><div class="dm-entry-title">{escape(name)}</div>'
        f'<div class="dm-entry-subtitle">{escape(subtitle)}</div></div>{status}</div>'
    )


def empty(title, hint):
    st.html(
        f'<div class="dm-empty"><strong>{escape(title)}</strong>'
        f'<p>{escape(hint)}</p></div>'
    )


def service_states(live, ready, values):
    def health_label(result):
        return ("Operational", "success") if result else ("Unavailable", "error")

    services = {
        "Backend": health_label(live),
        "PostgreSQL": health_label(ready) if live else ("Unknown", "muted"),
    }
    for label, key, positive in (
        ("Redis", "redis", ("reachable",)),
        ("Qdrant Cloud", "qdrant", ("collection ready", "reachable;")),
        ("DeepSeek", "deepseek", ("model reachable",)),
    ):
        value = values.get(key)
        if not value:
            services[label] = ("Unchecked", "muted")
        elif value.startswith(positive):
            services[label] = ("Connected", "success")
        elif value == "configured":
            services[label] = ("Configured", "info")
        else:
            services[label] = ("Needs attention", "error")
    return services


def service_list(services, details=None):
    details = details or {}
    for service, (label, tone) in services.items():
        detail = f'<small>{escape(details[service])}</small>' if details.get(service) else ""
        st.html(f'<div class="dm-service"><div><strong>{escape(service)}</strong>{detail}</div>{pill(label, tone)}</div>')


def service_overview(services):
    cells = "".join(f'<div><strong>{escape(service)}</strong>{pill(label, tone)}</div>'
                    for service, (label, tone) in services.items())
    st.html(f'<div class="dm-service-overview">{cells}</div>')


def job_progress(status):
    completed = {"queued": 1, "processing": 2, "ready": 3}.get(status)
    if completed:
        stage = ("Queued", "Extraction and indexing", "Complete")[completed - 1]
        bars = "".join('<span class="dm-checkpoint"></span>' if step <= completed else "<span></span>"
                       for step in range(1, 4))
        st.html(f'<div class="dm-job-progress" role="img" aria-label="Current stage: {stage}">{bars}</div>')


def text_html(content, citation_ids=None):
    # Permit formatting and local source anchors. Model-supplied URLs and HTML never become active.
    parser = MarkdownIt("commonmark", {"html": False}).disable(["link", "image", "autolink"])
    citation_ids = citation_ids or {}

    def render_text(tokens, index, options, env):
        text = tokens[index].content
        chunks = []
        last = 0
        for match in re.finditer(r"\[(\d+)\]", text):
            target = citation_ids.get(match[1])
            if target:
                chunks.append(escape(text[last:match.start()]))
                chunks.append(f'<a class="dm-cite" href="#{target}" aria-label="Go to source {match[1]}">{match[1]}</a>')
                last = match.end()
        chunks.append(escape(text[last:]))
        return "".join(chunks)

    parser.renderer.rules["text"] = render_text
    return parser.render(content)


def conversation_message(message, scope):
    role = message["role"]
    sources = message.get("citations", [])
    targets = {str(source["number"]): f"source-{scope}-{message['id']}-{source['number']}" for source in sources}
    with st.chat_message(role, avatar=":material/auto_awesome:" if role == "assistant" else ":material/person:"):
        author = "DocuMind" if role == "assistant" else "You"
        author_class = "dm-author-user" if role == "user" else "dm-author-assistant"
        st.html(f'<div class="dm-author {author_class}">{author}</div>'
                f'<div class="dm-prose">{text_html(message["content"], targets)}</div>')
        if sources:
            st.caption("Sources")
        for source in sources:
            target = targets[str(source["number"])]
            st.html(f'<div class="dm-source-anchor" id="{target}"></div>')
            # The title is escaped for Streamlit's Markdown-enabled label.
            location = re.sub(r"([\\\[\]()*_~`>#])", r"\\\1", source["location"])
            with st.expander(f"{source['number']:02d} · {location}", icon=":material/description:"):
                st.text(source["filename"])
                st.html(f'<blockquote class="dm-quote">{escape(source["quote"])}</blockquote>')
                st.caption("Retrieved passage")
                st.text(source["passage"])


def comparison_chart(averages):
    colors = ["#A78BFA", "#38BDF8", "#4ADE80", "#FBBF74"]
    fig = go.Figure()
    for (metric, label), color in zip(METRICS.items(), colors):
        fig.add_bar(
            name=label, x=[MODES.get(mode, mode) for mode in averages],
            y=[values.get(metric) for values in averages.values()], marker_color=color,
            hovertemplate="%{x}<br>%{y:.1%}<extra>" + label + "</extra>",
        )
    fig.update_layout(
        template="plotly_dark", font_color="#CBD5E1", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", barmode="group", bargap=0.28, height=350,
        margin=dict(l=42, r=14, t=50, b=42), legend=dict(orientation="h", y=1.16),
        yaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="#282E3D", zerolinecolor="#282E3D"),
        xaxis=dict(automargin=True, fixedrange=True),
    )
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})
