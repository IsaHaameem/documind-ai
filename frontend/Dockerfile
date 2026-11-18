FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY frontend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && pip check \
    && useradd --uid 10001 --create-home app
COPY frontend/streamlit_app.py frontend/workspace_api.py frontend/workspace_ui.py frontend/workspace.css ./
COPY frontend/.streamlit .streamlit
USER app
EXPOSE 8501
CMD ["sh", "-c", "exec streamlit run streamlit_app.py --server.address=0.0.0.0 --server.headless=true --server.maxUploadSize=${MAX_UPLOAD_SIZE_MB:-50} --browser.gatherUsageStats=false"]
