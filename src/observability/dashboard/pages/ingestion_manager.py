"""Ingestion Manager page – upload files, trigger ingestion, delete documents.

Layout:
1. File uploader + collection selector
2. Ingest button → progress bar (using on_progress callback)
3. Document list with delete buttons
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from src.observability.dashboard.services.data_service import DataService


def _run_ingestion(
    uploaded_file: "st.runtime.uploaded_file_manager.UploadedFile",
    collection: str,
    progress_bar: "st.delta_generator.DeltaGenerator",
    status_text: "st.delta_generator.DeltaGenerator",
) -> None:
    """Save the uploaded file to a temp location and run the pipeline."""
    from src.core.settings import load_settings
    from src.core.trace import TraceContext, TraceCollector
    from src.ingestion.pipeline import IngestionPipeline, PipelineResult

    settings = load_settings()

    # Write uploaded file to a temp location
    suffix = Path(uploaded_file.name).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    _STAGE_LABELS = {
        "integrity": "🔍 Checking file integrity…",
        "quality_check": "🔍 Checking document quality…",
        "load": "📄 Loading document…",
        "split": "✂️ Chunking document…",
        "transform": "🔄 Transforming chunks (LLM refine + enrich)…",
        "embed": "🔢 Encoding vectors…",
        "upsert": "💾 Storing to database…",
    }

    def on_progress(stage: str, current: int, total: int) -> None:
        frac = (current - 1) / total  # stage just started, show partial progress
        label = _STAGE_LABELS.get(stage, stage)
        progress_bar.progress(frac, text=f"[{current}/{total}] {label}")
        status_text.caption(label)

    trace = TraceContext(trace_type="ingestion")
    trace.metadata["source_path"] = uploaded_file.name
    trace.metadata["collection"] = collection
    trace.metadata["source"] = "dashboard"

    try:
        pipeline = IngestionPipeline(settings, collection=collection)
        result: PipelineResult = pipeline.run(
            file_path=tmp_path,
            trace=trace,
            on_progress=on_progress,
        )
        
        # Check quality result
        quality_result = result.stages.get("quality_check", {})
        
        if not result.success:
            error_msg = result.error or "Unknown error"
            if "quality" in error_msg.lower():
                # Quality check failed - show detailed error
                quality_metrics = quality_result.get("metrics", {})
                valid_ratio = quality_metrics.get("valid_ratio", 0)
                has_text_layer = quality_metrics.get("has_text_layer", True)
                garbage = quality_metrics.get("garbage_indicators", [])
                
                error_details = []
                if not has_text_layer:
                    error_details.append("📄 文档为扫描件或图片格式，无法提取文本")
                if valid_ratio < 0.8:
                    error_details.append(f"📝 有效字符占比过低: {valid_ratio:.1%} (阈值: 80%)")
                if garbage:
                    error_details.append(f"⚠️ 检测到问题: {', '.join(garbage)}")
                
                progress_bar.progress(0, text="❌ Quality Check Failed")
                status_text.error(
                    f"**文档质量检查未通过**\n\n"
                    f"文件: {uploaded_file.name}\n\n"
                    f"原因:\n" + "\n".join(f"- {e}" for e in error_details) + "\n\n"
                    f"💡 建议: 请确保文档可读，或联系文档管理员处理。"
                )
            else:
                progress_bar.progress(0, text="❌ Ingestion Failed")
                status_text.error(f"**摄入失败**: {error_msg}")
        else:
            progress_bar.progress(1.0, text="✅ Complete")
            status_text.success(f"Successfully ingested **{uploaded_file.name}** into collection **{collection}**.")
    except Exception as exc:
        status_text.error(f"Ingestion failed: {exc}")
    finally:
        TraceCollector().collect(trace)
        # Clean up temp file
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def render() -> None:
    """Render the Ingestion Manager page."""
    st.header("📥 Ingestion Manager")

    # ── Upload section ─────────────────────────────────────────────
    st.subheader("📤 Upload & Ingest")

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded = st.file_uploader(
            "Select a file to ingest",
            type=["pdf", "txt", "md", "docx"],
            key="ingest_uploader",
        )
    with col2:
        collection = st.text_input("Collection", value="default", key="ingest_collection")

    if uploaded is not None:
        if st.button("🚀 Start Ingestion", key="btn_ingest"):
            progress_bar = st.progress(0, text="Preparing…")
            status_text = st.empty()
            _run_ingestion(uploaded, collection.strip() or "default", progress_bar, status_text)

    st.divider()

    # ── Document management section ────────────────────────────────
    st.subheader("🗑️ Manage Documents")

    try:
        svc = DataService()
        docs = svc.list_documents()
    except Exception as exc:
        st.error(f"Failed to load documents: {exc}")
        return

    if not docs:
        st.info(
            "**No documents ingested yet.** "
            "Upload a PDF, TXT, MD, or DOCX file above and click \"Start Ingestion\" to begin."
        )
        return

    for idx, doc in enumerate(docs):
        col_info, col_btn = st.columns([4, 1])
        with col_info:
            st.markdown(
                f"**{doc['source_path']}** — "
                f"collection: `{doc.get('collection', '—')}` | "
                f"chunks: {doc['chunk_count']} | "
                f"images: {doc['image_count']}"
            )
        with col_btn:
            if st.button("🗑️ Delete", key=f"del_{idx}"):
                try:
                    result = svc.delete_document(
                        source_path=doc["source_path"],
                        collection=doc.get("collection", "default"),
                        source_hash=doc.get("source_hash"),
                    )
                    if result.success:
                        st.success(
                            f"Deleted: {result.chunks_deleted} chunks, "
                            f"{result.images_deleted} images removed."
                        )
                        st.rerun()
                    else:
                        st.warning(f"Partial delete. Errors: {result.errors}")
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")
