import json
from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import func

from src.app.deps import templates
from src.app.errors import friendly_error_message
from src.reason.chunking_toggle import get_active_strategy, set_active_strategy
from src.store.relational import get_session
from src.store.schema import Chunk, Paper

router = APIRouter()

# A7: the ablation script (evals/run_chunking_eval.py) writes real
# results here — the Corpus page reads them to render the comparison
# table. Absent until that script has actually been run once; shown as
# "not run yet", never inferred or faked.
_CHUNKING_RESULTS_PATH = Path(__file__).resolve().parents[3] / "evals" / "chunking_results.json"


def _load_chunking_comparison() -> dict | None:
    if not _CHUNKING_RESULTS_PATH.exists():
        return None
    return json.loads(_CHUNKING_RESULTS_PATH.read_text(encoding="utf-8"))


@router.post("/corpus/chunking-strategy")
def set_chunking_strategy(strategy: str = Form(...)):
    set_active_strategy(strategy)
    return RedirectResponse(url="/corpus", status_code=303)


@router.get("/corpus")
def corpus_page(request: Request):
    context = {
        "active": "corpus",
        "error": None,
        "chunking_comparison": _load_chunking_comparison(),
        "active_chunking_strategy": get_active_strategy(),
    }
    session = get_session()
    try:
        paper_count = session.query(func.count(Paper.arxiv_id)).scalar() or 0
        chunk_count = session.query(func.count(Chunk.chunk_id)).scalar() or 0

        papers_rows = (
            session.query(Paper, func.count(Chunk.chunk_id))
            .outerjoin(Chunk, Chunk.paper_id == Paper.arxiv_id)
            .group_by(Paper.arxiv_id)
            .order_by(Paper.ingested_at.desc())
            .all()
        )
        papers = [
            {
                "title": paper.title,
                "url": paper.url,
                "category": paper.category,
                "published_date": paper.published_date,
                "chunk_count": chunk_count_,
            }
            for paper, chunk_count_ in papers_rows
        ]

        categories = sorted({c for p in papers for c in p["category"]})
        dates = [p["published_date"] for p in papers if p["published_date"]]
        date_range = (min(dates), max(dates)) if dates else None

        context.update(
            paper_count=paper_count,
            chunk_count=chunk_count,
            papers=papers,
            categories=categories,
            date_range=date_range,
        )
    except Exception as exc:
        # This route had only a `finally: session.close()` before — closed
        # the session correctly but still let the exception crash to a raw
        # 500, the same bug class already found and fixed on Ask and
        # Pipeline. Caught while auditing for R6, not by a user report.
        context["error"] = friendly_error_message(exc)
    finally:
        session.close()

    return templates.TemplateResponse(request, "corpus.html", context)
