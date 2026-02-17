from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import KnowledgeDoc


def ingest_doc(
    db: Session,
    *,
    source: str,
    source_version: str,
    title: str,
    content: str,
) -> KnowledgeDoc:
    doc = KnowledgeDoc(
        source=source,
        source_version=source_version,
        title=title,
        content=content,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def search_docs(db: Session, *, query: str, top_k: int) -> list[tuple[KnowledgeDoc, float]]:
    q = query.strip().lower()
    if not q:
        return []

    rows = db.scalars(select(KnowledgeDoc)).all()

    terms = [term for term in q.split() if term]
    scored: list[tuple[KnowledgeDoc, float]] = []
    for row in rows:
        text = f"{row.title} {row.content}".lower()
        score = float(sum(text.count(term) for term in terms)) if terms else 0.0
        if score == 0 and q in text:
            score = 1.0
        if score > 0:
            scored.append((row, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]
