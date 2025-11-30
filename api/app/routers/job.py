from datetime import datetime
from enum import Enum
from textwrap import dedent
from typing import List, Optional, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator, ConfigDict, Field

from ..models.job import Job
from ..models.document import DocumentEntry
from ..worker_client import enqueue_ingress_batches
from ..config import settings


router = APIRouter()

class Status(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"
    partial = "partial"
    queued = "queued"

class ArticleIn(BaseModel):
    """Input model for a single PubMed article."""
    pmid: int = Field(
        ...,
        description="PubMed ID (PMID) of the article to analyze. Must be a valid positive integer.",
        examples=[10216320, 23462742, 37578046],
        gt=0
    )
    @field_validator("pmid", mode="before")
    @classmethod
    def coerce_int(cls, v):
        if v is None:
            raise ValueError("pmid required")
        return int(v)

class JobCreate(BaseModel):
    """Request model for creating a new biodiversity literature triage job."""
    use_fulltext: bool = Field(
        default=True,
        description="Whether to use full-text article content when available. "
                    "If True, the system will attempt to retrieve PMC full-text. "
                    "If False or full-text unavailable, abstracts will be used."
    )
    article_set: List[ArticleIn] = Field(
        ...,
        description="List of PubMed articles to analyze. Duplicate PMIDs will be automatically removed.",
        min_length=1
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "use_fulltext": True,
                    "article_set": [
                        {"pmid": 17663714},
                        {"pmid": 26293953},
                        {"pmid": 32999073},
                        {"pmid": 36008872}
                    ]
                }
            ]
        }
    )

@router.post(
    "/job",
    response_model=dict,
    summary="Create Triage Job",
    description=dedent("""
    Create a new biodiversity literature triage job with a list of PubMed IDs.

    The API will:
    1. Validate and deduplicate the submitted PMIDs
    2. Queue articles for text retrieval (abstract or full-text)
    3. Process articles through ML models for relevance scoring
    4. Return a job ID for tracking progress

    **Note**: Duplicate PMIDs are automatically removed and counted in the job statistics.
    """),
    responses={
        200: {
            "description": "Job created successfully",
            "content": {
                "application/json": {
                    "example": {"job_id": "550e8400-e29b-41d4-a716-446655440000"}
                }
            }
        },
        400: {
            "description": "Invalid request (empty article set)",
            "content": {
                "application/json": {
                    "example": {"detail": "article_set is empty"}
                }
            }
        },
        422: {
            "description": "Validation error (invalid PMID format)",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "article_set", 0, "pmid"],
                                "msg": "Input should be greater than 0",
                                "type": "greater_than"
                            }
                        ]
                    }
                }
            }
        }
    }
)
async def create_job(payload: JobCreate):
    if not payload.article_set:
        raise HTTPException(400, "article_set is empty")

    pmids = [a.pmid for a in payload.article_set]
    seen: set[int] = set()
    unique: List[int] = []
    for p in pmids:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    job = Job(
        name=None,
        submitted_order=unique,
        submitted_pmids=len(pmids),
        dedup_dropped=len(pmids) - len(unique),
        ingress_batch_size=settings.INGRESS_BATCH_SIZE,
        infer_batch_size=settings.INFER_BATCH_SIZE,
        status="queued",
    )
    await job.insert()

    docs = [
        DocumentEntry(job_id=job.job_id, pmid=int(p),
                      ingress_status="pending", infer_status="pending")
        for p in unique
    ]
    if docs:
        await DocumentEntry.insert_many(docs)

    # enqueue
    batches = [unique[i:i+settings.INGRESS_BATCH_SIZE] for i in range(0, len(unique), settings.INGRESS_BATCH_SIZE)]
    for b in batches:
        await enqueue_ingress_batches(job.job_id, b)

    job.ingress_queued = sum(len(b) for b in batches)
    job.status = "running"
    await job.save()

    return {"job_id": job.job_id}

class ArticleOut(BaseModel):
    """Output model for a scored article with relevance metrics."""
    pmid: int = Field(
        ...,
        description="PubMed ID of the article",
        examples=[10216320]
    )
    scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Relevance scores for different categories. Keys are category names, values are probability scores (0.0-1.0).",
        examples=[{"biodiversity": 0.92, "climate": 0.15, "conservation": 0.78}]
    )
    pmcid: Optional[str] = Field(
        default=None,
        description="PubMed Central ID if full-text was available",
        examples=["PMC1234567"]
    )
    text_source: Optional[str] = Field(
        default=None,
        description="Source of text used for analysis: 'abstract' or 'fulltext'",
        examples=["fulltext", "abstract"]
    )
    text: Optional[str] = Field(
        default=None,
        description="The actual text that was analyzed (abstract or full-text content)",
        examples=["This study investigates..."]
    )

class JobOut(BaseModel):
    """Complete job details including all scored articles."""
    id: str = Field(
        ...,
        description="Unique identifier for this triage job",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )
    use_fulltext: bool = Field(
        ...,
        description="Whether full-text retrieval was enabled for this job"
    )
    status: Status = Field(
        ...,
        description="Current processing status of the job"
    )
    job_created_at: datetime = Field(
        ...,
        description="Timestamp when the job was created"
    )
    process_start_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when processing began"
    )
    process_end_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when processing completed"
    )
    process_time: Optional[float] = Field(
        default=None,
        description="Total processing time in seconds",
        examples=[45.2]
    )
    article_set: List[ArticleOut] = Field(
        ...,
        description="List of articles with their relevance scores, in the order submitted"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "use_fulltext": True,
                    "status": "done",
                    "job_created_at": "2024-11-28T10:30:00Z",
                    "process_start_at": "2024-11-28T10:30:01Z",
                    "process_end_at": "2024-11-28T10:30:45Z",
                    "process_time": 44.5,
                    "article_set": [
                        {
                            "pmid": 10216320,
                            "scores": {"biodiversity": 0.92, "conservation": 0.78},
                            "pmcid": "PMC1234567",
                            "text_source": "fulltext",
                            "text": "This study examines biodiversity patterns..."
                        }
                    ]
                }
            ]
        }
    )

@router.get(
    "/job/{job_id}",
    response_model=JobOut,
    summary="Get Job Results",
    description=dedent("""
    Retrieve complete job details including all article scores and metadata.

    Returns:
    - Job metadata (status, timestamps, processing time)
    - Complete list of articles with relevance scores
    - Text sources used (abstract vs full-text)
    - PMC IDs when full-text was available

    **Note**: Articles are returned in the same order they were submitted.
    """),
    responses={
        200: {
            "description": "Job details retrieved successfully",
        },
        404: {
            "description": "Job not found",
            "content": {
                "application/json": {
                    "example": {"detail": "job not found"}
                }
            }
        }
    }
)
async def get_job(job_id: str):
    job = await Job.find_one(Job.job_id == job_id)
    if not job:
        raise HTTPException(404, "job not found")

    docs = await DocumentEntry.find(DocumentEntry.job_id == job_id).to_list()

    order_idx = {pmid: i for i, pmid in enumerate(job.submitted_order or [])}
    docs.sort(key=lambda d: (order_idx.get(d.pmid, 10**12), d.pmid)) 

    items: List[ArticleOut] = []
    for d in docs:
        scores = {}
        text_source = "abstract"
        if d.predictions:
            if isinstance(d.predictions, list) and d.predictions:
                # Legacy format or list of predictions
                pred = d.predictions[0]
                if isinstance(pred, dict):
                    scores = pred.get("scores", {})
            elif isinstance(d.predictions, dict):
                # New format with scores dict
                scores = d.predictions.get("scores", {})
        if d.pmc_text:
            text_source = "fulltext"

        items.append(ArticleOut(
            pmid=d.pmid,
            scores=scores,
            pmcid=d.pmcid,
            text_source=text_source,
            text=d.text_for_infer or None
        ))

    job_created_at = job.created_at
    process_start_at = job.created_at if job.status in {"running", "done", "partial"} else None
    process_end_at = job.updated_at if job.status in {"done", "partial"} else None
    process_time = (process_end_at - process_start_at).total_seconds() if (process_start_at and process_end_at) else None

    status_map = {
        "queued": "pending",
        "running": "running",
        "done": "done",
        "partial": "partial",
        "failed": "failed",
    }

    return JobOut(
        id=job.job_id,
        use_fulltext=True,
        status=status_map.get(job.status, "pending"),
        job_created_at=job_created_at,
        process_start_at=process_start_at,
        process_end_at=process_end_at,
        process_time=process_time,
        article_set=items
    )

class JobStatusResponse(BaseModel):
    """Detailed progress counters for a triage job."""
    job_id: str = Field(
        ...,
        description="Unique identifier for this job"
    )
    status: str = Field(
        ...,
        description="Current job status: 'pending', 'running', 'done', 'partial', or 'failed'"
    )
    submitted_pmids: int = Field(
        ...,
        description="Total number of PMIDs submitted (including duplicates)",
        ge=0
    )
    dedup_dropped: int = Field(
        ...,
        description="Number of duplicate PMIDs removed",
        ge=0
    )
    ingress_queued: int = Field(
        ...,
        description="Number of articles queued for text retrieval",
        ge=0
    )
    ingress_done: int = Field(
        ...,
        description="Number of articles successfully retrieved",
        ge=0
    )
    ingress_failed: int = Field(
        ...,
        description="Number of articles that failed text retrieval",
        ge=0
    )
    infer_queued: int = Field(
        ...,
        description="Number of articles queued for ML inference",
        ge=0
    )
    infer_done: int = Field(
        ...,
        description="Number of articles successfully scored",
        ge=0
    )
    infer_failed: int = Field(
        ...,
        description="Number of articles that failed ML inference",
        ge=0
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "job_id": "550e8400-e29b-41d4-a716-446655440000",
                    "status": "running",
                    "submitted_pmids": 100,
                    "dedup_dropped": 5,
                    "ingress_queued": 25,
                    "ingress_done": 65,
                    "ingress_failed": 5,
                    "infer_queued": 10,
                    "infer_done": 55,
                    "infer_failed": 0
                }
            ]
        }
    )

@router.get(
    "/job/{job_id}/status",
    response_model=JobStatusResponse,
    summary="Get Job Progress",
    description=dedent("""
    Get detailed progress counters for a triage job.

    This endpoint provides real-time tracking of:
    - **Ingress Stage**: Article text retrieval from PubMed/PMC
    - **Inference Stage**: ML model scoring

    Use this endpoint to monitor job progress without retrieving full article data.
    The status field indicates overall job state: 'pending', 'running', 'done', 'partial', or 'failed'.

    **Tip**: Poll this endpoint periodically to track job completion.
    """),
    responses={
        200: {
            "description": "Job status retrieved successfully"
        },
        404: {
            "description": "Job not found",
            "content": {
                "application/json": {
                    "example": {"detail": "job not found"}
                }
            }
        }
    }
)
async def get_job_status(job_id: str):
    job = await Job.find_one(Job.job_id == job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    ingress_done = await DocumentEntry.find(
        DocumentEntry.job_id == job_id, DocumentEntry.ingress_status == "fetched"
    ).count()
    ingress_failed = await DocumentEntry.find(
        DocumentEntry.job_id == job_id, DocumentEntry.ingress_status == "failed"
    ).count()
    infer_done = await DocumentEntry.find(
        DocumentEntry.job_id == job_id, DocumentEntry.infer_status == "done"
    ).count()
    infer_failed = await DocumentEntry.find(
        DocumentEntry.job_id == job_id, DocumentEntry.infer_status == "failed"
    ).count()
    total_ingress = await DocumentEntry.find(DocumentEntry.job_id == job_id).count()
    infer_queued = max(0, total_ingress - infer_done - infer_failed)
    ingress_queued = max(0, job.submitted_pmids - job.dedup_dropped - ingress_done - ingress_failed)

    job.ingress_done = ingress_done
    job.ingress_failed = ingress_failed
    job.ingress_queued = ingress_queued
    job.infer_done = infer_done
    job.infer_failed = infer_failed
    job.infer_queued = infer_queued
    job.status = job.compute_status()
    await job.save()

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        submitted_pmids=job.submitted_pmids,
        dedup_dropped=job.dedup_dropped,
        ingress_queued=job.ingress_queued,
        ingress_done=job.ingress_done,
        ingress_failed=job.ingress_failed,
        infer_queued=job.infer_queued,
        infer_done=job.infer_done,
        infer_failed=job.infer_failed
    )