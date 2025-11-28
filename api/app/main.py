import logging
from textwrap import dedent
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from .config import settings
from .db import init_db, close_db
from .routers.job import router as job_router

log = logging.getLogger("uvicorn")

app = FastAPI(
    title=f"{settings.APP_NAME}",
    description=dedent("""
    ## IPBES biodiversity Literature Triage API

    This API provides automated triage and relevance scoring for biodiversity literature from PubMed.

    ### Features

    * **Batch Processing**: Submit multiple PubMed IDs (PMIDs) for analysis
    * **Full-text Support**: Automatically retrieves and analyzes full-text articles when available
    * **Machine Learning Scoring**: Uses trained models to score article relevance
    * **Job Management**: Track the status of your triage jobs in real-time
    * **Deduplication**: Automatically removes duplicate PMIDs from submissions

    ### Workflow

    1. **Create a Job**: Submit a list of PMIDs to create a new triage job
    2. **Monitor Progress**: Use the status endpoint to track job progress
    3. **Retrieve Results**: Get scored articles with relevance metrics

    ### Rate Limits

    Processing time depends on the number of articles and availability of full-text content.
    """),
    version="0.1.0",
    contact={
        "name": "IPBES API Support",
        "url": "https://github.com/ipbes",
    },
    license_info={
        "name": "MIT License",
    },
    openapi_tags=[
        {
            "name": "Job",
            "description": "Operations for creating and managing biodiversity literature triage jobs. "
                          "Jobs process batches of PubMed articles and return relevance scores."
        },
        {
            "name": "default",
            "description": "System health and status endpoints."
        }
    ]
)

origins = [o.strip() for o in settings.CORS_ORIGINS.split(",")] if settings.CORS_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    await init_db()

@app.on_event("shutdown")
async def on_shutdown():
    await close_db()

app.include_router(job_router, prefix=settings.API_PREFIX, tags=["Job"])

@app.get(
    "/healthz",
    summary="Health Check",
    description="Returns the health status of the API. Use this endpoint to verify the service is running.",
    tags=["default"],
    responses={
        200: {
            "description": "Service is healthy and operational",
            "content": {
                "application/json": {
                    "example": {"status": "ok"}
                }
            }
        }
    }
)
async def healthz():
    """Check if the API service is healthy and responding to requests."""
    return {"status": "ok"}

# Redirect root to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs", status_code=308)