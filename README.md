# IPBES-Triage

IPBES-Triage is a pipeline for **literature triage**, combining
**FastAPI**, **Celery**, **MongoDB**, and **Redis** with an ensemble of fine-tuned RoBERTa models.

---

## Repository Structure

```
IPBES-api/
├── api/                      # API service (FastAPI + Celery workers)
├── model/                    # Model checkpoints (downloaded via script)
├── download_checkpoints.sh   # Script to download model checkpoints
└── README.md                 # (this file)
```

---

## Documentation

- **[API Service Documentation](./api/README.md)**
  Details on endpoints, job handling, Docker setup, and worker configuration.  


---

## Setup

### Download Model Checkpoints

Before running the API, you need to download the fine-tuned RoBERTa ensemble model checkpoints:

```bash
./download_checkpoints.sh
```

This script will:
- Download 5 cross-validation fold model checkpoints from the S3 bucket
- Store them in `model/checkpoints/`
- Models are based on RoBERTa-base architecture

---

## Quickstart

To launch the API stack with Docker:

```bash
cd api
docker compose up --build
```

The API will be available at [http://localhost:8501](http://localhost:8501).

---

## License
MIT License
