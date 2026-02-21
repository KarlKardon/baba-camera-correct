# Deploying on Akash

## Prerequisites

- Docker installed locally
- Docker Hub account (free)
- Kaggle API key (Settings → API → Create New Token → downloads `kaggle.json`)
- AKT tokens OR free trial credit from the hackathon

---

## Step 1 — Build & push the Docker image

```bash
# From the project root
docker build -t YOUR_DOCKERHUB_USER/lens-correction:latest .
docker push YOUR_DOCKERHUB_USER/lens-correction:latest
```

This bundles all the code and dependencies. The dataset is NOT baked in — it downloads on container start.

---

## Step 2 — Host the dataset at a direct URL

You already have the Kaggle zip locally. Upload it once to get a direct download URL:

**Option A — Google Drive (easiest):**
1. Upload `automatic-lens-correction.zip` to Google Drive
2. Right-click → Share → "Anyone with the link"
3. Get the file ID from the share URL and form:
   `https://drive.google.com/uc?export=download&id=YOUR_FILE_ID`
4. For large files, use [gdown](https://github.com/wkentaro/gdown) to verify the link works

**Option B — S3 pre-signed URL:**
```bash
aws s3 cp automatic-lens-correction.zip s3://your-bucket/
aws s3 presign s3://your-bucket/automatic-lens-correction.zip --expires-in 86400
```

Then edit [akash.sdl.yaml](akash.sdl.yaml) and replace:

| Placeholder | Replace with |
|---|---|
| `YOUR_DOCKERHUB_USER` | Your Docker Hub username |
| `YOUR_DIRECT_DOWNLOAD_URL` | The direct download URL from Step 2 |

---

## Step 3 — Deploy via Akash Console

1. Go to [console.akash.network](https://console.akash.network)
2. Click **Deploy** → **From a file** → upload `akash.sdl.yaml`
3. Review bids from providers — pick the cheapest with an RTX 4090 or better
4. Accept the bid and deploy

---

## Step 4 — Monitor training

In the Akash Console, go to your deployment → **Logs** to watch training progress.

Training time estimate on RTX 4090:
- Phase 1 (60 epochs, 512px): ~6–8 hrs
- Phase 2 (40 epochs, 768px): ~8–10 hrs
- Inference (1,000 images): ~5 min

---

## Step 5 — Download outputs

When training finishes, the container starts an HTTP server on port 8080.

In the Akash Console, find the **forwarded URI** for port 8080. Then:

```bash
# Download the submission zip
curl http://<akash-uri>/submissions/submission.zip -o submission.zip

# Download the best model checkpoint
curl http://<akash-uri>/checkpoints/phase2_full_best.pt -o phase2_full_best.pt
```

Or just open the URL in a browser to browse and download files.

---

## Step 6 — Submit

Upload `submission.zip` to **bounty.autohdr.com** manually. Done.

---

## GPU preference notes

The SDL requests `rtx4090` but Akash's bidding system will show you offers
from all providers. In order of preference for this workload:

1. **A100 80GB** — best, fits Phase 2 at 1024px
2. **RTX 4090 24GB** — great, use 768px for Phase 2
3. **RTX 3090 24GB** — fine, may need `PHASE2_RESOLUTION=512`
4. **P100 16GB** — workable, set `PHASE2_BATCH=2 PHASE2_RESOLUTION=512`

To target a different GPU, change the `model` field in the SDL.

---

## Cost estimate (with $100 credit)

RTX 4090 on Akash typically runs ~$0.30–0.50/hr.
Full training (~18 hrs) ≈ **$5–9**, leaving most of the $100 for re-runs.
