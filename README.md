# Credit Portfolio Optimization

This repository contains a set of **agentic AI-powered workflows** for smart feature engineering, interpretation, and monitoring in credit portfolio optimization. Each workflow uses:

- [`n8n`](https://n8n.io)
- [`FastAPI`](https://fastapi.tiangolo.com)
- `Gemini LLM` for interpretation and explanations

---

## Usage Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/tpaidich/credit-feature-optimization.git
cd credit-feature-optimization/n8n_data
```

---

## Running Use Case APIs with Docker

Each use-case has its own API and Dockerfile.

### Example: Feature Importances (Workflow 1)

```bash
cd workflow_1

docker build -t feature-api .

docker run -it --name feature-api-demo -p 8000:8000 feature-api
```

---

## Running n8n

### 1. Create the n8n Container

```bash
docker run -it --name n8n-container \
  -p 5678:5678 \
  -v /home/node/.n8n/ \
  -e N8N_BASIC_AUTH_ACTIVE=true \
  -e N8N_BASIC_AUTH_USER=admin \
  -e N8N_BASIC_AUTH_PASSWORD=pass123 \
  -e N8N_FILESYSTEM_ALLOW_ALL=true \
  -e N8N_BLOCK_FILE_ACCESS_TO_N8N_FILES=false \
  n8nio/n8n
```

This mounts your local folder into the container at `/home/node/.n8n/`, and enables file system access.

---

## Workflow Execution in n8n

1. Import the relevant **n8n workflow JSON**.
2. Click **"Execute Workflow"**.
3. The workflow will:
   - Use the **HTTP Request node** to POST the dataset to the appropriate FastAPI endpoint (e.g., `http://host.docker.internal:8000/feature-importances`)
   - Use **Gemini** to interpret and summarize the result
   - Store the response in a clean JSON or Markdown format

---

## Workflows

| Workflow    | Purpose                                                 |
|-------------|---------------------------------------------------------|
| Workflow 1  | Feature importance and LLM-based interpretation         |
| Workflow 2  | Downturn detection and calibration                      |
| Workflow 3  | Adaptive threshold monitoring of credit KPIs            |
| Workflow 4  | Model stability backtesting over time                   |
| Workflow 5  | Detection of override approval patterns                 |


