# ☁️ CloudTrace Lab

### Interactive Microservice Trace and Tail-Latency Simulator

**Developed by Nisar Ahmad**  
**COMSATS University Islamabad, Sahiwal Campus**

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Open%20Space-yellow)](https://huggingface.co/spaces/nisar-ai/cloudtrace-lab)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-Interactive%20UI-orange)](https://www.gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> An interactive discrete-event simulator for studying how workload, finite capacity, queueing, failures, retries, and latency variability affect microservice performance and tail latency.

## Live Demo

🚀 **Try CloudTrace Lab on Hugging Face Spaces:**

[**Open CloudTrace Lab →**](https://huggingface.co/spaces/nisar-ai/cloudtrace-lab)

## Overview

CloudTrace Lab is an educational cloud-systems simulation tool. It models requests flowing through a simplified microservice pipeline and analyzes how system behavior changes under different workload and capacity configurations.

The simulator helps explore:

- Request queueing under limited service capacity.
- End-to-end latency behavior.
- Tail latency through p50, p95, and p99 metrics.
- Service utilization and queue waiting time.
- Potential bottlenecks in a microservice pipeline.
- Retry behavior with exponential backoff and jitter.
- Reproducible experiments using a fixed random seed.

> **Accurate scope:** CloudTrace Lab is a statistical/discrete-event simulation. It does not collect live production telemetry, does not connect to real cloud services, and does not use machine learning.

## Microservice pipeline

The default simulation models requests moving through the following service path:

```text
Client Request
      │
      ▼
API Gateway
      │
      ▼
Authentication Service
      │
      ▼
Application Service
      │
      ▼
Database Service
      │
      ▼
Cache Service
      │
      ▼
Response
```

Each service can be configured with independent capacity and base latency.

## Features

- **Configurable workloads:** Constant, Burst, and Ramp request-arrival patterns.
- **Finite service capacity:** Each service has a configurable number of parallel processing slots.
- **Queue simulation:** Requests wait when all available service slots are busy.
- **Log-normal latency model:** Simulates positive, right-skewed request processing times and long-tail latency.
- **Failure modeling:** Configurable probability of simulated service-attempt failure.
- **Exponential-backoff retries:** Retries after failure using increasing delays.
- **Retry jitter:** Adds small random variation to retry delays to avoid synchronized retry spikes.
- **Tail-latency analysis:** Calculates mean, p50, p95, and p99 end-to-end latency.
- **Service-level metrics:** Measures utilization and average queue-wait time for every service.
- **Bottleneck detection:** Flags services at or above 80% estimated utilization as potential capacity constraints.
- **Trace explorer:** Shows request-level and service-stage-level trace data.
- **Interactive visualizations:** Includes latency-distribution, service-utilization, and queue-wait charts.
- **CSV export:** Downloads request summaries and complete trace data.
- **Reproducibility:** Uses a configurable random seed to recreate the same simulated workload conditions.

## Why tail latency matters

Average latency can look acceptable even when some users experience severe delays.

| Metric | Meaning |
|---|---|
| **Mean latency** | Average latency across all simulated requests |
| **p50 latency** | Median latency; half of requests are faster and half are slower |
| **p95 latency** | 95% of requests complete faster than this value |
| **p99 latency** | 99% of requests complete faster than this value; useful for identifying rare slow requests |

Tail-latency metrics are especially useful in distributed systems because one slow dependency can increase total end-to-end request latency.

## Modeling choices

### Log-normal latency

Service processing latency is modeled with a log-normal distribution.

This is an intentional approximation because:

- Processing latency cannot be negative.
- Most requests normally complete near a typical value.
- A smaller number of requests can be much slower due to contention, cache misses, storage delays, scheduling variation, or queueing.
- Log-normal distributions produce positive values with a right tail, making them useful for studying p95 and p99 behavior.

The simulator does not claim that every production service exactly follows a log-normal distribution. It uses this distribution as an adjustable learning model.

### Exponential backoff with jitter

After a simulated failure, a request may retry after an increasing delay:

```text
Retry 1 → approximately 100 ms
Retry 2 → approximately 200 ms
Retry 3 → approximately 400 ms
```

Small random jitter is added to each delay.

Exponential backoff reduces retry pressure during failures. Jitter prevents clients that failed together from retrying at the exact same time, reducing synchronized retry spikes known as the **thundering herd** problem. AWS documents exponential backoff with jitter as an approach for avoiding retry clustering. citeweb:230

### Utilization bottleneck warning

CloudTrace Lab flags a service as a potential bottleneck when its estimated utilization reaches or exceeds 80%.

This is a practical capacity-planning warning, not a strict universal rule. As service utilization approaches 100%, spare capacity falls and queueing delay can rise sharply. A service below 80% may still need investigation if its average queue wait is high.

## Reproducible experiments

The simulator includes a configurable random seed.

A fixed seed makes random behavior—latency samples, service failures, and retry jitter—repeatable. This supports controlled comparisons between two scenarios.

For example, compare a database capacity change while holding every other parameter constant:

| Parameter | Scenario A | Scenario B |
|---|---:|---:|
| Number of requests | 300 | 300 |
| Workload | Constant | Constant |
| Arrival rate | 70 req/s | 70 req/s |
| Database capacity | 2 slots | 5 slots |
| Retries | 1 | 1 |
| Random seed | 42 | 42 |

Then compare:

- Database utilization.
- Database queue-wait time.
- Mean latency.
- p50, p95, and p99 latency.
- Success rate.
- Simulated throughput.
- Primary bottleneck service.

## Technology stack

| Component | Technology |
|---|---|
| Programming language | Python |
| User interface | Gradio |
| Numerical simulation | NumPy |
| Data processing and CSV export | Pandas |
| Interactive charts | Plotly |
| Deployment | Hugging Face Spaces |

## Local setup

### 1. Clone the repository

```bash
git clone [https://github.com/YOUR-GITHUB-USERNAME/cloudtrace-lab.git](https://github.com/YOUR-GITHUB-USERNAME/cloudtrace-lab.git)
cd cloudtrace-lab
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Open the local URL printed in the terminal, usually:

```text
http://127.0.0.1:7860
```

## Project structure

```text
cloudtrace-lab/
├── app.py
├── requirements.txt
└── README.md
```

## Limitations

CloudTrace Lab is intended for learning and experimentation.

- It is not a production monitoring system.
- It does not collect real traces from deployed services.
- It does not use OpenTelemetry, Jaeger, Prometheus, Grafana, or real production metrics.
- Latency, capacity, failure rates, and workload patterns are model assumptions.
- The service pipeline is simplified and linear.
- The retry behavior repeats the full request pipeline; real systems may retry only a failed operation.
- The 80% utilization rule is a warning heuristic, not a universal bottleneck definition.
- Simulation results should not be presented as real measurements of a cloud provider or production system.

## Future improvements

- Add in-app side-by-side comparison of two scenarios.
- Add cache hit/miss probability.
- Add per-service failure-rate controls.
- Add request timeout and circuit-breaker simulation.
- Support dependency graphs instead of a fixed linear pipeline.
- Export charts alongside the CSV trace data.
- Add automated tests for queueing, retries, and percentile calculations.
- Add OpenTelemetry-compatible trace export for educational purposes.

## Author

**Nisar Ahmad**  
Computer Science Student  
COMSATS University Islamabad, Sahiwal Campus

- Hugging Face Space: [CloudTrace Lab](https://huggingface.co/spaces/nisar-ai/cloudtrace-lab)
- GitHub: https://github.com/nisar-ai

## License

This project is licensed under the MIT License.
