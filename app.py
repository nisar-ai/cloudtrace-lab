import math
import random
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# -------------------------------------------------------------------
# Project constants
# -------------------------------------------------------------------

PROJECT_TITLE = "CloudTrace Lab"
DEVELOPER_NAME = "Nisar Ahmad"
DEVELOPER_AFFILIATION = "COMSATS University Islamabad, Sahiwal Campus"

# Services are deliberately simple and fixed for the first version.
# A request flows through them in this order.
SERVICE_NAMES = [
    "API Gateway",
    "Authentication Service",
    "Application Service",
    "Database Service",
    "Cache Service",
]


@dataclass
class ServiceConfig:
    """
    Stores the configuration for one simulated service.

    capacity:
        Number of parallel service instances. If all instances are busy,
        incoming requests wait in a FIFO-like queue.

    base_latency_ms:
        Median-ish processing-time target used to parameterize the
        log-normal distribution.

    variability:
        Controls the spread of the log-normal distribution. Larger values
        create a heavier right tail, meaning occasional slow requests.

    failure_rate:
        Probability that an attempt fails at this service. This is only a
        simulation parameter, not an observed real-world failure rate.
    """
    name: str
    capacity: int
    base_latency_ms: float
    variability: float
    failure_rate: float


def generate_latency(base_latency_ms: float, variability: float) -> float:
    """
    Generate positive service time from a log-normal distribution.

    Why log-normal instead of a normal / Gaussian distribution?
    - Processing time cannot be negative.
    - Real service latency is often right-skewed: most requests are near
      the usual latency, but occasional requests are much slower.
    - A log-normal distribution naturally produces this positive,
      long-right-tail behavior.

    `base_latency_ms` is treated as the median of the distribution:
        median = exp(mu)
        so mu = log(base_latency_ms)

    `variability` is sigma. Higher sigma produces a wider, heavier tail.
    """
    safe_base = max(float(base_latency_ms), 0.1)
    sigma = max(float(variability), 0.01)
    mu = math.log(safe_base)

    latency = np.random.lognormal(mean=mu, sigma=sigma)

    # Keep rare extreme samples bounded so charts remain readable.
    return float(min(latency, safe_base * 50))


def exponential_backoff_delay_ms(attempt_number: int) -> float:
    """
    Compute exponential-backoff delay with small random jitter.

    Attempt 1 -> approximately 100 ms
    Attempt 2 -> approximately 200 ms
    Attempt 3 -> approximately 400 ms

    Why exponential backoff?
    If many clients retry immediately after a failure, they can create an
    additional overload spike. This is often called a thundering-herd
    pattern. Increasing the delay gives a stressed service time to recover.

    Jitter prevents all clients from retrying at exactly the same instant.
    """
    base_delay_ms = 100 * (2 ** (attempt_number - 1))
    jitter_multiplier = random.uniform(0.85, 1.15)
    return base_delay_ms * jitter_multiplier


def schedule_service(
    arrival_time_ms: float,
    service_available_times: List[float],
    base_latency_ms: float,
    variability: float,
) -> Tuple[float, float, float, int]:
    """
    Schedule a request at one service with finite parallel capacity.

    Each capacity slot has an availability time. The request chooses the
    slot that becomes free earliest. This approximates a FIFO queue:

    - If a slot is free on arrival, queue wait is zero.
    - If all slots are busy, the request waits until the earliest slot
      becomes available.
    - The service time is then sampled from a log-normal distribution.

    Returns:
        start_time_ms
        completion_time_ms
        queue_wait_ms
        selected_slot_index
    """
    selected_slot = int(np.argmin(service_available_times))
    earliest_available = service_available_times[selected_slot]

    start_time_ms = max(arrival_time_ms, earliest_available)
    queue_wait_ms = max(0.0, start_time_ms - arrival_time_ms)

    processing_time_ms = generate_latency(base_latency_ms, variability)
    completion_time_ms = start_time_ms + processing_time_ms

    service_available_times[selected_slot] = completion_time_ms

    return (
        start_time_ms,
        completion_time_ms,
        queue_wait_ms,
        selected_slot,
    )


def simulate_request(
    request_id: int,
    initial_arrival_ms: float,
    services: List[ServiceConfig],
    service_slots: Dict[str, List[float]],
    max_retries: int,
) -> Tuple[Dict, List[Dict]]:
    """
    Simulate one end-to-end request through every service in the pipeline.

    A request succeeds only if every service attempt succeeds.
    If an attempt fails, the complete pipeline is retried after an
    exponential-backoff delay, up to `max_retries`.

    This is intentionally a simplified model:
    - Retries repeat the full pipeline.
    - Real systems may retry only a single failed service or use more
      complex policies, deadlines, circuit breakers, and idempotency rules.
    """
    current_time_ms = initial_arrival_ms
    total_attempts = 0
    request_succeeded = False
    final_error_service = None

    request_trace_rows = []
    first_attempt_start_ms = initial_arrival_ms
    final_completion_ms = initial_arrival_ms

    for retry_index in range(max_retries + 1):
        total_attempts += 1
        attempt_number = retry_index + 1
        attempt_failed = False

        for service in services:
            start_ms, completion_ms, queue_wait_ms, selected_slot = schedule_service(
                arrival_time_ms=current_time_ms,
                service_available_times=service_slots[service.name],
                base_latency_ms=service.base_latency_ms,
                variability=service.variability,
            )

            processing_ms = completion_ms - start_ms
            failed_here = random.random() < service.failure_rate

            request_trace_rows.append(
                {
                    "request_id": request_id,
                    "attempt": attempt_number,
                    "service": service.name,
                    "arrival_ms": round(current_time_ms, 3),
                    "start_ms": round(start_ms, 3),
                    "completion_ms": round(completion_ms, 3),
                    "queue_wait_ms": round(queue_wait_ms, 3),
                    "processing_ms": round(processing_ms, 3),
                    "slot_index": selected_slot,
                    "service_outcome": "failed" if failed_here else "succeeded",
                }
            )

            final_completion_ms = completion_ms

            if failed_here:
                attempt_failed = True
                final_error_service = service.name
                break

            current_time_ms = completion_ms

        if not attempt_failed:
            request_succeeded = True
            break

        if retry_index < max_retries:
            retry_delay_ms = exponential_backoff_delay_ms(attempt_number)
            current_time_ms = final_completion_ms + retry_delay_ms

    end_to_end_latency_ms = final_completion_ms - first_attempt_start_ms

    request_summary = {
        "request_id": request_id,
        "arrival_ms": round(initial_arrival_ms, 3),
        "completion_ms": round(final_completion_ms, 3),
        "end_to_end_latency_ms": round(end_to_end_latency_ms, 3),
        "outcome": "succeeded" if request_succeeded else "failed",
        "attempts": total_attempts,
        "final_error_service": final_error_service or "",
    }

    return request_summary, request_trace_rows


def generate_arrival_times(
    number_of_requests: int,
    workload_type: str,
    request_rate: float,
) -> List[float]:
    """
    Create request arrival times in milliseconds.

    Constant:
        Requests arrive at a regular interval.

    Burst:
        All requests arrive at time zero. This creates immediate queueing
        pressure and lets the user observe burst behavior.

    Ramp:
        Arrival intervals shrink gradually, causing request pressure to
        increase over the simulation.
    """
    number_of_requests = int(number_of_requests)
    request_rate = max(float(request_rate), 0.1)

    if workload_type == "Burst":
        return [0.0] * number_of_requests

    if workload_type == "Ramp":
        arrivals = []
        current_time = 0.0

        for index in range(number_of_requests):
            progress = index / max(number_of_requests - 1, 1)

            # Start slower and become faster by the end of the run.
            current_rate = request_rate * (0.35 + 0.65 * progress)
            current_time += 1000 / max(current_rate, 0.1)
            arrivals.append(current_time)

        return arrivals

    # Default: constant request arrival rate.
    interval_ms = 1000 / request_rate
    return [index * interval_ms for index in range(number_of_requests)]


def calculate_service_metrics(
    trace_df: pd.DataFrame,
    services: List[ServiceConfig],
    simulation_end_ms: float,
) -> pd.DataFrame:
    """
    Calculate per-service queueing and utilization metrics.

    Utilization is estimated as total processing time divided by:
        simulation duration × number of service slots

    This is an approximation suitable for this educational discrete-event
    model. A value above 80% is marked as a capacity-planning warning.

    Why 80%?
    It is a common operational rule of thumb rather than a strict theorem.
    Queueing theory shows that waiting time can grow sharply as utilization
    approaches 100%, leaving little slack for bursts or latency variation.
    """
    rows = []

    for service in services:
        service_trace = trace_df[trace_df["service"] == service.name]

        total_processing_ms = service_trace["processing_ms"].sum()
        average_queue_wait_ms = (
            service_trace["queue_wait_ms"].mean()
            if not service_trace.empty
            else 0.0
        )

        denominator = max(
            simulation_end_ms * service.capacity,
            1.0,
        )

        utilization_pct = min(
            100.0,
            (total_processing_ms / denominator) * 100,
        )

        bottleneck_flag = (
            "⚠️ Potential bottleneck"
            if utilization_pct >= 80
            else "Healthy"
        )

        rows.append(
            {
                "service": service.name,
                "capacity": service.capacity,
                "avg_queue_wait_ms": round(float(average_queue_wait_ms), 2),
                "utilization_pct": round(float(utilization_pct), 2),
                "status": bottleneck_flag,
            }
        )

    return pd.DataFrame(rows)


def detect_bottleneck(service_metrics_df: pd.DataFrame) -> str:
    """
    Identify the service with the highest utilization.

    Utilization is used as the primary signal because a highly utilized
    resource has little spare capacity. Average queue wait is included as
    a secondary warning signal in the displayed table.
    """
    if service_metrics_df.empty:
        return "No service metrics were generated."

    bottleneck_row = service_metrics_df.loc[
        service_metrics_df["utilization_pct"].idxmax()
    ]

    return (
        f"**Primary capacity constraint:** {bottleneck_row['service']}  \n"
        f"- Estimated utilization: **{bottleneck_row['utilization_pct']}%**  \n"
        f"- Average queue wait: **{bottleneck_row['avg_queue_wait_ms']} ms**  \n"
        f"- Status: **{bottleneck_row['status']}**"
    )


def calculate_percentiles(request_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate tail-latency metrics from completed simulated requests.

    p50: median; half of requests are faster and half slower.
    p95: 95% of requests are faster than this value.
    p99: highlights rare slow requests and is useful for understanding
         user experience at the tail of the latency distribution.
    """
    if request_df.empty:
        return {
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "mean": float("nan"),
        }

    latencies = request_df["end_to_end_latency_ms"].to_numpy()

    return {
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
        "mean": float(np.mean(latencies)),
    }


def create_latency_histogram(request_df: pd.DataFrame) -> go.Figure:
    """Create an interactive end-to-end latency histogram."""
    figure = px.histogram(
        request_df,
        x="end_to_end_latency_ms",
        color="outcome",
        nbins=30,
        title="End-to-End Latency Distribution",
        labels={
            "end_to_end_latency_ms": "Latency (ms)",
            "outcome": "Request outcome",
        },
        color_discrete_map={
            "succeeded": "#36d399",
            "failed": "#f87272",
        },
        template="plotly_dark",
    )

    figure.update_layout(
        height=380,
        margin=dict(l=40, r=30, t=60, b=40),
        legend_title_text="Outcome",
    )

    return figure


def create_trace_chart(trace_df: pd.DataFrame) -> go.Figure:
    """
    Create a compact waterfall-style trace view for the first 20 requests.

    The bar begins at start_ms and spans processing_ms.
    Queue waiting is displayed as a lighter bar before processing.
    """
    trace_subset = trace_df[
        trace_df["request_id"] <= 20
    ].copy()

    if trace_subset.empty:
        return go.Figure()

    trace_subset["request_label"] = (
        "Request " + trace_subset["request_id"].astype(str)
        + " / Attempt " + trace_subset["attempt"].astype(str)
    )

    queue_figure = px.bar(
        trace_subset,
        x="queue_wait_ms",
        y="request_label",
        color="service",
        orientation="h",
        title="Trace View: Queue Wait per Stage (First 20 Requests)",
        labels={
            "queue_wait_ms": "Queue wait (ms)",
            "request_label": "Request / attempt",
        },
        template="plotly_dark",
    )

    queue_figure.update_layout(
        height=520,
        margin=dict(l=40, r=30, t=60, b=40),
        yaxis={"categoryorder": "total ascending"},
    )

    return queue_figure


def create_service_chart(service_metrics_df: pd.DataFrame) -> go.Figure:
    """Visualize utilization and queue wait by service."""
    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=service_metrics_df["service"],
            y=service_metrics_df["utilization_pct"],
            name="Utilization (%)",
            marker_color="#60a5fa",
            yaxis="y1",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=service_metrics_df["service"],
            y=service_metrics_df["avg_queue_wait_ms"],
            name="Average queue wait (ms)",
            mode="lines+markers",
            marker_color="#fbbf24",
            yaxis="y2",
        )
    )

    figure.add_hline(
        y=80,
        line_dash="dash",
        line_color="#f87171",
        annotation_text="80% utilization warning",
        annotation_position="top left",
    )

    figure.update_layout(
        title="Service Utilization and Queueing Delay",
        template="plotly_dark",
        height=400,
        margin=dict(l=40, r=40, t=60, b=80),
        yaxis=dict(
            title="Utilization (%)",
            range=[0, 105],
        ),
        yaxis2=dict(
            title="Average queue wait (ms)",
            overlaying="y",
            side="right",
        ),
        legend=dict(orientation="h"),
    )

    return figure


def write_csv_files(
    request_df: pd.DataFrame,
    trace_df: pd.DataFrame,
) -> Tuple[str, str]:
    """
    Export request summaries and full stage-level traces as CSV files.

    Separate exports make the experiment inspectable:
    - Request summary: one row per user-level request.
    - Full trace: one row per service stage and retry attempt.
    """
    request_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_request_summary.csv",
        delete=False,
        encoding="utf-8",
        newline="",
    )

    trace_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_full_trace.csv",
        delete=False,
        encoding="utf-8",
        newline="",
    )

    request_df.to_csv(request_file.name, index=False)
    trace_df.to_csv(trace_file.name, index=False)

    request_file.close()
    trace_file.close()

    return request_file.name, trace_file.name


def run_simulation(
    number_of_requests,
    workload_type,
    request_rate,
    api_capacity,
    auth_capacity,
    app_capacity,
    db_capacity,
    cache_capacity,
    api_latency,
    auth_latency,
    app_latency,
    db_latency,
    cache_latency,
    latency_variability,
    failure_rate_percent,
    max_retries,
    random_seed,
):
    """
    Main Gradio callback.

    It builds service configuration, generates a workload, runs a
    deterministic simulation when a seed is fixed, calculates metrics,
    creates charts, and returns downloadable experiment data.
    """
    # A seed lets the user reproduce a particular experiment.
    seed = int(random_seed)
    random.seed(seed)
    np.random.seed(seed)

    services = [
        ServiceConfig(
            "API Gateway",
            int(api_capacity),
            float(api_latency),
            float(latency_variability),
            float(failure_rate_percent) / 100,
        ),
        ServiceConfig(
            "Authentication Service",
            int(auth_capacity),
            float(auth_latency),
            float(latency_variability),
            float(failure_rate_percent) / 100,
        ),
        ServiceConfig(
            "Application Service",
            int(app_capacity),
            float(app_latency),
            float(latency_variability),
            float(failure_rate_percent) / 100,
        ),
        ServiceConfig(
            "Database Service",
            int(db_capacity),
            float(db_latency),
            float(latency_variability),
            float(failure_rate_percent) / 100,
        ),
        ServiceConfig(
            "Cache Service",
            int(cache_capacity),
            float(cache_latency),
            float(latency_variability),
            float(failure_rate_percent) / 100,
        ),
    ]

    # Each service slot begins free at time zero.
    service_slots = {
        service.name: [0.0] * service.capacity
        for service in services
    }

    arrival_times = generate_arrival_times(
        number_of_requests=int(number_of_requests),
        workload_type=workload_type,
        request_rate=float(request_rate),
    )

    request_summaries = []
    all_trace_rows = []

    for request_id, arrival_time_ms in enumerate(
        arrival_times,
        start=1,
    ):
        request_summary, trace_rows = simulate_request(
            request_id=request_id,
            initial_arrival_ms=arrival_time_ms,
            services=services,
            service_slots=service_slots,
            max_retries=int(max_retries),
        )

        request_summaries.append(request_summary)
        all_trace_rows.extend(trace_rows)

    request_df = pd.DataFrame(request_summaries)
    trace_df = pd.DataFrame(all_trace_rows)

    simulation_end_ms = max(
        float(request_df["completion_ms"].max()),
        1.0,
    )

    service_metrics_df = calculate_service_metrics(
        trace_df=trace_df,
        services=services,
        simulation_end_ms=simulation_end_ms,
    )

    percentiles = calculate_percentiles(request_df)

    successful_count = int(
        (request_df["outcome"] == "succeeded").sum()
    )
    failed_count = int(
        (request_df["outcome"] == "failed").sum()
    )

    total_requests = len(request_df)
    success_rate = (
        (successful_count / total_requests) * 100
        if total_requests > 0
        else 0.0
    )

    total_duration_seconds = simulation_end_ms / 1000
    throughput = (
        total_requests / total_duration_seconds
        if total_duration_seconds > 0
        else 0.0
    )

    metrics_df = pd.DataFrame(
        [
            {
                "Metric": "Total requests",
                "Value": total_requests,
            },
            {
                "Metric": "Succeeded",
                "Value": successful_count,
            },
            {
                "Metric": "Failed after retries",
                "Value": failed_count,
            },
            {
                "Metric": "Success rate",
                "Value": f"{success_rate:.2f}%",
            },
            {
                "Metric": "Mean latency",
                "Value": f"{percentiles['mean']:.2f} ms",
            },
            {
                "Metric": "p50 latency",
                "Value": f"{percentiles['p50']:.2f} ms",
            },
            {
                "Metric": "p95 latency",
                "Value": f"{percentiles['p95']:.2f} ms",
            },
            {
                "Metric": "p99 latency",
                "Value": f"{percentiles['p99']:.2f} ms",
            },
            {
                "Metric": "Observed throughput",
                "Value": f"{throughput:.2f} requests/sec",
            },
            {
                "Metric": "Simulation duration",
                "Value": f"{total_duration_seconds:.2f} sec",
            },
        ]
    )

    bottleneck_summary = detect_bottleneck(service_metrics_df)

    explanation = f"""
### Simulation interpretation

- **Workload:** {workload_type}
- **Requests simulated:** {total_requests}
- **Arrival rate setting:** {float(request_rate):.2f} requests/second
- **Latency model:** log-normal service times with variability σ = {float(latency_variability):.2f}
- **Retries allowed:** {int(max_retries)}
- **Per-attempt failure probability:** {float(failure_rate_percent):.2f}%

**Tail latency:** p50 describes a typical request; p95 and p99 reveal slower requests in the right tail.  
**Bottleneck warning:** services at or above 80% estimated utilization are flagged because queueing delay often increases sharply as a resource approaches full utilization.  
**Important:** results are simulated and should be interpreted as outcomes of this model, not measurements from a production cloud system.
"""

    histogram = create_latency_histogram(request_df)
    trace_chart = create_trace_chart(trace_df)
    service_chart = create_service_chart(service_metrics_df)

    request_csv_file, trace_csv_file = write_csv_files(
        request_df=request_df,
        trace_df=trace_df,
    )

    request_preview = request_df.head(100)
    trace_preview = trace_df.head(200)

    return (
        metrics_df,
        bottleneck_summary,
        explanation,
        histogram,
        service_chart,
        trace_chart,
        service_metrics_df,
        request_preview,
        trace_preview,
        request_csv_file,
        trace_csv_file,
    )


# -------------------------------------------------------------------
# Gradio interface
# -------------------------------------------------------------------

custom_css = """
body {
    background: radial-gradient(circle at top, #172554 0%, #020617 55%) !important;
}

.gradio-container {
    max-width: 1440px !important;
    margin: auto !important;
    background: transparent !important;
}

#hero {
    border: 1px solid rgba(96, 165, 250, 0.35);
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.82));
    border-radius: 22px;
    padding: 30px;
    margin-bottom: 18px;
    box-shadow: 0 18px 70px rgba(0, 0, 0, 0.32);
}

#hero h1 {
    font-size: 3rem !important;
    font-weight: 900 !important;
    letter-spacing: -0.04em;
    color: #e0f2fe !important;
    margin-bottom: 8px !important;
}

#hero .developer {
    font-size: 1.3rem;
    font-weight: 800;
    color: #fbbf24;
    margin-top: 14px;
}

#hero .subtitle {
    color: #cbd5e1;
    font-size: 1.05rem;
    line-height: 1.6;
}

.primary-button {
    background: linear-gradient(90deg, #2563eb, #0891b2) !important;
    color: white !important;
    border: none !important;
    font-size: 1.05rem !important;
    font-weight: 800 !important;
    min-height: 48px !important;
}

.note-box {
    border-left: 4px solid #38bdf8;
    padding: 12px 16px;
    border-radius: 8px;
    background: rgba(14, 116, 144, 0.15);
    color: #e0f2fe;
}
"""


with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
    ),
    css=custom_css,
    title="CloudTrace Lab | Nisar Ahmad",
) as demo:

    gr.HTML(
        f"""
        <div id="hero">
            <h1>☁️ CloudTrace Lab</h1>
            <div class="subtitle">
                Interactive Microservice Trace and Tail-Latency Simulator<br>
                A discrete-event learning tool for studying queues, retries,
                workload pressure, utilization, bottlenecks, and tail latency.
            </div>
            <div class="developer">
                Developed by {DEVELOPER_NAME} — {DEVELOPER_AFFILIATION}
            </div>
        </div>
        """
    )

    gr.Markdown(
        """
<div class="note-box">
<b>Accurate scope:</b> This is a statistical/discrete-event simulator.
It does not collect live production telemetry and does not use machine learning.
Use it to explore how design assumptions affect modeled microservice behavior.
</div>
        """
    )

    with gr.Tabs():
        with gr.Tab("⚙️ Configure Simulation"):
            with gr.Row():
                with gr.Column(scale=1):
                    number_of_requests_input = gr.Slider(
                        minimum=20,
                        maximum=2000,
                        value=300,
                        step=10,
                        label="Number of requests to simulate",
                    )

                    workload_type_input = gr.Radio(
                        choices=["Constant", "Burst", "Ramp"],
                        value="Constant",
                        label="Workload pattern",
                        info=(
                            "Constant = regular arrivals. "
                            "Burst = all requests arrive at once. "
                            "Ramp = request pressure increases over time."
                        ),
                    )

                    request_rate_input = gr.Slider(
                        minimum=1,
                        maximum=500,
                        value=70,
                        step=1,
                        label="Arrival rate setting (requests/second)",
                    )

                    max_retries_input = gr.Slider(
                        minimum=0,
                        maximum=3,
                        value=1,
                        step=1,
                        label="Maximum retries after a failed attempt",
                    )

                    failure_rate_input = gr.Slider(
                        minimum=0,
                        maximum=20,
                        value=2,
                        step=0.5,
                        label="Per-service attempt failure probability (%)",
                    )

                    random_seed_input = gr.Number(
                        value=42,
                        precision=0,
                        label="Random seed",
                        info=(
                            "Use the same seed and configuration to reproduce "
                            "the same simulation result."
                        ),
                    )

                with gr.Column(scale=1):
                    gr.Markdown(
                        """
### Service configuration

Capacity is the number of parallel service slots.  
Base latency is the median-ish processing-time target.  
All services use the same log-normal variability setting in this version.
                        """
                    )

                    latency_variability_input = gr.Slider(
                        minimum=0.05,
                        maximum=1.25,
                        value=0.45,
                        step=0.05,
                        label="Log-normal latency variability (σ)",
                    )

                    with gr.Group():
                        gr.Markdown("#### API Gateway")
                        api_capacity_input = gr.Slider(
                            1, 20, value=8, step=1,
                            label="Capacity",
                        )
                        api_latency_input = gr.Slider(
                            1, 150, value=8, step=1,
                            label="Base latency (ms)",
                        )

                    with gr.Group():
                        gr.Markdown("#### Authentication Service")
                        auth_capacity_input = gr.Slider(
                            1, 20, value=5, step=1,
                            label="Capacity",
                        )
                        auth_latency_input = gr.Slider(
                            1, 250, value=20, step=1,
                            label="Base latency (ms)",
                        )

                    with gr.Group():
                        gr.Markdown("#### Application Service")
                        app_capacity_input = gr.Slider(
                            1, 20, value=4, step=1,
                            label="Capacity",
                        )
                        app_latency_input = gr.Slider(
                            1, 300, value=35, step=1,
                            label="Base latency (ms)",
                        )

                    with gr.Group():
                        gr.Markdown("#### Database Service")
                        db_capacity_input = gr.Slider(
                            1, 20, value=2, step=1,
                            label="Capacity",
                        )
                        db_latency_input = gr.Slider(
                            1, 400, value=55, step=1,
                            label="Base latency (ms)",
                        )

                    with gr.Group():
                        gr.Markdown("#### Cache Service")
                        cache_capacity_input = gr.Slider(
                            1, 20, value=6, step=1,
                            label="Capacity",
                        )
                        cache_latency_input = gr.Slider(
                            1, 120, value=6, step=1,
                            label="Base latency (ms)",
                        )

            run_button = gr.Button(
                "Run CloudTrace Simulation",
                elem_classes=["primary-button"],
            )

        with gr.Tab("📊 Results"):
            gr.Markdown(
                """
### Latency metrics

- **p50:** typical request latency.
- **p95:** 95% of requests complete faster than this time.
- **p99:** tail latency; reveals rare slow requests that averages can hide.
                """
            )

            metrics_output = gr.Dataframe(
                headers=["Metric", "Value"],
                datatype=["str", "str"],
                interactive=False,
                label="Simulation summary",
            )

            bottleneck_output = gr.Markdown(
                value="Run a simulation to identify the primary capacity constraint."
            )

            interpretation_output = gr.Markdown()

            histogram_output = gr.Plot(
                label="Latency distribution"
            )

            service_chart_output = gr.Plot(
                label="Service utilization and queueing"
            )

            service_metrics_output = gr.Dataframe(
                interactive=False,
                label="Service-level metrics",
            )

        with gr.Tab("🔎 Trace Explorer"):
            gr.Markdown(
                """
The trace table stores stage-level data for every request attempt:
arrival time, queue wait, processing time, completion time, selected
capacity slot, and outcome.
                """
            )

            trace_chart_output = gr.Plot(
                label="Queue wait trace view"
            )

            request_preview_output = gr.Dataframe(
                interactive=False,
                label="Request summary preview (first 100 requests)",
            )

            trace_preview_output = gr.Dataframe(
                interactive=False,
                label="Full trace preview (first 200 service stages)",
            )

        with gr.Tab("⬇️ Export Data"):
            gr.Markdown(
                """
Download the raw CSV outputs to inspect results, build additional charts,
or use a fixed configuration and random seed for reproducible experiments.
                """
            )

            request_csv_output = gr.File(
                label="Download request summary CSV"
            )

            trace_csv_output = gr.File(
                label="Download complete trace CSV"
            )

        with gr.Tab("📚 Model Notes"):
            gr.Markdown(
                """
## Model assumptions

### Log-normal latency
Latency is modeled with a log-normal distribution because it is always positive
and can create a long right tail: many requests are near normal latency, while
a smaller number are much slower.

### Finite capacity and queues
Every service has a limited number of parallel slots. Requests wait when all
slots are busy. This approximates capacity pressure and queueing behavior.

### Exponential-backoff retries
After a failed attempt, a request waits roughly 100 ms, 200 ms, then 400 ms
before retrying. Small random jitter is added. Delaying retries reduces the
risk that many clients retry simultaneously and worsen an existing overload.

### 80% bottleneck warning
A service at or above 80% estimated utilization is flagged as a planning
warning. This is not a strict universal threshold, but a useful rule of thumb:
queueing delay can increase rapidly as resources approach full utilization.

### Limitations
This is an educational discrete-event simulation, not production monitoring.
It models service behavior from assumptions; real systems require real
telemetry, distributed tracing, observability pipelines, and workload data.
                """
            )

    all_inputs = [
        number_of_requests_input,
        workload_type_input,
        request_rate_input,
        api_capacity_input,
        auth_capacity_input,
        app_capacity_input,
        db_capacity_input,
        cache_capacity_input,
        api_latency_input,
        auth_latency_input,
        app_latency_input,
        db_latency_input,
        cache_latency_input,
        latency_variability_input,
        failure_rate_input,
        max_retries_input,
        random_seed_input,
    ]

    all_outputs = [
        metrics_output,
        bottleneck_output,
        interpretation_output,
        histogram_output,
        service_chart_output,
        trace_chart_output,
        service_metrics_output,
        request_preview_output,
        trace_preview_output,
        request_csv_output,
        trace_csv_output,
    ]

    run_button.click(
        fn=run_simulation,
        inputs=all_inputs,
        outputs=all_outputs,
    )


if __name__ == "__main__":
    demo.launch()
