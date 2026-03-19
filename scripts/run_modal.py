"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
    .env({"CUDA_HOME": "/usr/local/cuda"})
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(
    solution: Solution,
    config: BenchmarkConfig = None,
    max_workloads: int = 0,
    workload_uuid: str = "",
    debug_histogram: bool = False,
    debug_timing: bool = False,
) -> dict:
    """Run benchmark on Modal B200 and return results."""
    debug_mode = debug_histogram or debug_timing
    if config is None:
        if debug_mode:
            config = BenchmarkConfig(warmup_runs=0, iterations=1, num_trials=1)
        else:
            config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    def get_workload_uuid(item) -> str | None:
        if hasattr(item, "uuid"):
            return item.uuid
        workload = getattr(item, "workload", None)
        if workload is not None and hasattr(workload, "uuid"):
            return workload.uuid
        return None

    if debug_histogram:
        os.environ["MOE_DEBUG_HISTOGRAM"] = "1"
    else:
        os.environ.pop("MOE_DEBUG_HISTOGRAM", None)
    if debug_timing:
        os.environ["MOE_DEBUG_TIMING"] = "1"
    else:
        os.environ.pop("MOE_DEBUG_TIMING", None)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    if workload_uuid:
        workloads = [workload for workload in workloads if get_workload_uuid(workload) == workload_uuid]
        if not workloads:
            raise ValueError(f"Workload '{workload_uuid}' not found for definition '{solution.definition}'")

    if max_workloads > 0:
        workloads = workloads[:max_workloads]

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.log:
                entry["log"] = trace.evaluation.log
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

            if trace.evaluation.log and trace.evaluation.status.value != "SUCCESS":
                print(f"\nDetailed log for workload {trace.workload.uuid}:")
                print(trace.evaluation.log)

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()

            if result.get("log") and status != "SUCCESS":
                print("    log:")
                for line in result["log"].strip().splitlines():
                    print(f"      {line}")


@app.local_entrypoint()
def main(
    max_workloads: int = 0,
    workload_uuid: str = "",
    debug_histogram: bool = False,
    debug_timing: bool = False,
):
    """Pack solution and run benchmark on Modal."""
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark on Modal B200...")
    if max_workloads > 0:
        print(f"Debug mode: limiting run to {max_workloads} workload(s)")
    if workload_uuid:
        print(f"Filtering to workload: {workload_uuid}")
    if debug_histogram or debug_timing:
        enabled = []
        if debug_histogram:
            enabled.append("histogram")
        if debug_timing:
            enabled.append("timing")
        print(f"Kernel debug enabled: {', '.join(enabled)}")
        print("Using reduced benchmark config for diagnostics: warmup=0, iterations=1, trials=1")
    results = run_benchmark.remote(
        solution,
        max_workloads=max_workloads,
        workload_uuid=workload_uuid,
        debug_histogram=debug_histogram,
        debug_timing=debug_timing,
    )

    if not results:
        print("No results returned!")
        return

    print_results(results)
