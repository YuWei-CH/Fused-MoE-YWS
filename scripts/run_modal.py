"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import json
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

CUTLASS_INCLUDE_FALLBACKS = [
    "/opt/conda/envs/py312/lib/python3.12/site-packages/flashinfer/data/cutlass/include",
    "/opt/conda/envs/py312/lib/python3.12/site-packages/tilelang/3rdparty/cutlass/include",
]
CUTLASS_INCLUDE_PATH = ":".join(CUTLASS_INCLUDE_FALLBACKS)

image = (
    modal.Image.from_registry(
        "flashinfer/flashinfer-ci-cu132:latest",
    )
    .pip_install("flashinfer-bench")
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            # Expose preinstalled CUTLASS headers to the CUDA C++ builder.
            "CPLUS_INCLUDE_PATH": CUTLASS_INCLUDE_PATH,
            "C_INCLUDE_PATH": CUTLASS_INCLUDE_PATH,
        }
    )
)


def parse_workload_uuid_list(arg: str) -> list[str]:
    """Parse a comma-separated workload UUID list."""
    if not arg:
        return []
    uuids = []
    for item in arg.replace("\n", ",").split(","):
        item = item.strip()
        if item:
            uuids.append(item)
    return uuids


def dedupe_preserve_order(items: list[str]) -> list[str]:
    """Deduplicate workload UUIDs while preserving order."""
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={TRACE_SET_PATH: trace_volume},
    max_containers=10,
)
def run_benchmark(
    solution: Solution,
    config: BenchmarkConfig = None,
    max_workloads: int = 0,
    workload_uuid: str = "",
    kernel_profile: bool = False,
) -> dict:
    """Run benchmark on Modal B200 and return results."""
    debug_mode = kernel_profile
    profile_path = ""
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

    if kernel_profile:
        os.environ["FUSED_MOE_PROFILE"] = "1"
        workload_label = workload_uuid or "all"
        profile_path = f"/tmp/fused_moe_profile_{workload_label}.log"
        Path(profile_path).write_text("")
        os.environ["FUSED_MOE_PROFILE_PATH"] = profile_path
    else:
        os.environ.pop("FUSED_MOE_PROFILE", None)
        os.environ.pop("FUSED_MOE_PROFILE_PATH", None)

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

    kernel_profile_log = ""
    if kernel_profile and profile_path:
        profile_file = Path(profile_path)
        if profile_file.exists():
            kernel_profile_log = profile_file.read_text()

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
            if kernel_profile_log:
                if entry.get("log"):
                    entry["log"] = f"{entry['log'].rstrip()}\n{kernel_profile_log}"
                else:
                    entry["log"] = kernel_profile_log
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



def print_results(results: dict, show_logs: bool = False):
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

            if result.get("log") and (show_logs or status != "SUCCESS"):
                print("    log:")
                for line in result["log"].strip().splitlines():
                    print(f"      {line}")


@app.local_entrypoint()
def main(
    max_workloads: int = 0,
    workload_uuid: str = "",
    workload_uuids: str = "",
    kernel_profile: bool = False,
    json_out: str = "",
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
        print(f"Limiting run to {max_workloads} workload(s)")
    requested_workload_uuids = dedupe_preserve_order(
        [item for item in [workload_uuid, *parse_workload_uuid_list(workload_uuids)] if item]
    )
    if requested_workload_uuids:
        print("Filtering to workload(s):")
        for item in requested_workload_uuids:
            print(f"  - {item}")
    if kernel_profile:
        print("Kernel profiling enabled via FUSED_MOE_PROFILE=1")
        print("Using reduced benchmark config for diagnostics: warmup=0, iterations=1, trials=1")

    workload_uuids: list[str]
    if requested_workload_uuids:
        workload_uuids = requested_workload_uuids
    else:
        workload_file = (
            PROJECT_ROOT.parent
            / "mlsys26-contest"
            / "workloads"
            / "moe"
            / f"{solution.definition}.jsonl"
        )
        if not workload_file.exists():
            raise FileNotFoundError(
                f"Workload manifest not found: {workload_file}"
            )
        workload_uuids = []
        with workload_file.open() as f:
            for line in f:
                data = json.loads(line)
                workload_uuids.append(data["workload"]["uuid"])

    if max_workloads > 0:
        workload_uuids = workload_uuids[:max_workloads]

    print(
        f"Dispatching {len(workload_uuids)} workload(s) across up to 10 concurrent B200 containers..."
    )

    results = {}
    for res in run_benchmark.map(
        [solution] * len(workload_uuids),
        [None] * len(workload_uuids),
        [0] * len(workload_uuids),
        workload_uuids,
        [kernel_profile] * len(workload_uuids),
    ):
        if not res:
            continue
        for def_name, traces in res.items():
            if def_name not in results:
                results[def_name] = {}
            results[def_name].update(traces)

    if not results:
        print("No results returned!")
        return

    print_results(results, show_logs=kernel_profile)
    if json_out:
        output_path = Path(json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\nSaved JSON results to: {output_path}")
