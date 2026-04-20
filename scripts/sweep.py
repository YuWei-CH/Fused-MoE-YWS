"""Simple reproducible Modal sweep for GEMM tile tuning.

Usage:
    python scripts/sweep.py phase1
    python scripts/sweep.py phase2
    python scripts/sweep.py threshold

The candidate lists are intentionally hard-coded in this file.
This script only sweeps the large workloads and prints latency / speedup / timeout.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RUN_MODAL_SCRIPT = PROJECT_ROOT / "scripts" / "run_modal.py"
PACK_SOLUTION_SCRIPT = PROJECT_ROOT / "scripts" / "pack_solution.py"
TILE_CONFIG_PATH = PROJECT_ROOT / "solution" / "cuda" / "tile_config.h"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "tile_sweeps"
DEFINITION = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"

@dataclass(frozen=True)
class TileSet:
    m: int
    n: int
    k: int

    def __str__(self) -> str:
        return f"{self.m}x{self.n}x{self.k}"


@dataclass
class ConfigState:
    gemm1: TileSet
    gemm2: TileSet
    gemm1_threshold: int
    gemm2_threshold: int


LARGE_WORKLOADS = [
    "5e8dc11c-f2a9-42d5-8dce-9419cbf34d5d",
    "58a34f27-7995-4155-8b46-f60a7225e20e",
]

PHASE1_FIXED_GEMM2 = TileSet(8, 32, 64)
PHASE1_GEMM1_CANDIDATES = [
    TileSet(16, 32, 64),
    TileSet(8, 64, 64),
    TileSet(16, 32, 32),
    TileSet(8, 64, 32),
    TileSet(8, 32, 32),
]

PHASE2_FIXED_GEMM1 = TileSet(16, 32, 64)
PHASE2_GEMM2_CANDIDATES = [
    TileSet(16, 32, 64),
    TileSet(8, 64, 64),
    TileSet(16, 32, 32),
    TileSet(8, 64, 32),
    TileSet(8, 32, 32),
]

THRESHOLD_CANDIDATES = [
    32,
    64,
    128,
    256,
    512,
]


@dataclass
class WorkloadResult:
    workload_uuid: str
    status: str
    latency_ms: float | None
    speedup_factor: float | None
    log_path: Path
    json_path: Path


def read_tile_config() -> ConfigState:
    text = TILE_CONFIG_PATH.read_text()

    def extract(prefix: str) -> TileSet:
        pattern = (
            rf"inline constexpr int k{prefix}TileM = (\d+);.*?"
            rf"inline constexpr int k{prefix}TileN = (\d+);.*?"
            rf"inline constexpr int k{prefix}TileK = (\d+);"
        )
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            raise RuntimeError(f"could not parse {prefix} tiles from {TILE_CONFIG_PATH}")
        return TileSet(*(int(group) for group in match.groups()))

    gemm1_threshold_match = re.search(
        r"inline constexpr int kLargeGemm1TensorCoreThreshold = (\d+);", text
    )
    gemm2_threshold_match = re.search(
        r"inline constexpr int kLargeGemm2TensorCoreThreshold = (\d+);", text
    )
    if not gemm1_threshold_match or not gemm2_threshold_match:
        raise RuntimeError(f"could not parse tensor core thresholds from {TILE_CONFIG_PATH}")

    return ConfigState(
        gemm1=extract("Gemm1"),
        gemm2=extract("Gemm2"),
        gemm1_threshold=int(gemm1_threshold_match.group(1)),
        gemm2_threshold=int(gemm2_threshold_match.group(1)),
    )


def write_tile_config(
    gemm1: TileSet, gemm2: TileSet, gemm1_threshold: int, gemm2_threshold: int
) -> None:
    text = TILE_CONFIG_PATH.read_text()
    replacements = {
        r"inline constexpr int kGemm1TileM = \d+;": f"inline constexpr int kGemm1TileM = {gemm1.m};",
        r"inline constexpr int kGemm1TileN = \d+;": f"inline constexpr int kGemm1TileN = {gemm1.n};",
        r"inline constexpr int kGemm1TileK = \d+;": f"inline constexpr int kGemm1TileK = {gemm1.k};",
        r"inline constexpr int kGemm2TileM = \d+;": f"inline constexpr int kGemm2TileM = {gemm2.m};",
        r"inline constexpr int kGemm2TileN = \d+;": f"inline constexpr int kGemm2TileN = {gemm2.n};",
        r"inline constexpr int kGemm2TileK = \d+;": f"inline constexpr int kGemm2TileK = {gemm2.k};",
        r"inline constexpr int kLargeGemm1TensorCoreThreshold = \d+;":
            f"inline constexpr int kLargeGemm1TensorCoreThreshold = {gemm1_threshold};",
        r"inline constexpr int kLargeGemm2TensorCoreThreshold = \d+;":
            f"inline constexpr int kLargeGemm2TensorCoreThreshold = {gemm2_threshold};",
    }
    updated = text
    for pattern, replacement in replacements.items():
        updated, count = re.subn(pattern, replacement, updated, count=1)
        if count != 1:
            raise RuntimeError(f"failed to update tile config with pattern: {pattern}")
    TILE_CONFIG_PATH.write_text(updated)


def timestamped_dir(phase: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ARTIFACTS_ROOT / f"{phase}_{ts}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_header(title: str) -> None:
    print(f"\n== {title} ==")


def run_and_tee(cmd: list[str], cwd: Path, log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        lines: list[str] = []
        for line in process.stdout:
            print(line, end="")
            f.write(line)
            lines.append(line)
        return_code = process.wait()
    return return_code, "".join(lines)


def pack_solution(output_dir: Path) -> None:
    print_header("Pack Solution")
    log_path = output_dir / "pack_solution.log"
    cmd = [sys.executable, str(PACK_SOLUTION_SCRIPT)]
    return_code, _ = run_and_tee(cmd, PROJECT_ROOT, log_path)
    if return_code != 0:
        raise RuntimeError(f"pack_solution failed, see {log_path}")


def load_workload_result(json_path: Path, workload_uuid: str) -> WorkloadResult:
    payload = json.loads(json_path.read_text())
    definition_results = payload.get(DEFINITION, {})
    if workload_uuid not in definition_results:
        raise RuntimeError(f"workload {workload_uuid} not found in {json_path}")
    result = definition_results[workload_uuid]
    return WorkloadResult(
        workload_uuid=workload_uuid,
        status=result.get("status", "UNKNOWN"),
        latency_ms=result.get("latency_ms"),
        speedup_factor=result.get("speedup_factor"),
        log_path=Path(),
        json_path=json_path,
    )


def run_one_workload(workload_uuid: str, output_dir: Path) -> WorkloadResult:
    slug = workload_uuid.split("-")[0]
    json_path = output_dir / f"{slug}.json"
    log_path = output_dir / f"{slug}.log"
    cmd = [
        "modal",
        "run",
        str(RUN_MODAL_SCRIPT.relative_to(PROJECT_ROOT)),
        "--workload-uuid",
        workload_uuid,
        "--json-out",
        str(json_path),
    ]
    print_header(f"Run {workload_uuid}")
    return_code, _ = run_and_tee(cmd, PROJECT_ROOT, log_path)
    if return_code != 0:
        raise RuntimeError(f"modal run failed for {workload_uuid}, see {log_path}")
    result = load_workload_result(json_path, workload_uuid)
    result.log_path = log_path
    return result


def print_candidate_summary(
    index: int,
    gemm1: TileSet,
    gemm2: TileSet,
    gemm1_threshold: int,
    gemm2_threshold: int,
    results: list[WorkloadResult],
) -> None:
    pieces = [
        f"candidate {index}: GEMM1={gemm1} GEMM2={gemm2} "
        f"TH1={gemm1_threshold} TH2={gemm2_threshold}"
    ]
    for result in results:
        slug = result.workload_uuid.split("-")[0]
        if result.status == "PASSED" and result.latency_ms is not None and result.speedup_factor is not None:
            pieces.append(f"{slug} PASSED {result.latency_ms:.3f} ms {result.speedup_factor:.2f}x")
        else:
            pieces.append(f"{slug} {result.status}")
    print(" | ".join(pieces))


def sweep_phase(phase: str) -> None:
    original = read_tile_config()
    output_dir = timestamped_dir(phase)

    if phase == "phase1":
        candidates = [
            (gemm1, PHASE1_FIXED_GEMM2, original.gemm1_threshold, original.gemm2_threshold)
            for gemm1 in PHASE1_GEMM1_CANDIDATES
        ]
    elif phase == "phase2":
        candidates = [
            (PHASE2_FIXED_GEMM1, gemm2, original.gemm1_threshold, original.gemm2_threshold)
            for gemm2 in PHASE2_GEMM2_CANDIDATES
        ]
    else:
        candidates = [
            (original.gemm1, original.gemm2, threshold, threshold)
            for threshold in THRESHOLD_CANDIDATES
        ]

    try:
        print_header(f"Sweep {phase}")
        print(f"Artifacts: {output_dir}")
        for index, (gemm1, gemm2, gemm1_threshold, gemm2_threshold) in enumerate(candidates, start=1):
            candidate_dir = output_dir / (
                f"candidate_{index}_{gemm1}_{gemm2}_th1_{gemm1_threshold}_th2_{gemm2_threshold}"
            )
            print_header(
                f"Candidate {index}: GEMM1={gemm1} GEMM2={gemm2} "
                f"TH1={gemm1_threshold} TH2={gemm2_threshold}"
            )
            write_tile_config(gemm1, gemm2, gemm1_threshold, gemm2_threshold)
            pack_solution(candidate_dir)
            results = [run_one_workload(workload_uuid, candidate_dir) for workload_uuid in LARGE_WORKLOADS]
            print_candidate_summary(index, gemm1, gemm2, gemm1_threshold, gemm2_threshold, results)
    finally:
        write_tile_config(original.gemm1, original.gemm2,
                          original.gemm1_threshold, original.gemm2_threshold)
        print(
            f"\nRestored tile config to GEMM1={original.gemm1} "
            f"GEMM2={original.gemm2} TH1={original.gemm1_threshold} TH2={original.gemm2_threshold}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple Modal sweep for GEMM tile tuning.")
    parser.add_argument(
        "phase",
        choices=["phase1", "phase2", "threshold"],
        help="Sweep GEMM1, GEMM2, or the large GEMM1 Tensor Core threshold.",
    )
    args = parser.parse_args()
    sweep_phase(args.phase)


if __name__ == "__main__":
    main()
