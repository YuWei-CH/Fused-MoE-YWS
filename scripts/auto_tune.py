import subprocess
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TILE_CONFIG_PATH = PROJECT_ROOT / "solution" / "cuda" / "tile_config.h"

def run_sweep(phase):
    print(f"\n======================================")
    print(f"Running automated tuning for: {phase}")
    print(f"======================================")
    cmd = ["conda", "run", "--no-capture-output", "-n", "fi-bench", "python", "scripts/sweep.py", phase]
    process = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, text=True)
    
    best_candidate = None
    best_speedup = 0.0
    
    for line in process.stdout:
        print(line, end="")
        if "candidate" in line and "PASSED" in line:
            speedups = [float(s.replace("x", "")) for s in re.findall(r"(\d+\.\d+)x", line)]
            if speedups:
                avg_speedup = sum(speedups) / len(speedups)
                if avg_speedup > best_speedup:
                    best_speedup = avg_speedup
                    best_candidate = line
    
    process.wait()
    
    if best_candidate:
        print(f"\n>>> Best config for {phase} selected with avg speedup {best_speedup:.2f}x:")
        print(f"    {best_candidate.strip()}")
        
        m = re.search(r"GEMM1=(\d+)x(\d+)x(\d+)", best_candidate)
        m2 = re.search(r"GEMM2=(\d+)x(\d+)x(\d+)", best_candidate)
        th1 = re.search(r"TH1=(\d+)", best_candidate)
        th2 = re.search(r"TH2=(\d+)", best_candidate)
        
        text = TILE_CONFIG_PATH.read_text()
        if m:
            text = re.sub(r"inline constexpr int kGemm1TileM = \d+;", f"inline constexpr int kGemm1TileM = {m.group(1)};", text)
            text = re.sub(r"inline constexpr int kGemm1TileN = \d+;", f"inline constexpr int kGemm1TileN = {m.group(2)};", text)
            text = re.sub(r"inline constexpr int kGemm1TileK = \d+;", f"inline constexpr int kGemm1TileK = {m.group(3)};", text)
        if m2:
            text = re.sub(r"inline constexpr int kGemm2TileM = \d+;", f"inline constexpr int kGemm2TileM = {m2.group(1)};", text)
            text = re.sub(r"inline constexpr int kGemm2TileN = \d+;", f"inline constexpr int kGemm2TileN = {m2.group(2)};", text)
            text = re.sub(r"inline constexpr int kGemm2TileK = \d+;", f"inline constexpr int kGemm2TileK = {m2.group(3)};", text)
        if th1:
            text = re.sub(r"inline constexpr int kLargeGemm1TensorCoreThreshold = \d+;", f"inline constexpr int kLargeGemm1TensorCoreThreshold = {th1.group(1)};", text)
        if th2:
            text = re.sub(r"inline constexpr int kLargeGemm2TensorCoreThreshold = \d+;", f"inline constexpr int kLargeGemm2TensorCoreThreshold = {th2.group(1)};", text)
        TILE_CONFIG_PATH.write_text(text)
        print(">>> Updated tile_config.h with best parameters permanently.\n")
    else:
        print(f"\n>>> No valid candidate found for {phase}. Retaining previous configuration.\n")

def main():
    run_sweep("threshold")
    run_sweep("phase1")
    run_sweep("phase2")
    print("Auto-tuning fully complete! The optimal parameters are now saved in tile_config.h.")

if __name__ == "__main__":
    main()
