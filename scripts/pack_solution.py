"""
Pack solution source files into solution.json.

Reads configuration from config.toml and recursively packs the source tree
under the selected solution directory.
"""

import json
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def load_config() -> dict:
    """Load configuration from config.toml."""
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def pack_solution(output_path: Path = None) -> Path:
    """Pack solution files into a Solution JSON."""
    config = load_config()

    solution_config = config["solution"]
    build_config = config["build"]

    language = build_config["language"]
    entry_point = build_config["entry_point"]

    # Determine source directory based on language
    if language == "triton":
        source_dir = PROJECT_ROOT / "solution" / "triton"
    elif language == "cuda":
        source_dir = PROJECT_ROOT / "solution" / "cuda"
    else:
        raise ValueError(f"Unsupported language: {language}")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    spec = {
        "language": language,
        "target_hardware": ["cuda"],
        "entry_point": entry_point,
    }

    if "destination_passing_style" in build_config:
        spec["destination_passing_style"] = build_config["destination_passing_style"]

    if "binding" in build_config:
        spec["binding"] = build_config["binding"]

    if "dependencies" in build_config:
        spec["dependencies"] = build_config["dependencies"]
    else:
        spec["dependencies"] = []

    sources = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        sources.append(
            {
                "path": str(path.relative_to(source_dir)).replace("\\", "/"),
                "content": path.read_text(),
            }
        )

    solution = {
        "name": solution_config["name"],
        "definition": solution_config["definition"],
        "author": solution_config["author"],
        "spec": spec,
        "sources": sources,
        "description": solution_config.get("description", ""),
    }

    # Write to output file
    if output_path is None:
        output_path = PROJECT_ROOT / "solution.json"

    output_path.write_text(json.dumps(solution, indent=2))
    print(f"Solution packed: {output_path}")
    print(f"  Name: {solution['name']}")
    print(f"  Definition: {solution['definition']}")
    print(f"  Author: {solution['author']}")
    print(f"  Language: {language}")

    return output_path


def main():
    """Entry point for pack_solution script."""
    import argparse

    parser = argparse.ArgumentParser(description="Pack solution files into solution.json")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for solution.json (default: ./solution.json)"
    )
    args = parser.parse_args()

    try:
        pack_solution(args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
