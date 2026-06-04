import argparse
import fnmatch
import json
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "hf_source"
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "download_manifest.json"
DEFAULT_INDEX_PATH = DEFAULT_OUTPUT_DIR / "local_parquet_files.json"
DEFAULT_REPO_ID = "Skylion007/openwebtext"
DEFAULT_PATTERNS = [
    "*.parquet",
    "*.arrow",
    "*.jsonl",
    "*.jsonl.gz",
    "*.jsonl.zst",
    "*.txt",
]
SKIP_PATTERNS = [
    ".gitattributes",
    "README.md",
    "dataset_infos.json",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download OpenWebText dataset shards from Hugging Face one file at a time. "
            "The script is resumable: if it is interrupted, rerun it and the remaining "
            "files will continue downloading."
        )
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--repo-type",
        default="dataset",
        choices=["dataset", "model", "space"],
        help="Hugging Face repo type.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to store downloaded shards. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Manifest JSON path. Default: {DEFAULT_MANIFEST_PATH}",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help=f"JSON file with local parquet paths. Default: {DEFAULT_INDEX_PATH}",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        help=(
            "Glob pattern for dataset files inside the HF repo. "
            "Can be passed multiple times. Defaults cover parquet/jsonl/text shards."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Only download the first N matched files. Useful for smoke testing.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list matched files, do not download.",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Ask Hugging Face Hub to redownload files even if they already exist locally.",
    )
    return parser.parse_args()


def load_manifest(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True, sort_keys=True)


def write_local_index(index_path, output_dir):
    parquet_files = sorted(str(path.resolve()) for path in output_dir.rglob("*.parquet"))
    index_payload = {
        "generated_at_unix": int(time.time()),
        "root_dir": str(output_dir.resolve()),
        "parquet_files": parquet_files,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index_payload, f, indent=2, ensure_ascii=True)
    return parquet_files


def matches_any_pattern(path_str, patterns):
    return any(fnmatch.fnmatch(path_str, pattern) for pattern in patterns)


def discover_repo_files(api, repo_id, repo_type, patterns):
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    matched_files = []
    for repo_file in repo_files:
        if repo_file in SKIP_PATTERNS:
            continue
        if matches_any_pattern(repo_file, patterns):
            matched_files.append(repo_file)
    return sorted(matched_files)


def print_prepare_hint(index_path):
    print()
    print("Download index written to:")
    print(f"  {index_path}")
    print()
    print("After all shards are ready, you usually do not need to merge them physically.")
    print("`datasets.load_dataset` can load multiple local parquet shards directly.")
    print("Example change inside prepare.py:")
    print()
    print("  from datasets import load_dataset")
    print("  import json")
    print("  with open(r'%s', 'r', encoding='utf-8') as f:" % index_path.resolve())
    print("      data_files = json.load(f)['parquet_files']")
    print("  dataset = load_dataset('parquet', data_files={'train': data_files})")
    print()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    manifest_path = args.manifest_path.resolve()
    index_path = args.index_path.resolve()
    patterns = args.patterns or DEFAULT_PATTERNS

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    api = HfApi()

    print(f"Listing repo files from {args.repo_type}:{args.repo_id} ...")
    matched_files = discover_repo_files(api, args.repo_id, args.repo_type, patterns)

    if args.max_files is not None:
        matched_files = matched_files[: args.max_files]

    if not matched_files:
        print("No files matched the requested patterns.")
        print(f"Patterns: {patterns}")
        sys.exit(1)

    print(f"Matched {len(matched_files)} files.")
    for repo_file in matched_files[:10]:
        print(f"  {repo_file}")
    if len(matched_files) > 10:
        print(f"  ... and {len(matched_files) - 10} more")

    if args.list_only:
        print("List-only mode enabled, exiting without downloading.")
        return

    downloaded = 0
    skipped = 0
    failed = 0

    for index, repo_file in enumerate(matched_files, start=1):
        print(f"[{index}/{len(matched_files)}] {repo_file}")
        try:
            local_path = hf_hub_download(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                filename=repo_file,
                local_dir=output_dir,
                force_download=args.force_redownload,
            )
            file_record = {
                "repo_file": repo_file,
                "local_path": str(Path(local_path).resolve()),
                "status": "downloaded",
                "updated_at_unix": int(time.time()),
            }
            if manifest.get(repo_file, {}).get("local_path") == file_record["local_path"]:
                skipped += 1
            else:
                downloaded += 1
            manifest[repo_file] = file_record
            save_manifest(manifest_path, manifest)
        except Exception as exc:
            failed += 1
            manifest[repo_file] = {
                "repo_file": repo_file,
                "status": "failed",
                "error": repr(exc),
                "updated_at_unix": int(time.time()),
            }
            save_manifest(manifest_path, manifest)
            print(f"  FAILED: {exc}")

    parquet_files = write_local_index(index_path, output_dir)

    print()
    print("Done.")
    print(f"  Newly downloaded/updated: {downloaded}")
    print(f"  Already present:          {skipped}")
    print(f"  Failed:                   {failed}")
    print(f"  Local parquet files:      {len(parquet_files)}")
    print(f"  Manifest:                 {manifest_path}")
    print(f"  Local shard index:        {index_path}")

    if failed:
        print()
        print("Some files failed. You can rerun the same command and it will resume the rest.")
        sys.exit(2)

    print_prepare_hint(index_path)


if __name__ == "__main__":
    main()
