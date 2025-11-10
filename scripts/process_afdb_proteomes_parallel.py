#!/usr/bin/env python3
"""
AFDB Proteome Processing Pipeline (Parallel Version)

Processes downloaded .tar proteomes in parallel as they become available:
1. Monitors downloads directory for completed .tar files
2. Extracts relevant .cif.gz files based on UniProt IDs
3. Unzips and saves CIF files to output directory
4. Deletes tar file after processing

Expected workflow:
- Run gsutil download in separate terminal: gsutil -m cp gs://bucket/proteomes/*v4.tar downloads/
- Run this script to process files as they complete


gsutil -m cp gs://public-datasets-deepmind-alphafold-v4/proteomes/*v4.tar downloads/

# Terminal 2: Process as they complete
python process_afdb_proteomes_parallel.py \
--parquet clustered_afdb.parquet \
--downloads-dir downloads/ \
--output-dir cifs/ \
--workers 128
"""

import argparse
import gzip
import logging
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Set

import polars as pl
from tqdm import tqdm


def load_uniprot_ids(parquet_path: Path, column_name: str = "uniprot_id") -> Set[str]:
    """Load UniProt IDs from parquet file."""
    df = pl.read_parquet(parquet_path)
    return set(df[column_name].to_list())


def extract_uniprot_id(filename: str) -> str:
    """Extract UniProt ID from AFDB filename (AF-{id}-F1-model_v4.cif.gz or {id}.cif.gz)."""
    if filename.startswith("AF-"):
        return filename.split("-")[1]
    return filename.split(".")[0]


def process_tar_file(
    tar_path: Path,
    uniprot_ids: Set[str],
    cif_output_dir: Path,
    temp_base_dir: Path,
) -> dict:
    """
    Process a single downloaded tar file: extract, filter, ungzip, cleanup.

    Args:
        tar_path: Path to downloaded tar file
        uniprot_ids: Set of UniProt IDs to keep
        cif_output_dir: Output directory for CIF files
        temp_base_dir: Temporary extraction directory

    Returns:
        Statistics dictionary
    """
    proteome_name = tar_path.stem
    proteome_dir = temp_base_dir / proteome_name

    stats = {"proteome": proteome_name, "kept": 0, "extracted": 0, "error": None}

    try:
        # Extract tar (only .cif.gz files)
        proteome_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r") as tar:
            cif_members = [m for m in tar.getmembers() if m.name.endswith(".cif.gz")]
            for member in cif_members:
                tar.extract(member, path=proteome_dir)
                stats["extracted"] += 1

        # Process CIF files: filter by UniProt ID, ungzip, copy to output
        for cif_gz_path in proteome_dir.rglob("*.cif.gz"):
            uniprot_id = extract_uniprot_id(cif_gz_path.name)
            if uniprot_id in uniprot_ids:
                output_path = cif_output_dir / cif_gz_path.name.replace(".gz", "")
                with gzip.open(cif_gz_path, "rb") as f_in:
                    with open(output_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                stats["kept"] += 1

    except Exception as e:
        stats["error"] = str(e)
        logging.error(f"Error processing {proteome_name}: {e}")
    finally:
        # Cleanup temp extraction directory and tar file
        shutil.rmtree(proteome_dir, ignore_errors=True)
        tar_path.unlink(missing_ok=True)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Process downloaded AFDB proteomes in parallel"
    )
    parser.add_argument(
        "--parquet", type=Path, required=True, help="Parquet file with UniProt IDs"
    )
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        required=True,
        help="Directory with downloading/downloaded tar files",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory for CIF files"
    )
    parser.add_argument(
        "--temp-dir", type=Path, default="tmp", help="Temporary extraction directory"
    )
    parser.add_argument(
        "--uniprot-column", default="uniprot_id", help="UniProt ID column name"
    )
    parser.add_argument(
        "--workers", type=int, default=32, help="Number of processing worker threads"
    )
    parser.add_argument(
        "--check-interval",
        type=float,
        default=5.0,
        help="Seconds between checking for new files",
    )
    parser.add_argument(
        "--wait-for-new",
        action="store_true",
        help="Keep running and wait for new files (default: exit when no files remain)",
    )
    args = parser.parse_args()

    # Setup
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    args.downloads_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logging.info(f"Loading UniProt IDs from {args.parquet}")
    uniprot_ids = load_uniprot_ids(args.parquet, args.uniprot_column)
    logging.info(f"Loaded {len(uniprot_ids):,} UniProt IDs")

    logging.info(f"Monitoring {args.downloads_dir} for .tar files")
    logging.info(f"Processing with {args.workers} workers")

    stats_list = []
    processed_files = set()
    no_new_files_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}

        with tqdm(desc="Processing", unit=" proteome") as pbar:
            while True:
                # Find new completed tar files (without temp extensions)
                available_tars = [
                    p
                    for p in args.downloads_dir.glob("*.tar")
                    if p.name not in processed_files
                ]

                # Submit new jobs
                for tar_path in available_tars:
                    processed_files.add(tar_path.name)
                    future = executor.submit(
                        process_tar_file,
                        tar_path,
                        uniprot_ids,
                        args.output_dir,
                        args.temp_dir,
                    )
                    futures[future] = tar_path.name

                # Collect completed jobs
                done_futures = [f for f in futures if f.done()]
                for future in done_futures:
                    filename = futures.pop(future)
                    try:
                        stats = future.result()
                        stats_list.append(stats)
                        pbar.update(1)
                        pbar.set_postfix(
                            {
                                "success": sum(1 for s in stats_list if not s["error"]),
                                "kept": sum(s["kept"] for s in stats_list),
                            }
                        )
                    except Exception as e:
                        logging.error(f"Processing error for {filename}: {e}")
                        stats_list.append(
                            {
                                "proteome": filename,
                                "kept": 0,
                                "extracted": 0,
                                "error": str(e),
                            }
                        )

                # Check exit conditions
                if not available_tars and not futures:
                    if args.wait_for_new:
                        no_new_files_count += 1
                        if no_new_files_count >= 3:
                            logging.info(
                                f"No new files for {no_new_files_count * args.check_interval:.0f}s, "
                                "continuing to wait..."
                            )
                            no_new_files_count = 0
                    else:
                        logging.info("No more files to process, exiting")
                        break
                else:
                    no_new_files_count = 0

                time.sleep(args.check_interval)

    # Summary
    success = sum(1 for s in stats_list if not s["error"])
    total_kept = sum(s["kept"] for s in stats_list)
    total_extracted = sum(s["extracted"] for s in stats_list)

    logging.info(
        f"Complete: {success}/{len(stats_list)} successful, "
        f"{total_kept:,} CIFs kept from {total_extracted:,} extracted"
    )

    failures = [s for s in stats_list if s["error"]]
    if failures:
        logging.warning(f"{len(failures)} failures:")
        for s in failures[:10]:  # Show first 10
            logging.warning(f"  {s['proteome']}: {s['error']}")


if __name__ == "__main__":
    main()
