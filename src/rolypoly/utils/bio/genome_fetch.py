"""
Genome downloading using pre-computed rRNA to genome mappings.

Replaces ncbi-datasets + taxonkit dependencies with direct FTP downloads
using pre-computed taxonomy-based mappings. Prefers transcript files over
genome files and constrains matches to the same genus.

Usage:
    # Fetch genomes for a list of taxids
    fetch_genomes_from_mapping(
        taxids=[9606, 10090],
        taxid_lookup_path="data/contam/rrna/rrna_to_genome_mapping.parquet",
        output_file="genomes.fasta",
        prefer_transcript=True
    )

    # Drop-in replacement for old fetch_genomes() # internally it calls the one above...
    fetch_genomes_from_stats_file(
        stats_file="bbmap_stats.txt",
        taxid_lookup_path="mapping.parquet",
        output_file="genomes.fasta",
        max_genomes=5
    )

Test:
    python src/tests/test_genome_fetch_mapping.py
"""

import gzip
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse
from urllib.request import urlretrieve

import polars as pl
from requests import get
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from rolypoly.utils.bio.sequences import remove_duplicates
from rolypoly.utils.logging.loggit import get_logger

global data_dir, taxid_lookup_path
data_dir = Path(os.environ.get("ROLYPOLY_DATA", ""))
taxid_lookup_path = data_dir / "contam/rrna/rrna_to_genome_mapping.parquet"


def get_ftp_path_for_taxid(
    taxid: int, mapping_df: pl.DataFrame, get_relative_for_missing: bool = False
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Get the best FTP path for a given taxonomy ID with relationship info.

    Args:
        taxid: NCBI taxonomy ID
        mapping_df: The mapping DataFrame (simplified format)
        get_relative_for_missing: If False, only use 'self' data; if True, allow ancestor/relative

    Returns:
        Tuple of (ftp_path, relationship, reference_name) or (None, None, None) if no mapping found
        relationship: 'self' | 'ancestor' | 'relative' | None
    """
    # Query the mapping for this taxid
    result = mapping_df.filter(pl.col("query_tax_id") == taxid)

    if result.height == 0:
        return None, None, None

    row = result.row(0, named=True)
    relationship = row.get("relationship")
    ftp_path = row.get("ftp_path")
    reference_name = row.get("reference_name")

    # If self data available, use it
    if relationship == "self" and ftp_path:
        return ftp_path, relationship, reference_name

    # If allowing relatives, try ancestor/relative
    if (
        get_relative_for_missing
        and relationship in ["ancestor", "relative"]
        and ftp_path
    ):
        return ftp_path, relationship, reference_name

    # No suitable mapping found
    return None, None, None


def download_from_ftp_path(
    ftp_path: str,
    output_dir: Path,
    prefer_transcript: bool = True,
    overwrite: bool = False,
    logger=None,
) -> Tuple[Optional[Path], Optional[int]]:
    """Download sequence files from an NCBI FTP path.

    Args:
        ftp_path: Base FTP path (e.g., https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/...)
        output_dir: Directory to save downloaded files
        prefer_transcript: If True, try to download *_cds_from_genomic.fna.gz first
        overwrite: If True, re-download even if file exists

    Returns:
        Tuple of (Path to the downloaded file, file size in bytes) or (None, None) if download failed
    """
    logger = get_logger(logger)

    if not ftp_path:
        return None, None

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract the assembly name from the FTP path
    # e.g., GCF_000001405.40_GRCh38.p14 from the URL
    assembly_name = Path(urlparse(ftp_path).path).name

    # File types to try, in order of preference
    if prefer_transcript:
        file_suffixes = [
            "_cds_from_genomic.fna.gz",  # Transcript sequences
            "_genomic.fna.gz",  # Genome sequences
        ]
    else:
        file_suffixes = [
            "_genomic.fna.gz",  # Genome sequences
            "_cds_from_genomic.fna.gz",  # Transcript sequences
        ]

    for suffix in file_suffixes:
        filename = f"{assembly_name}{suffix}"
        url = f"{ftp_path}/{filename}"
        output_file = output_dir / filename

        # Check if file already exists
        if output_file.exists() and not overwrite:
            file_size = output_file.stat().st_size
            return output_file, file_size

        try:
            urlretrieve(url, output_file)

            # Verify the download (basic check)
            if output_file.exists() and output_file.stat().st_size > 0:
                file_size = output_file.stat().st_size
                return output_file, file_size
            else:
                logger.warning(f"Downloaded file is empty, trying next option")
                output_file.unlink(missing_ok=True)

        except Exception as e:
            logger.warning(f"Could not download {filename}: {e}")
            output_file.unlink(missing_ok=True)
            continue

    logger.error(f"No files could be downloaded from: {ftp_path}")
    return None, None


def fetch_genomes_by_taxid(
    taxids: List[int],
    taxid_lookup_path: str,
    output_file: str,
    temp_dir: Optional[str] = None,
    prefer_transcript: bool = True,
    get_relative_for_missing: bool = True,
    threads: int = 1,
    overwrite: bool = False,
    exclude_viral: bool = True,
    clean_headers=True,  # currently part of the exclude virus if else, for now no need to take it out.
    logger=None,
) -> None:
    """Fetch genome/transcript sequences for a list of taxids using pre-computed mappings.

    This replaces the ncbi-datasets + taxonkit workflow with a direct FTP download
    approach using pre-computed rRNA to genome mappings.

    Args:
        taxids: List of NCBI taxonomy IDs
        taxid_lookup_path: Path to the rRNA genome mapping parquet file
        output_file: Output fasta file path
        temp_dir: Temporary directory for downloads (default: creates one in current dir)
        prefer_transcript: If True, prefer transcript files over genome files
        threads: Number of parallel downloads
        overwrite: If True, re-download existing files
        exclude_viral: If True, filter out viral sequences from final output
        clean_headers: always true, not yet implemented as an option
        logger: Logger instance
        get_relative_for_missing: If True, attempt to use relative genome if no direct match found
    """
    if logger is None:
        logger = get_logger(logger)
    # breakpoint()
    # Setup directories
    if temp_dir is None:
        temp_dir = "tmp_genome_downloads"
    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)

    # Load mapping
    logger.debug(f"Loading mapping from: {taxid_lookup_path}")
    mapping_df = pl.read_parquet(taxid_lookup_path)

    subdf = mapping_df.filter(pl.col("query_tax_id").is_in(list(taxids)))
    avail_taxids = subdf.select("query_tax_id").unique().to_series().to_list()
    unmapped_names = list(set(taxids) - set(avail_taxids))
    if unmapped_names:
        logger.warning(
            f"Rolypoly pre-generated mapping missing for {len(unmapped_names)} taxids"
        )
        logger.warning(f"Examples: {unmapped_names[:5]}")

    # Track downloaded files
    downloaded_files: List[Path] = []
    unmapped_taxids: List[int] = []
    download_info: List[
        Tuple[int, Path, Optional[str], Optional[str], Optional[int]]
    ] = []

    # Worker function for parallel processing
    def process_taxid(
        taxid: int,
    ) -> Tuple[
        int, Optional[Path], Optional[str], Optional[str], Optional[int]
    ]:
        """Process a single taxid and return (taxid, downloaded_path, relationship, reference_name, file_size)."""
        ftp_path, relationship, reference_name = get_ftp_path_for_taxid(
            taxid, mapping_df, get_relative_for_missing
        )

        if ftp_path is None:
            return (taxid, None, None, None, None)

        # logger.debug(f"Downloading from FTP path: {ftp_path}")
        downloaded, file_size = download_from_ftp_path(
            ftp_path=ftp_path,
            output_dir=temp_path,
            prefer_transcript=prefer_transcript,
            overwrite=overwrite,
            logger=logger,
        )

        return (taxid, downloaded, relationship, reference_name, file_size)

    # Process all taxids
    logger.info(
        f"Fetching sequences for {len(taxids)} taxids using {threads} thread(s)..."
    )

    # Create progress bar
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            "[cyan]Downloading genomes...", total=len(taxids)
        )

        if threads > 0:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=threads) as executor:
                futures = {
                    executor.submit(process_taxid, taxid): taxid
                    for taxid in taxids
                }
                logger.debug(f"Submitted {len(futures)} download tasks")

                for future in as_completed(futures):
                    taxid = futures[future]
                    try:
                        (
                            result_taxid,
                            downloaded,
                            relationship,
                            reference_name,
                            file_size,
                        ) = future.result()
                        progress.update(task, advance=1)

                        if downloaded is None:
                            unmapped_taxids.append(result_taxid)
                        else:
                            downloaded_files.append(downloaded)
                            download_info.append(
                                (
                                    result_taxid,
                                    downloaded,
                                    relationship,
                                    reference_name,
                                    file_size,
                                )
                            )

                    except Exception as e:
                        logger.error(f"Error processing taxid {taxid}: {e}")
                        unmapped_taxids.append(taxid)
                        progress.update(task, advance=1)
            # # Sequential processing
            # for taxid in taxids:
            #     result_taxid, downloaded, relationship, reference_name, file_size = process_taxid(taxid)
            #     progress.update(task, advance=1)

            #     if downloaded is None:
            #         unmapped_taxids.append(result_taxid)
            #     else:
            #         downloaded_files.append(downloaded)
            #         download_info.append((result_taxid, downloaded, relationship, reference_name, file_size))

    # Report statistics
    logger.info(f"Downloaded {len(downloaded_files)} genome files")
    for taxid, path, rel, ref, size in download_info:
        size_mb = size / (1024 * 1024) if size else 0
        if rel != "self":
            logger.info(
                f"Taxid {taxid} - downloaded {path.name} ({size_mb:.1f} MB) using {rel} reference"
            )
        else:
            logger.info(
                f"Taxid {taxid} - downloaded {path.name} ({size_mb:.1f} MB)"
            )
    if unmapped_taxids:
        logger.warning(
            f"Could not map {len(unmapped_taxids)} taxids: {unmapped_taxids}"
        )

    if not downloaded_files:
        logger.error("No genome files were downloaded. Exiting.")
        return

    # Decompress and concatenate files
    logger.info("Processing downloaded files...")
    temp_concat = temp_path / "concat_genomes.fasta"

    with open(temp_concat, "w") as out_f:
        for i, gz_file in enumerate(downloaded_files, 1):
            logger.debug(
                f"Decompressing {gz_file.name} ({i}/{len(downloaded_files)})"
            )
            try:
                with gzip.open(gz_file, "rt") as in_f:
                    shutil.copyfileobj(in_f, out_f)
            except Exception as e:
                logger.error(f"Error processing {gz_file}: {e}")

    # Deduplicate sequences
    logger.info("Deduplicating sequences...")
    temp_dedup = temp_path / "dedup_genomes.fasta"

    try:
        remove_duplicates(
            input_file=str(temp_concat),
            output_file=str(temp_dedup),
            by="seq",
            logger=logger,
        )
    except Exception as e:
        logger.error(f"Error during deduplication: {e}")
        shutil.copy(temp_concat, temp_dedup)

    # Filter viral sequences if requested
    if exclude_viral:
        logger.info(
            "Filtering out headers with 'phage', 'virus', and 'viral' in headers..."
        )
        from rolypoly.utils.bio.sequences import (
            clean_fasta_headers,
            filter_fasta_by_headers,
        )

        temp_dedup1 = str(temp_dedup) + "1"
        counts = filter_fasta_by_headers(
            str(temp_dedup),
            ["virus", "viral", "phage"],
            temp_dedup1,
            wrap=True,
            invert=True,
            return_counts=True,
        )
        logger.info(
            f"Filtered sequences: removed {counts['records_processed'] - counts['records_written']} sequences; {counts['records_written']} written, {counts['records_processed']} processed"
        )

        clean_fasta_headers(
            fasta_file=temp_dedup1,
            drop_from_space=True,
            output_file=output_file,
            strip_prefix="lcl|",
            strip_suffix="bla bla bla",
        )

    else:
        shutil.copy(temp_dedup, output_file)

    logger.info(f"Final output written to: {output_file}")

    # Cleanup
    if not overwrite:  # Keep temp files if we might need them
        shutil.rmtree(temp_path, ignore_errors=True)


def fetch_genomes_from_stats_file(
    stats_file: str,
    taxid_lookup_path: str,
    output_file: str,
    max_genomes: int = 5,
    logger=None,
    **kwargs,
) -> None:
    """Fetch genomes based on a BBMap stats file, using the mapping table.

    Prioritizes taxids that have direct data available and limits to max_genomes.
    Reports which taxids were not available and which used relative references.

    Note:
        The reference file for the bb command needs to follow the @ notation,
        i.e. the first field if splitting on @, should be the ncbi taxid

    Args:
        stats_file: BBMap stats file with taxonomic information
        taxid_lookup_path: Path to the rRNA genome mapping parquet file
        output_file: Output fasta file path
        max_genomes: Maximum number of genomes to fetch (from available ones)
        **kwargs: Additional arguments passed to fetch_genomes_by_taxid
    """
    if logger is None:
        logger = get_logger()

    # Parse the stats file to extract taxon IDs
    logger.info(f"Parsing stats file: {stats_file}")

    # Read the BBMap stats file (print first 3 lines, then parse from line 4)
    with open(stats_file, "r") as f:
        lines = f.readlines()
    for line in lines[:3]:
        logger.info(f"{line.strip()}")

    # Parse all taxons from stats file
    all_taxons: List[int] = []
    for line in lines[4:]:
        taxon_str = line.split(sep="@")[0]
        try:
            taxon = int(taxon_str)
            all_taxons.append(taxon)
        except ValueError:
            logger.warning(f"Skipping non-integer taxon entry: {taxon_str}")
            continue

    logger.info(f"Found {len(all_taxons)} taxon entries in stats file")

    # Load the mapping to check availability
    mapping_df = pl.read_parquet(taxid_lookup_path)

    # Filter to taxons with data available
    available_taxons = []
    unavailable_taxons = []  # (taxid, name, relationship)
    taxon_details = {}  # Cache mapping details for later logging

    for taxon in all_taxons:
        result = mapping_df.filter(pl.col("query_tax_id") == taxon)

        if result.height > 0:
            row = result.row(0, named=True)
            relationship = row.get("relationship")
            ftp_path = row.get("ftp_path")
            reference_name = row.get("reference_name")
            query_name = row.get("query_name")

            if ftp_path and relationship:  # Has valid mapping
                available_taxons.append(taxon)
                taxon_details[taxon] = (
                    query_name,
                    relationship,
                    reference_name,
                )
            else:
                # Has entry but no FTP path
                unavailable_taxons.append((taxon, query_name, relationship))
        else:
            # No entry in mapping
            unavailable_taxons.append((taxon, "<unknown>", None))
    unavailable_taxons = list(set(unavailable_taxons))  # Remove duplicates
    available_taxons = list(set(available_taxons))  # Remove duplicates
    logger.info(f"Taxons with available data: {len(available_taxons)}")
    logger.info(f"Taxons without available data: {len(unavailable_taxons)}")

    if unavailable_taxons:
        logger.warning("Unavailable taxons:")
        for taxid, name, relationship in unavailable_taxons[:5]:  # Show first 5
            status = f"using {relationship}" if relationship else "not found"
            logger.warning(f"  - {taxid} ({name}): {status}")
        if len(unavailable_taxons) > 5:
            logger.warning(f"  ... and {len(unavailable_taxons) - 5} more")

    # Take up to max_genomes from available taxons
    taxons_to_fetch = available_taxons[:max_genomes]

    # Log relationship info only for taxons that will be fetched
    for taxon in taxons_to_fetch:
        if taxon in taxon_details:
            query_name, relationship, reference_name = taxon_details[taxon]
            if relationship != "self":
                logger.info(
                    f"Taxon {taxon} (is {query_name}) but not direct data available, "
                    f"using {relationship} reference: {reference_name}"
                )

    if not taxons_to_fetch:
        logger.error("No taxons with available data found. Exiting.")
        return

    logger.info(
        f"Fetching genomes for {len(taxons_to_fetch)} taxons (out of {len(available_taxons)} available)"
    )

    # Fetch genomes for the mapped taxids
    fetch_genomes_by_taxid(
        taxids=taxons_to_fetch,
        get_relative_for_missing=True,
        taxid_lookup_path=taxid_lookup_path,
        output_file=output_file,
        logger=logger,
        **kwargs,
    )
