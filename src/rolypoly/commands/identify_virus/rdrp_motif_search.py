import os
import json
import shutil
import multiprocessing
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Union

import polars as pl
from rich.console import Console
from rich_click import Choice, command, option

from rolypoly.utils.logging.config import BaseConfig
from rolypoly.utils.bio.sequences import guess_fasta_alpha, read_fasta_needletail
from rolypoly.utils.bio.alignments import search_hmmdb
from rolypoly.utils.bio.translation import translate_6frx_seqkit, pyro_predict_orfs
from rolypoly.utils.logging.loggit import log_start_info
from rolypoly.utils.logging.citation_reminder import remind_citations
from rolypoly.utils.various import ensure_memory, run_command_comp

console = Console(width=150)


class RdRpMotifSearchConfig(BaseConfig):
    """Configuration for RdRp motif search pipeline"""

    def __init__(self, **kwargs):
        # Always treat output as a directory
        output_path = Path(kwargs.get("output", "rdrp_motif_search_output"))
        kwargs["output_dir"] = str(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        super().__init__(
            input=kwargs.get("input", ""),
            output=kwargs.get("output", ""),
            keep_tmp=kwargs.get("keep_tmp", False),
            log_file=kwargs.get("log_file"),
            threads=kwargs.get("threads", 1),
            memory=kwargs.get("memory", "4gb"),
            config_file=kwargs.get("config_file", None),
            overwrite=kwargs.get("overwrite", False),
            log_level=kwargs.get("log_level", "INFO"),
            temp_dir=kwargs.get("temp_dir", "rdrp_motif_search_tmp/"),
        )

        # RdRp motif-specific parameters
        self.search_tool = kwargs.get("search_tool", "hmmsearch")
        self.evalue = kwargs.get("evalue", 1e-2)
        self.min_score = kwargs.get("min_score", None)
        self.max_distance = kwargs.get("max_distance", 200)
        self.aa_method = kwargs.get("aa_method", "six_frame")
        self.min_orf_length = kwargs.get("min_orf_length", 15)
        self.motif_filter = kwargs.get("motif_filter", None)  # Filter by specific motif type (A, B, C, D)
        # self.taxon_filter = kwargs.get("taxon_filter", None)  # Filter by specific taxon - DISABLED: no taxon data in profiles yet
        self.output_format = kwargs.get("output_format", "tsv")  # Output format: tsv or parquet
        self.output_structure = kwargs.get("output_structure", "flat")  # Output structure: flat or nested (default: flat)
        self.include_alignment = kwargs.get("include_alignment", True)  # Include aligned region in output (default: True)
        self.name = kwargs.get("name") or Path(self.input).stem
        
        # Database paths
        self.data_dir = kwargs.get("data_dir", None)
        if self.data_dir:
            self.motif_db_path = Path(self.data_dir) / "profiles" / "hmmdbs" / "rvmt_motifs.hmm"
            self.motif_metadata_path = Path(self.data_dir) / "profiles" / "rvmt_motifs" / "motif_metadata.json"
        else:
            # Try to find database in default locations
            self.motif_db_path = self._find_motif_database()
            self.motif_metadata_path = self._find_motif_metadata()

    def _find_motif_database(self) -> Path:
        """Find the RdRp motif HMM database"""
        possible_paths = [
            Path.cwd() / "data" / "profiles" / "hmmdbs" / "rvmt_motifs.hmm",
            Path("~/.rolypoly/data/profiles/hmmdbs/rvmt_motifs.hmm").expanduser(),
            Path("/data/profiles/hmmdbs/rvmt_motifs.hmm"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        raise FileNotFoundError(
            "RdRp motif database not found. "
            "Please run 'rolypoly build-data' first or specify --data-dir"
        )

    def _find_motif_metadata(self) -> Path:
        """Find the RdRp motif metadata file"""
        db_dir = self.motif_db_path.parent.parent / "rvmt_motifs"
        metadata_path = db_dir / "motif_metadata.json"
        
        if metadata_path.exists():
            return metadata_path
        
        raise FileNotFoundError(
            f"RdRp motif metadata not found at {metadata_path}. "
            "Please run 'rolypoly build-data' first."
        )


@command(
    epilog="""\n\nEXAMPLES:\n\n  # Basic search with default flat TSV output and alignment\n  rolypoly rdrp-motif-search -i sequences.fasta -o results_dir\n\n  # Nested structure for programmatic analysis\n  rolypoly rdrp-motif-search -i sequences.fasta -o results_dir --output-structure nested\n\n  # Parquet output with structured data for analysis\n  rolypoly rdrp-motif-search -i sequences.fasta -o results_dir --output-format parquet\n\n  # Disable alignment to reduce output size\n  rolypoly rdrp-motif-search -i sequences.fasta -o results_dir --no-include-alignment\n\n  # High sensitivity search with custom parameters\n  rolypoly rdrp-motif-search -i sequences.fasta -o results_dir -e 0.1 --max-distance 300\n\nOUTPUT FORMATS:\n\n  flat + tsv: separate columns (motif_a_start, motif_b_start, etc.) - DEFAULT\n  nested + tsv: motif_details column as JSON string\n  flat + parquet: separate columns with native data types\n  nested + parquet: motif_details as structured data types\n"""
)
@option(
    "-i",
    "--input",
    required=True,
    help="Input FASTA file with nucleotide or amino acid sequences",
)
@option(
    "-o",
    "--output",
    default="./rdrp_motif_search_output",
    help="Output directory path",
)
@option(
    "-t",
    "--threads",
    default=1,
    help="Number of threads for processing",
)
@option(
    "-g",
    "--log-file",
    default="./rdrp_motif_search_logfile.txt",
    help="Path to log file",
)
@option(
    "-M",
    "--memory",
    default="4gb",
    help="Memory limit in GB. Example: -M 8gb",
)
@option(
    "-e",
    "--evalue",
    default=1e-2,
    help="E-value threshold for motif searches",
)
@option(
    "--min-score",
    default=None,
    type=float,
    help="Minimum score threshold for motif matches",
)
@option(
    "--max-distance",
    default=200,
    help="Maximum distance between motifs in amino acids",
)
@option(
    "--search-tool",
    default="hmmsearch",
    type=Choice(["hmmsearch"], case_sensitive=False),
    help="Search tool to use (currently only hmmsearch supported)",
)
@option(
    "--aa-method",
    default="six_frame",
    type=Choice(["six_frame", "orffinder", "pyrodigal"], case_sensitive=False),
    help="Method for amino acid translation from nucleotides",
)
@option(
    "--min-orf-length",
    default=30,
    help="Minimum ORF length for gene prediction",
)
@option(
    "--motif-filter",
    default=None,
    type=Choice(["A", "B", "C", "D"], case_sensitive=False),
    help="Filter results by specific motif type",
)
# @option(
#     "--taxon-filter",
#     default=None,
#     help="Filter results by specific taxon name - DISABLED: no taxon data available yet",
# )
@option(
    "--no-include-alignment",
    is_flag=True,
    default=False,
    help="Disable including aligned region sequences in output (alignment included by default)",
)
@option(
    "--data-dir",
    default=None,
    help="Path to rolypoly data directory (if not in default location)",
)
@option(
    "--output-format",
    default="tsv",
    type=Choice(["tsv", "parquet"], case_sensitive=False),
    help="Output file format (tsv or parquet)",
)
@option(
    "--output-structure",
    default="nested",
    type=Choice(["nested", "flat"], case_sensitive=False),
    help="Output structure: nested (motif_details as JSON) or flat (separate columns per motif)",
)
@option(
    "-k",
    "--keep-tmp",
    is_flag=True,
    default=False,
    help="Keep temporary files",
)
@option(
    "-ow",
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite output directory if it exists",
)
@option(
    "-ll",
    "--log-level",
    default="INFO",
    type=Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Logging level",
)
@option(
    "-cf",
    "--config-file",
    default=None,
    help="JSON config file with parameters (overrides command line)",
)
def rdrp_motif_search(
    input,
    output,
    threads,
    log_file,
    memory,
    evalue,
    min_score,
    max_distance,
    search_tool,
    aa_method,
    min_orf_length,
    motif_filter,
    # taxon_filter,  # DISABLED: no taxon data available yet
    no_include_alignment,
    data_dir,
    output_format,
    output_structure,
    keep_tmp,
    overwrite,
    log_level,
    config_file,
):
    """Search for RdRp motifs (A, B, C, D) in nucleotide or amino acid sequences.
    
    This command searches input sequences for RNA-dependent RNA polymerase (RdRp)
    motif patterns using pre-built profile databases from the RVMT project.
    
    The output is a table with one row per input sequence, showing motif locations,
    scores, and conformations found.
    """
    import json
    
    # Load config from file if provided
    if config_file:
        with open(config_file) as f:
            config_dict = json.load(f)
        config = RdRpMotifSearchConfig(**config_dict)
    else:
        config = RdRpMotifSearchConfig(
            input=input,
            output=output,
            threads=threads,
            log_file=log_file,
            memory=ensure_memory(memory)["giga"],
            evalue=evalue,
            min_score=min_score,
            max_distance=max_distance,
            search_tool=search_tool,
            aa_method=aa_method,
            min_orf_length=min_orf_length,
            motif_filter=motif_filter,
            # taxon_filter=taxon_filter,  # DISABLED: no taxon data available yet
            include_alignment=not no_include_alignment,  # Invert the flag
            data_dir=data_dir,
            output_format=output_format,
            output_structure=output_structure,
            keep_tmp=keep_tmp,
            overwrite=overwrite,
            log_level=log_level,
        )

    # Start logging
    log_start_info(config.logger, config.to_dict())
    
    config.logger.info(f"Starting RdRp motif search with input: {config.input}")
    config.logger.info(f"Using motif database: {config.motif_db_path}")
    
    try:
        process_rdrp_motif_search(config)
        remind_citations(["hmmer", "pyhmmer"], config.logger)
    except Exception as e:
        config.logger.error(f"Error during RdRp motif search: {str(e)}")
        raise


def process_rdrp_motif_search(config: RdRpMotifSearchConfig):
    """Main processing function for RdRp motif search"""
    
    # Check input file format
    alphabet = guess_fasta_alpha(config.input)
    config.logger.info(f"Detected input alphabet: {alphabet}")
    
    # Load motif metadata
    motif_metadata = load_motif_metadata(config)
    
    # Prepare protein sequences
    if alphabet == "nucl":
        protein_file = prepare_protein_sequences(config)
    elif alphabet == "amino":
        protein_file = config.input
        config.logger.info("Input is amino acid sequences, using directly")
    else:
        raise ValueError(f"Could not determine sequence type from input file: {alphabet}")
    
    # Search for motifs
    search_results = search_motifs(config, protein_file)
    
    # Process and filter results
    processed_results = process_motif_results(config, search_results, motif_metadata)
    
    # Apply distance filtering
    filtered_results = apply_distance_filtering(config, processed_results)
    
    # Create summary output
    create_summary_output(config, filtered_results)
    
    # Cleanup temporary files
    if not config.keep_tmp:
        try:
            if os.path.exists(config.temp_dir):
                shutil.rmtree(config.temp_dir)
                config.logger.debug(f"Removed temporary directory: {config.temp_dir}")
        except OSError as e:
            config.logger.warning(f"Could not remove temporary directory {config.temp_dir}: {e}")
    
    config.logger.info("RdRp motif search completed successfully")


def load_motif_metadata(config: RdRpMotifSearchConfig) -> Dict:
    """Load motif metadata from JSON file"""
    try:
        with open(config.motif_metadata_path) as f:
            metadata = json.load(f)
        config.logger.info(f"Loaded metadata for {len(metadata)} motif profiles")
        return metadata
    except FileNotFoundError:
        config.logger.warning(f"Motif metadata not found at {config.motif_metadata_path}")
        return {}
    except Exception as e:
        config.logger.warning(f"Error loading motif metadata: {e}")
        return {}


def prepare_protein_sequences(config: RdRpMotifSearchConfig) -> str:
    """Prepare protein sequences from nucleotide input"""
    config.logger.info(f"Translating nucleotide sequences using {config.aa_method}")
    
    output_file = os.path.join(config.temp_dir, f"{config.name}_proteins.faa")
    os.makedirs(config.temp_dir, exist_ok=True)
    
    if config.aa_method == "six_frame":
        # Use six-frame translation
        translate_6frx_seqkit(
            input_file=config.input,
            output_file=output_file,
            threads=config.threads,
            min_orf_length=config.min_orf_length,
        )
    
    elif config.aa_method == "pyrodigal":
        # Use pyrodigal for gene prediction
        pyro_predict_orfs(
            input_file=config.input,
            output_file=output_file,
            threads=config.threads,
            genetic_code=11,  # Standard bacterial code
            min_gene_length=config.min_orf_length,
        )
    
    elif config.aa_method == "orffinder":
        # Use ORFfinder
        success = run_command_comp(
            base_cmd="ORFfinder",
            positional_args=["-in", config.input, "-out", output_file],
            params={
                "outfmt": "1",
                "ml": config.min_orf_length
            },
            logger=config.logger
        )
        
        if not success:
            raise RuntimeError("ORFfinder failed")
    
    config.logger.info(f"Protein sequences saved to: {output_file}")
    return output_file


def search_motifs(config: RdRpMotifSearchConfig, protein_file: str) -> pl.DataFrame:
    """Search for motifs using HMM profiles"""
    config.logger.info("Searching for RdRp motifs")
    
    output_file = os.path.join(config.temp_dir, f"{config.name}_motif_hits.tsv")
    
    # Use pyhmmer bindings through rolypoly utility
    output_path = search_hmmdb(
        amino_file=protein_file,
        db_path=str(config.motif_db_path),
        output=output_file,
        threads=config.threads,
        logger=config.logger,
        match_region=True,
        full_qseq=True,
        ali_str=True,
        inc_e=config.evalue,
        mscore=None,  # Use E-value filtering
    )
    
    # Read the results into a DataFrame
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        results = pl.read_csv(output_path, separator="\t", has_header=True)
        
        # Rename columns to match our expected names
        column_mapping = {
            "query_full_name": "query_name",
            "hmm_full_name": "target_name",
            "hmm_len": "target_length",
            "qlen": "query_length",
            "full_hmm_evalue": "evalue", 
            "full_hmm_score": "score",
            "full_hmm_bias": "bias",
            "this_dom_score": "dom_score",
            "this_dom_bias": "dom_bias",
            "hmm_from": "hmm_from",
            "hmm_to": "hmm_to",
            "q1": "ali_from",
            "q2": "ali_to",
            "env_from": "env_from",
            "env_to": "env_to",
            "hmm_cov": "acc",
            "ali_len": "ali_len",
            "dom_desc": "dom_desc"
        }
        
        # Rename columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in results.columns:
                results = results.rename({old_name: new_name})
        
        config.logger.info(f"Found {len(results)} motif hits")
        return results
    else:
        config.logger.info("No motif hits found")
        return pl.DataFrame()


def process_motif_results(config: RdRpMotifSearchConfig, 
                         search_results: pl.DataFrame, 
                         motif_metadata: Dict) -> pl.DataFrame:
    """Process and annotate motif search results"""
    
    if len(search_results) == 0:
        config.logger.warning("No motif hits found")
        return pl.DataFrame()
    
    config.logger.info("Processing motif search results")
    
    # Add motif metadata to results
    results = search_results.with_columns([
        pl.col("target_name").alias("motif_profile"),
        pl.lit(None).alias("motif_type"),
        # pl.lit(None).alias("taxon"),  # DISABLED: no taxon data in profiles yet
        pl.lit(None).alias("original_motif_name"),
    ])
    
    # Map profile names to motif metadata
    for profile_name, metadata in motif_metadata.items():
        mask = pl.col("motif_profile") == profile_name
        results = results.with_columns([
            pl.when(mask).then(pl.lit(metadata["motif_type"])).otherwise(pl.col("motif_type")).alias("motif_type"),
            # pl.when(mask).then(pl.lit(metadata["taxon"])).otherwise(pl.col("taxon")).alias("taxon"),  # DISABLED: no taxon data
            pl.when(mask).then(pl.lit(metadata["original_name"])).otherwise(pl.col("original_motif_name")).alias("original_motif_name"),
        ])
    
    # Apply filters
    if config.motif_filter:
        results = results.filter(pl.col("motif_type") == config.motif_filter.upper())
        config.logger.info(f"Filtered to motif type {config.motif_filter}: {len(results)} hits")
    
    # if config.taxon_filter:  # DISABLED: no taxon data available yet
    #     results = results.filter(pl.col("taxon").str.contains(config.taxon_filter, strict=False))
    #     config.logger.info(f"Filtered to taxon {config.taxon_filter}: {len(results)} hits")
    
    if config.min_score:
        results = results.filter(pl.col("score") >= config.min_score)
        config.logger.info(f"Filtered by minimum score {config.min_score}: {len(results)} hits")
    
    return results


def apply_distance_filtering(config: RdRpMotifSearchConfig, 
                           results: pl.DataFrame) -> pl.DataFrame:
    """Apply distance constraints between motifs"""
    
    if len(results) == 0:
        return results
    
    config.logger.info(f"Applying distance filtering (max distance: {config.max_distance} aa)")
    
    # Group by query sequence
    grouped = results.group_by("query_name", maintain_order=True)
    
    filtered_groups = []
    
    for group_key, group_df in grouped:
        query_name = group_key[0] if isinstance(group_key, tuple) else group_key
        query_hits = group_df.sort("ali_from")
        
        # Group hits by motif type
        motif_groups = defaultdict(list)
        for row in query_hits.to_dicts():
            motif_type = row.get("motif_type", "unknown")
            motif_groups[motif_type].append(row)
        
        # Find best combination within distance constraint
        best_combination = find_best_motif_combination(motif_groups, config.max_distance)
        
        if best_combination:
            filtered_groups.extend(best_combination)
    
    if filtered_groups:
        filtered_df = pl.DataFrame(filtered_groups)
        config.logger.info(f"Distance filtering result: {len(filtered_df)} hits retained")
        return filtered_df
    else:
        config.logger.info("No motif combinations passed distance filtering")
        return pl.DataFrame()


def find_best_motif_combination(motif_groups: Dict, max_distance: int) -> List[Dict]:
    """Find the best combination of motifs within distance constraint"""
    
    # Define minimum scores for each motif type (based on original script)
    min_score_thresholds = {"A": 10.0, "B": 11.0, "C": 5.9, "D": 5.0}
    
    # Filter by minimum scores
    filtered_groups = {}
    for motif_type, hits in motif_groups.items():
        threshold = min_score_thresholds.get(motif_type, 0.0)
        filtered_hits = [h for h in hits if h.get("score", 0) > threshold]
        if filtered_hits:
            filtered_groups[motif_type] = sorted(filtered_hits, key=lambda x: x.get("score", 0), reverse=True)
    
    if not filtered_groups:
        return []
    
    best_combination = []
    best_score = 0
    
    # Try all combinations
    for motif_type, hits in filtered_groups.items():
        for hit in hits:
            current_combination = [hit]
            current_score = hit.get("score", 0)
            
            # Try to add other motif types within distance
            for other_type, other_hits in filtered_groups.items():
                if other_type == motif_type:
                    continue
                
                for other_hit in other_hits:
                    if is_within_distance(current_combination, other_hit, max_distance):
                        current_combination.append(other_hit)
                        current_score += other_hit.get("score", 0)
                        break  # Only take the best hit of each type
            
            if current_score > best_score:
                best_score = current_score
                best_combination = current_combination
    
    return best_combination


def is_within_distance(existing_hits: List[Dict], new_hit: Dict, max_distance: int) -> bool:
    """Check if new hit is within distance of existing hits"""
    
    new_start = new_hit.get("ali_from", 0)
    new_end = new_hit.get("ali_to", 0)
    
    for hit in existing_hits:
        hit_start = hit.get("ali_from", 0)
        hit_end = hit.get("ali_to", 0)
        
        # Check for overlap (not allowed)
        if not (new_end < hit_start or new_start > hit_end):
            return False
        
        # Check distance
        distance = min(abs(new_start - hit_end), abs(hit_start - new_end))
        if distance > max_distance:
            return False
    
    return True


def create_summary_output(config: RdRpMotifSearchConfig, results: pl.DataFrame):
    """Create summary output with one row per input sequence"""
    
    config.logger.info("Creating summary output")
    
    if len(results) == 0:
        # Create empty output file
        file_extension = "parquet" if config.output_format == "parquet" else "tsv"
        output_file = os.path.join(config.output_dir, f"{config.name}_rdrp_motif_results.{file_extension}")
        
        if config.output_structure == "flat":
            base_columns = ["sequence_id", "sequence_length", "motif_conformation", "total_motifs"]
            motif_columns = []
            for motif in ["a", "b", "c", "d"]:
                motif_columns.extend([
                    f"motif_{motif}_start", f"motif_{motif}_end", f"motif_{motif}_score", f"motif_{motif}_evalue",
                    # f"motif_{motif}_taxon",  # DISABLED: no taxon data
                    f"motif_{motif}_profile"
                ])
                if config.include_alignment:
                    motif_columns.append(f"motif_{motif}_alignment")
            columns = base_columns + motif_columns
        else:
            columns = ["sequence_id", "sequence_length", "motif_conformation", "total_motifs", "motif_details"]
        
        empty_df = pl.DataFrame({col: [] for col in columns})
        
        if config.output_format == "parquet":
            empty_df.write_parquet(output_file)
        else:
            empty_df.write_csv(output_file, separator="\t")
            
        config.logger.warning(f"No results to output. Empty file created: {output_file}")
        return
    
    # Group by query sequence
    grouped = results.group_by("query_name", maintain_order=True)
    
    summary_rows = []
    
    for group_key, group_df in grouped:
        query_name = group_key[0] if isinstance(group_key, tuple) else group_key
        query_hits = group_df.sort("ali_from")
        
        # Create motif conformation string
        motif_types = [hit.get("motif_type", "unknown") for hit in query_hits.to_dicts()]
        motif_conformation = ",".join(sorted(set(motif_types)))
        
        # Calculate sequence length (approximate from alignment coordinates)
        max_coord = query_hits["ali_to"].max() if len(query_hits) > 0 else 0
        
        if config.output_structure == "flat":
            # Flat structure with separate columns per motif
            row = {
                "sequence_id": query_name,
                "sequence_length": max_coord,
                "motif_conformation": motif_conformation,
                "total_motifs": len(query_hits),
            }
            
            # Initialize all motif columns with None
            for motif in ["a", "b", "c", "d"]:
                row[f"motif_{motif}_start"] = None
                row[f"motif_{motif}_end"] = None
                row[f"motif_{motif}_score"] = None
                row[f"motif_{motif}_evalue"] = None
                # row[f"motif_{motif}_taxon"] = None  # DISABLED: no taxon data yet
                row[f"motif_{motif}_profile"] = None
                if config.include_alignment:
                    row[f"motif_{motif}_alignment"] = None
            
            # Group hits by motif type and handle multiple matches
            motif_groups = {}
            has_multiple_matches = False
            for hit in query_hits.to_dicts():
                motif_type = hit.get("motif_type", "unknown").lower()
                if motif_type not in motif_groups:
                    motif_groups[motif_type] = []
                motif_groups[motif_type].append(hit)
                
            # Check for multiple matches per motif type
            for motif_type, hits in motif_groups.items():
                if len(hits) > 1:
                    has_multiple_matches = True
                    break
            
            # If flat structure has multiple matches, fall back to nested for this sequence
            if has_multiple_matches and config.output_structure == "flat":
                config.logger.info(f"Multiple matches per motif type detected for {query_name}, falling back to nested structure for this sequence")
                # Process as nested structure instead
                motif_details = {}
                for hit in query_hits.to_dicts():
                    motif_type = hit.get("motif_type", "unknown")
                    if motif_type not in motif_details:
                        motif_details[motif_type] = []
                    
                    hit_entry = {
                        "start": hit.get("ali_from", 0),
                        "end": hit.get("ali_to", 0),
                        "score": hit.get("score", 0),
                        "evalue": hit.get("evalue", 0),
                        # "taxon": hit.get("taxon", "unknown"),  # DISABLED: no taxon data
                        "profile": hit.get("motif_profile", "unknown")
                    }
                    if config.include_alignment:
                        hit_entry["alignment"] = hit.get("ali_seq", "")
                        
                    motif_details[motif_type].append(hit_entry)
                
                nested_row = {
                    "sequence_id": query_name,
                    "sequence_length": max_coord,
                    "motif_conformation": motif_conformation,
                    "total_motifs": len(query_hits),
                }
                
                if config.output_format == "parquet":
                    nested_row["motif_details"] = motif_details
                else:
                    nested_row["motif_details"] = json.dumps(motif_details)
                
                summary_rows.append(nested_row)
                continue
            
            # Fill in motif data
            for motif_type, hits in motif_groups.items():
                if motif_type in ["a", "b", "c", "d"]:
                    if len(hits) == 1:
                        # Single match - use scalar values
                        hit = hits[0]
                        row[f"motif_{motif_type}_start"] = hit.get("ali_from", 0)
                        row[f"motif_{motif_type}_end"] = hit.get("ali_to", 0)
                        row[f"motif_{motif_type}_score"] = hit.get("score", 0)
                        row[f"motif_{motif_type}_evalue"] = hit.get("evalue", 0)
                        # row[f"motif_{motif_type}_taxon"] = hit.get("taxon", "unknown")  # DISABLED: no taxon data
                        row[f"motif_{motif_type}_profile"] = hit.get("motif_profile", "unknown")
                        if config.include_alignment:
                            row[f"motif_{motif_type}_alignment"] = hit.get("aligned_region", "")
                    else:
                        # Multiple matches - convert to JSON strings for TSV, keep as lists for Parquet
                        hit_data = []
                        for hit in hits:
                            hit_entry = {
                                "start": hit.get("ali_from", 0),
                                "end": hit.get("ali_to", 0),
                                "score": hit.get("score", 0),
                                "evalue": hit.get("evalue", 0),
                                # "taxon": hit.get("taxon", "unknown"),  # DISABLED: no taxon data
                                "profile": hit.get("motif_profile", "unknown")
                            }
                            if config.include_alignment:
                                hit_entry["alignment"] = hit.get("aligned_region", "")
                            hit_data.append(hit_entry)
                        
                        if config.output_format == "parquet":
                            # Keep as nested data for Parquet
                            row[f"motif_{motif_type}_start"] = [h["start"] for h in hit_data]
                            row[f"motif_{motif_type}_end"] = [h["end"] for h in hit_data]
                            row[f"motif_{motif_type}_score"] = [h["score"] for h in hit_data]
                            row[f"motif_{motif_type}_evalue"] = [h["evalue"] for h in hit_data]
                            # row[f"motif_{motif_type}_taxon"] = [h["taxon"] for h in hit_data]  # DISABLED: no taxon data
                            row[f"motif_{motif_type}_profile"] = [h["profile"] for h in hit_data]
                            if config.include_alignment:
                                row[f"motif_{motif_type}_alignment"] = [h.get("alignment", "") for h in hit_data]
                        else:
                            # Convert to JSON strings for TSV
                            row[f"motif_{motif_type}_start"] = json.dumps([h["start"] for h in hit_data])
                            row[f"motif_{motif_type}_end"] = json.dumps([h["end"] for h in hit_data])
                            row[f"motif_{motif_type}_score"] = json.dumps([h["score"] for h in hit_data])
                            row[f"motif_{motif_type}_evalue"] = json.dumps([h["evalue"] for h in hit_data])
                            # row[f"motif_{motif_type}_taxon"] = json.dumps([h["taxon"] for h in hit_data])  # DISABLED: no taxon data
                            row[f"motif_{motif_type}_profile"] = json.dumps([h["profile"] for h in hit_data])
                            if config.include_alignment:
                                row[f"motif_{motif_type}_alignment"] = json.dumps([h.get("alignment", "") for h in hit_data])
            
            summary_rows.append(row)
            
        else:
            # Nested structure with motif_details as JSON/dict
            motif_details = {}
            for hit in query_hits.to_dicts():
                motif_type = hit.get("motif_type", "unknown")
                if motif_type not in motif_details:
                    motif_details[motif_type] = []
                
                hit_entry = {
                    "start": hit.get("ali_from", 0),
                    "end": hit.get("ali_to", 0),
                    "score": hit.get("score", 0),
                    "evalue": hit.get("evalue", 0),
                    # "taxon": hit.get("taxon", "unknown"),  # DISABLED: no taxon data
                    "profile": hit.get("motif_profile", "unknown")
                }
                if config.include_alignment:
                    hit_entry["alignment"] = hit.get("aligned_region", "")
                    
                motif_details[motif_type].append(hit_entry)
            
            row = {
                "sequence_id": query_name,
                "sequence_length": max_coord,
                "motif_conformation": motif_conformation,
                "total_motifs": len(query_hits),
            }
            
            if config.output_format == "parquet":
                # Keep as nested dict for Parquet
                row["motif_details"] = motif_details
            else:
                # Convert to JSON string for TSV
                row["motif_details"] = json.dumps(motif_details)
            
            summary_rows.append(row)
    
    # Create output DataFrame
    summary_df = pl.DataFrame(summary_rows)
    
    # Write to file
    file_extension = "parquet" if config.output_format == "parquet" else "tsv"
    output_file = os.path.join(config.output_dir, f"{config.name}_rdrp_motif_results.{file_extension}")
    
    if config.output_format == "parquet":
        summary_df.write_parquet(output_file)
    else:
        summary_df.write_csv(output_file, separator="\t")
    
    config.logger.info(f"Summary results written to: {output_file}")
    config.logger.info(f"Found motifs in {len(summary_df)} sequences")
    config.logger.info(f"Output format: {config.output_format}, structure: {config.output_structure}")
    
    # Print summary statistics
    conformations = summary_df["motif_conformation"].value_counts().sort("count", descending=True)
    config.logger.info("Motif conformations found:")
    for row in conformations.to_dicts():
        config.logger.info(f"  {row['motif_conformation']}: {row['count']} sequences")


if __name__ == "__main__":
    rdrp_motif_search()