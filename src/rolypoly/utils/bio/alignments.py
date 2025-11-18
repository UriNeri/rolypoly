"""Alignments (MSAs, HMMs, and collection of them) and mapping utility functions."""

import re
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Union, Dict
import io

from rich.progress import track
import polars as pl
import pyhmmer 

from rolypoly.utils.logging.loggit import get_logger
from rolypoly.utils.various import find_files_by_extension, run_command_comp

def find_msa_files(
    input_path: Union[str, Path],
    extensions: List[str] = None,
    logger: Optional[logging.Logger] = None
) -> List[Path]:
    """Find all Multiple Sequence Alignment files in a directory or return single file.
    
    Args:
        input_path: Path to directory or file
        extensions: List of extensions to look for
        logger: Logger instance
        
    Returns:
        List of MSA file paths
    """
    if extensions is None:
        extensions = ["*.faa", "*.afa", "*.aln", "*.msa"]
    
    return find_files_by_extension(input_path, extensions, "MSA files", logger)

def calculate_percent_identity(cigar_string: str, num_mismatches: int) -> float:
    """Calculate sequence identity percentage from CIGAR string and edit distance.

    Computes the percentage identity between aligned sequences using the CIGAR
    string from an alignment and the number of mismatches (NM tag).

    Args:
        cigar_string (str): CIGAR string from sequence alignment
        num_mismatches (int): Number of mismatches (edit distance)

    Returns:
        float: Percentage identity between sequences (0-100)

    Note:
        The calculation considers matches (M), insertions (I), deletions (D),
        and exact matches (=) from the CIGAR string.

    Example:
         print(calculate_percent_identity("100M", 0))
         100.0
         print(calculate_percent_identity("100M", 2))
         98.0
    """

    cigar_tuples = re.findall(r"(\d+)([MIDNSHPX=])", cigar_string)
    matches = sum(int(length) for length, op in cigar_tuples if op in {"M", "=", "X"})
    total_length = sum(
        int(length) for length, op in cigar_tuples if op in {"M", "I", "D", "=", "X"}
    )
    return (matches - num_mismatches) / total_length * 100 



def find_hmm_files(
    input_path: Union[str, Path],
    extensions: List[str] = None,
    logger: Optional[logging.Logger] = None
) -> List[Path]:
    """Find all HMM files in a directory or return single file.
    
    Args:
        input_path: Path to directory or file
        extensions: List of extensions to look for
        logger: Logger instance
        
    Returns:
        List of HMM file paths
    """
    if extensions is None:
        extensions = ["*.hmm"]
    
    return find_files_by_extension(input_path, extensions, "HMM files", logger)



def validate_database_directory(
    database_path: Union[str, Path],
    expected_types: List[str] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Union[str, List[Path]]]:
    """Validate and categorize database directory contents.
    
    This function handles the common pattern of validating custom database directories
    that can contain either HMM files or MSA files that need to be converted to HMMs.
    
    Args:
        database_path: Path to database file or directory
        expected_types: List of expected file types ("hmm", "msa", "fasta")
        logger: Logger instance
        
    Returns:
        Dictionary containing:
        - type: "hmm_file", "hmm_directory", "msa_file", "msa_directory", "mixed", "invalid"
        - files: List of relevant files found
        - message: Human-readable description
    """
    logger = get_logger(logger)
    database_path = Path(database_path)
    
    if expected_types is None:
        expected_types = ["hmm", "msa"]
    
    result = {
        "type": "invalid",
        "files": [],
        "message": ""
    }
    
    if not database_path.exists():
        result["message"] = f"Database path {database_path} does not exist"
        return result
    
    if database_path.is_file():
        # Single file - determine type
        if database_path.suffix == ".hmm":
            result["type"] = "hmm_file"
            result["files"] = [database_path]
            result["message"] = f"Single HMM file: {database_path.name}"
        elif database_path.suffix in [".faa", ".afa", ".aln", ".msa"]:
            result["type"] = "msa_file"
            result["files"] = [database_path]
            result["message"] = f"Single MSA file: {database_path.name}"
        else:
            result["message"] = f"Unsupported file type: {database_path.suffix}"
        
        return result
    
    elif database_path.is_dir():
        # Directory - analyze contents
        hmm_files = find_hmm_files(database_path, logger=logger)
        msa_files = find_msa_files(database_path, logger=logger)
        
        if hmm_files and not msa_files:
            result["type"] = "hmm_directory"
            result["files"] = hmm_files
            result["message"] = f"Directory with {len(hmm_files)} HMM files"
        elif msa_files and not hmm_files:
            result["type"] = "msa_directory"
            result["files"] = msa_files
            result["message"] = f"Directory with {len(msa_files)} MSA files"
        elif hmm_files and msa_files:
            result["type"] = "mixed"
            result["files"] = hmm_files + msa_files
            result["message"] = f"Mixed directory: {len(hmm_files)} HMM files, {len(msa_files)} MSA files"
        else:
            result["message"] = "Directory contains no HMM or MSA files"
        
        return result
    
    result["message"] = f"Path {database_path} is neither file nor directory"
    return result


def get_hmmali_length(domain) -> int:
    """Get the alignment length of an HMM domain."""
    return domain.alignment.hmm_to - domain.alignment.hmm_from + 1


def get_hmm_coverage(domain) -> float:
    """Calculate the HMM coverage of a domain alignment."""
    return get_hmmali_length(domain) / domain.alignment.hmm_length


def search_hmmdb(
    amino_file : Union[str, Path],
    db_path : Union[str, Path],
    output : Union[str, Path],
    threads : int,
    logger=None,
    inc_e=0.05,
    mscore=20,
    match_region=False,
    full_qseq=False,
    ali_str=False,
    output_format="modomtblout",
    pyhmmer_hmmsearch_args={},
):
    """Search an HMM database using pyhmmer.

    Performs a profile HMM search against a database using pyhmmer, with configurable output formats
    and filtering options.

    Args:
      amino_file(str): Path to the amino acid sequence file in FASTA format
      db_path(str): Path to the HMM database file
      output(str): Path where the output file will be written
      threads(int): Number of CPU threads to use for the search
      logger(logging.Logger, optional): Logger object for debug messages. (Default value = None)
      inc_e(float, optional): Inclusion E-value threshold for reporting domains. (Default value = 0.05)
      mscore(float, optional): Minimum score threshold for reporting domains. (Default value = 20)
      match_region(bool, optional): Include aligned region in output. Only works with modomtblout format. (Default value = False)
      full_qseq(bool, optional): Include full query sequence in output. Only works with modomtblout format. (Default value = False)
      ali_str(bool, optional): Include alignment string in output. Only works with modomtblout format. (Default value = False)
      output_format(str, optional): Format of the output file. One of: "modomtblout", "domtblout", "tblout".

    Returns:
        str: Path to the output file containing search results

    Note:
      The modomtblout format is a modified domain table output that includes additional columns (like coverage, alignment string, query sequence, etc).
      match_region, full_qseq, and ali_str only work with modomtblout format. (Default value = "modomtblout")
      pyhmmer_hmmsearch_args(dict, optional): Additional arguments to pass to pyhmmer.hmmsearch. (Default value = {})

    Example:
      # Basic search with default parameters
      search_hmmdb("proteins.faa", "pfam.hmm", "results.txt", threads=4)
      # Search with custom settings and full alignment info
      search_hmmdb("proteins.faa", "pfam.hmm", "results.txt", threads=4,
      inc_e=0.01, match_region=True, ali_str=True)
    """

    if logger:
        logger.debug(
            f"Starting pyhmmer search against {db_path} with {threads} threads"
        )

    format_dict = {
        "tblout": "targets",
        "domtblout": "domains",
        "modomtblout": "modomtblout",
    }

    with pyhmmer.easel.SequenceFile(
        amino_file, digital=True, format="fasta"
    ) as seq_file:
        seqs = seq_file.read_block()
    seqs_dict = {}
    for seq in seqs:
        seqs_dict[seq.name.decode() + f" {seq.description.decode()}"] = (
            seq.textize().sequence
        )  # type: ignore

    if logger:
        logger.debug(f"loaded {len(seqs)} sequences from {amino_file}")
    # see https://pyhmmer.readthedocs.io/en/stable/api/plan7/results.html#pyhmmer.plan7.TopHits for format (though I changed it a bit)
    mod_title_domtblout = [
        "query_full_name",
        "hmm_full_name",
        "hmm_len",
        "qlen",
        "full_hmm_evalue",
        "full_hmm_score",
        "full_hmm_bias",
        "this_dom_score",
        "this_dom_bias",
        "hmm_from",
        "hmm_to",
        "q1",
        "q2",
        "env_from",
        "env_to",
        "hmm_cov",
        "ali_len",
        "dom_desc",
    ]
    mod_title_domtblout.extend(
        name
        for name, value in {
            "aligned_region": match_region,
            "full_qseq": full_qseq,
            "identity_str": ali_str,
        }.items()
        if value
    )
    og_domtblout_title = [
        "#                                                                                                                --- full sequence --- -------------- this domain -------------   hmm coord   ali coord   env coord",
        "# target name        accession   tlen query name                                               accession   qlen   E-value  score  bias   #  of  c-Evalue  i-Evalue  score  bias  from    to  from    to  from    to  acc description of target",
        "#------------------- ---------- -----                                     -------------------- ---------- ----- --------- ------ ----- --- --- --------- --------- ------ ----- ----- ----- ----- ----- ---- ---------------------",
    ]
    og_tblout = [
        "#                                                                                                   --- full sequence ---- --- best 1 domain ---- --- domain number estimation ----",
        "# target name        accession  query name                                               accession    E-value  score  bias   E-value  score  bias   exp reg clu  ov env dom rep inc description of target",
        "#------------------- ----------                                     -------------------- ---------- --------- ------ ----- --------- ------ -----   --- --- --- --- --- --- --- --- ---------------------",
    ]

    with open(output, "wb") as outfile:
        if output_format == "modomtblout":
            outfile.write("\t".join(mod_title_domtblout).encode("utf-8") + b"\n")
        else:
            outfile.write(
                "\n".join(
                    (
                        og_tblout if output_format == "tblout" else og_domtblout_title
                    )
                )
                + "\n"
            )
        with pyhmmer.plan7.HMMFile(db_path) as hmms:
            for hits in pyhmmer.hmmsearch(
                hmms, seqs, cpus=threads, T=mscore, E=inc_e, **pyhmmer_hmmsearch_args
            ):
                if output_format != "modomtblout":
                    # writes hits
                    hits.write(outfile, format=format_dict[output_format], header=False)
                    continue
                else:
                    if len(hits) >= 1:
                        for hit in hits:
                            hit_desc = hit.description or bytes("", "utf-8")
                            hit_name = hit.name.decode()
                            # join the prot name and acc into a single string because God knows why there are spaces in fasta headers
                            full_prot_name = f"{hit_name} {hit_desc.decode()}"
                            if full_qseq:
                                protein_seq = seqs_dict[full_prot_name]
                            for domain in hit.domains.included:
                                # Get alignment length
                                alignment_length = get_hmmali_length(domain)

                                # Calculate hmm_coverage
                                hmm_coverage = get_hmm_coverage(domain)

                                dom_desc = hits.query.description or bytes("", "utf-8")

                                outputline = [
                                    f"{full_prot_name}",  # query_full_name
                                    f"{hits.query.name.decode()}",  # hmm_full_name
                                    f"{domain.alignment.hmm_length}",  # hmm_len
                                    f"{hit.length}",  # qlen
                                    f"{hit.evalue}",  # full_hmm_evalue
                                    f"{hit.score}",  # full_hmm_score
                                    f"{hit.bias}",  # full_hmm_bias
                                    f"{domain.score}",  # this_dom_score
                                    f"{domain.bias}",  # this_dom_bias
                                    f"{domain.alignment.hmm_from}",  # hmm_from
                                    f"{domain.alignment.hmm_to}",  # hmm_to
                                    f"{domain.alignment.target_from}",  # q1
                                    f"{domain.alignment.target_to}",  # q2
                                    f"{domain.env_from}",  # env_from
                                    f"{domain.env_to}",  # env_to
                                    f"{hmm_coverage}",  # hmm_cov
                                    f"{alignment_length}",  # ali_len
                                    f"{dom_desc.decode()}",  # I think this is description of the target hit.
                                ]
                                if match_region:
                                    outputline.append(
                                        f"{domain.alignment.target_sequence}"
                                    )
                                if full_qseq:
                                    outputline.append(f"{protein_seq}")
                                if ali_str:
                                    outputline.append(
                                        f"{domain.alignment.identity_sequence}"
                                    )
                                outfile.write(("\t".join(outputline) + "\n").encode())
    return output


def hmm_from_msa(
    msa_file, output, alphabet="amino", set_ga=None, name=None, accession=None
):
    """Create an HMM from a multiple sequence alignment file.

    Args:
      msa_file: str or Path, path to the MSA file
      output: str or Path, path to save the HMM file
      alphabet: str, sequence alphabet type ("amino" or "dna") (Default value = "amino")
      set_ga: float or None, gathering threshold to set for the HMM (Default value = None)
      name: str or None, name for the HMM profile (Default value = None)
      accession: str or None, accession for the HMM profile (Default value = None)
    """

    # Set the alphabet
    if alphabet == "amino":
        alpha = pyhmmer.easel.Alphabet.amino()
    elif alphabet == "dna":
        alpha = pyhmmer.easel.Alphabet.dna()
    else:
        raise ValueError("alphabet must be either 'amino' or 'dna'")

    # Read the MSA file
    with pyhmmer.easel.MSAFile(msa_file, digital=True, alphabet=alpha) as msa_file:
        msa = msa_file.read()

    # Set name and accession if provided
    if name:
        msa.name = name.encode("utf-8")
    else:
        msa.name = msa.names[0] #.decode("utf-8")
    if accession:
        msa.accession = accession.encode("utf-8")

    # Build the HMM
    builder = pyhmmer.plan7.Builder(alpha)
    background = pyhmmer.plan7.Background(alpha)
    hmm, _, _ = builder.build_msa(msa, background)

    # Set gathering threshold if provided
    if set_ga:
        hmm.cutoffs.gathering = set_ga, set_ga

    # Write the HMM to file
    with open(output, "wb") as out_f:
        hmm.write(out_f)

    return output


def hmmdb_from_directory(
    msa_dir,
    output,
    msa_pattern="*.faa",
    info_table=None,
    name_col="MARKER",
    accs_col="ANNOTATION_ACCESSIONS",
    desc_col="ANNOTATION_DESCRIPTION",
    default_gath= "1",
    gath_col= None #"GATHERING_THRESHOLD",
):
    """Create a concatenated HMM database from a directory of MSA files.

    Args:
        msa_dir: str or Path, directory containing MSA files
        output: str or Path, path to save the concatenated HMM database
        msa_pattern: str, glob pattern to match MSA files
        info_table: str or Path, path to a table file containing information about the MSA files - name, accession, description. merge attempted based on the stem of the MSA file names to match the `name` column of the info table.
        name_col: str, column name in the info table to use for the HMM name
        accs_col: str, column name in the info table to use for the HMM accession
        desc_col: str, column name in the info table to use for the HMM description
        gath_col: str, column name for gathering threshold
        default_gath: str, default gathering threshold if none provided in info table
    """

    msa_dir = Path(msa_dir)
    output = Path(output)
    default_gath =  default_gath.encode("utf-8")  # default gathering threshold if none provided

    if info_table is not None:
        info_table = Path(info_table)
        info_table = pl.read_csv(info_table, has_header=True)
        if name_col not in info_table.columns:
            raise ValueError(f"info_table must contain a '{name_col}' column")
        some_bool = True
        cols_map = {accs_col: "accession", desc_col: "description"}
    else:
        some_bool = False

    hmms = []
    # create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        # Process each MSA file and collect HMMs
        for msa_file in track(
            msa_dir.glob(msa_pattern),
            description="Processing MSA files",
            total=len(list(msa_dir.glob(msa_pattern))),
        ):
            this_gath = default_gath
            with pyhmmer.easel.MSAFile(msa_file, digital=True) as msa_file_obj:
                msa = msa_file_obj.read()
            msa.name = msa_file.stem.encode("utf-8")
            # get info from the info table
            if some_bool:
                info = info_table.filter(
                    pl.col(name_col).str.contains(msa.name.decode())
                )
                if info.height == 1: # should only be one match but just in case
                    for col_key, col_val in cols_map.items():
                        if col_val is not None:
                            # print(col_val)
                            msa.__setattr__(
                                col_val,
                                info[col_key].item().encode("utf-8")
                                if info[col_key].item() is not None
                                else "None".encode("utf-8"),
                            )
                    if gath_col in info.columns:
                        this_gath = (
                            info[gath_col].item().encode("utf-8")
                            if info[gath_col].item() is not None
                            else "1".encode("utf-8")
                        )
                    else:
                        this_gath = default_gath # default gathering threshold
            else:
                msa.description = "None".encode("utf-8")
            # Build the HMM
            builder = pyhmmer.plan7.Builder(msa.alphabet)
            background = pyhmmer.plan7.Background(msa.alphabet)
            hmm, _, _ = builder.build_msa(msa, background)

            # Set gathering threshold if provided (or default to 1)
            hmm.cutoffs.gathering = (float(this_gath), float(this_gath))
            hmms.append(hmm)
            # write the hmm to a file
            fh = open(temp_dir / f"{msa.name.decode()}.hmm", "wb")
            hmm.write(fh, binary=False)
            fh.close()
        # pyhmmer.hmmer.hmmpress(iter(hmms), output=output) # this is broken
        # writing all the hmms to the output, new line as seperator
        with open(output, "wb") as out_f:
            for hmm in hmms:
                hmm.write(out_f, binary=False)
                out_f.write(b"\n")
    return output


def mmseqs_profile_db_from_directory(
    msa_dir,
    output,
    msa_pattern="*.faa",
    info_table=None,
    name_col="MARKER",
    accs_col="ANNOTATION_ACCESSIONS",
    desc_col="ANNOTATION_DESCRIPTION",
    match_mode: int = 1,
    match_ratio: float = 0.5,
):
    """Create a concatenated HMM database from a directory of MSA files.

    Args:
        msa_dir: str or Path, directory containing MSA files
        output: str or Path, path to save the concatenated HMM database
        msa_pattern: str, glob pattern to match MSA files
        info_table: str or Path, path to a table file containing information about the MSA files - name, accession, description. merge attempted based on the stem of the MSA file names to match the `name` column of the info table.
        name_col: str, column name in the info table to use for the profile name
        accs_col: str, column name in the info table to use for the profile accession
        desc_col: str, column name in the info table to use for the profile description
        match_mode: int, passed to `mmseqs msa2profile --match-mode` (0 = by first sequence, 1 = by gap fraction). Default is 1 (safer for diverse MSAs).
        match_ratio: float, passed to `mmseqs msa2profile --match-ratio` (threshold for gap fraction). Default is 0.5.

    Note:
        The function now uses `mmseqs msa2profile` with configurable `--match-mode` and `--match-ratio`.
        By default we set `match_mode=1` and `match_ratio=0.5` which treats columns with at least 50% residues as match states — safer for heterogeneous MSAs.
    """

    msa_dir = Path(msa_dir)
    output = Path(output)

    if info_table is not None:
        info_table = Path(info_table)
        info_table = pl.read_csv(info_table, has_header=True)
        if name_col not in info_table.columns:
            raise ValueError(f"info_table must contain a '{name_col}' column")
        some_bool = True
        cols_map = {accs_col: "accession", desc_col: "description"}
    else:
        some_bool = False
    

    all_msa_blocks = [] # List to hold the content of each complete MSA block
    # Process each MSA file
    for msa_file in track(
        msa_dir.glob(msa_pattern),
        description="Processing MSA files",
        total=len(list(msa_dir.glob(msa_pattern))),
    ):
        with pyhmmer.easel.MSAFile(msa_file, digital=True) as msa_file_obj:
            msa = msa_file_obj.read()

        # Build a combined display label to force into the profile header
        marker = msa_file.stem
        accs_val = "None"
        desc_val = "None"
        if some_bool:
            info = info_table.filter(pl.col(name_col).str.contains(marker))
            if info.height == 1:
                # prefer the accession and description columns from the info table
                if accs_col in info.columns:
                    accs_val = (
                        info[accs_col].item()
                        if info[accs_col].item() is not None
                        else "None"
                    )
                if desc_col in info.columns:
                    desc_val = (
                        info[desc_col].item()
                        if info[desc_col].item() is not None
                        else "None"
                    )

        # Normalize to strings and build display label
        accs_str = str(accs_val)
        desc_str = str(desc_val)
        display_label = f"{marker}|{accs_str}|{desc_str}"

        # Set MSA-level metadata so Stockholm writer will include GF tags
        try:
            msa.name = display_label.encode("utf-8")
        except Exception:
            msa.name = msa_file.stem.encode("utf-8")
        try:
            msa.accession = accs_str.encode("utf-8")
        except Exception:
            pass
        try:
            msa.description = desc_str.encode("utf-8")
        except Exception:
            pass

        # Also replace the first sequence header to the display label so mmseqs
        # will see a clear, unique identifier for the profile
        try:
            if hasattr(msa, "names") and len(msa.names) > 0:
                msa.names[0] = display_label.encode("utf-8")
        except Exception:
            # not critical; proceed
            pass

        # Write the complete MSA block content to a bytes buffer
        temp_buffer = io.BytesIO()

        # Rely on pyhmmer's msa.write to include the final '//' footer
        msa.write(format="stockholm", fh=temp_buffer)

        # Post-process the Stockholm block to explicitly include GF tags
        # for ID (display label), AC (accessions), and DE (description).
        # This ensures mmseqs convertmsa picks up the desired profile header.
        raw_block = temp_buffer.getvalue()
        try:
            text_block = raw_block.decode("utf-8")
        except Exception:
            text_block = raw_block.decode("latin-1")

        lines = text_block.splitlines()
        gf_lines = [f"#=GF ID {display_label}", f"#=GF AC {accs_str}", f"#=GF DE {desc_str}"]
        if len(lines) > 0 and lines[0].startswith("# STOCKHOLM"):
            # Insert GF lines after the STOCKHOLM header
            lines[1:1] = gf_lines
            new_text = "\n".join(lines) + "\n"
        else:
            # Prepend a STOCKHOLM header and the GF lines
            new_text = "# STOCKHOLM 1.0\n" + "\n".join(gf_lines) + "\n" + text_block

        msa_block = new_text.encode("utf-8")
        all_msa_blocks.append(msa_block)

    # create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        # Write all collected blocks to the final output file
        all_sto = temp_dir / "all_msas.sto"
        with open(all_sto, "wb") as msa_sto_fh:
            msa_sto_fh.write(b"".join(all_msa_blocks))

        # ensure output parent exists
        output.parent.mkdir(parents=True, exist_ok=True)

        # Convert the combined Stockholm MSAs into an MMseqs MSA DB and then into a profile DB
        msa_db = temp_dir / "msa_db"
        try:
            # mmseqs convertmsa <sto> <msa_db>
            run_command_comp(
                "mmseqs",
                positional_args=["convertmsa", str(all_sto), str(msa_db)],
                positional_args_location="start",
                return_final_cmd=True,
                check_status=True,
                check_output=False,
            )

            # mmseqs msa2profile <msa_db> <output> --match-mode X --match-ratio Y
            run_command_comp(
                "mmseqs",
                positional_args=["msa2profile", str(msa_db), str(output)],
                params={"match-mode": int(match_mode), "match-ratio": float(match_ratio)},
                positional_args_location="start",
                return_final_cmd=True,
                check_status=True,
                check_output=False,
                output_file=str(output),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to build mmseqs profile DB: {e}")

        return output

def mmseqs_search(
    query_db: Union[str, Path],
    target_db: Union[str, Path],
    result_db: Union[str, Path],
    tmp_dir: Union[str, Path],
    sensitivity: int = 5,
    threads: int = 1,
    threads_opt_name: str = "--threads",
    extra_opts: str = "",
):
    """Run `mmseqs search` with sensible defaults and return the result DB path.

    This is a thin wrapper around the mmseqs CLI to centralize invocation.
    """
    positional = [
        "search",
        str(query_db),
        str(target_db),
        str(result_db),
        str(tmp_dir)
    ]
    params ={
        "threads": int(threads),
        "sensitivity": int(sensitivity), # TODO: check if this should be float?
        "a" : True # TODO: find out if this has a long form name so less chance of fudging the param prefix.
        }

    if extra_opts:
        # split extra options naively on whitespace
        positional.extend(str(extra_opts).split())

    run_command_comp(
        "mmseqs",
        positional_args=positional,
        params=params,
        positional_args_location="start",
        return_final_cmd=True,
        check_status=True,
        check_output=False,
    )
    return result_db


def mmseqs_convertalis(
    query_db: Union[str, Path],
    target_db: Union[str, Path],
    alignment_db: Union[str, Path],
    out_file: Union[str, Path],
    format_mode: int = 0,
    format_output: Optional[Union[str, List[str]]] = None,
    compressed: int = 0,
    threads: Optional[int] = None,
):
    """Wrapper for `mmseqs convertalis` that supports custom output columns.

    Use `format_output` to include `theader` (target header) which often contains
    richer label information (for example the GF ID/AC/DE we injected into Stockholm).
    Example format_output: 'query,theader,evalue,bits,alnlen'
    """
    params = {"format-mode": int(format_mode)}
    if format_output:
        if isinstance(format_output, list):
            fmt = ",".join(format_output)
        else:
            fmt = str(format_output)
        params["format-output"] = fmt
    if compressed:
        params["compressed"] = int(compressed)
    if threads is not None:
        params["threads"] = int(threads)

    run_command_comp(
        "mmseqs",
        positional_args=["convertalis", str(query_db), str(target_db), str(alignment_db), str(out_file)],
        params=params,
        check_status=True,
        check_output=True,
        output_file=str(out_file),
    )
    return out_file


def msas_to_stockholm(
    msa_dir,
    out_sto: Union[str, Path],
    msa_pattern: str = "*.faa",
    info_table: Optional[Union[str, Path]] = None,
    name_col: str = "MARKER",
    accs_col: str = "ANNOTATION_ACCESSIONS",
    desc_col: str = "ANNOTATION_DESCRIPTION",
):
    """Write a combined Stockholm file from MSAs in `msa_dir` including GF tags.

    Returns the path to the written Stockholm file (`out_sto`). This re-uses the
    logic used by `mmseqs_profile_db_from_directory` to create explicit `#=GF`
    ID/AC/DE tags so downstream `mmseqs convertmsa`/`msa2profile` can see them.
    """
    msa_dir = Path(msa_dir)
    out_sto = Path(out_sto)

    if info_table is not None:
        info_table = Path(info_table)
        info_table = pl.read_csv(info_table, has_header=True)
        some_bool = True
    else:
        some_bool = False

    all_msa_blocks = []
    for msa_file in track(
        msa_dir.glob(msa_pattern),
        description="Processing MSA files to stockholm",
        total=len(list(msa_dir.glob(msa_pattern))),
    ):
        with pyhmmer.easel.MSAFile(msa_file, digital=True) as msa_file_obj:
            msa = msa_file_obj.read()

        marker = msa_file.stem
        accs_val = "None"
        desc_val = "None"
        if some_bool:
            info = info_table.filter(pl.col(name_col).str.contains(marker))
            if info.height == 1:
                if accs_col in info.columns:
                    accs_val = info[accs_col].item() or "None"
                if desc_col in info.columns:
                    desc_val = info[desc_col].item() or "None"

        accs_str = str(accs_val)
        desc_str = str(desc_val)
        display_label = f"{marker}|{accs_str}|{desc_str}"

        try:
            msa.name = display_label.encode("utf-8")
        except Exception:
            msa.name = msa_file.stem.encode("utf-8")
        try:
            msa.accession = accs_str.encode("utf-8")
        except Exception:
            pass
        try:
            msa.description = desc_str.encode("utf-8")
        except Exception:
            pass
        try:
            if hasattr(msa, "names") and len(msa.names) > 0:
                msa.names[0] = display_label.encode("utf-8")
        except Exception:
            pass

        temp_buffer = io.BytesIO()
        msa.write(format="stockholm", fh=temp_buffer)
        raw_block = temp_buffer.getvalue()
        try:
            text_block = raw_block.decode("utf-8")
        except Exception:
            text_block = raw_block.decode("latin-1")

        lines = text_block.splitlines()
        gf_lines = [f"#=GF ID {display_label}", f"#=GF AC {accs_str}", f"#=GF DE {desc_str}"]
        if len(lines) > 0 and lines[0].startswith("# STOCKHOLM"):
            lines[1:1] = gf_lines
            new_text = "\n".join(lines) + "\n"
        else:
            new_text = "# STOCKHOLM 1.0\n" + "\n".join(gf_lines) + "\n" + text_block

        all_msa_blocks.append(new_text.encode("utf-8"))

    out_sto.parent.mkdir(parents=True, exist_ok=True)
    with open(out_sto, "wb") as fh:
        fh.write(b"".join(all_msa_blocks))

    return out_sto


