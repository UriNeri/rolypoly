import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import polars as pl
import rich_click as click
from needletail import parse_fastx_file
from rich.console import Console
# from rolypoly.rolypoly import rolypoly
# import polars_hash as plh #TODO: Use this instead of hashlib (specifiically for non-cryptographic hash)
# import polars_pairing as plp
# __all__ = [
#     "read_fasta_df",
#     "process_sequences",
#     "is_nucl_string",
#     "SequenceExpr",
#     "RNAStructureExpr",
#     "FastxNameSpace",
# ]

global datadir
datadir = Path(os.environ["ROLYPOLY_DATA"])

# Register custom expressions for sequence analysis
@pl.api.register_expr_namespace("seq")
class SequenceExpr:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def gc_content(self) -> pl.Expr:
        """Calculate GC content of sequence"""
        return (
            self._expr.str.count_matches("G") + self._expr.str.count_matches("C")
        ) / self._expr.str.len_chars()

    def n_count(self) -> pl.Expr:
        """Count N's in sequence"""
        return self._expr.str.count_matches("N")

    def length(self) -> pl.Expr:
        """Get sequence length"""
        return self._expr.str.len_chars()

    def codon_usage(self) -> pl.Expr:
        """Calculate codon usage frequencies"""

        def _calc_codons(seq: str) -> dict:
            codons = defaultdict(int)
            for i in range(0, len(seq) - 2, 3):
                codon = seq[i : i + 3].upper()
                if "N" not in codon:
                    codons[codon] += 1
            total = sum(codons.values())
            return {k: v / total for k, v in codons.items()} if total > 0 else {}

        return self._expr.map_elements(_calc_codons, return_dtype=pl.Struct)

    def generate_hash(self, length: int = 32) -> pl.Expr:
        """Generate a hash for a sequence"""
        import hashlib

        def _hash(seq: str) -> str:
            return hashlib.md5(seq.encode()).hexdigest()[:length]

        return self._expr.map_elements(_hash, return_dtype=pl.String)

    def calculate_kmer_frequencies(self, k: int = 3) -> pl.Expr:
        """Calculate k-mer frequencies in the sequence"""

        def _calc_kmers(seq: str, k: int) -> dict:
            if not seq or len(seq) < k:
                return {}
            kmers = defaultdict(int)
            for i in range(len(seq) - k + 1):
                kmer = seq[i : i + k].upper()
                if "N" not in kmer:
                    kmers[kmer] += 1
            total = sum(kmers.values())
            return {k: v / total for k, v in kmers.items()} if total > 0 else {}

        return self._expr.map_elements(
            lambda x: _calc_kmers(x, k), return_dtype=pl.Struct
        )

    def is_valid_codon(self) -> pl.Expr:
        """Check if sequence length is divisible by 3 (valid for codon analysis)"""
        return (self._expr.str.len_chars() % 3) == 0


def _scan_fastx_file(input_file: Union[str, Path], batch_size: int = 512) -> pl.LazyFrame:
    """Internal function to scan a FASTA/FASTQ file into a lazy polars DataFrame."""
    
    def file_has_quality(file: Union[str, Path]) -> bool:
        try:
            first_record = next(parse_fastx_file(file))
            return hasattr(first_record, 'qual') and getattr(first_record, 'qual', None) is not None  # type: ignore
        except StopIteration:
            return False

    has_quality = file_has_quality(input_file)
    if has_quality:
        schema = pl.Schema(
            {"header": pl.String, "sequence": pl.String, "quality": pl.String}
        )
    else:
        schema = pl.Schema({"header": pl.String, "sequence": pl.String})

    def read_chunks():
        reader = parse_fastx_file(input_file)
        while True:
            chunk = []
            for _ in range(batch_size):
                try:
                    record = next(reader)
                    row = [str(getattr(record, 'id', '')), str(getattr(record, 'seq', ''))]  # type: ignore
                    if has_quality and hasattr(record, 'qual'):
                        row.append(str(getattr(record, 'qual', '')))  # type: ignore
                    chunk.append(row)
                except StopIteration:
                    if chunk:
                        yield pl.LazyFrame(chunk, schema=schema, orient="row")
                    return
            yield pl.LazyFrame(chunk, schema=schema, orient="row")

    return pl.concat(read_chunks(), how="vertical")


@pl.api.register_lazyframe_namespace("from_fastx")
class FastxLazyFrameAccessor:
    def __init__(self, ldf: pl.LazyFrame):
        self._ldf = ldf

    def scan(self, input_file: Union[str, Path], batch_size: int = 512) -> pl.LazyFrame:
        """Scan a FASTA/FASTQ file into a lazy polars DataFrame.

        This function extends polars with the ability to lazily read FASTA/FASTQ files.
        It can be used directly as pl.LazyFrame.from_fastx.scan("sequences.fasta").

        Args:
            input_file (Union[str, Path]): Path to the FASTA/FASTQ file
            batch_size (int, optional): Number of records to read per batch. Defaults to 512.

        Returns:
            pl.LazyFrame: Lazy DataFrame with columns:
                - header: Sequence headers (str)
                - sequence: Sequences (str) 
                - quality: Quality scores (only for FASTQ)
        """
        return _scan_fastx_file(input_file, batch_size)


@pl.api.register_dataframe_namespace("from_fastx")
class FastxDataFrameAccessor:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def scan(self, file: Union[str, Path], batch_size: int = 512) -> pl.DataFrame:
        return _scan_fastx_file(file, batch_size).collect()


@pl.api.register_expr_namespace("rna")
class RNAStructureExpr:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def predict_structure(self) -> pl.Expr:
        """Predict RNA structure and minimum free energy"""
        import RNA

        def _predict(seq: str) -> dict:
            if len(seq) > 10000:
                return {"structure": None, "mfe": None}
            try:
                ss_string, mfe = RNA.fold_compound(seq).mfe()
                return {"structure": ss_string, "mfe": mfe}
            except Exception:
                return {"structure": None, "mfe": None}

        return self._expr.map_elements(
            _predict,
            # strategy="threading",
            return_dtype=pl.Struct({"structure": pl.String, "mfe": pl.Float64}),
        )

    def predict_structure_with_tool(self, tool: str = "ViennaRNA") -> pl.Expr:
        """Predict RNA structure using specified tool"""

        def _predict_vienna(seq: str) -> dict:
            if len(seq) > 10000:
                return {"structure": None, "mfe": None}
            try:
                import RNA

                ss_string, mfe = RNA.fold_compound(seq).mfe()
                return {"structure": ss_string, "mfe": mfe}
            except Exception:
                return {"structure": None, "mfe": None}

        def _predict_linearfold(seq: str) -> dict:
            import subprocess
            import tempfile

            if len(seq) > 10000:
                return {"structure": None, "mfe": None}
            try:
                # Convert T to U for RNA folding
                seq = seq.replace("T", "U")
                with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_in:
                    # Write sequence directly without FASTA header
                    temp_in.write(seq)
                    temp_in.flush()
                    temp_in_name = temp_in.name

                # Create temp file for output
                with tempfile.NamedTemporaryFile(mode="r+", delete=False) as temp_out:
                    temp_out_name = temp_out.name

                # Run LinearFold with default beam size (100)
                subprocess.run(
                    ["linearfold", temp_in_name],
                    stdout=open(temp_out_name, "w"),
                    stderr=subprocess.PIPE,
                )

                # Parse the results
                with open(temp_out_name, "r") as f:
                    result = f.read().strip().split("\n")
                    if len(result) >= 2:
                        structure, mfe_str = result[1].split()
                        mfe = float(mfe_str.replace("(", "").replace(")", ""))

                        # Clean up temp files
                        os.unlink(temp_in_name)
                        os.unlink(temp_out_name)

                        return {"structure": structure, "mfe": mfe}

                # Clean up temp files
                os.unlink(temp_in_name)
                if os.path.exists(temp_out_name):
                    os.unlink(temp_out_name)

                return {"structure": None, "mfe": None}

            except Exception:
                return {"structure": None, "mfe": None}

        if tool.lower() == "linearfold":
            return self._expr.map_elements(
                _predict_linearfold,
                return_dtype=pl.Struct({"structure": pl.String, "mfe": pl.Float64}),
            )
        else:  # Default to ViennaRNA
            return self._expr.map_elements(
                _predict_vienna,
                return_dtype=pl.Struct({"structure": pl.String, "mfe": pl.Float64}),
            )



def read_fasta_needletail(fasta_file: str) -> tuple[list[str], list[str]]:
    """Read sequences from a FASTA/FASTQ file using needletail"""

    seqs = []
    seq_ids = []
    for record in parse_fastx_file(fasta_file):
        seqs.append(getattr(record, 'seq', ''))  # type: ignore
        seq_ids.append(getattr(record, 'id', ''))  # type: ignore
    return seq_ids, seqs


def add_fasta_to_gff(config, gff_file):
    """Add FASTA section to GFF file"""

    with open(gff_file, "a") as f:
        f.write("##FASTA\n")
        write_fasta_file(
            records=parse_fastx_file(config.input),
            output_file=f,
            format="fasta",
        )


def filter_fasta_by_headers(
    fasta_file: str,
    headers: Union[str, List[str]],
    output_file: str,
    invert: bool = False,
) -> None:
    """Filter sequences in a FASTA file based on their headers.

    Extracts sequences whose headers match (or don't match if inverted) any of
    the provided header patterns.

    Args:
        fasta_file (str): Path to input FASTA file
        headers (Union[str, List[str]]): Either a file containing headers (one per line)
            or a list of header patterns to match
        output_file (str): Path to write filtered sequences
        invert (bool, optional): If True, keep sequences that don't match.
    """

    headers_list = []
    if not isinstance(headers, list):
        with open(headers, "r") as f:
            for line in f:
                headers_list.append(line.strip())
    else:
        headers_list = headers

    with open(output_file, "w") as out_f:
        for record in parse_fastx_file(fasta_file):
            matches = any(header in str(getattr(record, 'id', '')) for header in headers_list)  # type: ignore
            if (
                matches ^ invert
            ):  # XOR operation: write if (matches and not invert) or (not matches and invert)
                out_f.write(f">{getattr(record, 'id', '')}\n{getattr(record, 'seq', '')}\n")  # type: ignore


# def translate_6frx_seqkit(input_file: str, output_file: str, threads: int) -> None:
#     """Translate nucleotide sequences in all 6 reading frames using Rust implementation.

#     Args:
#         input_file (str): Path to input nucleotide FASTA file
#         output_file (str): Path to output amino acid FASTA file
#         threads (int): Number of CPU threads to use
#     """
#     # from rolypoly import translate_six_frame_file
    
#     # Use our Rust implementation
#     # translate_six_frame_file(input_file, output_file, None, 11, threads)

def translate_6frx_seqkit(input_file: str, output_file: str, threads: int) -> None:
    """Translate nucleotide sequences in all 6 reading frames using seqkit.

    Args:
        input_file (str): Path to input nucleotide FASTA file
        output_file (str): Path to output amino acid FASTA file
        threads (int): Number of CPU threads to use

    Note:
        Requires seqkit to be installed and available in PATH.
        The output sequences are formatted with 20000bp line width.
    """
    import subprocess as sp

    command = f"seqkit translate -x -F --clean -w 0 -f 6 {input_file} --id-regexp '(\\*)' --clean  --threads {threads} > {output_file}"
    sp.run(command, shell=True, check=True)


def translate_with_bbmap(input_file: str, output_file: str, threads: int) -> None:
    """Translate nucleotide sequences using BBMap's callgenes.sh

    Args:
        input_file (str): Path to input nucleotide FASTA file
        output_file (str): Path to output amino acid FASTA file
        threads (int): Number of CPU threads to use

    Note:
        - Requires BBMap to be installed and available in PATH (should be done via bbmapy)
        - Generates both protein sequences (.faa) and gene annotations (.gff)
        - The GFF output file is named by replacing .faa with .gff
    """
    import subprocess as sp

    gff_o = output_file.replace(".faa", ".gff")
    command = (
        f"callgenes.sh threads={threads} in={input_file} outa={output_file} out={gff_o}"
    )
    sp.run(command, shell=True, check=True)


def pyro_predict_orfs(
    input_file: str,
    output_file: str,
    threads: int,
    # gv_or_else: str = "gv",
    genetic_code: int = 11,  # NOT USED YET # TODO: add SUPPORT for this.
    model: str = "1",  # NOT USED YET/at all.
) -> None:
    """Predict and translate Open Reading Frames using Pyrodigal.

    Uses either Pyrodigal-GV (optimized for viruses) or standard Pyrodigal
    to predict and translate ORFs from nucleotide sequences.

    Args:
        input_file (str): Path to input nucleotide FASTA file
        output_file (str): Path to output amino acid FASTA file
        threads (int): Number of CPU threads to use
        gv_or_else (str, optional): Uses "gv" for viral genes or any other value for standard gene prediction.
        genetic_code (int, optional): Genetic code table to use (Standard/Bacterial) (NOT USED YET).

    Note:
        - Creates both protein sequences (.faa) and gene annotations (.gff)
        - genetic_code is 11 for standard/bacterial

    """
    import multiprocessing.pool

    import pyrodigal_gv as pyro_gv

    sequences = []
    ids = []
    for record in parse_fastx_file(input_file):
        sequences.append((record.seq)) # type: ignore
        ids.append((record.id)) # type: ignore

    gene_finder = pyro_gv.ViralGeneFinder(
        meta=True
    )  # a single gv gene finder object

    with multiprocessing.pool.Pool(processes=threads) as pool:
        orfs = pool.map(gene_finder.find_genes, sequences)

    with open(output_file, "w") as dst:
        for i, orf in enumerate(orfs):
            orf.write_translations(dst, sequence_id=ids[i], width=111110)

    with open(output_file.replace(".faa", ".gff"), "w") as dst:
        for i, orf in enumerate(orfs):
            orf.write_gff(dst, sequence_id=ids[i], full_id=True)


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
    import re

    cigar_tuples = re.findall(r"(\d+)([MIDNSHPX=])", cigar_string)
    matches = sum(int(length) for length, op in cigar_tuples if op in {"M", "=", "X"})
    total_length = sum(
        int(length) for length, op in cigar_tuples if op in {"M", "I", "D", "=", "X"}
    )
    return (matches - num_mismatches) / total_length * 100


def mask_sequence_mp(seq: str, start: int, end: int, is_reverse: bool) -> str:
    """Mask a portion of a mappy (minimap2) aligned sequence with N's.

    Args:
        seq (str): Input sequence to mask
        start (int): Start position of the region to mask (0-based)
        end (int): End position of the region to mask (exclusive)
        is_reverse (bool): Whether the sequence is reverse complemented

    Returns:
        str: Sequence with the specified region masked with N's

    Note:
        Handles reverse complement if needed by using mappy's revcomp function.
    """
    import mappy as mp

    is_reverse = is_reverse == -1
    if is_reverse:
        seq = str(mp.revcomp(seq))
    masked_seq = seq[:start] + "N" * (end - start) + seq[end:]
    return str(mp.revcomp(masked_seq)) if is_reverse else masked_seq


def is_gzipped(file_path: str) -> bool:
    with open(file_path, "rb") as test_f:
        return test_f.read(2).startswith(b"\x1f\x8b")


def guess_fastq_properties(file_path: str, mb_to_read: int = 20) -> dict:
    """Analyze a FASTQ file to determine its properties.

    Examines the first 20MB of a FASTQ file to determine if it's gzipped,
    paired-end, and calculate average read length.

    Args:
        file_path (str): Path to the FASTQ file

    Returns:
        dict: Dictionary containing:
            - is_gzipped (bool): Whether file is gzip compressed
            - paired_end (bool): Whether reads appear to be paired-end
            - average_read_length (float): Average length of reads
    """
    import gzip

    bytes_to_read = mb_to_read * 1024 * 1024
    is_gz = is_gzipped(file_path)
    file_size = os.path.getsize(file_path)
    if file_size < bytes_to_read:
        bytes_to_read = file_size
    paired_end = False
    average_read_length = 0
    total_length = 0
    read_count = 0

    # Open the file accordingly
    if is_gz:
        with gzip.open(file_path, "rb") as f:
            data = f.read(bytes_to_read)
    else:
        with open(file_path, "rb") as f:
            data = f.read(bytes_to_read)

    # Decode the data
    data = data.decode("utf-8", errors="ignore")

    # Split the data into lines
    lines = data.splitlines()

    # Process the lines to determine properties
    for i in range(0, len(lines) - len(lines) % 4, 4):
        if lines[i].startswith("@"):
            read_id = lines[i][1:].split()[0]
            if "/1" in read_id or "/2" in read_id:
                paired_end = True
            read_length = len(lines[i + 1])
            total_length += read_length
            read_count += 1

    if read_count > 0:
        average_read_length = total_length / read_count

    return {
        "is_gzipped": is_gzipped,
        "paired_end": paired_end,
        "average_read_length": average_read_length,
    }


def guess_fasta_alpha(input_file) -> str:
    # only peek at the first sequence
    with open(input_file, "rb") as fin:
        input_string = get_sequence_between_newlines(
            fin.peek(2)[:1110].decode().replace(r"*/\n", "")
        )
    if is_nucl_string(input_string):
        return "nucl"
    elif is_aa_string(input_string):
        return "amino"
    else:
        return "nothing_good"


def is_nucl_string(sequence, extended=False):
    valid_characters = set({"A", "T", "G", "C", "U", "N"})
    if extended:
        valid_characters.update({"M", "R", "W", "S", "Y", "K", "V", "H", "D", "B"})
    return all(char in valid_characters for char in sequence.upper())


def is_aa_string(sequence, extended=False):
    valid_characters = set(
        {
            "A",
            "R",
            "N",
            "D",
            "C",
            "Q",
            "E",
            "G",
            "H",
            "I",
            "L",
            "K",
            "M",
            "F",
            "P",
            "S",
            "T",
            "W",
            "Y",
            "V",
            "O",
            "U",
            "B",
            "Z",
            "X",
            "J",
        }
    )
    if extended:
        valid_characters.update({"B", "J", "X", "Z", "*", "-", "."})
    return all(char in valid_characters for char in sequence.upper())


def get_sequence_between_newlines(input_string):
    import re

    newline_pattern = re.compile(r"\n")
    newline_positions = [
        match.start() for match in newline_pattern.finditer(input_string)
    ]
    if len(newline_positions) < 2:
        return input_string[newline_positions[0] + 1 :]
    return input_string[newline_positions[0] + 1 : newline_positions[1]]


def ensure_faidx(input_file: str) -> None:
    """Ensure a FASTA file has a pyfastx index.

    Creates a pyfastx index for the input FASTA file if it doesn't exist.

    Args:
        input_file (str): Path to the FASTA file

    """
    import pyfastx

    if not os.path.exists(f"{input_file}.fxi"):
        console.print(f"[yellow]Indexing {input_file} with pyfastx    [/yellow]")
        pyfastx.Fasta(str(input_file))
        console.print(f"[green]Indexing complete.[/green]")


def download_genome(taxid: str) -> None:
    """Download genome data from NCBI for a given taxon ID.

    Args:
        taxid (str): NCBI taxonomy ID for the organism

    Note:
        Uses the NCBI datasets command-line tool to download genome data.
        Downloads RNA and genome data, excluding atypical sequences.
    """
    import subprocess as sp

    # import shutil
    # print(shutil.which("datasets"))
    sp.run(
        [
            "datasets",
            "download",
            "genome",
            "taxon",
            taxid,
            "--include",
            "rna,genome",
            "--filename",
            f"{taxid}_fetched_genomes.zip",
            "--assembly-version",
            "latest",
            "--exclude-atypical",
            "--assembly-source",
            "RefSeq",
            "--no-progressbar",
        ],
        stdout=sp.DEVNULL,
        stderr=sp.DEVNULL,
        # shell=True
    )


def process_with_timeout(func: callable, arg: any, timeout: int) -> any: #type: ignore
    """Execute a function with a timeout.

    Args:
        func (callable): Function to execute
        arg (any): Argument to pass to the function
        timeout (int): Timeout in seconds

    Returns:
        any: Result of the function call, or None if timeout occurred

    Note:
        Uses ProcessPoolExecutor to run the function in a separate process.
    """
    from concurrent.futures import ProcessPoolExecutor, TimeoutError

    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, arg)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            print(f"Task for {arg} timed out after {timeout} seconds")
            return None


def fetch_genomes(
    input_file: str,
    output_file: str,
    threads: int = 1,
    max2take: int = 25,
    timeout: int = 600,
) -> None:
    """Fetch genomes from NCBI for masking purposes.

    Downloads and processes genomes from NCBI (via datasets) based on a BBMap stats file.
    Filters out metagenomic, uncultured, and synthetic sequences.

    Args:
        input_file (str): Path to BBMap stats file
        output_file (str): Path to save the processed sequences
        threads (int, optional): Number of threads to use.
        max2take (int, optional): Maximum number of genomes to process.
        timeout (int, optional): Timeout in seconds for each download.

    Note:
        - Excludes sequences with keywords indicating non-natural sources
        - Concatenates and deduplicates sequences
        - Removes viral sequences from the final output
    TODO:
        - replace taxonkit with taxopy
        - replace datasets.
    """
    import concurrent.futures
    import shutil
    import subprocess as sp
    from concurrent.futures import ProcessPoolExecutor

    from rolypoly.utils.various import extract_zip

    with open(input_file, "r") as f:
        lines = f.readlines()[4:]

    taxons = set()
    for line in lines:
        taxon = line.split(sep=";")[-1]
        taxon = taxon.split(sep="\t")[0]
        if any(
            word in taxon.lower()
            for word in [
                "meta",
                "uncultured",
                "unidentified",
                "synthetic",
                "construct",
                "coli",
            ]
        ):
            print(
                f"{line} contains a keyword that indicates it is not an actual organism, skipping"
            )
        else:
            if "PREDICTED: " in taxon:
                taxon = taxon.split(sep=": ")[1]
                taxon = taxon.split(sep=" 28S")[0]
                taxon = taxon.split(sep=" 18S")[0]
                taxon = taxon.split(sep=" small subunit ")[0]
                taxon = taxon.split(sep=" large subunit ")[0]
            taxons.update([taxon])
            if len(taxons) > max2take:
                break

    if not taxons:
        print("No valid taxons found in the input file. Skipping genome fetching.")
        return

    # Use taxonkit to get taxids
    with open("tmp_gbs_50m_taxids.lst", "w") as f:
        sp.run(["taxonkit", "name2taxid", f"--data-dir {datadir}/taxdump"], input="\n".join(taxons).encode(), stdout=f)

    # Use datasets to download genomes
    with open("tmp_gbs_50m_taxids.lst", "r") as f:
        taxids = [
            line.split(sep="\t")[1].replace(" ", "_").strip()
            for line in f
            if line != ""
        ]
    taxids = list(set(taxids).difference(["", "562"]))  # Remove empty and E. coli

    if not taxids:
        print("No valid taxids found. Skipping genome fetching.")
        return

    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(
                process_with_timeout, download_genome, taxid, timeout
            )  # TODO: replace with something else.
            for taxid in taxids
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

    zip_files = list(Path(".").glob("*.zip"))
    if not zip_files:
        print("No genome zip files were downloaded. Skipping extraction.")
        return

    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(process_with_timeout, extract_zip, zip_file, timeout)
            for zip_file in zip_files
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred during extraction: {e}")

    # Concatenate and deduplicate sequences
    ref_seqs = set()
    ncbi_dataset_dir = Path("ncbi_dataset")
    if not ncbi_dataset_dir.exists():
        print("NCBI dataset directory not found. Skipping sequence processing.")
        return

    for folder in ncbi_dataset_dir.rglob("*/"):
        fna_files = list(folder.rglob("*.fna"))
        if not fna_files:
            continue
        rna_file = next((f for f in fna_files if "rna" in f.name.lower()), None)
        if rna_file:
            chosen_file = rna_file
        else:
            chosen_file = fna_files[0]  # choose the first file if no RNA file found
        ref_seqs.add(str(chosen_file))

    if not ref_seqs:
        print(
            "No FNA files found in the downloaded genomes. Skipping sequence processing."
        )
        return

    with open("tmp.lst", "w") as outf:
        for f in ref_seqs:
            outf.write(f + "\n")
    from shutil import which

    print(which("seqkit"))
    sp.call(
        ["seqkit", "rmdup", "-s", "--infile-list", "tmp.lst"],
        stdout=open("tmp.fasta", "w"),
    )

    # Remove any fasta entries that have "virus", "viral", "phage" in the header
    filter_fasta_by_headers(
        "tmp.fasta", ["virus", "viral", "phage"], output_file, invert=True
    )

    # Clean up
    for item in Path(".").glob("ncbi_dataset*"):
        if item.is_dir():
            shutil.rmtree(item)
        else:
            os.remove(item)
    remove_files = ["tmp.fasta", "tmp.lst"]
    for file in remove_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error removing {file}: {e}")


def mask_nuc_range(input_fasta: str, input_table: str, output_fasta: str) -> None:
    """Mask nucleotide sequences in a FASTA file based on provided range table.

    Args:
        input_fasta (str): Path to the input FASTA file
        input_table (str): Path to the table file with the ranges to mask
            (tab-delimited with columns: seq_id, start, stop, strand)
        output_fasta (str): Path to the output FASTA file

    Note:
        The ranges in the table should be 1-based coordinates.
        Handles both forward and reverse strand masking.
    """
    # Read ranges
    ranges = {}
    with open(input_table, "r") as f:
        for line in f:
            seq_id, start, stop, strand = line.strip().split("\t")
            if seq_id not in ranges:
                ranges[seq_id] = []
            ranges[seq_id].append((int(start), int(stop), strand))

    # Process FASTA file
    with open(input_fasta, "r") as in_f, open(output_fasta, "w") as out_f:
        current_id = ""
        current_seq = ""
        for line in in_f:
            if line.startswith(">"):
                if current_id:
                    if current_id in ranges:
                        for start, stop, strand in ranges[current_id]:
                            if start > stop:
                                start, stop = stop, start
                            if strand == "-":
                                current_seq = revcomp(current_seq)
                            current_seq = (
                                current_seq[: start - 1]
                                + "N" * (stop - start + 1)
                                + current_seq[stop:]
                            )
                            if strand == "-":
                                current_seq = revcomp(current_seq)
                    out_f.write(f">{current_id}\n{current_seq}\n")
                current_id = line[1:].strip()
                current_seq = ""
            else:
                current_seq += line.strip()

        if current_id:
            if current_id in ranges:
                for start, stop, strand in ranges[current_id]:
                    if start > stop:
                        start, stop = stop, start
                    if strand == "-":
                        current_seq = revcomp(current_seq)
                    current_seq = (
                        current_seq[: start - 1]
                        + "N" * (stop - start + 1)
                        + current_seq[stop:]
                    )
                    if strand == "-":
                        current_seq = revcomp(current_seq)
            out_f.write(f">{current_id}\n{current_seq}\n")


def read_fasta_polars(
    fasta_file, idcol="seq_id", seqcol="seq", add_length=False, add_gc_content=False
):
    seq_ids, seqs = read_fasta_needletail(fasta_file)
    df = pl.DataFrame({idcol: seq_ids, seqcol: seqs})
    if add_length:
        df = df.with_columns(pl.col(seqcol).str.len_chars().alias("length")) # type: ignore
    if add_gc_content:
        df = df.with_columns(pl.col(seqcol).str.count_matches("G|C").alias("gc_content") / pl.col(seqcol).str.len_chars().alias("length")) 
    return df


# def write_fasta_file(
#     records=None, seqs=None, headers=None, output_file=None, format: str = "fasta"
# ) -> None:
#     """
#     Note:
#         Either records or both seqs and headers must be provided.
#     """
#     import sys
#     if format == "fasta":
#         seq_delim = "\n"
#         header_delim = "\n>"
#     elif format == "tab":
#         seq_delim = "\t"
#         header_delim = "\n>"
#     else:
#         raise ValueError(f"Invalid format: {format}")

#     if output_file is None:
#         output_file = sys.stdout
#     elif isinstance(output_file, (str, Path)):
#         output_file = open(output_file, "w")

#     try:
#         if records:
#             for record in records:
#                 output_file.write(
#                     f"{header_delim}{record.id}{seq_delim}{str(record.seq)}"
#                 )
#         elif seqs is not None and headers is not None:
#             for header, seq in zip(headers, seqs):
#                 output_file.write(f"{header_delim}{header}{seq_delim}{seq}")
#         else:
#             raise ValueError("No records, seqs, or headers provided")
#     finally:
#         if output_file is not sys.stdout:
#             output_file.close()


def write_fasta_file(
    records=None, seqs=None, headers=None, output_file=None, format: str = "fasta"
) -> None:
    """Write sequences to a FASTA file or stdout if no output file is provided"""
    import sys

    if format == "fasta":
        seq_delim = "\n"
        header_delim = "\n>"
    elif format == "tab":
        seq_delim = "\t"
        header_delim = "\n>"
    else:
        raise ValueError(f"Invalid format: {format}")

    if output_file is None:
        output_file = sys.stdout
    else:
        output_file = open(output_file, "w")

    if records:
        for record in records:
            output_file.write(f"{header_delim}{record.id}{seq_delim}{str(record.seq)}")
    elif seqs is not None and headers is not None:
        for i, seq in enumerate(seqs):
            output_file.write(f"{header_delim}{headers[i]}{seq_delim}{seq}")
    else:
        raise ValueError("No records, seqs, or headers provided")


def read_fasta_df(file_path: str) -> pl.DataFrame:
    """Reads a FASTA file into a Polars DataFrame.

    Args:
        file_path (str): Path to the input FASTA file

    Returns:
        polars.DataFrame: DataFrame with columns:
            - header: Sequence headers
            - sequence: Sequence strings
            - group: Internal grouping number (used for sequence concatenation)

    """

    # First read the raw sequences using the registered from_fastx namespace
    raw_df = pl.LazyFrame.from_fastx.init(file_path).collect() # type: ignore defined above

    # The from_fastx namespace already returns header and sequence columns
    df = raw_df

    # Drop quality column if it exists
    if "quality" in df.columns:
        df = df.drop("quality")

    # Create a running length column for sequence concatenation
    df = df.with_columns([pl.col("header").is_not_null().cum_sum().alias("group")])

    # Create a new dataframe concatenating sequences
    result = (
        df.filter(pl.col("header") != "")
        .select("header", "group")
        .join(
            df.group_by("group").agg(
                [
                    pl.when(pl.col("sequence").str.contains(">") == False)
                    .then(pl.col("sequence"))
                    .str.concat(delimiter="")
                    .alias("sequence")
                ]
            ),
            on="group",
        )
    )

    return result


def rename_sequences(
    df: pl.DataFrame, prefix: str = "CID", use_hash: bool = False
) -> Tuple[pl.DataFrame, Dict[str, str]]:
    """Rename sequences with consistent IDs.

    Args:
        df (pl.DataFrame): DataFrame with 'header' and 'sequence' columns
        prefix (str, optional): Prefix for new IDs. Defaults to "CID".
        use_hash (bool, optional): Use hash instead of numbers. Defaults to False.

    Returns:
        Tuple[pl.DataFrame, Dict[str, str]]:
            - DataFrame with renamed sequences
            - Dictionary mapping old IDs to new IDs
    """

    if use_hash:
        # Use polars expressions for hash generation
        df_with_hashes = df.with_columns(
            pl.col("sequence").str.__hash__.alias("seq_hash")
        )
        new_headers = [f"{prefix}_{h}" for h in df_with_hashes["seq_hash"]]
    else:
        # Calculate padding based on total number of sequences
        padding = len(str(len(df)))
        new_headers = [f"{prefix}_{str(i + 1).zfill(padding)}" for i in range(len(df))]

    # Create mapping dictionary
    id_map = dict(zip(df["header"], new_headers))

    return df.with_columns(pl.Series("header", new_headers)), id_map


def process_sequences(df: pl.DataFrame) -> pl.DataFrame:
    """Process sequences and calculate statistics.

    Args:
        df (pl.DataFrame): DataFrame with sequence column
        structure (bool): Whether to include RNA structure prediction

    Returns:
        pl.DataFrame: DataFrame with added statistics columns
    """

    # Calculate basic stats
    df = df.with_columns(
        [
            pl.col("sequence").str.len_chars().alias("length"),
            pl.col("sequence").str.count_matches("G|C").alias("gc_content") / pl.col("sequence").str.len_chars().alias("length"),
            pl.col("sequence").str.count_matches("N").alias("n_count"),
        ]
    )

    return df


console = Console(width=150)


@click.command()
@click.option("-t", "--threads", default=1, help="Number of threads to use")
@click.option("-M", "--memory", default="6gb", help="Memory in GB")
@click.option("-o", "--output", required=True, help="Output file name")
@click.option(
    "-f", "--flatten", is_flag=True, help="Attempt to kcompress.sh the masked file"
)
@click.option("-i", "--input", required=True, help="Input fasta file")
@click.option("-F", "--mmseqs", is_flag=True, help="use mmseqs2 instead of bbmap.sh")
# @click.option('-lm', '--low-mem', is_flag=True, help='use strobealign instead of bbmap.sh')
@click.option("-lm", "--low-mem", is_flag=True, help="use minimap2 instead of bbmap.sh")
@click.option("-bt", "--bowtie", is_flag=True, help="use bowtie1 instead of bbmap.sh")
@click.option(
    "-r",
    "--reference",
    default=datadir / "masking/RVMT_NCBI_Ribo_Japan_for_masking.fasta",
    help="Provide an input fasta file to be used for masking, instead of the pre-generated collection of RNA viral sequences",
)
def mask_dna(
    threads, memory, output, flatten, input, mmseqs, low_mem, bowtie, reference
):
    """Mask an input fasta file for sequences that could be RNA viral (or mistaken for such).

    Args:
      threads: (int) Number of threads to use
      memory: (str) Memory in GB
      output: (str) Output file name
      flatten: (bool) Attempt to kcompress.sh the masked file
      input: (str) Input fasta file
      mmseqs: (bool) use mmseqs2 instead of bbmap.sh
      low_mem: (bool) use minimap2 instead of bbmap.sh
      bowtie: (bool) use bowtie1 instead of bbmap.sh
      reference: (str) Provide an input fasta file to be used for masking, instead of the pre-generated collection of RNA viral sequences

    Returns:
      None
    """
    import shutil
    import subprocess as sp

    import mappy as mp
    from bbmapy import bbmap, bbmask, kcompress

    from rolypoly.utils.various import ensure_memory

    input_file = Path(input).resolve()
    output_file = Path(output).resolve()
    # datadir = Path(os.environ['datadir'])
    memory = ensure_memory(memory)["giga"]
    reference = Path(reference).absolute().resolve()
    tmpdir = str(output_file.parent) + "/tmp"

    try:
        Path.mkdir(Path(tmpdir), exist_ok=True)
    except:
        console.print(f"couldn't create {tmpdir}")
        exit(123)

    if low_mem:
        console.print("Using minimap2 (low memory mode)")

        # Create a mappy aligner object
        aligner = mp.Aligner(str(reference), k=11, n_threads=threads, best_n=150)
        if not aligner:
            raise Exception("ERROR: failed to load/build index")

        # Perform alignment, write results to SAM file, and mask sequences
        masked_sequences = {}
        # with open(f"{tmpdir}/tmp_mapped.sam", "w") as sam_out: # masking directly so no need to save the sam.
        #     sam_out.write("@HD\tVN:1.6\tSO:unsorted\n")
        for name, seq, qual in mp.fastx_read(str(input_file)):
            masked_sequences[name] = seq
            for hit in aligner.map(seq):
                percent_id = calculate_percent_identity(hit.cigar_str, hit.NM)
                console.print(f"{percent_id}")
                if percent_id > 70:
                    masked_sequences[name] = mask_sequence_mp(
                        masked_sequences[name], hit.q_st, hit.q_en, hit.strand
                    )
                    # sam_out.write(f"{hit.ctg}\t{hit.r_st+1}\t{hit.mapq}\t{hit.cigar_str}\t*\t0\t0\t{seq}\t*\tNM:i:{hit.NM}\tms:i:{hit.mlen}\tmm:c:{hit.blen-hit.mlen}\n")

        # Write masked sequences to output file
        with open(output_file, "w") as out_f:
            for name, seq in masked_sequences.items():
                out_f.write(f">{name}\n{seq}\n")
        console.print(
            f"[green]Masking completed. Output saved to {output_file}[/green]"
        )
        shutil.rmtree(f"{tmpdir}", ignore_errors=True)
        return

    elif bowtie:
        index_command = [
            "bowtie-build",
            "--threads",
            str(threads),
            reference,
            f"{tmpdir}/contigs_index",
        ]
        sp.run(index_command, check=True)
        align_command = [
            "bowtie",
            "--threads",
            str(threads),
            "-f",
            "-a",
            "-v",
            "3",
            f"{tmpdir}/contigs_index",
            input_file,
            "-S",
            f"{tmpdir}/tmp_mapped.sam",
        ]
        sp.run(align_command, check=True)

    elif mmseqs:
        console.print(
            "Note! using mmseqs instead of bbmap is not a tight drop in replacement."
        )
        mmseqs_search_cmd = [
            "mmseqs",
            "easy-search",
            str(reference),
            str(input_file),
            f"{tmpdir}/tmp_mapped.sam",
            f"{tmpdir}",
            "--min-seq-id",
            "0.7",
            "--min-aln-len",
            "80",
            "--threads",
            str(threads),
            "-a",
            "--search-type",
            "3",
            "-v",
            "1",
            "--format-mode",
            "1",
        ]
        sp.run(mmseqs_search_cmd, check=True)

    else:
        console.print("Using bbmap.sh (default)")
        bbmap(
            ref=input_file,
            in_file=reference,
            outm=f"{tmpdir}/tmp_mapped.sam",
            minid=0.7,
            overwrite="true",
            threads=threads,
            Xmx=memory,
        )

    # Mask using the sam files
    bbmask(
        in_file=input_file,
        out=output_file,
        sam=f"{tmpdir}/tmp_mapped.sam",
        entropy=0.2,
        overwrite="true",
        threads=threads,
        Xmx=memory,
    )

    # os.remove(f"{tmpdir}/tmp_mapped.sam")
    shutil.rmtree(str(tmpdir), ignore_errors=True)

    if flatten:
        kcompress(
            in_file=output_file,
            out=f"{output_file}_flat.fa",
            fuse=2000,
            k=31,
            prealloc="true",
            overwrite="true",
            threads=threads,
            Xmx=memory,
        )
        os.rename(f"{output_file}_flat.fa", output_file)
    shutil.rmtree("ref", ignore_errors=True)
    # os.remove("tmp_target.fas")
    console.print(f"[green]Masking completed. Output saved to {output_file}[/green]")


def revcomp(seq: str) -> str:
    import mappy as mp

    return mp.revcomp(seq)


def get_hmmali_length(domain) -> int:
    return domain.alignment.hmm_to - domain.alignment.hmm_from + 1


def get_hmm_coverage(domain) -> float:
    return get_hmmali_length(domain) / domain.alignment.hmm_length


def search_hmmdb(
    amino_file,
    db_path,
    output,
    threads,
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
    import pyhmmer

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
            # print(hmms.read().name)
            # print(hmms.)
            for hits in pyhmmer.hmmsearch(
                hmms, seqs, cpus=threads, T=mscore, E=inc_e, **pyhmmer_hmmsearch_args
            ):
                if output_format != "modomtblout":
                    # writes hits
                    hits.write(outfile, format=format_dict[output_format], header=False)
                    continue
                else:
                    if len(hits) >= 1:
                        # print(hits.query_name.decode())
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
                                # TODO: add these two directly into pyhmmer/domain class.

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
                                    # f"{domain.c_evalue}",
                                    # f"{domain.i_evalue}",
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
    import pyhmmer

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
    gath_col="GATHERING_THRESHOLD",
):  # alphabet="guess",  msa_format="fasta"
    """Create a concatenated HMM database from a directory of MSA files.

    Args:
        msa_dir: str or Path, directory containing MSA files
        output: str or Path, path to save the concatenated HMM database
        set_ga: float or None, gathering threshold to set for all HMMs
        msa_format: str, format of the MSA files (e.g. "fasta", "stockholm")
        msa_pattern: str, glob pattern to match MSA files
        info_table: str or Path, path to a table file containing information about the MSA files - name, accession, description. merge attempted based on the stem of the MSA file names to match the `name` column of the info table.
        name_col: str, column name in the info table to use for the HMM name
        accs_col: str, column name in the info table to use for the HMM accession
        desc_col: str, column name in the info table to use for the HMM description

    """
    import tempfile
    from subprocess import run as runc

    import pyhmmer
    from rich.progress import track

    msa_dir = Path(msa_dir)
    output = Path(output)

    if info_table != None:
        info_table = Path(info_table)
        info_table = pl.read_csv(info_table, has_header=True)
        if name_col not in info_table.columns:
            raise ValueError(f"info_table must contain a '{name_col}' column")
        some_bool = True
        cols_map = {accs_col: "accession", desc_col: "description"}
    else:
        some_bool = False

    # create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        hmms = []
        # Process each MSA file and collect HMMs
        for msa_file in track(
            msa_dir.glob(msa_pattern),
            description="Processing MSA files",
            total=len(list(msa_dir.glob(msa_pattern))),
        ):
            with pyhmmer.easel.MSAFile(msa_file, digital=True) as msa_file_obj:
                msa = msa_file_obj.read()
            msa.name = msa_file.stem.encode("utf-8")
            # get info from the info table
            if some_bool:
                info = info_table.filter(
                    pl.col(name_col).str.contains(msa.name.decode())
                )
                if info.height == 1:
                    for col_key, col_val in cols_map.items():
                        if col_val != None:
                            msa.__setattr__(
                                col_val,
                                info[col_key].item().encode("utf-8")
                                if info[col_key].item() != None
                                else "None".encode("utf-8"),
                            )
                    if gath_col in info.columns:
                        this_gath = (
                            info[gath_col].item().encode("utf-8")
                            if info[gath_col].item() != None
                            else "1".encode("utf-8")
                        )
            else:
                msa.description = "None".encode("utf-8")
            # Build the HMM
            builder = pyhmmer.plan7.Builder(msa.alphabet)
            background = pyhmmer.plan7.Background(msa.alphabet)
            hmm, _, _ = builder.build_msa(msa, background)

            # Set gathering threshold if provided
            if gath_col in info.columns:
                hmm.cutoffs.gathering = (float(this_gath), float(this_gath))
            # write the hmm to a file
            # hmms.append(hmm)
            fh = open(temp_dir / f"{msa.name.decode()}.hmm", "wb")
            hmm.write(fh, binary=False)
            # runc(f"head {fh.name}", shell=True)
            # break
            fh.close()
        runc(f"cat {temp_dir}/*.hmm > {output}", shell=True)
    # Press all HMMs into a database
    # pyhmmer.hmmer.hmmpress(hmms, output) # this is bugged =\ using cat as a workaround for now.


def populate_pldf_withseqs_needletail(
    pldf,
    seqfile,
    chunk_size=20000000,
    trim_to_region=False,
    reverse_by_strand_col=False,
    idcol="contig_id",
    seqcol="contig_seq",
    start_col="start",
    end_col="end",
    strand_col="strand",
):
    import subprocess

    merge_cols = [idcol]
    if reverse_by_strand_col:
        merge_cols.append(strand_col)
    if trim_to_region:
        merge_cols.extend([start_col, end_col])

    print(f"Initial pldf shape: {pldf.shape}")
    minipldf = pldf.select(merge_cols).unique()
    print(f"Unique entries in minipldf: {minipldf.shape}")

    minipldf = minipldf.filter(~pl.col(idcol).is_in([None, "", "nan"]))
    print(f"After filtering nulls: {minipldf.shape}")

    minipldf = minipldf.with_columns(pl.lit(None).alias(seqcol))

    seqs = []
    seq_ids = []

    # Get actual sequence count from file
    seq_count = int(
        subprocess.run(
            f"grep -F '>'  {seqfile} -c ", shell=True, capture_output=True, text=True
        ).stdout.strip()
    )
    # seq_count = 0
    # for _ in parse_fastx_file(seqfile):
    #     seq_count += 1
    print(f"Actual number of sequences in file: {seq_count}")

    # Reset file iterator
    index = 0
    for record in parse_fastx_file(seqfile):
        seqs.append(record.seq) # type: ignore
        seq_ids.append(record.id) # type: ignore
        index += 1

        # Process chunk when we hit chunk_size or end of file
        if len(seqs) >= chunk_size or index == seq_count:
            print(f"\nProcessing chunk {index}/{seq_count}")
            print(f"Number of sequences in chunk: {len(seqs)}")

            chunk_seqs = pl.DataFrame({idcol: seq_ids, seqcol: seqs})

            chunk_seqs = chunk_seqs.join(
                minipldf.select(merge_cols), on=idcol, how="inner"
            )  # this join get's the info columns (start, end, strand) if needed, only for the entires in this chunk that are in the minipldf.

            if trim_to_region:
                print("Trimming sequences")
                # print(chunk_seqs.columns)
                chunk_seqs = chunk_seqs.with_columns(
                    pl.struct(pl.col(seqcol), pl.col(start_col), pl.col(end_col))
                    .map_elements(
                        lambda x: str(x[seqcol][x[start_col] : x[end_col]])
                        if x[seqcol] is not None
                        else None,
                        return_dtype=pl.Utf8,
                    )
                    .alias(seqcol)
                )

            if reverse_by_strand_col:
                print("Reversing sequences")
                # print(chunk_seqs.columns)
                chunk_seqs = chunk_seqs.with_columns(
                    pl.when(pl.col(strand_col))
                    .then(
                        pl.col(seqcol).map_elements(
                            lambda x: revcomp(x) if x is not None else None,
                            return_dtype=pl.Utf8,
                        )
                    )
                    .otherwise(pl.col(seqcol))
                    .alias(seqcol)
                )

            print("Joining with nascent df")
            minipldf = minipldf.join(chunk_seqs, on=merge_cols, how="left")
            minipldf = minipldf.with_columns(
                pl.coalesce([pl.col(seqcol), pl.col(f"{seqcol}_right")]).alias(seqcol)
            ).drop(f"{seqcol}_right")

            print(f"Null count in seqcol after chunk: {minipldf[seqcol].null_count()}")

            seqs = []
            seq_ids = []
            # get count for remaining nulls, if zero, break - should be useful when fetching just a few sequences from a large file, at least if the needed seqs are closer to the start of the input fasta.
            if minipldf[seqcol].null_count() == 0:
                break

    print("\nFinal merge with original df")
    pldf = pldf.join(minipldf, on=merge_cols, how="left")
    print(f"Final null count in seqcol: {pldf[seqcol].null_count()}")

    return pldf


def normalize_column_names(df):
    """Normalize common column name variations to standard names.
    
    Maps various column names to standard annotation schema:
    - begin/from/seq_from -> start
    - to/seq_to -> end  
    - qseqid/sequence_ID/contig_id -> sequence_id
    - etc.
    """
    import polars as pl
    
    # Define column name mappings
    column_mappings = {
        # Start position variations
        'begin': 'start',
        'from': 'start', 
        'seq_from': 'start',
        'query_start': 'start',
        'qstart': 'start',
        
        # End position variations
        'to': 'end',
        'seq_to': 'end',
        'query_end': 'end',
        'qend': 'end',
        
        # Sequence ID variations
        'qseqid': 'sequence_id',
        'sequence_ID': 'sequence_id',
        'contig_id': 'sequence_id',
        'contig': 'sequence_id',
        'query': 'sequence_id',
        'id': 'sequence_id',
        'name': 'sequence_id',
        
        # Score variations
        'bitscore': 'score',
        'bit_score': 'score',
        'bits': 'score',
        'evalue': 'evalue',
        'e_value': 'evalue',
        
        # Source variations
        'tool': 'source',
        'method': 'source',
        'db': 'source',
        'database': 'source',
        
        # Type variations
        'feature': 'type',
        'annotation': 'type',
        'category': 'type',
    }
    
    # Rename columns if they exist
    rename_dict = {}
    for old_name, new_name in column_mappings.items():
        if old_name in df.columns:
            rename_dict[old_name] = new_name
    
    if rename_dict:
        df = df.rename(rename_dict)
    
    return df


def create_minimal_annotation_schema(df, annotation_type, source, tool_specific_cols=None):
    """Create a minimal standardized annotation schema.
    
    Args:
        df: Input DataFrame
        annotation_type: Type of annotation (e.g., 'ribozyme', 'tRNA', 'IRES')
        source: Source tool name
        tool_specific_cols: List of tool-specific columns to preserve
        
    Returns:
        DataFrame with standardized minimal schema
    """
    import polars as pl
    
    # First normalize column names
    df = normalize_column_names(df)
    
    # Define minimal required columns with defaults
    minimal_schema = {
        'sequence_id': pl.Utf8,
        'type': pl.Utf8, 
        'start': pl.Int64,
        'end': pl.Int64,
        'score': pl.Float64,
        'source': pl.Utf8,
        'strand': pl.Utf8,
        'phase': pl.Utf8
    }
    
    # Add missing columns with appropriate defaults
    for col, dtype in minimal_schema.items():
        if col not in df.columns:
            if col == 'type':
                default_val = annotation_type
            elif col == 'source':
                default_val = source
            elif col in ['start', 'end']:
                default_val = 0
            elif col == 'score':
                default_val = 0.0
            elif col == 'strand':
                default_val = '+'
            elif col == 'phase':
                default_val = '.'
            else:
                default_val = ''
            
            df = df.with_columns(pl.lit(default_val).alias(col).cast(dtype))
    
    # Select minimal columns plus any tool-specific ones
    columns_to_keep = list(minimal_schema.keys())
    if tool_specific_cols:
        for col in tool_specific_cols:
            if col in df.columns and col not in columns_to_keep:
                columns_to_keep.append(col)
    
    # Only select columns that actually exist
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df.select(existing_columns)
    
    # Ensure all minimal schema columns exist even if not in original data
    for col, dtype in minimal_schema.items():
        if col not in df.columns:
            if col == 'type':
                default_val = annotation_type
            elif col == 'source':
                default_val = source
            elif col in ['start', 'end']:
                default_val = 0
            elif col == 'score':
                default_val = 0.0
            elif col == 'strand':
                default_val = '+'
            elif col == 'phase':
                default_val = '.'
            else:
                default_val = ''
            
            df = df.with_columns(pl.lit(default_val).alias(col).cast(dtype))
    
    return df


def ensure_unified_schema(dataframes):
    """Ensure all DataFrames have the same unified schema.
    
    Args:
        dataframes: List of (name, dataframe) tuples
        
    Returns:
        List of DataFrames with unified schema
    """
    import polars as pl
    
    if not dataframes:
        return []
    
    # Define the unified schema
    unified_schema = {
        'sequence_id': pl.Utf8,
        'type': pl.Utf8, 
        'start': pl.Int64,
        'end': pl.Int64,
        'score': pl.Float64,
        'source': pl.Utf8,
        'strand': pl.Utf8,
        'phase': pl.Utf8
    }
    
    # Add common tool-specific columns
    tool_specific_columns = {
        'profile_name': pl.Utf8,
        'evalue': pl.Float64,
        'ribozyme_description': pl.Utf8,
        'tRNA_type': pl.Utf8,
        'anticodon': pl.Utf8,
        'motif_type': pl.Utf8,
        'structure': pl.Utf8,
        'sequence': pl.Utf8
    }
    
    # Combine schemas
    full_schema = {**unified_schema, **tool_specific_columns}
    
    unified_dataframes = []
    
    for name, df in dataframes:
        # Add missing columns with appropriate defaults
        for col, dtype in full_schema.items():
            if col not in df.columns:
                if col == 'type':
                    default_val = name
                elif col == 'source':
                    default_val = 'unknown'
                elif col in ['start', 'end']:
                    default_val = 0
                elif col == 'score':
                    default_val = 0.0
                elif col == 'evalue':
                    default_val = 1.0
                elif col == 'strand':
                    default_val = '+'
                elif col == 'phase':
                    default_val = '.'
                else:
                    default_val = ''
                
                df = df.with_columns(pl.lit(default_val).alias(col).cast(dtype))
        
        # Ensure column order is consistent
        ordered_columns = list(full_schema.keys())
        df = df.select([col for col in ordered_columns if col in df.columns])
        
        unified_dataframes.append(df)
    
    return unified_dataframes


def identify_fastq_files(
    input_path: Union[str, Path], return_rolypoly: bool = True
) -> dict:
    """Identify and categorize FASTQ files from input path.

    Args:
        input_path: Path to input directory or file
        return_rolypoly: Whether to look for and return rolypoly-formatted files first

    Returns:
        dict: Dictionary containing:
            - rolypoly_data: {lib_name: {'interleaved': path, 'merged': path}}
            - R1_R2_pairs: [(r1_path, r2_path), ...]
            - interleaved_files: [path, ...]
            - single_end: [path, ...]

    Note:
        When return_rolypoly is True and rolypoly files are found, other files
        are ignored to maintain consistency with rolypoly pipeline.
    """

    input_path = Path(input_path)
    file_info = {
        "rolypoly_data": {},
        "R1_R2_pairs": [],
        "interleaved_files": [],
        "single_end": [],
    }

    def is_paired_filename(filename: str) -> tuple[bool, str]:
        """Check if filename indicates paired-end data and extract pair info."""
        import re

        patterns = [
            (r".*_R?1[._].*", r".*_R?2[._].*"),  # Matches _R1/_R2, _1/_2
            (r".*_1\.f.*q.*", r".*_2\.f.*q.*"),  # Matches _1.fastq/_2.fastq
            (r".*\.1\.f.*q.*", r".*\.2\.f.*q.*"),  # Matches .1.fastq/.2.fastq
        ]

        for pat in patterns:
            if re.match(pat, filename): # type: ignore
                pair_file = (
                    filename.replace("_R1", "_R2")
                    .replace("_1.", "_2.")
                    .replace(".1.", ".2.")
                )
                return True, pair_file
        return False, ""

    if input_path.is_dir():
        # First look for rolypoly output files
        if return_rolypoly:
            rolypoly_files = list(input_path.glob("*_final_*.f*q*"))
            if rolypoly_files:
                for file in rolypoly_files:
                    lib_name = file.stem.split("_final_")[0]
                    if lib_name not in file_info["rolypoly_data"]:
                        file_info["rolypoly_data"][lib_name] = {
                            "interleaved": None,
                            "merged": None,
                        }
                    if "interleaved" in file.name:
                        file_info["rolypoly_data"][lib_name]["interleaved"] = file
                    elif "merged" in file.name:
                        file_info["rolypoly_data"][lib_name]["merged"] = file
                return file_info

        # Process all fastq files
        all_fastq = list(input_path.glob("*.f*q*"))
        processed_files = set()

        # First pass - identify paired files by name
        for file in all_fastq:
            if file in processed_files:
                continue

            is_paired, pair_file = is_paired_filename(file.name)
            if is_paired and (file.parent / pair_file).exists():
                file_info["R1_R2_pairs"].append((file, file.parent / pair_file))
                processed_files.add(file)
                processed_files.add(file.parent / pair_file)
                continue

            # Check remaining files obsolete
            props = guess_fastq_properties(str(file))
            if props["paired_end"]:  # Interleaved paired-end
                file_info["interleaved_files"].append(file)
            else:  # Single-end
                file_info["single_end"].append(file)
            processed_files.add(file)

    else:
        # Single file input
        props = guess_fastq_properties(str(input_path))
        if props["paired_end"]:  # Interleaved paired-end
            file_info["interleaved_files"].append(input_path)
        else:  # Single-end
            file_info["single_end"].append(input_path)

    return file_info
