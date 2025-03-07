"""Sequence analysis and utilities (io, fasta/fastq) for RolyPoly.

Provides functions for nucleotide and protein sequences, including sequence masking, pyHMMER stuff.
"""

import os
from pathlib import Path
import rich_click as click
from rich.console import Console
from typing import Union

# from rolypoly.utils.various import run_command_comp

global datadir
datadir = Path(os.environ['ROLYPOLY_DATA'])

def read_fasta_needletail(fasta_file: str) -> tuple[list[str], list[str]]:
    """Read sequences from a FASTA/FASTQ file using needletail.
    
    A fast and memory-efficient function to parse FASTA/FASTQ files using
    the Rust-based needletail library.

    Args:
        fasta_file (str): Path to the input FASTA/FASTQ file

    Returns:
        tuple[list[str], list[str]]: A tuple containing:
            - List of sequence identifiers
            - List of sequences
    Example:
         ids, seqs = read_fasta_needletail("sequences.fasta")
         print(f"Found {len(ids)} sequences")
    """
    from needletail import parse_fastx_file
    seqs = []
    seq_ids = []
    for record in parse_fastx_file(fasta_file):
        seqs.append(record.seq)
        seq_ids.append(record.id)
    return seq_ids, seqs

def filter_fasta_by_headers(fasta_file: str, headers: Union[str, list[str]], output_file: str, invert: bool = False) -> None:
    """Filter sequences in a FASTA file based on their headers.
    
    Extracts sequences whose headers match (or don't match if inverted) any of
    the provided header patterns.

    Args:
        fasta_file (str): Path to input FASTA file
        headers (Union[str, list[str]]): Either a file containing headers (one per line)
            or a list of header patterns to match
        output_file (str): Path to write filtered sequences
        invert (bool, optional): If True, keep sequences that don't match. 
    
    Example:
         # Keep only sequences with specified headers
         filter_fasta_by_headers("all.fasta", ["seq1", "seq2"], "filtered.fasta")
         # Exclude sequences with specified headers
         filter_fasta_by_headers("all.fasta", "headers.txt", "filtered.fasta", invert=True)
    """
    import subprocess as sp
    from needletail import parse_fastx_file
    headers_list = []
    if not isinstance(headers, list):
        with open(headers, 'r') as f:
            for line in f:
                headers_list.append(line.strip())
    else:
        headers_list = headers
        
    with open(output_file, 'w') as out_f:
        for record in parse_fastx_file(fasta_file):
            matches = any(header in str(record.id) for header in headers_list)
            if matches ^ invert:  # XOR operation: write if (matches and not invert) or (not matches and invert)
                out_f.write(f">{record.id}\n{record.seq}\n")

def read_fasta_polars(fasta_file: str, idcol: str = "contig_id", seqcol: str = "contig_seq")  :
    """Read a FASTA file into a Polars DataFrame, Uses needletail for fast FASTA parsing and returns a Polars DataFrame with customizable column names for sequence IDs and sequences.

    Args:
        fasta_file (str): Path to input FASTA file
        idcol (str, optional): Name for the sequence ID column.
        seqcol (str, optional): Name for the sequence column. 

    Returns:
        polars.DataFrame: DataFrame with two columns for IDs and sequences

    Example:
         df = read_fasta_polars("proteins.fasta", "prot_id", "aa_seq")
    """
    import polars as pl
    seq_ids, seqs = read_fasta_needletail(fasta_file)
    return pl.DataFrame({idcol: seq_ids, seqcol: seqs})

def translate_6frx_seqkit(input_file: str, output_file: str, threads: int) -> None:
    """Translate nucleotide sequences in all 6 reading frames using seqkit.

    Args:
        input_file (str): Path to input nucleotide FASTA file
        output_file (str): Path to output amino acid FASTA file
        threads (int): Number of CPU threads to use

    Note:
        Requires seqkit to be installed and available in PATH.
        The output sequences are formatted with 20000bp line width.

    Example:
         translate_6frx_seqkit("genes.fna", "proteins.faa", 4)
    """
    import subprocess as sp
    command = f"seqkit translate -x -F --clean -w 20000 -f 6 {input_file} --id-regexp '(\\*)' --clean  --threads {threads} > {output_file}"
    sp.run(command, shell=True, check=True)

def translate_with_bbmap(input_file: str, output_file: str, threads: int) -> None:
    """Translate nucleotide sequences using BBMap's callgenes.sh.
    
    Predicts and translates genes from nucleotide sequences using BBMap's
    gene prediction and translation tools.

    Args:
        input_file (str): Path to input nucleotide FASTA file
        output_file (str): Path to output amino acid FASTA file
        threads (int): Number of CPU threads to use

    Note:
        - Requires BBMap to be installed and available in PATH
        - Generates both protein sequences (.faa) and gene annotations (.gff)
        - The GFF output file is named by replacing .faa with .gff

    """
    import subprocess as sp
    gff_o = output_file.replace(".faa", ".gff")
    command = f"callgenes.sh threads={threads} in={input_file} outa={output_file} out={gff_o}"
    sp.run(command, shell=True, check=True)

def pyro_predict_orfs(input_file: str, output_file: str, threads: int, gv_or_else: str = "gv", genetic_code: int = 11) -> None:
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
    import pyrodigal_gv as pyro_gv
    from pyrodigal_gv import pyrodigal as pyro
    from Bio import SeqIO
    import multiprocessing.pool

    sequences = []
    ids = []
    with open(input_file, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            sequences.append(bytes(record.seq))
            ids.append((record.id))

    if gv_or_else == "gv":
        gene_finder = pyro_gv.ViralGeneFinder(meta=True)  # a single gv gene finder object
    else:   
        gene_finder = pyro.GeneFinder(meta=True)  # a single gene finder object

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
    cigar_tuples = re.findall(r'(\d+)([MIDNSHPX=])', cigar_string)
    matches = sum(int(length) for length, op in cigar_tuples if op in {'M', '=', 'X'})
    total_length = sum(int(length) for length, op in cigar_tuples if op in {'M', 'I', 'D', '=', 'X'})
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
    mp.revcomp(seq)
    is_reverse = (is_reverse == -1)
    if is_reverse:
        seq = str(mp.revcomp(seq))
    masked_seq = seq[:start] + 'N' * (end - start) + seq[end:]
    return str(mp.revcomp(masked_seq)) if is_reverse else masked_seq

def is_gzipped(file_path: str) -> bool:
    """Check if a file is gzipped.

    Args:
        file_path (str): Path to the file to check

    Returns:
        bool: True if file is gzipped, False otherwise

    Example:
         is_gzipped("sequences.fa.gz")
         True
         is_gzipped("sequences.fa")
         False
    """
    with open(file_path, 'rb') as test_f:
        return test_f.read(2).startswith(b'\x1f\x8b')

def guess_fastq_properties(file_path: str) -> dict:
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

    Example:
         props = guess_fastq_properties("reads.fastq")
         print(props['paired_end'])
        True
    """
    is_gz = is_gzipped(file_path)
    paired_end = False
    average_read_length = 0
    total_length = 0
    read_count = 0

    # Open the file accordingly
    if is_gz:
        import gzip
        with gzip.open(file_path, 'rb') as f:
            data = f.read(20 * 1024 * 1024)  # Read the first 20MB
    else:
        with open(file_path, 'rb') as f:
            data = f.read(20 * 1024 * 1024)  # Read the first 20MB

    # Decode the data
    data = data.decode('utf-8', errors='ignore')

    # Split the data into lines
    lines = data.splitlines()

    # Process the lines to determine properties
    for i in range(0, len(lines) - len(lines) % 4, 4):
        if lines[i].startswith('@'):
            read_id = lines[i][1:].split()[0]
            if '/1' in read_id or '/2' in read_id:
                paired_end = True
            read_length = len(lines[i + 1])
            total_length += read_length
            read_count += 1

    if read_count > 0:
        average_read_length = total_length / read_count

    return {
        'is_gzipped': is_gzipped,
        'paired_end': paired_end,
        'average_read_length': average_read_length
    }

def guess_fasta_alpha(input_file: str) -> str:
    """Guess the alphabet of a FASTA file.
    
    Args:
        input_file (str): Path to the FASTA file

    Returns:
        str: Alphabet of the FASTA file

    Example:
         guess_fasta_alpha("sequences.fasta")
        "nucl"
    """
    # only peek at the first sequence
    with open(input_file, "rb") as fin:
        input_string = get_sequence_between_newlines(fin.peek(2)[:1110].decode().replace(r"*/\n",""))
    if(is_nucl_string(input_string)):
        return("nucl")
    elif(is_aa_string(input_string)):
         return("amino")
    else:
        return("nothing_good")


def is_nucl_string(sequence):
    valid_characters = set({'A','T','G','C','U','N'})
    return all(char in valid_characters for char in sequence.upper())

def is_aa_string(sequence):
    valid_characters = set({"A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V","O","U","B","Z","X","J","*","-","."})
    return all(char in valid_characters for char in sequence.upper())

def get_sequence_between_newlines(input_string):
    import re
    newline_pattern = re.compile(r'\n')
    newline_positions = [match.start() for match in newline_pattern.finditer(input_string)]
    if len(newline_positions) < 2:
        return (input_string[newline_positions[0]+1:])
    return (input_string[newline_positions[0] + 1:newline_positions[1]])


def ensure_faidx(input_file: str) -> None:
    """Ensure a FASTA file has a pyfastx index.
    
    Creates a pyfastx index for the input FASTA file if it doesn't exist.

    Args:
        input_file (str): Path to the FASTA file

    Example:
         ensure_faidx("sequences.fasta")
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
    sp.run(['datasets', 'download', 'genome', 'taxon', taxid,
            '--include', 'rna,genome', '--filename', f'{taxid}_fetched_genomes.zip',
            '--assembly-version', 'latest', '--exclude-atypical',
            '--assembly-source', 'RefSeq', '--no-progressbar'],stdout=sp.DEVNULL, stderr=sp.DEVNULL)

def process_with_timeout(func: callable, arg: any, timeout: int) -> any:
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

def fetch_genomes(input_file: str, output_file: str, threads: int = 1, max2take: int = 75, timeout: int = 600) -> None:
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
    import shutil
    import subprocess as sp
    from concurrent.futures import ProcessPoolExecutor
    import concurrent.futures
    from rolypoly.utils.various import extract_zip
    
    with open(input_file, 'r') as f:
        lines = f.readlines()[4:]
    
    taxons = set()
    for line in lines:
        taxon = line.split(sep=";")[-1]
        taxon = taxon.split(sep="\t")[0]
        if any(word in taxon.lower() for word in ['meta', 'uncultured', 'unidentified', 'synthetic', 'construct', "coli"]):
            print(f"{line} contains a keyword that indicates it is not an actual organism, skipping")
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

    # Use taxonkit to get taxids
    with open('tmp_gbs_50m_taxids.lst', 'w') as f:
        sp.run(['taxonkit', 'name2taxid'], input='\n'.join(taxons).encode(), stdout=f)
    
    # Use datasets to download genomes
    with open('tmp_gbs_50m_taxids.lst', 'r') as f:
        taxids = [line.split(sep="\t")[1].replace(" ", "_").strip() for line in f if line != ""]
    taxids = list(set(taxids).difference(['', "562"]))  # Remove empty and E. coli

    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(process_with_timeout, download_genome, taxid, timeout) for taxid in taxids]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

    zip_files = list(Path('.').glob('*.zip'))
    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(process_with_timeout, extract_zip, zip_file, timeout) for zip_file in zip_files]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred during extraction: {e}")

    # Concatenate and deduplicate sequences
    ref_seqs = set()
    for folder in Path('ncbi_dataset/').rglob('*/'):
        fna_files = list(folder.rglob('*.fna'))
        rna_file = next((f for f in fna_files if 'rna' in f.name.lower()), None)
        if rna_file:
            chosen_file = rna_file
        else:
            chosen_file = fna_files[0]  # choose the first file if no RNA file found
        ref_seqs.add(str(chosen_file))

    with open("tmp.lst", 'w') as outf:
        for f in ref_seqs:
            outf.write(f + "\n")
    sp.call(['seqkit', 'rmdup', '-s', '--infile-list', "tmp.lst"], stdout=open("tmp.fasta", "w"))
    
    # Remove any fasta entries that have "virus", "viral", "phage" in the header
    filter_fasta_by_headers("tmp.fasta", ["virus", "viral", "phage"], output_file, invert=True)

    # Clean up
    for item in Path('.').glob('ncbi_dataset*'):
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
    with open(input_table, 'r') as f:
        for line in f:
            seq_id, start, stop, strand = line.strip().split('\t')
            if seq_id not in ranges:
                ranges[seq_id] = []
            ranges[seq_id].append((int(start), int(stop), strand))

    # Process FASTA file
    with open(input_fasta, 'r') as in_f, open(output_fasta, 'w') as out_f:
        current_id = ''
        current_seq = ''
        for line in in_f:
            if line.startswith('>'):
                if current_id:
                    if current_id in ranges:
                        for start, stop, strand in ranges[current_id]:
                            if start > stop:
                                start, stop = stop, start
                            if strand == '-':
                                current_seq = revcomp(current_seq)
                            current_seq = current_seq[:start-1] + 'N' * (stop - start + 1) + current_seq[stop:]
                            if strand == '-':
                                current_seq = revcomp(current_seq)
                    out_f.write(f'>{current_id}\n{current_seq}\n')
                current_id = line[1:].strip()
                current_seq = ''
            else:
                current_seq += line.strip()
        
        if current_id:
            if current_id in ranges:
                for start, stop, strand in ranges[current_id]:
                    if start > stop:
                        start, stop = stop, start
                    if strand == '-':
                        current_seq = revcomp(current_seq)
                    current_seq = current_seq[:start-1] + 'N' * (stop - start + 1) + current_seq[stop:]
                    if strand == '-':
                        current_seq = revcomp(current_seq)
            out_f.write(f'>{current_id}\n{current_seq}\n')

def read_fasta2polars_df(file_path: str):   
    """Reads a FASTA file into a Polars DataFrame.

    Args:
        file_path (str): Path to the input FASTA file
    
    Returns:
        polars.DataFrame: DataFrame with columns:
            - header: Sequence headers
            - sequence: Sequence strings
            - group: Internal grouping number

    Example:
         df = read_fasta2polars_df("sequences.fasta")
         print(df.head())
    """
    import polars as pl
    df = pl.read_csv(file_path, has_header=False, separator="\n", new_columns=["seq"])
    #  header column
    df = df.with_columns([
        pl.when(pl.col("seq").str.starts_with(">"))
        .then(pl.col("seq").str.replace(">",""))
        .alias("header")
    ])
    # Create a running length column
    df = df.with_columns([
        pl.col("header").is_not_null().cum_sum().alias("group")
    ])
    # Create a new dataframe concatenating sequences
    result = df.filter(pl.col("header") != "").select("header", "group").join(
        df.group_by("group").agg([
        pl.when(pl.col("seq").str.contains(">") == False)
        .then(pl.col("seq"))
                    .str.concat(delimiter="").alias("sequence")
        ]),
        on="group",
    ).drop("group")

    return result

console = Console(width=150)
@click.command()
@click.option('-t', '--threads', default=1, help='Number of threads to use')
@click.option('-M', '--memory', default="6gb", help='Memory in GB')
@click.option('-o', '--output', required=True, help='Output file name')
@click.option('-f', '--flatten', is_flag=True, help='Attempt to kcompress.sh the masked file')
@click.option('-i', '--input', required=True, help='Input fasta file')
@click.option('-F', '--mmseqs', is_flag=True, help='use mmseqs2 instead of bbmap.sh')
# @click.option('-lm', '--low-mem', is_flag=True, help='use strobealign instead of bbmap.sh')
@click.option('-lm', '--low-mem', is_flag=True, help='use minimap2 instead of bbmap.sh')
@click.option('-bt', '--bowtie', is_flag=True, help='use bowtie1 instead of bbmap.sh')
@click.option('-r', '--reference',default = datadir/"masking/RVMT_NCBI_Ribo_Japan_for_masking.fasta", help='Provide an input fasta file to be used for masking, instead of the pre-generated collection of RNA viral sequences')
def mask_dna(threads, memory, output, flatten, input, mmseqs, low_mem, bowtie, reference):
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
    from bbmapy import bbmask, kcompress,bbmap
    import subprocess as sp
    from rolypoly.utils.various import ensure_memory
    import mappy as mp
    import shutil
    
    input_file = Path(input).resolve()
    output_file = Path(output).resolve()
    # datadir = Path(os.environ['datadir'])
    memory = ensure_memory(memory)["giga"]
    reference = Path(reference).absolute().resolve()
    tmpdir = (str(output_file.parent) + "/tmp")

    try:
        Path.mkdir(Path(tmpdir),exist_ok=True)
    except:
        console.print(f"couldn't create {tmpdir}") 
        exit(123)

    if low_mem:
        console.print("Using minimap2 (low memory mode)")
        
        # Create a mappy aligner object
        aligner = mp.Aligner(str(reference), k=11, n_threads=threads,best_n=150)
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
        console.print(f"[green]Masking completed. Output saved to {output_file}[/green]")
        shutil.rmtree(f"{tmpdir}", ignore_errors=True)
        return 

    elif bowtie:
        index_command = ["bowtie-build", reference, f"{tmpdir}/contigs_index"]
        sp.run(index_command, check=True)
        align_command = ["bowtie", "--threads", str(threads), "-f", "-a", "-v", "3", f"{tmpdir}/contigs_index", input_file, "-S", f"{tmpdir}/tmp_mapped.sam"]
        sp.run(align_command, check=True)

    elif mmseqs:
        console.print("Note! using mmseqs instead of bbmap is not a tight drop in replacement.")
        mmseqs_search_cmd = ["mmseqs", "easy-search", str(reference), str(input_file),
                             f"{tmpdir}/tmp_mapped.sam", f'{tmpdir}',
                            "--min-seq-id", "0.7", "--min-aln-len", "80",
                            "--threads", str(threads),
                            "-a", "--search-type", "3", "-v","1","--format-mode", "1"]
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
        Xmx=memory)

    # Mask using the sam files    
    bbmask(
        in_file=input_file,
        out=output_file,
        sam=f"{tmpdir}/tmp_mapped.sam",
        entropy=0.2,
        overwrite="true",
        threads=threads,
        Xmx=memory
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
            Xmx=memory
        )
        os.rename(f"{output_file}_flat.fa", output_file)
    shutil.rmtree("ref", ignore_errors=True)
    # os.remove("tmp_target.fas")
    console.print(f"[green]Masking completed. Output saved to {output_file}[/green]")

def bowtie_build(reference: str, output_base: str) -> None:
    """Build a Bowtie index from a reference sequence.

    Args:
        reference (str): Path to the reference FASTA file
        output_base (str): Base name for the output index files

    Example:
         bowtie_build("reference.fasta", "ref_index")
    """
    import subprocess as sp
    command = ["bowtie-build", reference, output_base]
    sp.run(command, check=True)

def revcomp(seq: str) -> str:
    """Get the reverse complement of a DNA sequence.
    
    Uses mappy.revcomp() for efficient computation.

    Args:
        seq (str): Input DNA sequence

    Returns:
        str: Reverse complement of the input sequence

    Example:
         revcomp("ATGC")
         'GCAT'
    """
    import mappy as mp
    return mp.revcomp(seq)
    
def get_hmmali_length(domain) -> int:
    """Get the length of a HMM domain alignment.

    Args:
        domain (pyhmmer.plan7.Domain): A pyhmmer domain object

    Returns:
        int: Length of the HMM alignment

    """
    return (domain.alignment.hmm_to - domain.alignment.hmm_from + 1)

def get_hmm_coverage(domain) -> float:
    """Calculate the coverage of a HMM domain alignment.

    Args:
        domain (pyhmmer.plan7.Domain): A pyhmmer domain object

    Returns:
        float: Coverage of the HMM alignment (0-1)

    Example:
    ```python
         coverage = get_hmm_coverage(domain)
    ```
    """
    return (get_hmmali_length(domain) / domain.alignment.hmm_length)

def search_hmmdb(amino_file, db_path, output, threads, logger = None,inc_e=0.05, mscore=20, match_region=False, full_qseq=False, ali_str=False, output_format="modomtblout", pyhmmer_hmmsearch_args={}):
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
        logger.debug(f"Starting pyhmmer search against {db_path} with {threads} threads")
    
    format_dict = {"tblout": "targets", "domtblout": "domains", "modomtblout": "modomtblout"}
    
    with pyhmmer.easel.SequenceFile(amino_file, digital=True,format="fasta") as seq_file:
        seqs = seq_file.read_block()
    seqs_dict = {}
    for seq in seqs:
        seqs_dict[seq.name.decode()+ f" {seq.description.decode()}"] = seq.textize().sequence # type: ignore
        
    if logger:
        logger.debug(f"loaded {len(seqs)} sequences from {amino_file}")
    # see https://pyhmmer.readthedocs.io/en/stable/api/plan7/results.html#pyhmmer.plan7.TopHits for format (though I changed it a bit)
    mod_title_domtblout = ["query_full_name","hmm_full_name", 
                           "hmm_len", "qlen", "full_hmm_evalue",
                           "full_hmm_score", "full_hmm_bias", "this_dom_score", "this_dom_bias", "hmm_from", "hmm_to", "q1", "q2", "env_from", "env_to",
                           "hmm_cov", "ali_len", "dom_desc"]
    mod_title_domtblout.extend(name for name, value in 
        {'aligned_region': match_region, 'full_qseq': full_qseq, 'identity_str': ali_str}.items() if value)
    og_domtblout_title = ["#                                                                                                                --- full sequence --- -------------- this domain -------------   hmm coord   ali coord   env coord",
                          "# target name        accession   tlen query name                                               accession   qlen   E-value  score  bias   #  of  c-Evalue  i-Evalue  score  bias  from    to  from    to  from    to  acc description of target",
                          "#------------------- ---------- -----                                     -------------------- ---------- ----- --------- ------ ----- --- --- --------- --------- ------ ----- ----- ----- ----- ----- ----- ---- ---------------------"]
    og_tblout = ["#                                                                                                   --- full sequence ---- --- best 1 domain ---- --- domain number estimation ----",
                 "# target name        accession  query name                                               accession    E-value  score  bias   E-value  score  bias   exp reg clu  ov env dom rep inc description of target",
                 "#------------------- ----------                                     -------------------- ---------- --------- ------ ----- --------- ------ -----   --- --- --- --- --- --- --- --- ---------------------"]
    
    
    with open(output, 'wb') as outfile:
        if output_format == "modomtblout":
            outfile.write("\t".join(mod_title_domtblout).encode("utf-8") + b"\n")
        else:
            outfile.write("\n".join((og_tblout if output_format == "tblout" else og_domtblout_title).encode("utf-8") + b"\n"))
        with pyhmmer.plan7.HMMFile(db_path) as hmms:
            # print(hmms.read().name)
            # print(hmms.)
            for hits in pyhmmer.hmmsearch(hmms, seqs, cpus=threads, T=mscore, E=inc_e, **pyhmmer_hmmsearch_args): 
                if  output_format != "modomtblout":
                    # writes hits
                    hits.write(outfile, format=format_dict[output_format], header=False)
                    continue
                else:
                    if len(hits) >= 1:
                        
                        # print(hits.query_name.decode())
                        for hit in hits:
                            hit_desc = hit.description or bytes("", "utf-8")
                            hit_name  = hit.name.decode()
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
                                    f"{full_prot_name}", # query_full_name
                                    f"{hits.query.name.decode()}", # hmm_full_name
                                    f"{domain.alignment.hmm_length}", # hmm_len   
                                    f"{hit.length}", # qlen 
                                    f"{hit.evalue}", # full_hmm_evalue
                                    f"{hit.score}", # full_hmm_score
                                    f"{hit.bias}", # full_hmm_bias
                                    f"{domain.score}", # this_dom_score
                                    f"{domain.bias}", # this_dom_bias
                                    # f"{domain.c_evalue}",
                                    # f"{domain.i_evalue}",
                                    f"{domain.alignment.hmm_from}", # hmm_from
                                    f"{domain.alignment.hmm_to}", # hmm_to
                                    f"{domain.alignment.target_from}", # q1
                                    f"{domain.alignment.target_to}", # q2
                                    f"{domain.env_from}", # env_from
                                    f"{domain.env_to}", # env_to
                                    f"{hmm_coverage}", # hmm_cov
                                    f"{alignment_length}", # ali_len
                                    f"{dom_desc.decode()}" # I think this is description of the target hit.
                                ]
                                if match_region:
                                    outputline.append(f"{domain.alignment.target_sequence}")
                                if full_qseq:
                                    outputline.append(f"{protein_seq}")
                                if ali_str:
                                    outputline.append(f"{domain.alignment.identity_sequence}")
                                outfile.write(("\t".join(outputline) + "\n").encode())
    return output
                        

def hmm_from_msa(msa_file, output, alphabet="amino", set_ga=None, name=None, accession=None):
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
        msa.name = msa.names[0].encode("utf-8")
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

def hmmdb_from_directory(msa_dir, output, msa_pattern="*.faa",info_table=None,name_col="MARKER",accs_col="ANNOTATION_ACCESSIONS",desc_col="ANNOTATION_DESCRIPTION", gath_col="GATHERING_THRESHOLD"): # alphabet="guess",  msa_format="fasta"
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
        # alphabet: str, sequence alphabet type ("amino" or "dna", or "guess")
    import pyhmmer
    from pathlib import Path
    import tempfile
    import polars as pl
    from rich.progress import track
    from subprocess import run as runc
    msa_dir = Path(msa_dir)
    output = Path(output)
    
    if info_table != None:
        info_table = Path(info_table)
        info_table = pl.read_csv(info_table,has_header=True)
        if name_col not in info_table.columns:
            raise ValueError(f"info_table must contain a '{name_col}' column")
        some_bool = True
        cols_map = {accs_col: "accession", desc_col: "description"}
    else:
        some_bool = False

    # create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        hmms = []
        # Process each MSA file and collect HMMs
        for msa_file in track(msa_dir.glob(msa_pattern), description="Processing MSA files", total=len(list(msa_dir.glob(msa_pattern)))):
            with pyhmmer.easel.MSAFile(msa_file, digital=True) as msa_file_obj:
                msa = msa_file_obj.read()
            msa.name = msa_file.stem.encode("utf-8")
            #get info from the info table   
            if some_bool:
                info = info_table.filter(pl.col(name_col).str.contains(msa.name.decode()))
                if info.height == 1:
                    for col_key,col_val in cols_map.items():
                        if col_val != None:
                            msa.__setattr__(col_val, info[col_key].item().encode("utf-8") if info[col_key].item() != None else "None".encode("utf-8"))
                    if gath_col in info.columns:
                        this_gath = info[gath_col].item().encode("utf-8") if info[gath_col].item() != None else "1".encode("utf-8")
            else:
                msa.description = "None".encode("utf-8")
            # Build the HMM
            builder = pyhmmer.plan7.Builder(msa.alphabet)
            background = pyhmmer.plan7.Background(msa.alphabet)
            hmm, _, _ = builder.build_msa(msa, background)
            
            # Set gathering threshold if provided
            if gath_col in info.columns:
                hmm.cutoffs.gathering = this_gath, this_gath
            # write the hmm to a file
            # hmms.append(hmm)
            fh = open(tmp_dir / f"{msa.name.decode()}.hmm", "wb")
            hmm.write(fh, binary=False)
            # runc(f"head {fh.name}", shell=True)
            # break
            fh.close()
        runc(f"cat {tmp_dir}/*.hmm > {output}", shell=True)
    # Press all HMMs into a database
    # pyhmmer.hmmer.hmmpress(hmms, output) # this is bugged =\ using cat as a workaround for now.

        
        
#ValueError: Index contains duplicate keys.
# seen = set()
# new_hmms = []
# for hmm in hmms:
#     if hmm.name in seen:
#         print(f"Duplicate HMM name: {hmm.name}")
#     else:
#         seen.add(hmm.name)
#         print(hmm.accession)
#         hmm.accession = ''.encode("utf-8")
#         hmm.description = ''.encode("utf-8")
#         new_hmms.append(hmm)


 #### OBSOLETE: ####

# def hasty_wrap_up(some_path, keep_tmp=False):
#     """Something went wrong and you got here"""
#     if not keep_tmp:
#         shutil.rmtree(str(some_path)+"/tmp", ignore_errors=True)
#         for tmp_file in Path(str(some_path)).glob("tmp*"):
#             if tmp_file.is_dir():
#                 shutil.rmtree(tmp_file)
#             else:
#                 tmp_file.unlink()
#     exit(127) # TODO: make a file or list somewhere with what error codes ought to be and try to be consistent please    


# def reverse_complement(seq):
#     # complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
#     # return ''.join(complement.get(base.upper(), base) for base in reversed(seq))
#     # CPU times: user 719 μs, sys: 64 μs, total: 783 μs for a 6169nt seq vs 
#     # CPU times: user 11 μs, sys: 1 μs, total: 12 μs for the same seq with: mappy.revcomp()

# def download_genome(taxid):
# # Define the API endpoint URL
#     url = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome"

#     # Define the request payload
#     payload = {
#         "accessions": [taxid],
#         "include_annotation_type": ["RNA_FASTA","GENOME_FASTA"],
#         "hydrated": "FULLY_HYDRATED",
#         "include_tsv": False,
#         "_exp_debug_values": "debug_value"
#     }
#     # Convert the payload to JSON
#     payload_json = json.dumps(payload)

#     # Set the request headers
#     headers = {"Content-Type": "application/json"}

#     # Make the POST request
#     response = requests.post(url, headers=headers, data=payload_json)
#     response
#     # Check the response status code
#     if response.status_code == 200:
#         # Parse the response JSON
#         response_json = response.json()
#         print(response_json)
#     else:
#         print(f"Error: {response.status_code}")




# def download_genome_ebi(accession: str, output_dir: str = ".") -> None:
#     """Download genome data from ENA for a given accession.
    
#     Args:
#         accession (str): ENA/EBI accession for the organism
#         output_dir (str): Directory to save downloaded files
        
#     Returns:
#         str or None: Path to downloaded file if successful, None otherwise
        
#     Note:
#         Uses ENA programmatic access to download data.
#         Supports accession types:
#         - Study: PRJxxxxx
#         - Sample: SAMEAxxxxxx, SAMNxxxxxxxx, SAMDxxxxxxxx
#         - Assembly: GCA_xxxxxx
#         - Run: ERRxxxxxx, SRRxxxxxx, DRRxxxxxx
#     """
#     import requests
#     from pathlib import Path
#     import subprocess as sp
    
#     # Create output directory if it doesn't exist
#     Path(output_dir).mkdir(exist_ok=True)
    
#     try:
#         # Determine accession type and use appropriate endpoint
#         if accession.startswith(('GCA_', 'GCF_')):
#             # Assembly accession
#             url = f"https://www.ebi.ac.uk/ena/portal/api/filereport?accession={accession}&result=assembly&fields=submitted_ftp,assembly_level"
#         elif accession.startswith(('ERR', 'SRR', 'DRR')):
#             # Run accession
#             url = f"https://www.ebi.ac.uk/ena/portal/api/filereport?accession={accession}&result=read_run&fields=fastq_ftp,submitted_ftp"
#         elif accession.startswith('PRJ'):
#             # Study accession
#             url = f"https://www.ebi.ac.uk/ena/portal/api/filereport?accession={accession}&result=assembly&fields=submitted_ftp,assembly_level"
#         elif accession.startswith(('SAME', 'SAMN', 'SAMD')):
#             # Sample accession
#             url = f"https://www.ebi.ac.uk/ena/portal/api/filereport?accession={accession}&result=assembly&fields=submitted_ftp,assembly_level"
#         else:
#             print(f"Unsupported accession format: {accession}")
#             return None
            
#         response = requests.get(url)
#         if response.status_code != 200:
#             print(f"Failed to get file report: {response.status_code}")
#             return None
            
#         # Skip header and get first data line
#         lines = response.text.strip().split('\n')
#         if len(lines) < 2:
#             print(f"No files found for accession {accession}")
#             return None
            
#         # Parse file URLs
#         data = lines[1].split('\t')
#         if len(data) < 1:
#             print(f"Invalid data format for accession {accession}")
#             return None
            
#         # Get FTP URLs from the response
#         ftp_fields = data[0].split(';')
#         if not any(ftp_fields):
#             print(f"No download URLs found for accession {accession}")
#             return None
            
#         # Download each file
#         output_files = []
#         for ftp_url in ftp_fields:
#             if not ftp_url:
#                 continue
                
#             # Convert FTP URL to HTTPS
#             if ftp_url.startswith('ftp://'):
#                 ftp_url = 'https://' + ftp_url[6:]
                
#             filename = Path(ftp_url).name
#             output_path = Path(output_dir) / filename
            
#             print(f"Downloading {filename}...")
#             try:
#                 # Use wget for reliable downloads
#                 sp.run(['wget', '-q', '-O', str(output_path), ftp_url], check=True)
#                 output_files.append(str(output_path))
#             except sp.CalledProcessError as e:
#                 print(f"Failed to download {filename}: {e}")
#                 continue
                
#         if output_files:
#             print(f"Successfully downloaded {len(output_files)} files to {output_dir}")
#             return output_files[0]  # Return path to first downloaded file
#         else:
#             print("No files were successfully downloaded")
#             return None
            
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None

# def download_cds_ebi(accession: str, output_dir: str = ".") -> None:
#     """Download coding sequences (CDS) from ENA for a given accession.
    
#     Args:
#         accession (str): ENA/EBI accession for the organism
#         output_dir (str): Directory to save downloaded files
        
#     Returns:
#         str or None: Path to downloaded file if successful, None otherwise
        
#     Note:
#         Uses ENA programmatic access to download CDS data.
#         Only supports coding accessions in format: XXXxxxxx[.version]
#         where X is letter and x is digit, optional version number.
#     """
#     import requests
#     from pathlib import Path
#     import subprocess as sp
#     import re
    
#     # Create output directory if it doesn't exist
#     Path(output_dir).mkdir(exist_ok=True)
    
#     # Validate coding accession format
#     if not re.match(r'^[A-Z]{3}[0-9]{5}(\.[0-9]{1,2})?$', accession):
#         print(f"Invalid coding accession format: {accession}")
#         print("Coding accessions must be in format: XXXxxxxx[.version]")
#         return None
    
#     try:
#         # Get coding sequence data
#         url = f"https://www.ebi.ac.uk/ena/browser/api/embl/{accession}"
#         response = requests.get(url)
        
#         if response.status_code != 200:
#             print(f"Failed to get CDS data: {response.status_code}")
#             return None
            
#         # Save the EMBL format file
#         output_path = Path(output_dir) / f"{accession}.embl"
#         with open(output_path, 'wb') as f:
#             f.write(response.content)
            
#         print(f"Successfully downloaded CDS data to {output_path}")
#         return str(output_path)
            
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None
