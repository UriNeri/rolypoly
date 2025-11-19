"""Translation and ORF prediction functions."""

import multiprocessing.pool
from needletail import parse_fastx_file
import re
from typing import Union, Dict, Tuple, List

from rolypoly.utils.various import run_command_comp
from pathlib import Path


def translate_6frx_seqkit(
    input_file: str,
    output_file: str,
    threads: int,
    min_orf_length: int = 0,
) -> None:
    """Translate nucleotide sequences in all 6 reading frames using seqkit.

    Args:
        input_file (str): Path to input nucleotide FASTA file
        output_file (str): Path to output amino acid FASTA file
        threads (int): Number of CPU threads to use

    Note:
        Requires seqkit to be installed and available in PATH.
        The output sequences are formatted with 20000bp line width.
    """
    # import subprocess as sp

    # command = f"seqkit translate -x -F --clean --min-len {min_orf_length} -w 0 -f 6 {input_file} --id-regexp '(\\*)' --clean  --threads {threads} > {output_file}"
    run_command_comp(base_cmd="seqkit translate",
                     assign_operator="=",
                     prefix_style="double",
                     params={
                         "allow-unknown-codon": True,
                         "clean": True, 
                         "min-len": min_orf_length,
                         "line-width": 0,
                         "frame": 6,
                         "id-regexp": "'(\\*)'",  # Properly quote the regex
                         "threads": threads},
                     positional_args=[f"{input_file} --out-file {output_file}"],
                     positional_args_location="end")
    # sp.run(command, shell=True, check=True)


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
    min_gene_length: int = 30,
    genetic_code: int = 11,  # NOT USED 
) -> None:
    """Predict and translate Open Reading Frames using Pyrodigal.

    Uses either Pyrodigal-GV (optimized for viruses) or standard Pyrodigal
    to predict and translate ORFs from nucleotide sequences.

    Args:
        input_file (str): Path to input nucleotide FASTA file
        output_file (str): Path to output amino acid FASTA file
        threads (int): Number of CPU threads to use
        genetic_code (int, optional): Genetic code table to use (Standard/Bacterial) (NOT USED YET).

    Note:
        - Creates both protein sequences (.faa) and gene annotations (.gff)
        - genetic_code is 11 for standard/bacterial
    """
    import pyrodigal_gv as pyro_gv

    sequences = []
    ids = []
    for record in parse_fastx_file(input_file):
        sequences.append((record.seq)) # type: ignore
        ids.append((record.id)) # type: ignore

    gene_finder = pyro_gv.ViralGeneFinder(
        meta=True,
        min_gene=min_gene_length,
        max_overlap=min(30, min_gene_length - 1) if min_gene_length > 30 else 20,  # Ensure max_overlap < min_gene
        
    )  # a single gv gene finder object

    with multiprocessing.pool.Pool(processes=threads) as pool:
        orfs = pool.map(gene_finder.find_genes, sequences)

    with open(output_file, "w") as dst:
        for i, orf in enumerate(orfs):
            orf.write_translations(dst, sequence_id=ids[i], width=111110)

    with open(output_file.replace(".faa", ".gff"), "w") as dst:
        for i, orf in enumerate(orfs):
            orf.write_gff(dst, sequence_id=ids[i], full_id=True) 
            
            
def predict_orfs_orffinder(
    input_fasta: Union[str, Path],
    output_file: Union[str, Path],
    min_orf_length: int,
    genetic_code: int,
    start_codon: int = 1,
    strand: str = "both",
    outfmt: int = 1,
    ignore_nested: bool = False,
) -> None:
    run_command_comp(
        "ORFfinder",
        params={
            "in": str(input_fasta),
            "out": str(output_file),
            "ml": min_orf_length, # orfinder automatically replaces values below 30 to 30.
            "s": start_codon, # ORF start codon to use, 0 is atg only, 1 atg + alt start codons
            "g": genetic_code,
            "n": "false" if ignore_nested else "true", # do not ignore nested ORFs
            "strand": strand, # both is plus and minus.
            "outfmt": outfmt, # 1 is fasta, 3 is feature table
        },prefix_style="single"
    )


# Genetic codes / variables sourced from Seals2 by Yuri Wolf (https://github.com/YuriWolf-ncbi/seals-2/blob/master/bin/misc/orf)
# Original Perl implementation for comprehensive genetic code support


DEFAULT_CODE = 1
DEFAULT_FRAME = 1
DEFAULT_PMODE = 0
DEFAULT_NMODE = 0
DEFAULT_LMIN = 16
DEFAULT_IDWRD = 2
DEFAULT_DELIM = r"[ ,;:|]"
DEFAULT_UPPER = False

GENETIC_CODES_AA = {
    "1": "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "2": "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNKKSS**VVVVAAAADDEEGGGG",
    "3": "FFLLSSSSYY**CCWWTTTTPPPPHHQQRRRRIIMMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "4": "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "5": "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNKKSSSSVVVVAAAADDEEGGGG",
    "6": "FFLLSSSSYYQQCC*WLLLLPPPPHHQQRRRRIIIMTTTTNKKSSRRVVVVAAAADDEEGGGG",
    "9": "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNNKSSSSVVVVAAAADDEEGGGG",
    "10": "FFLLSSSSYY**CCCWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "11": "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "12": "FFLLSSSSYY**CC*WLLLSPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "13": "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNKKSSGGVVVVAAAADDEEGGGG",
    "14": "FFLLSSSSYYY*CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNNKSSSSVVVVAAAADDEEGGGG",
    "15": "FFLLSSSSYY*QCC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "16": "FFLLSSSSYY*LCC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "21": "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNNKSSSSVVVVAAAADDEEGGGG",
    "22": "FFLLSS*SYY*LCC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "23": "FF*LSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "24": "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSSKVVVVAAAADDEEGGGG",
    "25": "FFLLSSSSYY**CCGWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "26": "FFLLSSSSYY**CC*WLLLAPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "27": "FFLLSSSSYYQQCCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "28": "FFLLSSSSYYQQCCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "29": "FFLLSSSSYYYYCC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "30": "FFLLSSSSYYEECC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "31": "FFLLSSSSYYEECCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "33": "FFLLSSSSYYY*CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSSKVVVVAAAADDEEGGGG",
    "6000": "FFLLSSSSYYQQCCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "7001": "FFLLSSSSYY*QCCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "7010": "FFLLSSSSYYQ*CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "7100": "FFLLSSSSYYQQCC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "8011": "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "8101": "FFLLSSSSYY*QCC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "8110": "FFLLSSSSYYQ*CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "9111": "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "666": "FFLLSSSSYY12CC3WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
}

GENETIC_CODES_START = {
    "1": "---M---------------M---------------M----------------------------",
    "2": "--------------------------------MMMM---------------M------------",
    "3": "----------------------------------MM----------------------------",
    "4": "--MM---------------M------------MMMM---------------M------------",
    "5": "---M----------------------------MMMM---------------M------------",
    "6": "-----------------------------------M----------------------------",
    "9": "-----------------------------------M---------------M------------",
    "10": "-----------------------------------M----------------------------",
    "11": "---M---------------M------------MMMM---------------M------------",
    "12": "-------------------M---------------M----------------------------",
    "13": "-----------------------------------M----------------------------",
    "14": "-----------------------------------M----------------------------",
    "15": "-----------------------------------M----------------------------",
    "16": "-----------------------------------M----------------------------",
    "21": "-----------------------------------M---------------M------------",
    "22": "-----------------------------------M----------------------------",
    "23": "--------------------------------M--M---------------M------------",
    "24": "---M---------------M---------------M---------------M------------",
    "25": "---M-------------------------------M---------------M------------",
    "26": "-------------------M---------------M----------------------------",
    "27": "-----------------------------------M----------------------------",
    "28": "-----------------------------------M----------------------------",
    "29": "-----------------------------------M----------------------------",
    "30": "-----------------------------------M----------------------------",
    "31": "-----------------------------------M----------------------------",
    "33": "---M---------------M---------------M---------------M------------",
    "6000": "---M---------------M------------MMMM---------------M------------",
    "7001": "---M---------------M------------MMMM---------------M------------",
    "7010": "---M---------------M------------MMMM---------------M------------",
    "7100": "---M---------------M------------MMMM---------------M------------",
    "8011": "---M---------------M------------MMMM---------------M------------",
    "8101": "---M---------------M------------MMMM---------------M------------",
    "8110": "---M---------------M------------MMMM---------------M------------",
    "9111": "---M---------------M------------MMMM---------------M------------",
    "666": "---M---------------M---------------M----------------------------",
}

GENETIC_CODE_NAMES = {
    "1": "Standard",
    "2": "Vertebrate Mitochondrial",
    "3": "Yeast Mitochondrial",
    "4": "Mold Mitochondrial; Protozoan Mitochondrial; Coelenterate Mitochondrial; Mycoplasma; Spiroplasma",
    "5": "Invertebrate Mitochondrial",
    "6": "Ciliate Nuclear; Dasycladacean Nuclear; Hexamita Nuclear",
    "9": "Echinoderm Mitochondrial; Flatworm Mitochondrial",
    "10": "Euplotid Nuclear",
    "11": "Bacterial and Plant Plastid",
    "12": "Alternative Yeast Nuclear",
    "13": "Ascidian Mitochondrial",
    "14": "Alternative Flatworm Mitochondrial",
    "15": "Blepharisma Macronuclear",
    "16": "Chlorophycean Mitochondrial",
    "21": "Trematode Mitochondrial",
    "22": "Scenedesmus obliquus mitochondrial",
    "23": "Thraustochytrium mitochondrial",
    "24": "Pterobranchia mitochondrial",
    "25": "Candidate Division SR1 and Gracilibacteria",
    "26": "Pachysolen tannophilus Nuclear Code",
    "27": "Karyorelict Nuclear Code",
    "28": "Condylostoma Nuclear Code",
    "29": "Mesodinium Nuclear Code",
    "30": "Peritrich Nuclear Code",
    "31": "Blastocrithidia Nuclear Code",
    "33": "Cephalodiscidae Mitochondrial UAA-Tyr Code",
    "6000": "generic no-stop code",
    "7001": "generic TAA-stop code",
    "7010": "generic TAG-stop code",
    "7100": "generic TGA-stop code",
    "8011": "generic TAA+TAG-stop code",
    "8101": "generic TAA+TGA-stop code",
    "8110": "generic TAG+TGA-stop code",
    "9111": "generic TAA+TAG+TGA-stop (standard) code",
    "666": "TAA->1 TAG->2 TGA->3 (standard) code",
}

NT_NAME = "TCAG"
NT_COMP = "AGTC"

def make_translation_table(genetic_code: int) -> Tuple[Dict[str, str], Dict[str, bool]]:
    """Create codon translation and start codon lookup tables.
    
    Args:
        genetic_code: Genetic code number to use
        
    Returns:
        Tuple of (amino_acid_table, start_codon_table)
    """
    code_str = str(genetic_code)
    if code_str not in GENETIC_CODES_AA:
        raise ValueError(f"Unknown genetic code: {genetic_code}")
    
    aa_string = GENETIC_CODES_AA[code_str]
    start_string = GENETIC_CODES_START[code_str]
    
    tranaa = {}
    transt = {}
    
    # Build codon tables
    for i in range(4):
        for j in range(4):
            for k in range(4):
                codon = NT_NAME[i] + NT_NAME[j] + NT_NAME[k]
                idx = i * 16 + j * 4 + k
                tranaa[codon] = aa_string[idx]
                transt[codon] = start_string[idx] != "-"
    
    tranaa["---"] = "-"
    return tranaa, transt

def translate_sequence(seq: str, frame: int, genetic_code: int = 11) -> str:
    """Translate a nucleotide sequence in a specific frame.
    
    Args:
        seq: DNA sequence (uppercase, T not U)
        frame: Reading frame (1-3 for forward, -1 to -3 for reverse)
        genetic_code: Genetic code table to use
        
    Returns:
        Translated amino acid sequence
    """
    tranaa, transt = make_translation_table(genetic_code)
    
    # Handle reverse frames
    if frame < 0:
        # Reverse complement
        seq = seq[::-1]  # reverse
        seq = seq.translate(str.maketrans("AGCT", "TCGA"))  # complement
        frame = abs(frame)
    
    offset = frame - 1
    translated = ""
    
    for i in range(offset, len(seq) - 2, 3):
        codon = seq[i:i+3]
        aa = tranaa.get(codon, "X")
        # Make start codons lowercase if specified
        if transt.get(codon, False):
            aa = aa.lower()
        translated += aa
    
    return translated

def print_translation_results(
    definition: str,
    translated_seq: str,
    frame: int,
    nt_length: int,
    seq_id: str,
    pmode: int = 0,
    nmode: int = 0,
    lmin: int = 16,
    upper: bool = False
) -> List[str]:
    """Format and return translation results.
    
    Args:
        definition: Original sequence definition line
        translated_seq: Translated amino acid sequence
        frame: Reading frame used
        nt_length: Length of original nucleotide sequence
        seq_id: Sequence identifier
        pmode: Translation mode (0: full frame, 1: stop-to-stop, 2: from first start)
        nmode: Naming mode (0-3, different naming schemes)
        lmin: Minimum ORF length in amino acids
        upper: Whether to uppercase the sequence
        
    Returns:
        List of FASTA formatted strings
    """
    results = []
    
    if pmode == 1 or pmode == 2:
        # Stop-to-stop or start-to-stop mode
        pattern = r"([A-Za-z]+)" if pmode == 1 else r"([a-z][A-Za-z]*)"
        
        for match in re.finditer(pattern, translated_seq):
            seq = match.group(1).upper()
            if len(seq) < lmin:
                continue
                
            end_pos = match.end()
            start_pos = end_pos - len(seq)
            
            # Calculate nucleotide positions
            r1 = start_pos * 3 + abs(frame)
            r2 = end_pos * 3 + abs(frame) - 1
            
            if frame < 0:
                r1 = nt_length - r1 + 1
                r2 = nt_length - r2 + 1
            
            # Format name based on naming mode
            if nmode == 1:
                name = f"{seq_id}_{r1}_{r2} {definition} [frame {frame}] [range {r1}..{r2}] [len {len(seq)}]"
            elif nmode >= 2:
                name = f"{seq_id}.{r1}-{r2} {definition} [frame {frame}] [range {r1}..{r2}] [len {len(seq)}]"
            else:
                name = f"{definition} [frame {frame}] [range {r1}..{r2}] [len {seq}]"
            
            results.append(f">{name}\n{seq}")
    
    else:
        # Full frame translation
        seq = translated_seq.upper() if upper else translated_seq
        
        if nmode == 1 or nmode == 2:
            name = f"{seq_id}.{frame} {definition} [frame {frame}]"
        elif nmode == 3:
            frame_num = frame if frame > 0 else abs(frame) + 3
            name = f"{seq_id}_fr{frame_num} {definition} [frame {frame}]"
        else:
            name = f"{definition} [frame {frame}]"
        
        results.append(f">{name}\n{seq}")
    
    return results

def translate_fasta_sequences(
    sequences: List[Tuple[str, str]],  # (header, sequence) pairs
    frame: int = 0,
    genetic_code: int = 11,
    pmode: int = 0,
    nmode: int = 0,
    lmin: int = 16,
    idwrd: int = 2,
    upper: bool = False,
    delim: str = r"[ ,;:|]"
) -> List[str]:
    """Translate multiple FASTA sequences.
    
    Args:
        sequences: List of (header, sequence) tuples
        frame: Reading frame (0 for all 6 frames, 1-3 forward, -1 to -3 reverse)
        genetic_code: Genetic code table number
        pmode: Translation mode
        nmode: Naming mode
        lmin: Minimum ORF length
        idwrd: Which word of ID to use (0 for all)
        upper: Uppercase output sequences
        delim: Delimiter pattern for parsing sequence IDs
        
    Returns:
        List of FASTA formatted translation results
    """
    all_results = []
    
    for header, seq in sequences:
        # Clean up sequence
        seq = seq.replace(" ", "").replace("\t", "").upper().replace("U", "T")
        seq_len = len(seq)
        
        # Parse sequence ID
        seq_id = header.split()[0] if header.split() else header
        if idwrd > 0:
            id_parts = re.split(delim, seq_id)
            if len(id_parts) >= idwrd:
                seq_id = id_parts[idwrd - 1]
        
        # Determine frames to translate
        frames = []
        if frame > 0:
            frames = [frame]
        elif frame < 0:
            frames = [frame]
        else:  # frame == 0, translate all 6 frames
            frames = [1, 2, 3, -1, -2, -3]
        
        # Translate in each frame
        for f in frames:
            translated = translate_sequence(seq, f, genetic_code)
            results = print_translation_results(
                header, translated, f, seq_len, seq_id, pmode, nmode, lmin, upper
            )
            all_results.extend(results)
    
    return all_results

# Native simple translation 
def translate(sequence: str, genetic_code: int = 11) -> str:
    """Translate a nucleotide sequence to amino acids using the specified genetic code.
    
    Args:
        sequence: DNA sequence string (will be converted to uppercase, U->T)
        genetic_code: Genetic code table number (default 11 for bacterial/plastid)
        
    Returns:
        Translated amino acid sequence
    """
    # Clean and prepare sequence
    seq = sequence.replace(" ", "").replace("\t", "").upper().replace("U", "T")
    
    # Get translation table
    tranaa, transt = make_translation_table(genetic_code)
    
    translated = ""
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3]
        aa = tranaa.get(codon, "X")
        # Make start codons lowercase
        if transt.get(codon, False):
            aa = aa.lower()
        translated += aa
    
    return translated
    
    