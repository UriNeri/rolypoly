"""Translation and ORF prediction functions."""

import multiprocessing.pool
import re
from contextlib import nullcontext
from functools import lru_cache
from itertools import product
from string import Formatter

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple, Union

from needletail import parse_fastx_file

from rolypoly.utils.various import run_command_comp


def translate_6frx_seqkit(
    input_file: str, output_file: str, threads: int, min_orf_length: int = 0
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
    run_command_comp(
        base_cmd="seqkit translate",
        assign_operator="=",
        prefix_style="double",
        params={
            "allow-unknown-codon": True,
            "append-frame": True,
            "clean": True,
            "min-len": min_orf_length,
            "line-width": 0,
            "frame": 6,
            "threads": threads,
        },
        positional_args=[f"{input_file} --out-file {output_file}"],
        positional_args_location="end",
    )
    # sp.run(command, shell=True, check=True)


def translate_with_bbmap(
    input_file: str, output_file: str, threads: int
) -> None:
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
    command = f"callgenes.sh threads={threads} in={input_file} outa={output_file} out={gff_o}"
    sp.run(command, shell=True, check=True)


def pyro_predict_orfs(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    threads: int,
    min_gene_length: int = 30,
    genetic_code: int = 11,  # NOT USED
) -> None:
    """Predict and translate Open Reading Frames using Pyrodigal.

    Uses either Pyrodigal-rv (optimized for viruses) or standard Pyrodigal
    to predict and translate ORFs from nucleotide sequences.

    Args:
        input_file (str): Path to input nucleotide FASTA file
        output_file (str): Path to output amino acid FASTA file
        threads (int): Number of CPU threads to use
        genetic_code (int, optional): Genetic code table to use (Standard/Bacterial) (NOT USED YET).

    Note:
        - Creates both protein sequences (.faa) and gene annotations (.gff)
        - Uses each input header's first whitespace-delimited token as the
          sequence ID in generated FASTA/GFF output. This keeps ORF identifiers
          compatible with search tools that also key FASTA records by the first
          token.
        - genetic_code is 11 for standard/bacterial
    """
    # import pyrodigal_gv as pyro_gv
    import pyrodigal_rv as pyro_rv

    output_path = Path(output_file)
    sequences = []
    ids = []
    seen_ids = set()
    for index, record in enumerate(parse_fastx_file(input_file), start=1):
        source_id = (
            record.id.decode()
            if isinstance(record.id, bytes)
            else str(record.id)
        )
        source_tokens = source_id.split()
        sequence_id = source_tokens[0] if source_tokens else f"seq{index}"
        candidate_id = sequence_id
        suffix = 1
        while candidate_id in seen_ids:
            candidate_id = (
                f"{sequence_id}_seq{index}"
                if suffix == 1
                else f"{sequence_id}_seq{index}_{suffix}"
            )
            suffix += 1
        seen_ids.add(candidate_id)
        sequences.append((record.seq))  # type: ignore
        ids.append(candidate_id)

    gene_finder = pyro_rv.ViralGeneFinder(
        meta=True,
        min_gene=min_gene_length,
        max_overlap=(
            min(30, min_gene_length - 1) if min_gene_length > 30 else 20
        ),  # Ensure max_overlap < min_gene
    )  # a single gv gene finder object

    with multiprocessing.pool.Pool(processes=threads) as pool:
        orfs = pool.map(gene_finder.find_genes, sequences)

    with open(output_path, "w") as dst:
        for i, orf in enumerate(orfs):
            orf.write_translations(dst, sequence_id=ids[i], width=111110)

    gff_path = output_path.with_suffix(".gff")
    with open(gff_path, "w") as dst:
        dst.write("##gff-version 3\n")
        for i, orf in enumerate(orfs):
            orf.write_gff(
                dst, sequence_id=ids[i], header=False, full_id=True
            )


def predict_orfs_orffinder(
    input_fasta: Union[str, Path],
    output_file: Union[str, Path],
    min_orf_length: int,
    genetic_code: int,
    start_codon: int = 1,
    strand: str = "both",
    outfmt: int = 0,
    ignore_nested: bool = False,
) -> None:
    run_command_comp(
        "ORFfinder",
        params={
            "in": str(input_fasta),
            "out": str(output_file),
            "ml": min_orf_length,  # orfinder automatically replaces values below 30 to 30.
            "s": start_codon,  # ORF start codon to use, 0 is atg only, 1 atg + alt start codons
            "g": genetic_code,
            "n": "true"
            if ignore_nested
            else "false",
            "strand": strand,  # both is plus and minus.
            "outfmt": outfmt,  # 0 is protein FASTA, 1 is nucleotide CDS FASTA
        },
        prefix_style="single",
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
    "6": "FFLLSSSSYYQQCC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
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
SIX_FRAME_TRANSLATION_VERSION = "numpy-iupac-v2"
SIX_FRAMES = (1, 2, 3, -1, -2, -3)
SEQKIT_SIX_FRAME_DEFLINE = "{id}_frame={frame} {description}"
CANONICAL_SIX_FRAME_DEFLINE = "{canonical_id}_frame_{frame_token}"
IUPAC_BASES = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "R": "AG",
    "Y": "CT",
    "S": "CG",
    "W": "AT",
    "K": "GT",
    "M": "AC",
    "B": "CGT",
    "D": "AGT",
    "H": "ACT",
    "V": "ACG",
    "N": "ACGT",
}
IUPAC_COMPLEMENT = str.maketrans("ACGTRYSWKMBDHVN.-", "TGCAYRSWMKVHDBN.-")
_NUMPY_CODON_RADIX = 17
_NUMPY_INVALID_BASE = 16


def make_translation_table(
    genetic_code: int,
) -> Tuple[Dict[str, str], Dict[str, bool]]:
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


@lru_cache(maxsize=None)
def numpy_translation_tables(
    genetic_code: int = 1, clean: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Build cached byte and codon lookup arrays with exhaustive IUPAC support.

    An ambiguous codon is translated when every concrete codon it represents
    has the same amino acid. Otherwise it becomes ``X``. This deliberately
    avoids SeqKit's order-dependent ambiguity-table expansion.
    """
    canonical, _ = make_translation_table(genetic_code)
    symbols = tuple(IUPAC_BASES)
    base_codes = np.full(256, _NUMPY_INVALID_BASE, dtype=np.uint16)
    for code, base in enumerate(symbols):
        base_codes[ord(base)] = code
    gap_code = len(symbols)
    base_codes[ord("-")] = gap_code
    base_codes[ord(".")] = gap_code

    lookup = np.full(_NUMPY_CODON_RADIX**3, ord("X"), dtype=np.uint8)
    for ambiguous_codon in product(symbols, repeat=3):
        amino_acids = {
            canonical["".join(codon)]
            for codon in product(
                *(IUPAC_BASES[base] for base in ambiguous_codon)
            )
        }
        amino_acid = next(iter(amino_acids)) if len(amino_acids) == 1 else "X"
        if clean and amino_acid == "*":
            amino_acid = "X"
        first, second, third = (
            int(base_codes[ord(base)]) for base in ambiguous_codon
        )
        index = first * _NUMPY_CODON_RADIX**2 + second * _NUMPY_CODON_RADIX + third
        lookup[index] = ord(amino_acid)

    gap_index = (
        gap_code * _NUMPY_CODON_RADIX**2
        + gap_code * _NUMPY_CODON_RADIX
        + gap_code
    )
    lookup[gap_index] = ord("-")
    return base_codes, lookup


def normalize_nucleotide_sequence(sequence: str) -> str:
    """Normalize case and RNA uracil before translation."""
    return sequence.replace(" ", "").replace("\t", "").upper().replace("U", "T")


def reverse_complement_iupac(sequence: str) -> str:
    """Return a reverse complement that preserves all IUPAC ambiguity codes."""
    return normalize_nucleotide_sequence(sequence).translate(IUPAC_COMPLEMENT)[
        ::-1
    ]


def translate_sequence_numpy(
    sequence: str, frame: int = 1, genetic_code: int = 1, clean: bool = True
) -> str:
    """Translate one signed reading frame with a vectorized NumPy lookup."""
    if frame not in SIX_FRAMES:
        raise ValueError(f"frame must be one of {SIX_FRAMES}, got {frame}")
    sequence = normalize_nucleotide_sequence(sequence)
    if frame < 0:
        sequence = sequence.translate(IUPAC_COMPLEMENT)[::-1]
    sequence = sequence[abs(frame) - 1 :]
    try:
        encoded = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    except UnicodeEncodeError as error:
        raise ValueError(
            "Nucleotide sequences must contain ASCII characters"
        ) from error
    length = len(encoded) // 3 * 3
    if length == 0:
        return ""
    base_codes, lookup = numpy_translation_tables(genetic_code, clean)
    codes = base_codes[encoded[:length]]
    indices = (
        codes[::3] * _NUMPY_CODON_RADIX**2
        + codes[1::3] * _NUMPY_CODON_RADIX
        + codes[2::3]
    )
    return lookup[indices].tobytes().decode("ascii")


def translate_six_frames_numpy(
    sequence: str, genetic_code: int = 1, clean: bool = True
) -> List[Tuple[int, str]]:
    """Translate a nucleotide sequence in signed frame order 1,2,3,-1,-2,-3."""
    sequence = normalize_nucleotide_sequence(sequence)
    reverse = sequence.translate(IUPAC_COMPLEMENT)[::-1]
    base_codes, lookup = numpy_translation_tables(genetic_code, clean)
    translations = []
    for frame in SIX_FRAMES:
        frame_sequence = (sequence if frame > 0 else reverse)[abs(frame) - 1 :]
        encoded = np.frombuffer(frame_sequence.encode("ascii"), dtype=np.uint8)
        length = len(encoded) // 3 * 3
        if length:
            codes = base_codes[encoded[:length]]
            indices = (
                codes[::3] * _NUMPY_CODON_RADIX**2
                + codes[1::3] * _NUMPY_CODON_RADIX
                + codes[2::3]
            )
            amino_acids = lookup[indices].tobytes().decode("ascii")
        else:
            amino_acids = ""
        translations.append((frame, amino_acids))
    return translations


def validate_six_frame_defline(
    defline_template: str, require_unique_ids: bool = True
) -> None:
    """Validate supported fields and optionally require unique FASTA IDs."""
    allowed = {
        "id",
        "canonical_id",
        "description",
        "frame",
        "frame_abs",
        "strand",
        "frame_token",
    }
    fields = {
        name
        for _, name, _, _ in Formatter().parse(defline_template)
        if name is not None
    }
    unknown = fields - allowed
    if unknown:
        raise ValueError(
            f"Unknown six-frame defline fields: {sorted(unknown)}"
        )
    rendered = [
        format_six_frame_header(
            f"parent_{parent} description", frame, defline_template
        )
        for parent in (1, 2)
        for frame in SIX_FRAMES
    ]
    if any(not defline.split() for defline in rendered):
        raise ValueError("Six-frame deflines must not be empty")
    if not require_unique_ids:
        return
    identifiers = {defline.split()[0] for defline in rendered}
    if len(identifiers) != 12:
        raise ValueError(
            "The first whitespace-delimited defline token must uniquely "
            "identify every parent and frame"
        )


def format_six_frame_header(
    header: str,
    frame: int,
    defline_template: str = SEQKIT_SIX_FRAME_DEFLINE,
) -> str:
    """Format one translated-frame defline from a source FASTA header."""
    from urllib.parse import quote

    sequence_id, separator, description = header.partition(" ")
    if not separator:
        sequence_id, separator, description = header.partition("\t")
    return defline_template.format(
        id=sequence_id,
        canonical_id=quote(sequence_id, safe="_.-"),
        description=description,
        frame=frame,
        frame_abs=abs(frame),
        strand="+" if frame > 0 else "-",
        frame_token=f"{'p' if frame > 0 else 'm'}{abs(frame)}",
    )


def six_frame_header(header: str, frame: int) -> str:
    """Append a signed frame to a FASTA ID using SeqKit-compatible naming."""
    return format_six_frame_header(header, frame)


def translate_6frx_numpy(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    threads: int = 1,
    min_orf_length: int = 0,
    genetic_code: int = 1,
    stops_as_x: bool = True,
    defline_template: str = SEQKIT_SIX_FRAME_DEFLINE,
    require_unique_ids: bool = True,
) -> None:
    """Stream FASTA through the NumPy six-frame translator.

    Long-contig batches can use record-level threads because NumPy releases the
    GIL for its array operations. Short-contig batches remain sequential because
    thread scheduling costs more than their translations.
    ``min_orf_length`` filters complete translated frames, because this function
    does not split translations into stop-delimited ORFs.
    ``stops_as_x`` selects ``X`` rather than ``*`` for resolved stop codons.
    ``defline_template`` may use id, canonical_id, description, frame,
    frame_abs, strand, and frame_token fields.
    """
    from concurrent.futures import ThreadPoolExecutor

    validate_six_frame_defline(defline_template, require_unique_ids)
    if Path(input_file).stat().st_size == 0:
        Path(output_file).write_text("")
        return
    threads = max(1, int(threads))
    batch_size = max(1, threads * 2)

    def input_records():
        for record in parse_fastx_file(str(input_file)):
            yield (
                record.id.decode()
                if isinstance(record.id, bytes)
                else str(record.id),
                record.seq.decode()
                if isinstance(record.seq, bytes)
                else str(record.seq),
            )

    def write_batch(output, batch, executor):
        sequences = [sequence for _, sequence in batch]
        # Threads helped for long records but substantially hurt short records.
        use_threads = (
            executor is not None
            and len(batch) > 1
            and sum(map(len, sequences)) >= len(batch) * 32_768
        )
        if use_threads:
            translated = executor.map(
                translate_six_frames_numpy,
                sequences,
                [genetic_code] * len(batch),
                [stops_as_x] * len(batch),
            )
        else:
            translated = (
                translate_six_frames_numpy(
                    sequence, genetic_code, stops_as_x
                )
                for sequence in sequences
            )
        for (header, _), frames in zip(batch, translated):
            for frame, amino_acids in frames:
                if len(amino_acids) < min_orf_length:
                    continue
                output.write(
                    f">{format_six_frame_header(header, frame, defline_template)}\n"
                    f"{amino_acids}\n"
                )

    with (
        Path(output_file).open("w") as output,
        ThreadPoolExecutor(max_workers=threads)
        if threads > 1
        else nullcontext() as executor,
    ):
        batch = []
        for record in input_records():
            batch.append(record)
            if len(batch) == batch_size:
                write_batch(output, batch, executor)
                batch.clear()
        if batch:
            write_batch(output, batch, executor)


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
        codon = seq[i : i + 3]
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
    upper: bool = False,
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
    delim: str = r"[ ,;:|]",
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
                header,
                translated,
                f,
                seq_len,
                seq_id,
                pmode,
                nmode,
                lmin,
                upper,
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
        codon = seq[i : i + 3]
        aa = tranaa.get(codon, "X")
        # Make start codons lowercase
        if transt.get(codon, False):
            aa = aa.lower()
        translated += aa

    return translated


TRANSLATION_SCHEMA = {
    "translation_id": pl.String, "source_seq_id": pl.String,
    "contig_length": pl.Int64, "translation_length_aa": pl.Int64,
    "translation_method": pl.String, "translation_nt_start": pl.Int64,
    "translation_nt_end": pl.Int64, "strand": pl.Int64, "frame_id": pl.Int64,
    "orf_nt_start": pl.Int64, "orf_nt_end": pl.Int64,
    "original_translation_id": pl.String, "original_header": pl.String,
    "original_source_header": pl.String, "prediction_attributes": pl.String,
    "original_gff_record": pl.String,
}


def build_translation_metadata(input_fasta, protein_fasta, method):
    """Describe the genomic interval represented by each protein query.

    Full-frame translations are not ORFs. Frame is signed 1..3, measured from
    the corresponding end of the original contig, and is distinct from GFF phase.
    Protein input has no inferred nucleotide origin, even if its ID looks like
    an ORF header. All coordinates are 1-based inclusive.
    """
    from rolypoly.utils.bio.interval_ops import normalize_oriented_interval
    from rolypoly.utils.bio.polars_fastx import scan_protein_gff_records

    def records(path):
        if Path(path).stat().st_size == 0:
            return
        for record in parse_fastx_file(str(path)):
            header = record.id.decode() if isinstance(record.id, bytes) else str(record.id)
            sequence = record.seq.decode() if isinstance(record.seq, bytes) else str(record.seq)
            yield header, sequence

    source_headers = {header.split()[0]: header for header, _ in records(input_fasta)}
    lengths = {}
    if method != "input_protein":
        for header, seq in records(input_fasta):
            key = header.split()[0]
            if key in lengths:
                raise ValueError(f"Duplicate contig ID prevents unambiguous coordinate mapping: {key}")
            lengths[key] = len(seq)
    gff = {}
    gff_path = Path(protein_fasta).with_suffix(".gff")
    if method in ("pyrodigal", "bbmap") and gff_path.exists():
        try:
            gff_records = scan_protein_gff_records(gff_path).collect()
        except pl.exceptions.NoDataError:
            gff_records = pl.DataFrame()
        for row in gff_records.iter_rows(named=True):
            if row["type"] == "CDS":
                if row["protein"] in gff:
                    raise ValueError("Split/duplicate CDS records require a segment-aware mapping")
                gff[row["protein"]] = row
    rows, seen = [], set()
    for header, seq in records(protein_fasta):
        key = header.split()[0]
        if key in seen:
            raise ValueError(f"Duplicate translation ID: {key}")
        seen.add(key)
        row = dict.fromkeys(TRANSLATION_SCHEMA)
        import json
        attributes = dict(re.findall(r"(?:^|[; ])([A-Za-z_][\w]*)=([^;]+)", header.split(" # ")[-1])) if " # " in header else {}
        original_gff = gff.get(key)
        if original_gff:
            attributes.update(dict(item.split("=", 1) for item in original_gff["attributes"].split(";") if "=" in item))
        row.update(translation_id=key, translation_length_aa=len(seq), translation_method=method,
                   original_translation_id=key, original_header=header,
                   prediction_attributes=json.dumps(attributes, sort_keys=True),
                   original_gff_record=json.dumps(original_gff, sort_keys=True) if original_gff else None)
        if method == "input_protein":
            row["source_seq_id"] = key
            row["original_source_header"] = source_headers[key]
            rows.append(row)
            continue
        orf = method not in ("six-frame", "six_frame")
        if not orf:
            match = re.fullmatch(r"(.+)_frame=([+-]?[123])", key)
            if match:
                contig, frame = match[1], int(match[2])
            else:
                from urllib.parse import unquote

                match = re.fullmatch(r"(.+)_frame_([pm])([123])", key)
                if not match:
                    raise ValueError(
                        f"Unrecognized six-frame header: {header}"
                    )
                contig = unquote(match[1])
                frame = int(match[3]) * (1 if match[2] == "p" else -1)
            direction = 1 if frame > 0 else -1
            length = lengths[contig]
            lo = abs(frame) if direction == 1 else length - abs(frame) + 2 - len(seq) * 3
            hi = lo + len(seq) * 3 - 1
        else:
            if key in gff:
                record = gff[key]
                contig = record["seqid"]
                lo, hi, direction = normalize_oriented_interval(record["start"], record["end"], record["strand"])
                if str(record["phase"]) not in ("0", ".", "None"):
                    raise ValueError("Nonzero CDS phase requires an explicit translation offset")
            elif method == "ORFfinder":
                # ORFfinder -outfmt 0 protein IDs carry ZERO-based inclusive
                # oriented endpoints. -outfmt 1 nucleotide CDS IDs differ.
                match = re.fullmatch(r"lcl\|ORF[0-9]+_(.+):([0-9]+):([0-9]+)", key)
                if not match:
                    raise ValueError(f"Unrecognized ORFfinder protein header: {header}")
                contig = match[1]
                lo, hi, direction = normalize_oriented_interval(
                    int(match[2]) + 1, int(match[3]) + 1, descending_encodes_strand=True)
            else:
                match = re.match(r"(.+)_([0-9]+) # ([0-9]+) # ([0-9]+) # (-?1)(?: #|$)", header)
                if not match:
                    raise ValueError(f"No coordinate mapping for translation: {header}")
                contig = match[1]
                lo, hi, direction = normalize_oriented_interval(match[3], match[4], match[5])
            length = lengths[contig]
            frame = ((lo - 1) % 3 + 1) if direction == 1 else -((length - hi) % 3 + 1)
            row.update(orf_nt_start=lo, orf_nt_end=hi)
            # An ORF may include a terminal stop omitted from the protein FASTA.
            if len(seq) * 3 > hi - lo + 1:
                raise ValueError(f"Translation exceeds its nucleotide bounds: {key}")
            if direction == 1:
                hi = lo + len(seq) * 3 - 1
            else:
                lo = hi - len(seq) * 3 + 1
        if not 1 <= lo <= hi <= length:
            raise ValueError(f"Translation outside contig bounds: {key}")
        row.update(source_seq_id=contig, original_source_header=source_headers[contig], contig_length=length,
                   translation_nt_start=lo, translation_nt_end=hi,
                   strand=direction, frame_id=frame)
        rows.append(row)
    return pl.DataFrame(rows, schema=TRANSLATION_SCHEMA)


def translation_records(path):
    """Yield complete headers (including descriptions) and unwrapped sequences."""
    if Path(path).stat().st_size == 0:
        return
    for record in parse_fastx_file(str(path)):
        header = record.id.decode() if isinstance(record.id, bytes) else str(record.id)
        sequence = record.seq.decode() if isinstance(record.seq, bytes) else str(record.seq)
        yield header, sequence


def translation_signature(method, parameters):
    """The prediction implementation and effective parameters, independent of search."""
    import importlib.metadata
    import subprocess

    method = method.replace("six_frame", "six-frame")
    if method == "pyrodigal":
        versions = {name: importlib.metadata.version(name) for name in ("pyrodigal-rv", "pyrodigal")}
    elif method == "six-frame":
        versions = {
            "implementation": SIX_FRAME_TRANSLATION_VERSION,
            "numpy": np.__version__,
        }
    elif method == "ORFfinder":
        command = ["ORFfinder", "-version"]
        result = subprocess.run(command, text=True, capture_output=True, check=True, timeout=15)
        versions = {command[0]: (result.stdout + result.stderr).strip()}
    elif method == "input_protein":
        versions = {}
    else:
        # No verified version fingerprint for this backend: do not authorize reuse.
        versions = None
    return {"method": method, "parameters": parameters, "versions": versions}


def translation_input_fingerprints(input_fasta):
    import hashlib

    result = {}
    for header, sequence in translation_records(input_fasta):
        key = header.split()[0]
        if key in result:
            raise ValueError(f"Duplicate input ID: {key}")
        result[key] = {"header": header, "sha256": hashlib.sha256(sequence.encode()).hexdigest()}
    return result


def write_translation_gff(metadata, output):
    """Write normalized parent features, with CDS phase distinct from frame."""
    from urllib.parse import quote
    from rolypoly.utils.bio.polars_fastx import write_gff3_dataframe

    rows = []
    for row in metadata.iter_rows(named=True):
        if row["translation_nt_start"] is None:
            continue
        is_orf = row["orf_nt_start"] is not None
        rows.append({
            "sequence_id": quote(row["source_seq_id"], safe="_.:-"),
            "source": row["translation_method"], "type": "CDS" if is_orf else "translated_region",
            "start": row["orf_nt_start"] if is_orf else row["translation_nt_start"],
            "end": row["orf_nt_end"] if is_orf else row["translation_nt_end"],
            "score": ".", "strand": "+" if row["strand"] == 1 else "-",
            "phase": "0" if is_orf else ".", "ID": row["translation_id"],
            "original_translation_id": row["original_translation_id"],
            "frame_id": row["frame_id"],
        })
    write_gff3_dataframe(pl.DataFrame(rows), output)


def translation_file_digest(path):
    import hashlib
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_translation_manifest(input_fasta, output_dir, signature, reused_from=None):
    import hashlib
    import json

    output_dir = Path(output_dir)
    files = [output_dir / name for name in ("predicted_orfs.faa", "predicted_orfs.gff", "translation_metadata.tsv")]
    files.extend(sorted((output_dir / "tool_outputs").glob("*")))
    manifest = {
        "schema_version": 1, "signature": signature,
        "inputs": translation_input_fingerprints(input_fasta),
        "files": {str(path.relative_to(output_dir)): translation_file_digest(path)
                  for path in files if path.is_file()},
        "reused_from": reused_from,
    }
    (output_dir / "translation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def normalize_translation_output(input_fasta, protein_fasta, output_dir, method, parameters):
    """Preserve native output and produce a canonical FASTA/GFF/mapping bundle.

    Ordinals are local to each parent contig and ordered by genomic position.
    Frames have explicit signed labels. Original complete headers and GFF records
    stay in the mapping even when a search backend discards header descriptions.
    """
    import shutil
    from collections import defaultdict
    from urllib.parse import quote

    method = method.replace("six_frame", "six-frame")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "translation_manifest.json"
    if manifest_path.exists() and Path(protein_fasta).resolve() == (output_dir / "predicted_orfs.faa").resolve():
        import json
        manifest = json.loads(manifest_path.read_text())
        if manifest["files"].get("predicted_orfs.faa") == translation_file_digest(protein_fasta):
            if manifest["signature"] != translation_signature(method, parameters) or manifest["inputs"] != translation_input_fingerprints(input_fasta):
                raise ValueError("Existing normalized translations do not match input or prediction settings; regenerate them")
            for name, digest in manifest["files"].items():
                path = (output_dir / name).resolve()
                if not path.is_relative_to(output_dir.resolve()) or not path.is_file() or translation_file_digest(path) != digest:
                    raise ValueError("Existing translation bundle was modified; regenerate it")
            return pl.read_csv(output_dir / "translation_metadata.tsv", separator="\t", schema_overrides={**TRANSLATION_SCHEMA, "orf_ordinal": pl.Int64})
    metadata = build_translation_metadata(input_fasta, protein_fasta, method)
    raw_dir = output_dir / "tool_outputs"
    raw_dir.mkdir(exist_ok=True)
    raw_fasta = raw_dir / "predicted_orfs.faa"
    if Path(protein_fasta).resolve() != raw_fasta.resolve():
        shutil.copyfile(protein_fasta, raw_fasta)
    raw_gff = Path(protein_fasta).with_suffix(".gff")
    if method in ("pyrodigal", "bbmap") and raw_gff.exists() and raw_gff.resolve() != (raw_dir / "predicted_orfs.gff").resolve():
        shutil.copyfile(raw_gff, raw_dir / "predicted_orfs.gff")
    elif method not in ("pyrodigal", "bbmap"):
        (raw_dir / "predicted_orfs.gff").unlink(missing_ok=True)
    counters = defaultdict(int)
    ids, labels, ordinals = {}, {}, {}
    for row in metadata.sort(["source_seq_id", "translation_nt_start", "translation_nt_end", "strand", "translation_id"]).iter_rows(named=True):
        original = row["translation_id"]
        parent = quote(row["source_seq_id"], safe="_.-")
        if method in ("six-frame", "six_frame"):
            frame = row["frame_id"]
            suffix = f"frame_{'p' if frame > 0 else 'm'}{abs(frame)}"
            ordinals[original] = None
        else:
            counters[parent] += 1
            ordinal = counters[parent]
            kind = "protein" if method == "input_protein" else "orf"
            suffix = f"{kind}_{ordinal}"
            ordinals[original] = ordinal
        ids[original] = f"{parent}_{suffix}"
        labels[original] = ids[original]
    if len(set(ids.values())) != len(ids):
        raise ValueError("Normalized translation IDs are not unique")
    metadata = metadata.with_columns(
        pl.col("translation_id").replace_strict(labels, return_dtype=pl.String).alias("translation_label"),
        pl.col("translation_id").replace_strict(ordinals, return_dtype=pl.Int64).alias("orf_ordinal"),
        pl.col("translation_id").replace_strict(ids, return_dtype=pl.String),
    )
    canonical_fasta = output_dir / "predicted_orfs.faa"
    if all(original == normalized for original, normalized in ids.items()):
        # Native translation can emit final IDs, avoiding another FASTA parse and
        # rewrite. A temporary source still needs copying into the bundle.
        if Path(protein_fasta).resolve() != canonical_fasta.resolve():
            shutil.copyfile(raw_fasta, canonical_fasta)
    else:
        with canonical_fasta.open("w") as handle:
            for header, sequence in translation_records(raw_fasta):
                handle.write(f">{ids[header.split()[0]]}\n{sequence}\n")
    metadata.write_csv(output_dir / "translation_metadata.tsv", separator="\t")
    write_translation_gff(metadata, output_dir / "predicted_orfs.gff")
    write_translation_manifest(input_fasta, output_dir, translation_signature(method, parameters))
    return metadata


def reuse_translation_bundle(source, input_fasta, output_dir, method, parameters):
    """Reuse all translations for an exact input or a verified contig subset."""
    import hashlib
    import json
    import shutil

    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    if source == output_dir:
        raise ValueError("Reuse requires a distinct output directory")
    if not (source / "translation_manifest.json").is_file():
        raise ValueError("Translation reuse requires a normalized bundle with translation_manifest.json; rerun the source prediction")
    manifest = json.loads((source / "translation_manifest.json").read_text())
    signature = translation_signature(method, parameters)
    if manifest.get("schema_version") != 1 or signature["versions"] is None or manifest["signature"] != signature:
        raise ValueError("Translation reuse rejected: method, parameters, version, or schema differ")
    inputs = translation_input_fingerprints(input_fasta)
    if any(manifest["inputs"].get(key) != value for key, value in inputs.items()):
        raise ValueError("Translation reuse rejected: input IDs, headers, or sequences differ")
    for name, expected in manifest["files"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError("Translation reuse rejected: bundle file missing or outside bundle")
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise ValueError(f"Translation reuse rejected: modified bundle file {name}")
    metadata = pl.read_csv(source / "translation_metadata.tsv", separator="\t", schema_overrides={**TRANSLATION_SCHEMA, "orf_ordinal": pl.Int64})
    metadata = metadata.filter(pl.col("source_seq_id").is_in(list(inputs)))
    selected = set(metadata["translation_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predicted_orfs.faa").open("w") as handle:
        for header, sequence in translation_records(source / "predicted_orfs.faa"):
            if header in selected:
                handle.write(f">{header}\n{sequence}\n")
    if (source / "tool_outputs").exists():
        shutil.copytree(source / "tool_outputs", output_dir / "tool_outputs", dirs_exist_ok=True)
    metadata.write_csv(output_dir / "translation_metadata.tsv", separator="\t")
    write_translation_gff(metadata, output_dir / "predicted_orfs.gff")
    write_translation_manifest(input_fasta, output_dir, signature, str(source))
    return metadata
