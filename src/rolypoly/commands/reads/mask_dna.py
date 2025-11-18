import os
import shutil
from pathlib import Path

import mappy as mp
import rich_click as click
from bbmapy import bbmap, bbmask, kcompress
from rich.console import Console

from rolypoly.utils.various import ensure_memory
from rolypoly.utils.bio.interval_ops import mask_sequence_mp
from rolypoly.utils.bio.alignments import calculate_percent_identity

global datadir
datadir = Path(os.environ.get("ROLYPOLY_DATA", "")) # THIS IS A HACK, I need to figure out how to set the datadir if code is accessed from outside the package (currently it's set in the rolypoly.py  and exported into the env).


@click.command()
@click.option("-t", "--threads", default=1, help="Number of threads to use")
@click.option("-M", "--memory", default="6gb", help="Memory in GB")
@click.option("-o", "--output", required=True, help="Output file name")
@click.option(
    "-f", "--flatten", is_flag=True, help="Attempt to kcompress.sh the masked file"
)
@click.option("-i", "--input", required=True, help="Input fasta file")
@click.option("-F", "--mmseqs", is_flag=True, help="use mmseqs2 instead of bbmap.sh")
@click.option("-lm", "--low-mem", is_flag=True, help="use minimap2 instead of bbmap.sh")
@click.option("-bt", "--bowtie", is_flag=True, help="use bowtie1 instead of bbmap.sh")
@click.option(
    "-r",
    "--reference",
    default=datadir / "masking/masking/RVMT_NCBI_Ribo_Japan_for_masking.fasta",
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
    import subprocess as sp

    console = Console(width=150)

    input_file = Path(input).resolve()
    output_file = Path(output).resolve()
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
        for name, seq, qual in mp.fastx_read(str(input_file)):
            masked_sequences[name] = seq
            for hit in aligner.map(seq):
                percent_id = calculate_percent_identity(hit.cigar_str, hit.NM)
                console.print(f"{percent_id}")
                if percent_id > 70:
                    masked_sequences[name] = mask_sequence_mp(
                        masked_sequences[name], hit.q_st, hit.q_en, hit.strand
                    )

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
    console.print(f"[green]Masking completed. Output saved to {output_file}[/green]") 