import os
import shutil
from pathlib import Path

import rich_click as click

# from rich.console import Console
from rolypoly.utils.bio.alignments import calculate_percent_identity
from rolypoly.utils.bio.interval_ops import mask_nuc_range, mask_sequence_mp
from rolypoly.utils.logging.loggit import get_logger
from rolypoly.utils.various import (  # TODO: Replace sp.run with run_command_comp.
    ensure_memory,
    run_command_comp,
)

global datadir
datadir = Path(
    os.environ.get("ROLYPOLY_DATA_DIR", "")
)  # THIS IS A HACK, I need to figure out how to set the datadir if code is accessed from outside the package (currently it's set in the rolypoly.py and exported into the env).


@click.command()
@click.option("-t", "--threads", default=1, help="Number of threads to use")
@click.option("-M", "--memory", default="6gb", help="Memory in GB")
@click.option("-o", "--output", required=True, help="Output file name")
@click.option(
    "-f",
    "--flatten",
    is_flag=True,
    help="Attempt to kcompress.sh the masked file",
)
@click.option("-i", "--input", required=True, help="Input fasta file")
@click.option(
    "-a",
    "--aligner",
    required=False,
    default="mmseqs2",
    help="Which tool to use for identifying shared sequence (minimap2, mmseqs2, diamond, bowtie1, bbmap)",
)
@click.option(
    "-mlc",
    "--mask-low-complexity",
    is_flag=True,
    help="Whether to mask low complexity regions using bbduks entropy masking",
)
@click.option(
    "-r",
    "--reference",
    default=datadir / "contam/masking/combined_entropy_masked.fasta",
    help="Provide an input fasta file to be used for masking, instead of the pre-generated collection of RNA viral sequences",
)
@click.option(
    "--tmpdir",
    default=None,
    help="Temporary directory to use (default: output file's parent/tmp - if you have enough RAM, you can set this to /dev/shm/ or /tmp/ for faster I/O)",
)
def mask_dna(
    threads,
    memory,
    output,
    flatten,
    input,
    aligner,
    reference,
    mask_low_complexity,
    tmpdir,
):
    """Mask an input fasta file for sequences that could be RNA viral (or mistaken for such).

    Args:
      threads: (int) Number of threads to use
      memory: (str) Memory in GB
      output: (str) Output file name
      flatten: (bool) Attempt to kcompress.sh the masked file
      input: (str) Input fasta file
      aligner: (str) Which tool to use for identifying shared sequence (minimap2, mmseqs2, diamond, bowtie1, bbmap)
      reference: (str) Provide an input fasta file to be used for masking, instead of the pre-generated collection of RNA viral sequences
      mask_low_complexity: (bool) Whether to mask low complexity regions using bbduks entropy masking

    Returns:
      None
    """
    logger = get_logger()
    logger.debug(f"datadir used: {datadir}")

    input_file = Path(input).resolve()
    output_file = Path(output).resolve()
    aligner = str(aligner).lower()
    if aligner not in ["minimap2", "mmseqs2", "diamond", "bowtie1", "bbmap"]:
        logger.error(
            f"{aligner} not recognised as one of minimap2, mmseqs2, diamond, bowtie1 or bbmap"
        )
        exit
    needs_bbmask_only = aligner in ["bowtie1", "bbmap", "mmseqs2"]
    memory = ensure_memory(memory)["giga"]
    reference = Path(reference).absolute().resolve()
    if tmpdir is None:
        tmpdir = output_file.parent / "tmp_mask_dna"
    tmpdir = Path(tmpdir).absolute().resolve()
    Path.mkdir(Path(tmpdir), exist_ok=True)

    if aligner == "minimap2":
        logger.info("Using minimap2 (low memory mode)")
        import mappy as mp

        # Create a mappy aligner object
        mpaligner = mp.Aligner(
            str(reference), k=11, n_threads=threads, best_n=15000
        )
        if not mpaligner:
            raise Exception("ERROR: failed to load/build index")

        # Perform alignment, write results to SAM file, and mask sequences
        masked_sequences = {}
        for name, seq, qual in mp.fastx_read(str(input_file)):
            masked_sequences[name] = seq
            for hit in mpaligner.map(seq):
                percent_id = calculate_percent_identity(
                    hit.cigar_str, hit.NM
                )  # this make some assumptions
                logger.info(f"{percent_id}")
                if percent_id > 70:
                    masked_sequences[name] = mask_sequence_mp(
                        masked_sequences[name], hit.q_st, hit.q_en, hit.strand
                    )

        # Write masked sequences to output file
        with open(f"{tmpdir}/tmp_masked.fasta", "w") as out_f:
            for name, seq in masked_sequences.items():
                out_f.write(f">{name}\n{seq}\n")
        logger.info(
            f"Masking completed. Output saved to {tmpdir}/tmp_masked.fasta"
        )
        shutil.rmtree(f"{tmpdir}", ignore_errors=True)
    elif aligner == "bowtie1":
        import subprocess as sp

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
    elif aligner == "mmseqs2":
        # logger.info(
        #     "Note! using mmseqs2 instead of bbmap is not a tight drop in replacement."
        # )
        # v=1
        v = 3 if logger.level == "DEBUG" or logger.level == 10 else 1
        # v=3
        run_command_comp(
            assign_operator=" ",
            base_cmd="mmseqs easy-linsearch",
            check_output=True,
            output_file=f"{tmpdir}/tmp_mapped.sam",
            positional_args=[
                str(reference),
                str(input_file),
                f"{tmpdir}/tmp_mapped.sam",
                f"{tmpdir}",
            ],
            positional_args_location="start",
            param_sep=" ",
            params={
                "min-seq-id": str(0.7),
                "min-aln-len": str(80),
                # "subject-cover": "40",
                "threads": threads,
                "format-mode": 1,
                # "headers-split-mode": "1",
                # "alt-ali":123123123,
                "search-type": "3",
                "v": v,
                "max-accept": "1231",
                # "max-seqs": "1231",
                # "dbtype": 2,
                "a": "",
            },
        )
    elif aligner == "diamond":
        logger.info(
            "Note! using diamond blastx - NOTE - SWITCHING TO A PROTEIN SEQ instead of default REFERENCE"
        )
        reference = (
            reference
            if str(reference)
            != str(datadir / "contam/masking/combined_entropy_masked.fasta")
            else str(datadir / "contam/masking/combined_deduplicated_orfs.faa")
        )
        logger.info(f"Note! using as reference: {reference} ")
        run_command_comp(
            assign_operator=" ",
            base_cmd="diamond blastx",
            positional_args=["qseqid qstart qend qstrand"],
            positional_args_location="end",
            param_sep=" ",
            params={
                "query": str(input_file),
                "db": str(reference),
                "out": f"{tmpdir}/tmp_mapped.tsv",
                "id": "70",
                "subject-cover": "40",
                "min-query-len": "20",
                "threads": threads,
                "max-target-seqs": 123123123,
                "outfmt": "6",
            },
        )
        logger.info(f"Finished diamond blastx step")
        mask_nuc_range(
            input_fasta=str(input_file),
            input_table=f"{tmpdir}/tmp_mapped.tsv",
            output_fasta=f"{tmpdir}/tmp_masked.fasta",
        )
        # TODO: Check if diamond blastx reports qstrand needs to be adjusted based on frame?
        # TODO: Maybe drop the entry query contig if qcov > 80%  (would require adding qcov to the output table)
    elif aligner == "bbmap":
        logger.info("Using bbmap.sh")
        from bbmapy import bbmap

        bbmap(
            ref=input_file,
            in_file=reference,
            outm=f"{tmpdir}/tmp_mapped.sam",
            minid=0.7,
            overwrite="true",
            threads=threads,
            Xmx=memory,
            simd="true",
        )

    logger.info(f"Finished running aligner {aligner}")
    logger.info(f"beginning bbmask (masking + entropy) step")

    if needs_bbmask_only:
        # Mask using the sam files, for aligners that need it...
        # approximate to bbmask.sh in=input.fasta out=output.fasta sam=mapped.sam overwrite=true threads=8 Xmx=16g
        logger.debug(
            f"equivalent to bbmask.sh in={input_file} out={tmpdir}/tmp_masked.fasta sam={tmpdir}/tmp_mapped.sam overwrite=true threads={threads} Xmx={memory}"
        )
        from bbmapy import bbmask

        bbmask(
            in_file=input_file,
            out=f"{tmpdir}/tmp_masked.fasta",
            sam=f"{tmpdir}/tmp_mapped.sam",
            overwrite="true",
            threads=threads,
            Xmx=memory,
        )
        logger.info(f"Finished bbmask step")

    last_file = f"{tmpdir}/tmp_masked.fasta"

    if mask_low_complexity:
        logger.info(f"Proceeding to entropy masking step")
        from bbmapy import bbduk

        # # Apply entropy masking
        bbduk(
            in1=last_file,
            out=f"{tmpdir}/tmp_masked_mle.fasta",
            entropy=0.4,
            entropyk=4,
            entropywindow=24,
            maskentropy=True,
            ziplevel=9,
        )
        last_file = f"{tmpdir}/tmp_masked_mle.fasta"
        logger.info(f"Finished entropy masking step")

    if flatten:
        from bbmapy import kcompress

        kcompress(
            in_file=last_file,
            out=f"{tmpdir}/tmp_masked_mle_flat.fa",
            fuse=2000,
            k=31,
            prealloc="true",
            overwrite="true",
            threads=threads,
            Xmx=memory,
        )
        last_file = f"{tmpdir}/tmp_masked_mle_flat.fa"

    os.rename(f"{last_file}", output_file)  # this is like mv i think...
    shutil.rmtree("ref", ignore_errors=True)
    shutil.rmtree(str(tmpdir), ignore_errors=True)

    logger.info(f"Masking completed. Output saved to {output_file}")
