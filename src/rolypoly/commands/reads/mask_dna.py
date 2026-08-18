import logging
import os
import shutil
from pathlib import Path

import rich_click as click

# from rich.console import Console
from rolypoly.utils.bio.alignments import calculate_percent_identity
from rolypoly.utils.bio.interval_ops import mask_nuc_range, mask_sequence_mp
from rolypoly.utils.logging.loggit import get_logger, setup_logging
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
    help="Which tool to use for identifying shared sequence (minimap2, mmseqs2, diamond,  bbmap)",
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
    default=datadir / "contam/masking/combined_entropy_masked.fasta.gz",
    help="Provide an input fasta file to be used for masking, instead of the pre-generated collection of RNA viral sequences",
)
@click.option(
    "-drvh",
    "--dont-remove-viral-headers",
    is_flag=True,
    default=False,
    help="Whether to remove entried from the fasta input with 'virus' sounding headers (sequence names/ids in the fasta input).",
)
@click.option(
    "-dcs",
    "--dont-clean-spaces-from-input",
    is_flag=True,
    default=False,
    help="removed everything after the first space in the fasta header (sequence name/id). USEFUL AS mmseqs sam output doesn't seem to retain that, and then bbnmask won't find the seqid without the space...",
)
@click.option(
    "--tmpdir",
    default=None,
    help="Temporary directory to use (default: output file's parent/tmp - if you have enough RAM, you can set this to /dev/shm/ or /tmp/ for faster I/O)",
)
@click.option(
    "-ll", "--log-level", hidden=True, default="INFO", help="Log level"
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
    dont_remove_viral_headers,
    dont_clean_spaces_from_input,
    tmpdir,
    log_level,
):
    """Mask an input fasta file for sequences that could be RNA viral (or mistaken for such).

    Args:
      threads: (int) Number of threads to use
      memory: (str) Memory in GB
      output: (str) Output file name
      flatten: (bool) Attempt to kcompress.sh the masked file
      input: (str) Input fasta file
      aligner: (str) Which tool to use for identifying shared sequence (minimap2, mmseqs2, diamond,  bbmap)
      reference: (str) Provide an input fasta file to be used for masking, instead of the pre-generated collection of RNA viral sequences
      mask_low_complexity: (bool) Whether to mask low complexity regions using bbduks entropy masking

    Returns:
      None
    """
    # Reuse existing root logger configuration when invoked from another
    # rolypoly command (e.g., filter_reads) to avoid switching log files.
    if not logging.getLogger().handlers:
        setup_logging(None, log_level)
    logger = get_logger()
    logger.debug(f"datadir used: {datadir}")

    input_file = Path(input).resolve()
    output_file = Path(output).resolve()
    aligner = str(aligner).lower()
    if aligner not in ["minimap2", "mmseqs2", "diamond", "bbmap"]:
        logger.error(
            f"{aligner} not recognised as one of minimap2, mmseqs2, diamond or bbmap"
        )
        exit
    needs_bbmask_only = aligner in ["bbmap", "mmseqs2"]
    memory = ensure_memory(memory)["giga"]
    reference = Path(reference).expanduser().resolve()
    if tmpdir is None:
        tmpdir = output_file.parent / "tmp_mask_dna"
    tmpdir = Path(tmpdir).absolute().resolve()
    Path.mkdir(Path(tmpdir), exist_ok=True)
    sam_output = Path(tmpdir) / "tmp_mapped.sam"

    if not dont_remove_viral_headers:
        from rolypoly.utils.bio.sequences import filter_fasta_by_headers

        no_viral_headers_file = tmpdir / "tmp_no_viral.fasta"

        counts = filter_fasta_by_headers(
            str(input_file),
            ["virus", "viral", "phage"],
            str(no_viral_headers_file),
            wrap=True,
            invert=True,
            return_counts=True,
        )
        logger.info(
            f"Filtered sequences: removed {counts['records_processed'] - counts['records_written']} sequences; {counts['records_written']} written, {counts['records_processed']} processed"
        )

        input_file = no_viral_headers_file
        logger.info(
            f"New input file after removing viral headers: {input_file}"
        )
        if not dont_clean_spaces_from_input:
            logger.info(
                "Also dropping everything after the first space in fasta headers"
            )
            from rolypoly.utils.bio.sequences import clean_fasta_headers

            headers_cleaned = tmpdir / "tmp_no_viral_cleaned.fasta"
            clean_fasta_headers(
                fasta_file=str(no_viral_headers_file),
                drop_from_space=True,
                output_file=str(headers_cleaned),
                strip_prefix="lcl|",
                strip_suffix="bla bla bla",
            )

            input_file = headers_cleaned
            logger.info(f"New input file after cleaning: {input_file}")

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
    elif aligner == "mmseqs2":
        # logger.info(
        #     "Note! using mmseqs2 instead of bbmap is not a tight drop in replacement."
        # )
        # v=1
        v = 3 if logger.level == "DEBUG" or logger.level == 10 else 1
        # v=3
        mmseqs_ok = True
        try:
            run_command_comp(
                assign_operator=" ",
                base_cmd="mmseqs easy-linsearch",
                check_output=True,
                output_file=str(sam_output),
                positional_args=[
                    str(reference),
                    str(input_file),
                    str(sam_output),
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
        except Exception as e:
            mmseqs_ok = False
            logger.warning("mmseqs2 mapping failed: %s", str(e))

        if (
            not mmseqs_ok
            or (not sam_output.exists())
            or sam_output.stat().st_size == 0
        ):
            logger.warning(
                "mmseqs2 did not produce a SAM file, falling back to bbmap for masking input"
            )
            from bbmapy import bbmap

            bbmap(
                ref=reference,
                in_file=input_file,
                outm=str(sam_output),
                minid=0.7,
                overwrite="true",
                threads=threads,
                Xmx=memory,
                simd="true",
            )
    elif aligner == "diamond":
        logger.info(
            "Note! using diamond blastx - NOTE - SWITCHING TO A PROTEIN SEQ instead of default REFERENCE"
        )
        reference = (
            reference
            if str(reference)
            != str(datadir / "contam/masking/combined_entropy_masked.fasta.gz")
            else str(datadir / "contam/masking/combined_deduplicated_orfs.faa.gz")
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
        logger.info("Finished diamond blastx step")
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
            ref=reference,
            in_file=input_file,
            outm=str(sam_output),
            minid=0.7,
            overwrite="true",
            threads=threads,
            Xmx=memory,
            simd="true",
        )

    logger.info(f"Finished running aligner {aligner}")
    logger.info("beginning bbmask (masking + entropy) step")

    if needs_bbmask_only:
        if not sam_output.exists() or sam_output.stat().st_size == 0:
            raise click.ClickException(
                f"Expected SAM file for bbmask was not produced: {sam_output}"
            )

        # Mask using the sam files, for aligners that need it...
        # approximate to bbmask.sh in=input.fasta out=output.fasta sam=mapped.sam overwrite=true threads=8 Xmx=16g
        logger.debug(
            f"equivalent to bbmask.sh in={input_file} out={tmpdir}/tmp_masked.fasta sam={sam_output} overwrite=true threads={threads} Xmx={memory}"
        )
        from bbmapy import bbmask

        bbmask(
            in_file=input_file,
            out=f"{tmpdir}/tmp_masked.fasta",
            sam=str(sam_output),
            overwrite="true",
            threads=threads,
            Xmx=memory,
        )
        logger.info("Finished bbmask step")

    last_file = f"{tmpdir}/tmp_masked.fasta"

    if mask_low_complexity:
        logger.info("Proceeding to entropy masking step")
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
        logger.info("Finished entropy masking step")

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
