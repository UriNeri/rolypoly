import os
import shutil
from pathlib import Path

import mappy as mp
import rich_click as click
from bbmapy import bbmap, bbmask, kcompress
# from rich.console import Console

from rolypoly.utils.bio.alignments import calculate_percent_identity
from rolypoly.utils.bio.interval_ops import mask_sequence_mp, mask_nuc_range
from rolypoly.utils.various import ensure_memory, run_command_comp #TODO: Replace sp.run with run_command_comp.
from rolypoly.utils.logging.loggit import get_logger
global datadir
datadir = Path(
    os.environ.get("ROLYPOLY_DATA", "")
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
@click.option("-a", "--aligner", required=False,default="diamond", help="Which tool to use for identifying shared sequence (minimap2, mmseqs2, diamond, bowtie1, bbmap)")
@click.option(
    "-r",
    "--reference",
    default=datadir / "contam/masking/combined_entropy_masked.fasta",
    help="Provide an input fasta file to be used for masking, instead of the pre-generated collection of RNA viral sequences",
)
def mask_dna(
    threads, memory, output, flatten, input, aligner, reference
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

    Returns:
      None
    """
    import subprocess as sp
    logger = get_logger()

    input_file = Path(input).resolve()
    output_file = Path(output).resolve()
    aligner = str(aligner).lower()
    if aligner not in ["minimap2", "mmseqs2", "diamond", "bowtie1", "bbmap"]:
        logger.error(f"{aligner} not recognised as one of minimap2, mmseqs2, diamond, bowtie1 or bbmap")
        exit
    memory = ensure_memory(memory)["giga"]
    reference = Path(reference).absolute().resolve()
    tmpdir = str(output_file.parent) + "/tmp"

    try:    
        Path.mkdir(Path(tmpdir), exist_ok=True)
    except:
        exit(123) # one day, figure out what error codes to return...

    if aligner == "minimap2":
        logger.info("Using minimap2 (low memory mode)")

        # Create a mappy aligner object
        mpaligner = mp.Aligner(
            str(reference), k=11, n_threads=threads, best_n=15000,
        )
        if not mpaligner:
            raise Exception("ERROR: failed to load/build index")

        # Perform alignment, write results to SAM file, and mask sequences
        masked_sequences = {}
        for name, seq, qual in mp.fastx_read(str(input_file)):
            masked_sequences[name] = seq
            for hit in mpaligner.map(seq):
                percent_id = calculate_percent_identity(hit.cigar_str, hit.NM) # this make some assumptions
                logger.info(f"{percent_id}")
                if percent_id > 70:
                    masked_sequences[name] = mask_sequence_mp(
                        masked_sequences[name], hit.q_st, hit.q_en, hit.strand
                    )

        # Write masked sequences to output file
        with open(output_file, "w") as out_f:
            for name, seq in masked_sequences.items():
                out_f.write(f">{name}\n{seq}\n")
        logger.info(
            f"Masking completed. Output saved to {output_file}"
        )
        shutil.rmtree(f"{tmpdir}", ignore_errors=True)
        return # no sam file/bbmask needed

    elif aligner=="bowtie1":
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

    elif aligner=="mmseqs2":
        logger.info(
            "Note! using mmseqs2 instead of bbmap is not a tight drop in replacement."
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
            "--headers-split-mode",
            "1",
            "--format-mode",
            "1",
        ]
        sp.run(mmseqs_search_cmd, check=True)

    elif aligner=="diamond":
        logger.info(
            "Note! using diamond blastx - NOTE - SWITCHING TO A PROTEIN SEQ instead of default REFERENCE"
        )
        reference = reference if str(reference) != str(datadir / "contam/masking/combined_entropy_masked.fasta") else str(datadir / "contam/masking/combined_deduplicated_orfs.faa")
        logger.info(
            f"Note! using as reference: {reference} "
        )
        test = run_command_comp(
            assign_operator=" ",
            base_cmd="diamond blastx",
            positional_args=["qseqid qstart qend qstrand"],
            positional_args_location="end",
            param_sep=" ",
            params={
                "query":str(input_file),
                "db":str(reference),
                "out":f"{tmpdir}/tmp_mapped.tsv",
                "id": "70",
                "subject-cover": "40",
                "min-query-len": "20",
                "threads": threads,
                "max-target-seqs":123123123,
                "outfmt": "6"
            }
        )
        # dimamond_search_cmd = [
        #     "diamond",
        #     "blastx",
        #     "--query",
        #     str(input_file),
        #     "--db",
        #     str(reference),
        #     "--out",
        #     f"{tmpdir}/tmp_mapped.tsv",
        #     "--id",
        #     "70",
        #     "--subject-cover",
        #     "40",
        #     "--min-query-len",
        #     "20",
        #     "--threads",
        #     str(threads),
        #     "--max-target-seqs",
        #     "10000000",
        #     "--unal",
        #     "0",
        #     "--outfmt",
        #     "6 "
        # ]
        # logger.info(f"Running command: {' '.join(dimamond_search_cmd)}")
        # sp.run(dimamond_search_cmd, check=True, shell=True)
        
        mask_nuc_range(
            input_fasta=str(input_file),
            input_table=f"{tmpdir}/tmp_mapped.tsv",
            output_fasta=output_file)
        return


    elif aligner == "bbmap":
        logger.info("Using bbmap.sh")
        bbmap(
            ref=input_file,
            in_file=reference,
            outm=f"{tmpdir}/tmp_mapped.sam",
            minid=0.7,
            overwrite="true",
            threads=threads,
            Xmx=memory,
            simd="true"
        )
    
    logger.info(f"Finished running aligner {aligner}")

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
    logger.info(
        f"Masking completed. Output saved to {output_file}"
    )
