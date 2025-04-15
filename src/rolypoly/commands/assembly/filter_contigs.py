import os
import shutil
from pathlib import Path

import rich_click as click

from rolypoly.utils.config import BaseConfig

# TODO: cleaning assembly graph directly? by mmseqs nucleic / diamond amino searching against user supplied host sequence
# TODO: precompiled contamination DB? Masked RefSeq?

global tools
tools = []


class FilterContigsConfig(BaseConfig):
    def __init__(self, **kwargs):
        # in this case output_dir and output are NOT the same, so we only explicitly make sure output_dir exists, and just "touch" the output file.
        if not Path(kwargs.get("output")).exists():
            kwargs["output_dir"] = Path(kwargs.get("output")).parent
            Path(kwargs.get("output")).parent.mkdir(parents=True, exist_ok=True)
            Path(kwargs.get("output")).touch()

        super().__init__(
            input=kwargs.get("input"),
            output=kwargs.get("output"),
            keep_tmp=kwargs.get("keep_tmp"),
            log_file=kwargs.get("log_file"),
            threads=kwargs.get("threads"),
            memory=kwargs.get("memory"),
            config_file=kwargs.get("config_file"),
            overwrite=kwargs.get("overwrite"),
            log_level=kwargs.get("log_level"),
        )  # initialize the BaseConfig class
        # initialize the rest of the parameters (i.e. the ones that are not in the BaseConfig class)

        self.host = Path(kwargs.get("host")).absolute().resolve()
        self.mode = kwargs.get("mode") or "both"
        self.dont_mask = kwargs.get("dont_mask") or False
        self.filter1_nuc = kwargs.get("filter1_nuc") or "alnlen >= 120 & pident>=75"
        self.filter2_nuc = kwargs.get("filter2_nuc") or "qcov >= 0.95 & pident>=95"
        self.mmseqs_args = (
            kwargs.get("mmseqs_args") or "--min-seq-id 0.5 --min-aln-len 80"
        )
        self.filter1_aa = kwargs.get("filter1_aa") or "length >= 80 & pident>=75"
        self.filter2_aa = kwargs.get("filter2_aa") or "qcovhsp >= 95 & pident>=80"
        self.diamond_args = kwargs.get("diamond_args") or "--id 50 --min-orf 50"


@click.command(name="filter_contigs")
@click.option(
    "-i",
    "--input",
    required=True,
    type=click.Path(exists=True),
    help="Input path to fasta file",
)
@click.option(
    "-d",
    "--known-dna",
    "--host",
    required=True,
    type=click.Path(exists=True),
    help="Path to the user-supplied host/contamination fasta",
)
@click.option(
    "-o",
    "--output",
    default=os.getcwd() + "/filtered_contigs.fasta",
    help="Output file location. ",
)
@click.option(
    "-m",
    "--mode",
    type=click.Choice(["nuc", "aa", "both"]),
    default="both",
    help="Filtering mode: nucleotide, amino acid, or both (nuc / aa / both)",
)
@click.option("-t", "--threads", default=1, help="Number of threads")
@click.option("-M", "--memory", default="6g", help="Memory. Can be specified in gb ")
@click.option("--keep-tmp", is_flag=True, default=False, help="Keep temporary files")
@click.option("-g", "--log-file", default=None, help="Path to log file")
@click.option(
    "-Fm1",
    "--filter1_nuc",
    default="alnlen >= 120 & pident>=75",
    help="First set of rules for nucleic filtering by aligned stats",
)
@click.option(
    "-Fm2",
    "--filter2_nuc",
    default="qcov >= 0.95 & pident>=95",
    help="Second set of rules for nucleic match filtering",
)
@click.option(
    "-Fd1",
    "--filter1_aa",
    default="length >= 80 & pident>=75",
    help="First set of rules for amino (protein) match filtering",
)
@click.option(
    "-Fd2",
    "--filter2_aa",
    default="qcovhsp >= 95 & pident>=80",
    help="Second set of rules for protein match filtering",
)
@click.option(
    "--dont-mask",
    is_flag=True,
    help="If set, host fasta won't be masked for potential RNA virus-like seqs",
)
@click.option(
    "--mmseqs-args",
    default="--min-seq-id 0.5 --min-aln-len 80",
    help="Additional arguments for MMseqs2",
)
@click.option(
    "--diamond-args",
    default="--id 50 --min-orf 50",
    help="Additional arguments for Diamond",
)
@click.option(
    "-ow",
    "--overwrite",
    is_flag=True,
    default=False,
    help="Do not overwrite the output directory if it already exists",
)
@click.option(
    "-ll",
    "--log-level",
    default="info",
    hidden=True,
    help="Log level. Options: debug, info, warning, error, critical",
)
def filter_contigs(
    input,
    known_dna,
    output,
    mode,
    threads,
    memory,
    keep_tmp,
    log_file,
    filter1_nuc,
    filter2_nuc,
    filter1_aa,
    filter2_aa,
    dont_mask,
    mmseqs_args,
    diamond_args,
    overwrite,
    log_level,
):
    """
    Filter contigs based on user-supplied host sequences. Wrapper for both nucleotide and amino acid filtering.
    """
    from rolypoly.utils.citation_reminder import remind_citations
    from rolypoly.utils.loggit import log_start_info

    output = Path(output).absolute().resolve()
    host = Path(known_dna).absolute().resolve()
    if not output.parent.exists():
        print("asdasdasdasds")
        output.parent.mkdir(parents=True, exist_ok=True)
    print(output)
    config = FilterContigsConfig(
        input=Path(input).absolute().resolve(),
        host=Path(host).absolute().resolve(),
        output=Path(output).absolute().resolve(),
        threads=threads,
        log_file=Path(log_file)
        if log_file
        else Path(output).parent / f"rolypoly_filter_contigs_log.txt",
        memory=memory,
        mode=mode,
        keep_tmp=keep_tmp,
        overwrite=overwrite,
        log_level=log_level,
        dont_mask=dont_mask,
        filter1_nuc=filter1_nuc,
        filter2_nuc=filter2_nuc,
        mmseqs_args=mmseqs_args,
        filter1_aa=filter1_aa,
        filter2_aa=filter2_aa,
        diamond_args=diamond_args,
    )

    log_start_info(config.logger, config.__dict__)

    config.logger.info(f"Starting contig filtering in {mode} mode")

    if config.mode == "nuc":
        filter_contigs_nuc(config)
        tools.append("mmseqs")
        tools.append("pyfastx")
        tools.append("bbmap")

    elif config.mode == "aa":
        filter_contigs_aa(config)
        tools.append("diamond")
        tools.append("pyfastx")
        tools.append("bbmap")

    elif config.mode == "both":
        og_output = config.output
        config.output = config.temp_dir / "filtered_contigs_nuc.fasta"
        filter_contigs_nuc(config)
        config.input = config.temp_dir / "filtered_contigs_nuc.fasta"
        config.output = og_output
        filter_contigs_aa(config)
        tools.append("diamond")
        tools.append("mmseqs")
        tools.append("pyfastx")
        tools.append("bbmap")

    if not config.keep_tmp:
        shutil.rmtree(config.temp_dir, ignore_errors=True)
    config.logger.info(
        f"Contig filtering completed. Final output saved to {config.output}"
    )
    remind_citations(tools)
    with open(config.log_file, "w") as f_out:
        f_out.write(remind_citations(tools, return_as_text=True))


def filter_contigs_nuc(config: FilterContigsConfig):
    import subprocess

    import polars as pl
    import pyfastx
    from rich_click import Context

    from rolypoly.utils.fax import ensure_faidx, mask_dna
    from rolypoly.utils.various import apply_filter, ensure_memory

    config.logger.info(f"Started nucleotide host filtering for: {config.input}")

    # Ensure input and host fasta files are indexed
    ensure_faidx(str(config.input))
    ensure_faidx(str(config.host))

    # Create folders for MMseqs2 to use
    tmpdir = config.temp_dir / "tmp_nuc"
    resdb = config.temp_dir / "filter_assembly_mmdb"
    tmpdir.mkdir(parents=True, exist_ok=True)
    resdb.mkdir(parents=True, exist_ok=True)

    # Convert input to MMseqs2 DB if it's a fasta file
    input_db = config.input
    if config.input.suffix.endswith((".faa", ".fasta", ".fas", ".fna")):
        input_db = config.temp_dir / "contig_db" / "cmmdb"
        input_db.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "mmseqs",
                "createdb",
                str(config.input),
                str(input_db),
                "--dbtype",
                "2",
                "-v",
                "1",
            ],
            check=True,
        )

    # Process host file
    host_db = config.host
    if config.host.suffix.endswith((".faa", ".fasta", ".fas", ".fna", ".fa")):
        host_db = config.temp_dir / "host_db" / "dnammdb"
        host_db.parent.mkdir(parents=True, exist_ok=True)

        if not config.dont_mask:
            host_fasta = config.temp_dir / "masked_host.fasta"
            mask_args = {
                "threads": config.threads,
                "memory": ensure_memory(config.memory)["giga"],
                "output": host_fasta,
                "flatten": False,
                "input": config.host,
            }
            context = Context(mask_dna, ignore_unknown_options=True)
            context.invoke(mask_dna, **mask_args)
        else:
            host_fasta = config.host

        subprocess.run(
            [
                "mmseqs",
                "createdb",
                str(host_fasta),
                str(host_db),
                "--dbtype",
                "2",
                "-v",
                "1",
            ],
            check=True,
        )

    # Perform MMseqs2 search
    # config.logger.info(f"Searching against {host_db}")
    mmseqs_search_cmd = [
        "mmseqs",
        "search",
        str(input_db),
        str(host_db),
        f"{resdb}/res",
        str(config.temp_dir),
        "--threads",
        str(config.threads),
        "-a",
        "--search-type",
        "3",
        "-v",
        "1",
    ]
    config.logger.info(
        f"Running mmseqs2 search with command:  {' '.join(mmseqs_search_cmd)}"
    )
    mmseqs_search_cmd.extend(config.mmseqs_args.split())
    subprocess.run(mmseqs_search_cmd, check=True)

    # Convert results to desired format
    result_file = config.temp_dir / f"{config.input.stem}_vs_host.tab"
    config.logger.info(f"Converting results to desired format: {result_file}")
    subprocess.run(
        [
            "mmseqs",
            "convertalis",
            str(input_db),
            str(host_db),
            f"{resdb}/res",
            str(result_file),
            "--format-mode",
            "4",
            "--format-output",
            "qheader,theader,qlen,tlen,qstart,qend,tstart,tend,alnlen,mismatch,qcov,tcov,bits,evalue,gapopen,pident,nident",
            "-v",
            "1",
        ],
        check=True,
    )

    # Apply filters
    config.logger.info(f"reading results from {result_file}")
    y = pl.read_csv(result_file, separator="\t", has_header=True)
    if y.shape[0] == 0:
        config.logger.warning(
            f"No nucleic hits found for {config.input} against {config.host}, copying input to output"
        )
        shutil.copy(config.input, config.output)
        return

    config.logger.info(f"Applying filters:")
    config.logger.info(f"Filter 1: {config.filter1_nuc}")
    filtered1 = apply_filter(y, config.filter1_nuc)
    config.logger.info(f"Filter 2: {config.filter2_nuc}")
    filtered2 = apply_filter(y, config.filter2_nuc)
    y_set = set(filtered1["qheader"]).union(set(filtered2["qheader"]))

    # Write filtered sequences
    fa = pyfastx.Fasta(str(config.input))
    with open(config.output, "w") as out_file:
        for seq in fa:
            if seq.name not in y_set:
                out_file.write(f">{seq.name}\n{seq.seq}\n")

    # Print filtering statistics
    total_sequences = len(fa)
    filtered_sequences = total_sequences - len(y_set)
    percentage_filtered = (len(y_set) / total_sequences) * 100
    config.logger.info(f"Filtered {len(y_set)} sequences.")
    config.logger.info(
        f"Kept {filtered_sequences} sequences ({percentage_filtered:.2f}% filtered)."
    )
    config.logger.info(
        f"Nucleotide filtering completed. Output saved to {config.output}"
    )

    # Clean up
    if not config.keep_tmp:
        shutil.rmtree(tmpdir, ignore_errors=True)
        shutil.rmtree(resdb, ignore_errors=True)
        if input_db != config.input:
            shutil.rmtree(input_db.parent, ignore_errors=True)
        if host_db != config.host:
            shutil.rmtree(host_db.parent, ignore_errors=True)
        result_file.unlink(missing_ok=True)


def filter_contigs_aa(config: FilterContigsConfig):
    import subprocess

    import polars as pl
    import pyfastx
    from bbmapy import callgenes

    from rolypoly.utils.fax import ensure_faidx, guess_fasta_alpha
    from rolypoly.utils.various import apply_filter, ensure_memory

    config.logger.info(f"Started amino acid host filtering for: {config.input}")

    # Ensure input fasta file is indexed
    ensure_faidx(str(config.input))

    # Create folders for Diamond to use
    tmpdir = config.temp_dir / "tmp_aa"
    tmpdir.mkdir(parents=True, exist_ok=True)
    res_tab = config.temp_dir / "filter_assembly_aa_diamondout.tsv"

    # Process host file
    host_fasta = config.host
    if config.host.suffix.endswith((".faa", ".fasta", ".fas", ".fna", ".fa")):
        host_alpha = guess_fasta_alpha(config.host)
        if host_alpha == "nucl":
            host_fasta = tmpdir / "host_genes.fasta"
            callgenes(
                in_file=config.host,
                outa=host_fasta,
                threads=config.threads,
                overwrite="true",
                Xmx=ensure_memory(config.memory)["giga"],
            )
            subprocess.run(
                f"sed 's|\t|__|g' -i {str(host_fasta)}", check=True, shell=True
            )
        elif host_alpha == "amino":
            host_fasta = config.host
        else:
            config.logger.error(
                f"Can't guess the alphabet (doesn't look like nucl or amino fasta) of \n {config.host}"
            )
            return

        if not config.dont_mask:
            masked_fasta = config.temp_dir / "masked_host.fasta"
            rna_virus_prots = (
                Path(os.environ.get("ROLYPOLY_DATA")) / "prots_for_masking.faa"
            )
            diamond_mask_cmd = [
                "diamond",
                "blastp",
                "--query",
                str(host_fasta),
                "--db",
                str(rna_virus_prots),
                "--tmpdir",
                str(config.temp_dir),
                "--threads",
                str(config.threads),
                "--un",
                str(masked_fasta),
            ]
            diamond_mask_cmd.extend(config.diamond_args.split())
            diamond_mask_cmd.extend(
                [
                    "--header",
                    "simple",
                    "--out",
                    f"{config.temp_dir}/diamond_out_for_masking.tab",
                    "--outfmt",
                    "6",
                    "qseqid sseqid pident length mismatch gapopen qlen qstart qend sstart send slen evalue bitscore qcovhsp",
                ]
            )
            with open(config.log_file, "a") as log_file:
                subprocess.run(
                    " ".join(diamond_mask_cmd),
                    check=True,
                    shell=True,
                    stdout=log_file,
                    stderr=log_file,
                )
            host_fasta = masked_fasta

    # Perform the Diamond search
    config.logger.info(f"Searching against {host_fasta}")
    diamond_search_cmd = [
        "diamond",
        "blastx",
        "--query",
        str(config.input),
        "--db",
        str(host_fasta),
        "--out",
        str(res_tab),
        "--tmpdir",
        str(config.temp_dir),
        "--threads",
        str(config.threads),
    ]
    diamond_search_cmd.extend(config.diamond_args.split())
    diamond_search_cmd.extend(
        [
            "--header",
            "simple",
            "--outfmt",
            "6",
            "qtitle sseqid pident length mismatch gapopen qstart qend qlen sstart send slen evalue bitscore qstrand qframe qcovhsp",
        ]
    )
    with open(config.log_file, "a") as log_file:
        subprocess.run(
            " ".join(diamond_search_cmd),
            check=True,
            shell=True,
            stdout=log_file,
            stderr=log_file,
        )

    # Apply filters
    config.logger.info(f"reading results from {res_tab}")
    y = pl.read_csv(res_tab, separator="\t", has_header=True)
    if y.shape[0] == 0:
        config.logger.warning(
            f"No amino acid hits found for {config.input} against {config.host}, proceeding to copy paste the input as the output."
        )
        shutil.copy(config.input, config.output)
        return
    config.logger.info(f"Filter 1: {config.filter1_aa}")
    filtered1 = apply_filter(y, config.filter1_aa)
    config.logger.info(f"Filter 2: {config.filter2_aa}")
    filtered2 = apply_filter(y, config.filter2_aa)
    y_set = set(filtered1["qtitle"]).union(set(filtered2["qtitle"]))

    # Write filtered sequences
    fa = pyfastx.Fasta(str(config.input))
    with open(config.output, "w") as out_file:
        for seq in fa:
            if seq.name not in y_set:
                out_file.write(f">{seq.name}\n{seq.seq}\n")

    # Print filtering statistics
    total_sequences = len(fa)
    filtered_sequences = total_sequences - len(y_set)
    percentage_filtered = (len(y_set) / total_sequences) * 100
    config.logger.info(f"Filtered {len(y_set)} sequences.")
    config.logger.info(
        f"Kept {filtered_sequences} sequences ({percentage_filtered:.2f}% filtered)."
    )
    config.logger.info(
        f"Amino acid filtering completed. Output saved to {config.output}"
    )

    # Clean up
    if not config.keep_tmp:
        shutil.rmtree(config.temp_dir, ignore_errors=True)
        res_tab.unlink(missing_ok=True)
