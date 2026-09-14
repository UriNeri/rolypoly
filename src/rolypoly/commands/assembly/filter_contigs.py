import os
import shutil
import subprocess

import polars as pl
from pathlib import Path

import rich_click as click

from rolypoly.utils.bio.interval_ops import inclusive_union_length, normalize_oriented_interval
from rolypoly.utils.bio.sequences import retain_contigs
from rolypoly.utils.bio.translation import translation_records
from rolypoly.utils.cli_options import shared_command_context
from rolypoly.utils.logging.config import BaseConfig

# TODO: cleaning assembly graph directly? by mmseqs nucleic / diamond amino searching against user supplied host sequence
# TODO: precompiled contamination DB? Masked RefSeq?
# TODO: replace all the subprocess calls with the run_command_comp.


global tools
tools = []


class FilterContigsConfig(BaseConfig):
    # initialize the BaseConfig class
    def __init__(self, **kwargs):
        output_path = Path(kwargs.get("output", "filtered_contigs.fasta"))
        # in this case output_dir and output are NOT the same, so we only explicitly make sure output_dir exists, and just "touch" the output file.
        if not output_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()

        super().__init__(
            input=kwargs.get("input", ""),
            output=output_path.parent,
            keep_tmp=kwargs.get("keep_tmp", False),
            log_file=kwargs.get("log_file", "filter_contigs_log.txt"),
            threads=kwargs.get("threads", 1),
            memory=kwargs.get("memory"),
            config_file=kwargs.get("config_file", None),
            overwrite=kwargs.get("overwrite", False),
            temp_dir=kwargs.get("temp_dir", None),
            log_level=kwargs.get("log_level", "INFO"),
        )

        self.output = output_path
        self.output_dir = output_path.parent

        # initialize the rest of the parameters (i.e. the ones that are not in the BaseConfig class)
        self.host = Path(kwargs["host"]).resolve() if kwargs.get("host") else None
        self.flag_only = kwargs.get("flag_only", False)
        self.rrna = kwargs.get("rrna", False)
        self.rrna_db = kwargs.get("rrna_db") or Path(os.environ.get("ROLYPOLY_DATA", "")) / "rrna/rrna.cm"
        self.rrna_min_fraction = kwargs.get("rrna_min_fraction", 0.8)
        self.filter_evidence = []
        self.evidence_output = output_path
        self.mode = kwargs.get("mode", "both")
        self.dont_mask = kwargs.get("dont_mask", False)
        self.filter1_nuc = kwargs.get(
            "filter1_nuc", "alnlen >= 120 & pident>=75"
        )
        self.filter2_nuc = kwargs.get(
            "filter2_nuc", "qcov >= 0.95 & pident>=95"
        )
        self.mmseqs_args = kwargs.get(
            "mmseqs_args", "--min-seq-id 0.5 --min-aln-len 80"
        )
        self.filter1_aa = kwargs.get("filter1_aa", "length >= 80 & pident>=75")
        self.filter2_aa = kwargs.get("filter2_aa", "qcovhsp >= 95 & pident>=80")
        self.diamond_args = kwargs.get("diamond_args", "--id 50 --min-orf 50")


@click.command(name="filter-contigs")
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
    required=False,
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
@click.option("--flag-only", is_flag=True, default=False,
              help="Retain matching contigs and record warning intervals instead of discarding them.")
@click.option("--rrna", is_flag=True, default=False,
              help="Opt in to an rRNA-only cmscan after host filtering (also works without --host).")
@click.option("--rrna-db", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Override the bundled rrna/rrna.cm database; requires --rrna.")
@click.option("--rrna-min-fraction", type=click.FloatRange(min=0, max=1, min_open=True), default=0.8, show_default=True,
              help="Minimum contig fraction covered by accepted rRNA hits for removal; local hits are flagged. Ignored for removal with --flag-only.")
@click.option(
    "-ow",
    "--overwrite",
    is_flag=True,
    default=False,
    help="Do not overwrite the output directory if it already exists",
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
    temp_dir,
    flag_only=False,
    rrna=False,
    rrna_db=None,
    rrna_min_fraction=0.8,
):
    """
    Filter contigs against user-supplied host/contamination references.

    Depending on `--mode`, the command applies nucleotide filtering, protein
    filtering, or both, using two-stage rule sets (`filter1_*` and
    `filter2_*`) to retain likely non-host contigs.

    Host references can be masked first (default) unless `--dont-mask` is set.
    Use `--rrna` for optional rRNA screening after host filtering, or without a
    host reference. `--flag-only` retains matches from all enabled filters and
    records warning intervals for reports; it is disabled by default.
    """
    from rolypoly.utils.logging.citation_reminder import remind_citations
    from rolypoly.utils.logging.loggit import log_start_info

    output = Path(output).absolute().resolve()
    host = Path(known_dna).resolve() if known_dna else None
    if not host and not rrna:
        raise click.UsageError("Supply --known-dna/--host or enable --rrna")
    if rrna_db and not rrna:
        raise click.UsageError("--rrna-db requires --rrna")
    if output == Path(input).resolve():
        raise click.UsageError("Input and output FASTA must differ")
    if not output.parent.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
    config = FilterContigsConfig(
        input=Path(input).absolute().resolve(),
        host=host,
        flag_only=flag_only, rrna=rrna, rrna_db=rrna_db, rrna_min_fraction=rrna_min_fraction,
        output=Path(output).absolute().resolve(),
        threads=threads,
        log_file=Path(log_file)
        if log_file
        else Path(output).parent / "rolypoly_filter_contigs_log.txt",
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
        temp_dir=Path(temp_dir).absolute().resolve() if temp_dir else None,
    )

    log_start_info(config.logger, config.__dict__)

    config.logger.info(f"Starting contig filtering in {mode} mode")

    original_input = config.input
    final_output = config.output
    host_output = config.temp_dir / "host_filtered.fasta" if config.rrna else final_output
    config.output = host_output
    if config.host is None:
        retain_contigs(original_input, host_output)
    elif config.mode == "nuc":
        filter_contigs_nuc(config)
        tools.append("mmseqs")
    elif config.mode == "aa":
        filter_contigs_aa(config)
        tools.append("diamond")
    else:
        config.output = config.temp_dir / "filtered_contigs_nuc.fasta"
        filter_contigs_nuc(config)
        config.input = config.output
        config.output = host_output
        if config.input.stat().st_size:
            filter_contigs_aa(config)
        else:
            config.output.write_text("")
        tools.extend(["mmseqs", "diamond"])
    if config.rrna:
        rrna_filter(config, host_output, final_output)
        tools.append("Infernal")
    config.output = final_output
    if not config.rrna:
        Path(str(final_output)+".rrna.tblout").unlink(missing_ok=True)
    write_filter_evidence(config.filter_evidence, final_output)
    import json
    Path(str(final_output)+'.filter_run.json').write_text(json.dumps({
        'rrna': config.rrna, 'flag_only': config.flag_only,
        'rrna_min_fraction': config.rrna_min_fraction,
        'rrna_db': str(config.rrna_db) if config.rrna else None,
        'host': str(config.host) if config.host else None,
    }, indent=2)+'\n')

    if not config.keep_tmp:
        shutil.rmtree(config.temp_dir, ignore_errors=True)
    config.logger.info(
        f"Contig filtering completed. Final output saved to {config.output}"
    )
    if config.log_level != 10:
        with open(f"{config.log_file}", "a") as f_out:
            f_out.write(remind_citations(tools, return_bibtex=True) or "")


def filter_contigs_nuc(config: FilterContigsConfig):
    import subprocess

    import polars as pl
    import pyfastx
    from rolypoly.commands.reads.mask_dna import mask_dna
    from rolypoly.utils.bio.sequences import ensure_faidx
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
    if config.input.suffix.endswith((".faa", ".fasta", ".fas", ".fna", ".fa")):  # type: ignore - an initalized config.input is a path
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
            context = shared_command_context(
                mask_dna, ignore_unknown_options=True
            )
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
    result_file = config.temp_dir / f"{config.input.stem}_vs_host.tab"  # type: ignore - an initalized config.input is a path
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

    config.logger.info("Applying filters:")
    config.logger.info(f"Filter 1: {config.filter1_nuc}")
    filtered1 = apply_filter(y, config.filter1_nuc)
    config.logger.info(f"Filter 2: {config.filter2_nuc}")
    filtered2 = apply_filter(y, config.filter2_nuc)
    y_set = host_filter_evidence(config, filtered1, filtered2, "host_nucleotide")
    retain_contigs(config.input, config.output, () if config.flag_only else y_set)
    fa = pyfastx.Fasta(str(config.input))

    # Print filtering statistics
    total_sequences = len(fa)
    filtered_sequences = total_sequences if config.flag_only else total_sequences - len(y_set)
    percentage_filtered = (len(y_set) / total_sequences) * 100
    config.logger.info("%s %d matched sequences.", "Flagged" if config.flag_only else "Filtered", len(y_set))
    config.logger.info(
        f"Kept {filtered_sequences} sequences ({percentage_filtered:.2f}% matched)."
    )
    config.logger.info(
        f"Nucleotide filtering completed. Output saved to {config.output}"
    )

    # Clean up
    if not config.keep_tmp:
        shutil.rmtree(tmpdir, ignore_errors=True)
        shutil.rmtree(resdb, ignore_errors=True)
        if input_db != config.input:
            shutil.rmtree(input_db.parent, ignore_errors=True)  # type: ignore - an initalized input_db is a path
        if host_db != config.host:
            shutil.rmtree(host_db.parent, ignore_errors=True)  # type: ignore - an initalized host_db is a path
        result_file.unlink(missing_ok=True)


def filter_contigs_aa(config: FilterContigsConfig):
    import subprocess

    import polars as pl
    import pyfastx
    from bbmapy import (
        callgenes,  # TODO use something that can handle meta/euks better.
    )

    # from rolypoly.utils.bio.translation import
    from rolypoly.utils.bio.sequences import ensure_faidx, guess_fasta_alpha
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
                Path(os.environ.get("ROLYPOLY_DATA", ""))
                / "contam/masking/combined_deduplicated_orfs.faa.gz"
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
            with open(config.log_file, "a") as log_file:  # type: ignore
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
    with open(config.log_file, "a") as log_file:  # type: ignore
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
    y_set = host_filter_evidence(config, filtered1, filtered2, "host_protein")
    retain_contigs(config.input, config.output, () if config.flag_only else y_set)
    fa = pyfastx.Fasta(str(config.input))

    # Print filtering statistics
    total_sequences = len(fa)
    filtered_sequences = total_sequences if config.flag_only else total_sequences - len(y_set)
    percentage_filtered = (len(y_set) / total_sequences) * 100
    config.logger.info("%s %d matched sequences.", "Flagged" if config.flag_only else "Filtered", len(y_set))
    config.logger.info(
        f"Kept {filtered_sequences} sequences ({percentage_filtered:.2f}% matched)."
    )
    config.logger.info(
        f"Amino acid filtering completed. Output saved to {config.output}"
    )

    # Clean up
    if not config.keep_tmp:
        shutil.rmtree(tmpdir, ignore_errors=True)
        res_tab.unlink(missing_ok=True)


QC_SCHEMA = {
    'contig_id': pl.String, 'contig_length': pl.Int64, 'start': pl.Int64,
    'end': pl.Int64, 'strand': pl.String, 'kind': pl.String,
    'profile': pl.String, 'accession': pl.String, 'source': pl.String,
    'score': pl.Float64, 'evalue': pl.Float64, 'rule': pl.String,
    'action': pl.String, 'description': pl.String,
}


def write_filter_evidence(rows, output):
    pl.DataFrame(rows, schema=QC_SCHEMA).write_csv(str(output)+'.filter_hits.tsv', separator='\t')


def host_filter_evidence(config, first, second, kind):
    """Persist only alignments satisfying an existing rejection rule."""
    rows = []
    query_col = 'qheader' if kind == 'host_nucleotide' else 'qtitle'
    for table, rule in ((first, getattr(config, 'filter1_nuc' if kind == 'host_nucleotide' else 'filter1_aa')),
                        (second, getattr(config, 'filter2_nuc' if kind == 'host_nucleotide' else 'filter2_aa'))):
        for hit in table.iter_rows(named=True):
            lo, hi, direction = normalize_oriented_interval(hit['qstart'], hit['qend'],
                hit.get('qstrand'), descending_encodes_strand=True)
            rows.append(dict(contig_id=hit[query_col].split()[0], contig_length=int(hit['qlen']),
                start=lo, end=hi, strand='+' if direction == 1 else '-', kind=kind,
                profile=hit.get('theader', hit.get('sseqid', '')), accession='',
                source='MMseqs2' if kind == 'host_nucleotide' else 'DIAMOND blastx',
                score=float(hit.get('bits', hit.get('bitscore', 0))), evalue=float(hit['evalue']),
                rule=rule, action='flagged' if config.flag_only else 'removed',
                description='Host/contamination reference match; not proof of a chimera'))
    config.filter_evidence.extend(rows)
    return {r['contig_id'] for r in rows}


def rrna_filter(config, input_fasta, output):
    """Scan the rRNA-only CM set; remove only predominantly rRNA contigs by default."""
    from rolypoly.utils.various import read_cmscan_tblout

    records = list(translation_records(input_fasta))
    lengths = {header.split()[0]: len(seq) for header, seq in records}
    if len(lengths) != len(records):
        raise ValueError('Duplicate contig IDs in rRNA input')
    if not lengths:
        retain_contigs(input_fasta, output)
        return
    database = Path(config.rrna_db)
    if not database.is_file():
        raise FileNotFoundError(f'rRNA covariance models not found: {database}')
    if not all(Path(str(database)+suffix).exists() for suffix in ('.i1f','.i1i','.i1m','.i1p')):
        local = config.temp_dir/'rrna.cm'
        shutil.copyfile(database, local)
        subprocess.run(['cmpress', str(local)], check=True)
        database = local
    raw = Path(str(config.evidence_output)+'.rrna.tblout')
    subprocess.run(['cmscan', '--cpu', str(config.threads), '--cut_ga',
                    '--tblout', str(raw), '-o', str(config.temp_dir/'rrna.log'),
                    str(database), str(input_fasta)], check=True)
    # Infernal writes comments only for a valid scan with no accepted hits.
    # The shared parser cannot infer an empty CSV; distinguish this from failure.
    with raw.open() as handle:
        has_hits = any(line.strip() and not line.startswith("#") for line in handle)
    if not has_hits:
        retain_contigs(input_fasta, output)
        return
    hits = read_cmscan_tblout(raw)
    rows = []
    for hit in hits.iter_rows(named=True):
        if hit['inc'] != '!':
            continue
        parent = hit['query_name']
        lo, hi, direction = normalize_oriented_interval(hit['seq_from'], hit['seq_to'], hit['strand'], descending_encodes_strand=True)
        if parent not in lengths or not 1 <= lo <= hi <= lengths[parent]:
            raise ValueError(f'Invalid rRNA coordinates: {parent}:{lo}-{hi}')
        rows.append(dict(contig_id=parent, contig_length=lengths[parent], start=lo, end=hi,
            strand='+' if direction == 1 else '-', kind='rRNA', profile=hit['target_name'],
            accession=hit['target_accession'], source='cmscan rrna.cm', score=hit['score'],
            evalue=hit['e_value'], rule='Rfam model gathering threshold (--cut_ga)',
            action='flagged', description=hit['description']))
    removed = rrna_removal_candidates(rows, config.rrna_min_fraction)
    for row in rows:
        if row['contig_id'] in removed and not config.flag_only:
            row['action'] = 'removed'
    config.filter_evidence.extend(rows)
    retain_contigs(input_fasta, output, () if config.flag_only else removed)


def rrna_removal_candidates(rows, min_fraction):
    """Union coverage across models and strands, avoiding double-counted overlaps."""
    spans = {}
    for row in rows:
        spans.setdefault(row['contig_id'], []).append(row)
    removed = set()
    for parent, hits in spans.items():
        covered = inclusive_union_length((hit['start'], hit['end']) for hit in hits)
        if covered / hits[0]['contig_length'] >= min_fraction:
            removed.add(parent)
    return removed
