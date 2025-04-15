import os
from pathlib import Path
from typing import Dict, Tuple, Union

import rich_click as click

from rolypoly.utils.config import BaseConfig
from rolypoly.utils.loggit import log_start_info

global tools
tools = []


class AssemblyConfig(BaseConfig):
    def __init__(self, **kwargs):
        # in this case output_dir and output are the same, so need to explicitly make sure it exists.
        if not Path(kwargs.get("output")).exists():
            kwargs["output_dir"] = kwargs.get("output")
            Path(kwargs.get("output")).mkdir(parents=True, exist_ok=True)

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
        self.assembler = kwargs.get("assembler")
        self.keep_tmp = kwargs.get("keep_tmp")
        self.step_params = {
            "spades": {
                "k": "21,33,45,57,63,69,71,83,95,103,107,111,119",
                "mode": "meta",
            },
            "megahit": {"k-min": 21, "k-max": 147, "k-step": 8, "min-contig-len": 30},
            "penguin": {"min-contig-len": 150, "num-iterations": "aa:1,nucl:12"},
            "seqkit": {},
            "bbwrap": {"maxindel": 200, "minid": 90, "untrim": True, "ambig": "best"},
            "bowtie": {},
        }
        self.skip_steps = (
            kwargs.get("skip_steps")
            if isinstance(kwargs.get("skip_steps"), list)
            else kwargs.get("skip_steps").split(",")
            if isinstance(kwargs.get("skip_steps"), str)
            else []
        )
        if kwargs.get("override_parameters") is not None:
            self.logger.info(
                f"override_parameters: {kwargs.get('override_parameters')}"
            )
            for step, params in kwargs.get("override_parameters").items():
                if step in self.step_params:
                    self.step_params[step].update(params)
                else:
                    self.logger.warning(
                        f"Warning: Unknown step '{step}' in override_parameters. Ignoring."
                    )


class LibraryInfo:
    def __init__(self):
        self.paired_end = {}  # {lib_num: (R1_path, R2_path)}
        self.single_end = {}  # {lib_num: path}
        self.merged = {}  # {lib_num: path}
        self.long_read = {}  # {lib_num: path}
        self.raw_fasta = []  # [paths]
        self.rolypoly_data = {}  # {lib_name: {'interleaved': path, 'merged': path}}

    def add_paired(self, lib_num: int, r1_path: str, r2_path: str):
        self.paired_end[lib_num] = (r1_path, r2_path)

    def add_single(self, lib_num: int, path: str):
        self.single_end[lib_num] = path

    def add_merged(self, lib_num: int, path: str):
        self.merged[lib_num] = path

    def add_long_read(self, lib_num: int, path: str):
        self.long_read[lib_num] = path

    def add_raw_fasta(self, path: str):
        self.raw_fasta.append(path)

    def add_rolypoly_data(
        self, lib_name: str, interleaved: str = None, merged: str = None
    ):
        if lib_name not in self.rolypoly_data:
            self.rolypoly_data[lib_name] = {"interleaved": None, "merged": None}
        if interleaved:
            self.rolypoly_data[lib_name]["interleaved"] = interleaved
        if merged:
            self.rolypoly_data[lib_name]["merged"] = merged

    def to_assembly_dict(self) -> dict:
        """Convert to format expected by assembly functions"""
        libraries = {}

        # Add rolypoly data first
        libraries.update(self.rolypoly_data)

        # Add other data types
        for lib_num, (r1, r2) in self.paired_end.items():
            lib_name = f"lib_{lib_num}_paired"
            libraries[lib_name] = {"interleaved": None, "merged": None}
            # Convert to interleaved format
            libraries[lib_name]["interleaved"] = (
                r1  # Will need to be interleaved during processing
            )

        for lib_num, path in self.merged.items():
            lib_name = f"lib_{lib_num}_merged"
            libraries[lib_name] = {"interleaved": None, "merged": path}

        for lib_num, path in self.single_end.items():
            lib_name = f"lib_{lib_num}_single"
            libraries[lib_name] = {"interleaved": None, "merged": path}

        return libraries


def handle_input_files(
    input_path: Union[str, Path], library_info: LibraryInfo = None
) -> Tuple[Dict, int]:
    """Process input files and identify libraries.

    Args:
        input_path: Path to input directory or file
        library_info: Optional pre-populated LibraryInfo object

    Returns:
        Tuple containing libraries dict and number of libraries
    """
    import re
    from pathlib import Path

    if library_info is None:
        library_info = LibraryInfo()

    input_path = Path(input_path)
    libraries = {}

    if input_path.is_dir():
        # Look for rolypoly output first
        rolypoly_files = list(input_path.glob("*_final_*.fq.gz"))
        if rolypoly_files:
            for file in rolypoly_files:
                lib_name = file.stem.split("_final_")[0]
                if "interleaved" in file.name:
                    library_info.add_rolypoly_data(lib_name, interleaved=str(file))
                elif "merged" in file.name:
                    library_info.add_rolypoly_data(lib_name, merged=str(file))

        # Look for other fastq files
        all_fastq = list(input_path.glob("*.f*q*"))
        r1_pattern = re.compile(r".*_R1[._].*")
        r2_pattern = re.compile(r".*_R2[._].*")

        # Group paired files
        r1_files = [f for f in all_fastq if r1_pattern.match(f.name)]
        for r1 in r1_files:
            r2 = r1.parent / r1.name.replace("_R1", "_R2")
            if r2.exists():
                lib_num = len(library_info.paired_end) + 1
                library_info.add_paired(lib_num, str(r1), str(r2))

        # Handle remaining files
        processed = set(r1_files)
        processed.update([f.parent / f.name.replace("_R1", "_R2") for f in r1_files])

        for file in all_fastq:
            if file not in processed:
                if any(x in file.name.lower() for x in ["merged", "single"]):
                    lib_num = len(library_info.merged) + 1
                    library_info.add_merged(lib_num, str(file))
                else:
                    lib_num = len(library_info.single_end) + 1
                    library_info.add_single(lib_num, str(file))

        # Handle raw fasta files
        fasta_files = list(input_path.glob("*.fa*"))
        for fasta in fasta_files:
            library_info.add_raw_fasta(str(fasta))

    else:
        # Single file input - treat as merged/single-end
        lib_name = input_path.stem.split("_final_")[
            0
        ]  # Handle rolypoly naming if present
        library_info.add_merged(1, str(input_path))
        libraries[f"lib_1_merged"] = {"interleaved": None, "merged": str(input_path)}

    # Convert library_info to the expected libraries format
    libraries = library_info.to_assembly_dict()

    return libraries, len(libraries)


def run_spades(config, libraries):
    import subprocess

    from rolypoly.utils.various import ensure_memory

    spades_output = config.output_dir / f"spades_meta_output"
    spades_cmd = f"spades.py --{config.step_params['spades']['mode']} -o {spades_output} --threads {config.threads} --only-assembler -k {config.step_params['spades']['k']} --phred-offset 33 -m {ensure_memory(config.memory)['bytes'][:-1]}"

    if len(libraries) > 9:
        config.logger.info(f"Running SPAdes on concatenated reads")
        with open(f"{config.output_dir}/all_merged.fq.gz", "wb") as outfile:
            for lib in libraries.values():
                if lib["merged"]:
                    with open(lib["merged"], "rb") as infile:
                        outfile.write(infile.read())
        with open(f"{config.output_dir}/all_interleaved.fq.gz", "wb") as outfile:
            for lib in libraries.values():
                if lib["interleaved"]:
                    with open(lib["interleaved"], "rb") as infile:
                        outfile.write(infile.read())
        spades_cmd += f" --pe-12 1 {config.output_dir}/all_interleaved.fq.gz --s 1 {config.output_dir}/all_merged.fq.gz"
    else:
        for i, (lib_name, lib) in enumerate(libraries.items(), 1):
            if lib["interleaved"]:
                spades_cmd += f" --pe-12 {i} {lib['interleaved']}"
            if lib["merged"]:
                if config.step_params["spades"]["mode"] == "meta":
                    # metaSPAdes only works with paired-end data, so switch to regular mode
                    spades_cmd = spades_cmd.replace("--meta", "")
                spades_cmd += f" --s {i} {lib['merged']}"

    subprocess.run(spades_cmd, shell=True, check=True)
    config.logger.info(f"Finished SPAdes assembly")

    return spades_output / "scaffolds.fasta"


def run_megahit(config, libraries):
    """Run MEGAHIT assembly."""
    import glob
    import subprocess

    from rolypoly.utils.various import ensure_memory

    config.logger.info(f"Started Megahit assembly")
    megahit_output = config.output_dir / "megahit_custom_out"

    interleaved = ",".join(
        str(lib["interleaved"]) for lib in libraries.values() if lib["interleaved"]
    )
    merged = ",".join(str(lib["merged"]) for lib in libraries.values() if lib["merged"])

    megahit_cmd = [
        f"megahit",
        f"--k-min {config.step_params['megahit']['k-min']}",
        f"--k-max {config.step_params['megahit']['k-max']}",
        f"--k-step {config.step_params['megahit']['k-step']}",
        f"--min-contig-len {config.step_params['megahit']['min-contig-len']}",
    ]
    if len(interleaved) > 0:
        megahit_cmd.extend([f"--12 {interleaved}"])
    if len(merged) > 0:
        megahit_cmd.extend([f"--read {merged}"])
    megahit_cmd.extend(
        [
            f"--out-dir {megahit_output}",
            f"--num-cpu-threads {config.threads} --memory {ensure_memory(config.memory)['bytes'][:-1]}",
        ]
    )
    config.logger.info(
        f"Running Megahit assembly with command: {' '.join(megahit_cmd)}"
    )
    subprocess.run(" ".join(megahit_cmd), shell=True, check=True)

    final_k = max(
        int(os.path.basename(file).split("k")[1].split(".")[0])
        for file in glob.glob(
            f"{megahit_output}/intermediate_contigs/*.final.contigs.fa"
        )
    )

    subprocess.run(
        f"megahit_toolkit contig2fastg {final_k} {megahit_output}/final.contigs.fa > "
        f"{megahit_output}/final_megahit_assembly_k{final_k}.fastg",
        shell=True,
        check=True,
    )

    return megahit_output / "final.contigs.fa"


def run_penguin(config, libraries):
    """Run Penguin assembler."""
    import subprocess

    config.logger.info(f"Started Penguin assembly")
    penguin_output = config.output_dir / "penguin_Fguided_1_nuclassemble_c0.fasta"
    interleaved = " ".join(
        str(lib["interleaved"]) for lib in libraries.values() if lib["interleaved"]
    )
    merged = " ".join(str(lib["merged"]) for lib in libraries.values() if lib["merged"])

    penguin_cmd = (
        f"penguin guided_nuclassemble {interleaved} {merged} "
        f"{penguin_output} ./tmp/ --min-contig-len {config.step_params['penguin']['min-contig-len']} "
        f"--contig-output-mode 0 --num-iterations {config.step_params['penguin']['num-iterations']} "
        f"--min-seq-id nucl:0.9,aa:0.99 --min-aln-len nucl:31,aa:150 "
        f"--clust-min-seq-id 0.99 --clust-min-cov 0.99 --threads {config.threads}"
    )
    subprocess.run(penguin_cmd, shell=True, check=True)
    return penguin_output


@click.command()
@click.option("-t", "--threads", default=1, help="Threads ")
@click.option(
    "-M",
    "--memory",
    default="6gb",
    help=" RAM limit  (more is betterer, see the docs for more info)",
)
@click.option(
    "-o",
    "--output",
    default="RP_assembly_output",
    help="Output path (folder will be created if it doesn't exist)",
)
@click.option(
    "-k", "--keep-tmp", is_flag=True, default=False, help="Keep temporary files"
)
@click.option(
    "-g",
    "--log-file",
    default=lambda: f"{os.getcwd()}/assemble_logfile.txt",
    help="Path to a logfile, should exist and be writable (permission wise)",
)
@click.option("-i", "--input", help="Input directory containing fastq files")
@click.option(
    "--paired-end",
    multiple=True,
    nargs=3,
    help="Library number and paired FASTQ files: <lib_num> <R1> <R2>",
)
@click.option(
    "--single-end",
    multiple=True,
    nargs=2,
    help="Library number and single-end FASTQ: <lib_num> <fastq>",
)
@click.option(
    "--merged",
    multiple=True,
    nargs=2,
    help="Library number and merged FASTQ: <lib_num> <fastq>",
)
@click.option(
    "--long-read",
    multiple=True,
    nargs=2,
    help="Library number and long read FASTQ: <lib_num> <fastq>",
)
@click.option("--raw-fasta", multiple=True, help="Raw FASTA file(s) to include")
@click.option(
    "-A",
    "--assembler",
    default="spades,megahit,penguin",
    help="Assembler choice. for multiple, give a comma-seperated list e.g. 'spades,penguin')",
)
@click.option(
    "-op",
    "--override-parameters",
    default="{}",
    help='JSON-like string of parameters to override. Example: --override-parameters \'{"spades": {"k": "21,33,55"}, "megahit": {"k-min": 31}}\'',
)
@click.option(
    "-ss",
    "--skip-steps",
    default="",
    help="Comma-separated list of steps to skip. Example: --skip-steps seqkit,bowtie",
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
def assembly(
    input=None,
    paired_end=None,
    single_end=None,
    merged=None,
    long_read=None,
    raw_fasta=None,
    **kwargs,
):
    """Assembly wrapper - takes in (presumably filtered) reads, and assembles them using one or more assemblers.
    Currently supported assemblers are:
    • SPAdes (metaSPAdes)
    • MEGAHIT
    • Penguin
    """
    import shutil

    import polars as pl
    from bbmapy import bbmap

    from rolypoly.utils.citation_reminder import remind_citations
    from rolypoly.utils.fax import process_sequences, read_fasta_df, rename_sequences
    from rolypoly.utils.various import run_command_comp

    if not kwargs.get("overwrite"):
        if Path(kwargs.get("output")).exists():
            raise ValueError(
                f"Output directory {kwargs.get('output')} already exists. Use -ow to overwrite."
            )
    else:
        shutil.rmtree(kwargs.get("output"), ignore_errors=True)

    Path(kwargs.get("output")).mkdir(parents=True, exist_ok=True)
    # print(kwargs.get("output"))

    config = AssemblyConfig(
        input=Path(input),
        output=Path(kwargs.get("output")),
        threads=kwargs.get("threads"),
        log_file=Path(kwargs.get("log_file")),
        memory=(kwargs.get("memory")),
        assembler=kwargs.get("assembler"),
        keep_tmp=kwargs.get("keep_tmp"),
        override_params=(kwargs.get("override_parameters")),
        skip_steps=(kwargs.get("skip_steps")),
        log_level=kwargs.get("log_level"),
    )

    config.logger.info(f"Starting assembly process    ")
    log_start_info(config.logger, config_dict=config.__dict__)
    config.logger.info(f"Saving config to {config.output_dir / 'assembly_config.json'}")
    config.save(config.output_dir / "assembly_config.json")

    library_info = LibraryInfo()

    # Handle explicit library specifications
    if paired_end:
        for lib_num, r1, r2 in paired_end:
            library_info.add_paired(int(lib_num), r1, r2)
    if single_end:
        for lib_num, path in single_end:
            library_info.add_single(int(lib_num), path)
    if merged:
        for lib_num, path in merged:
            library_info.add_merged(int(lib_num), path)
    if long_read:
        for lib_num, path in long_read:
            library_info.add_long_read(int(lib_num), path)
    if raw_fasta:
        for path in raw_fasta:
            library_info.add_raw_fasta(path)

    # Process input directory if provided
    if input:
        libraries, n_libraries = handle_input_files(input, library_info)
    else:
        libraries = {}
        for lib_name, data in library_info.rolypoly_data.items():
            libraries[lib_name] = data
        n_libraries = len(libraries)

    config.logger.info(f"Found {n_libraries} libraries")
    config.logger.info(f"Libraries: {libraries}")
    contigs4eval = []

    if "spades" in config.assembler.lower() and "spades" not in config.skip_steps:
        contigs4eval.append(run_spades(config, libraries))
        tools.append("spades")
    if "megahit" in config.assembler.lower() and "megahit" not in config.skip_steps:
        contigs4eval.append(run_megahit(config, libraries))
        tools.append("megahit")
    if "penguin" in config.assembler.lower() and "penguin" not in config.skip_steps:
        contigs4eval.append(run_penguin(config, libraries))
        tools.append("penguin")

    # First concatenate and rename all contigs
    if len(contigs4eval) > 0:
        # Concatenate all contigs into one file
        concat_file = str(config.output_dir / "all_contigs.fasta")
        config.logger.info(
            f"Concatenating {len(contigs4eval)} contig files into {concat_file}"
        )
        with open(concat_file, "w") as outfile:
            for contig_file in contigs4eval:
                with open(str(contig_file), "r") as infile:
                    outfile.write(infile.read())

        try:
            # Rename sequences
            config.logger.info("Reading and parsing FASTA file")
            df = read_fasta_df(concat_file)
            config.logger.info(f"Found {len(df)} sequences")

            config.logger.info("Renaming sequences")
            df_renamed, id_map = rename_sequences(df, prefix="CID", use_hash=False)
            config.logger.info("Calculating sequence statistics")
            df_renamed = process_sequences(df_renamed)

            # Write renamed sequences
            renamed_file = str(config.output_dir / "all_contigs_renamed.fasta")
            config.logger.info(f"Writing renamed sequences to {renamed_file}")
            with open(renamed_file, "w") as f:
                for header, seq in zip(df_renamed["header"], df_renamed["sequence"]):
                    f.write(f">{header}\n{seq}\n")

            # Update contigs4eval to use renamed file
            contigs4eval = [renamed_file]

            # Save mapping file
            mapping_file = str(config.output_dir / "contigs_id_map.tsv")
            config.logger.info(f"Saving ID mapping to {mapping_file}")
            mapping_df = pl.DataFrame(
                {
                    "old_id": list(id_map.keys()),
                    "new_id": list(id_map.values()),
                    "length": df_renamed["length"],
                    "gc_content": df_renamed["gc_content"].round(2),
                }
            )
            mapping_df.write_csv(mapping_file, separator="\t")

        except Exception as e:
            config.logger.error(f"Error during sequence renaming: {str(e)}")
            config.logger.warning("Continuing with original contig files")
            # Keep original contigs4eval if renaming fails

    # Deduplication step # TODO: add as optional a linclust step.
    if "seqkit" not in config.skip_steps and len(contigs4eval) > 0:
        tools.append("seqkit")
        dedup_output = str(config.output_dir / "rmdup_contigs.fasta")
        run_command_comp(
            "seqkit rmdup",
            positional_args=[str(contigs4eval[0])],
            positional_args_location="end",
            params={
                "by-seq": True,  # Use sequence for deduplication
                "line-width": "0",
                "threads": str(config.threads),
                "out-file": dedup_output,
                "dup-num-file": str(config.output_dir / "Redundancy_lookup.txt"),
            },
            logger=config.logger,
            prefix_style="double",
        )
        config.logger.info(f"Finished deduplicating: {contigs4eval}")

        # Verify dedup output exists before proceeding
        if not os.path.exists(dedup_output) or os.path.getsize(dedup_output) == 0:
            config.logger.error(
                f"Deduplication failed: {dedup_output} not found or empty"
            )
            return

    # Map reads back to contigs using either bbmap_skimmer (default) or bowtie (low-mem)
    if os.path.exists(str(config.output_dir / "rmdup_contigs.fasta")):
        interleaved = ",".join(
            str(lib["interleaved"]) for lib in libraries.values() if lib["interleaved"]
        )
        merged = ",".join(
            str(lib["merged"]) for lib in libraries.values() if lib["merged"]
        )

        # Use bbmap_skimmer by default
        if "bbmap" not in config.skip_steps:
            tools.append("bbmap")
            config.logger.info("Running bbmap_skimmer for read mapping")

            # Combine all input reads
            input_reads = []
            if interleaved:
                input_reads.extend(interleaved.split(","))
            if merged:
                input_reads.extend(merged.split(","))

            bbmap(
                ref=str(config.output_dir / "rmdup_contigs.fasta"),
                in_file=",".join(input_reads),
                out=str(config.output_dir / "assembly_bbmap.sam"),
                threads=str(config.threads),
                Xmx=str(config.memory["giga"]),
                ignorefrequentkmers="f",
                vslow=True,
                maxsites="1500",
                maxsites2="1500",
                sam="1.4",
                minid="0.8",
                nodisk=True,
                ambiguous="all",
                overwrite="t",
                secondary=True,
            )

            # Compress SAM file
            if os.path.exists(str(config.output_dir / "assembly_bbmap.sam")):
                run_command_comp(
                    "pigz",
                    params={"p": str(config.threads)},
                    positional_args=[str(config.output_dir / "assembly_bbmap.sam")],
                    logger=config.logger,
                    prefix_style="single",
                )

        # Use bowtie as low-memory alternative
        elif "bowtie" not in config.skip_steps:
            tools.append("bowtie")
            config.logger.info("Running bowtie (low-memory mode) for read mapping")

            bowtie_index = str(config.output_dir / "bowtie_index")
            os.makedirs(bowtie_index, exist_ok=True)

            # Build bowtie index
            index_success = run_command_comp(
                "bowtie-build",
                positional_args=[
                    str(config.output_dir / "rmdup_contigs.fasta"),
                    str(config.output_dir / "bowtie_index/contigs"),
                ],
                params={"threads": str(config.threads)},
                logger=config.logger,
                prefix_style="double",
            )

            if index_success:
                try:
                    if len(interleaved) > 0:
                        # Align paired-end interleaved reads
                        align_success = run_command_comp(
                            "bowtie",
                            params={
                                "p": str(config.threads),
                                "S": True,
                                "x": str(config.output_dir / "bowtie_index/contigs"),
                            },
                            positional_args=[
                                "--12",
                                interleaved,
                                str(
                                    config.output_dir
                                    / "assembly_bowtie_interleaved.sam"
                                ),
                            ],
                            logger=config.logger,
                            prefix_style="single",
                        )
                        if align_success and os.path.exists(
                            str(config.output_dir / "assembly_bowtie_interleaved.sam")
                        ):
                            run_command_comp(
                                "pigz",
                                params={"p": str(config.threads)},
                                positional_args=[
                                    str(
                                        config.output_dir
                                        / "assembly_bowtie_interleaved.sam"
                                    )
                                ],
                                logger=config.logger,
                                prefix_style="single",
                            )

                    if len(merged) > 0:
                        # Align single-end/merged reads
                        align_success = run_command_comp(
                            "bowtie",
                            params={
                                "p": str(config.threads),
                                "S": True,
                                "x": str(config.output_dir / "bowtie_index/contigs"),
                            },
                            positional_args=[
                                merged,
                                str(
                                    config.output_dir
                                    / "assembly_bowtie_merged_reads.sam"
                                ),
                            ],
                            logger=config.logger,
                            prefix_style="single",
                        )
                        if align_success and os.path.exists(
                            str(config.output_dir / "assembly_bowtie_merged_reads.sam")
                        ):
                            run_command_comp(
                                "pigz",
                                params={"p": str(config.threads)},
                                positional_args=[
                                    str(
                                        config.output_dir
                                        / "assembly_bowtie_merged_reads.sam"
                                    )
                                ],
                                logger=config.logger,
                                prefix_style="single",
                            )
                except Exception as e:
                    config.logger.warning(f"Failed to align reads to contigs: {e}")
            else:
                config.logger.error(
                    "Failed to build bowtie index, skipping alignment steps"
                )

    config.logger.info(f"Finished assembly evaluation on: {contigs4eval}")

    if not config.keep_tmp:
        files_to_remove = [
            "tmp",
            str(config.output_dir / "all_interleaved.fq.gz"),
            str(config.output_dir / "all_merged.fq.gz"),
        ]
        folders_to_remove = [
            str(config.output_dir / "megahit_custom_out/intermediate_contigs")
        ]
        for file in files_to_remove:
            if os.path.exists(file):
                if os.path.isdir(file):
                    shutil.rmtree(file, ignore_errors=True)
                else:
                    os.unlink(file)
        for folder in folders_to_remove:
            if os.path.exists(folder):
                shutil.rmtree(folder, ignore_errors=True)
        if os.path.exists(str(config.output_dir / "spades_meta_output")):
            for spades_folder in os.listdir(
                str(config.output_dir / "spades_meta_output")
            ):
                if Path(
                    str(config.output_dir / "spades_meta_output") / spades_folder
                ).is_dir():
                    shutil.rmtree(
                        str(config.output_dir / "spades_meta_output" / spades_folder)
                    )

    config.logger.info("Assembly process completed successfully.")
    config.logger.info(
        f"Final redundancy filtered contigs from the assemblers used are in {config.output_dir}/rmdup_contigs.fasta"
    )
    config.logger.info(
        f"Reads unassembled from the assembly are in {config.output_dir}/assembly_bbw_unassembled.fq.gz"
    )
    config.logger.info(
        f"Reads aligned to the assembly (interleaved and merged) are in {config.output_dir}/assembly_bowtie_interleaved.sam.gz and {config.output_dir}/assembly_bowtie_merged_reads.sam.gz"
    )

    with open(f"{config.log_file}", "w") as f_out:
        f_out.write(remind_citations(tools, return_bibtex=True))


if __name__ == "__main__":
    assembly()
