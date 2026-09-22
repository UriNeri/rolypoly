import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import rich_click as click

from rolypoly.utils.logging.config import BaseConfig

# TODO: replace all the subprocess calls with the run_command_comp.
# TODO: figure out how to not require Console AND logging.


global tools
tools = []

# Keys: preset name → dict with keys:
#   assembler       – list of assembler IDs to run
#   spades_mode     – SPAdes mode override (only for the generic 'spades' assembler)
#   step_params     – per-tool param overrides, merged on top of AssemblyConfig defaults
#   dereplicate     – whether to deduplicate identical multi-assembler output
#   description     – human-readable summary shown in --help and logs
ASSEMBLY_PRESETS: dict[str, dict[str, Any]] = {
    "rna_virus": {
        # Recommended for RNA virus metatranscriptomics.
        # rnaviralSPAdes is tuned for RNA viruses; MEGAHIT provides complementary contigs.
        "assembler": ["spades_rnaviral", "megahit"],
        "step_params": {
            "spades": {"k": "21,33,55,77,99,121,127"},
            "megahit": {"k-min": 21, "k-max": 127, "k-step": 10},
        },
        "dereplicate": True,
        "description": (
            "RNA virus-focused: rnaviralSPAdes + MEGAHIT, broad k-mer range. "
            "Removes duplicate contigs (rmdup). Recommended for viral metatranscriptomes."
        ),
    },
    "metatranscriptome": {
        # For poly-A selected or mixed transcriptome libraries.
        # rnaSPAdes handles splice-aware assembly; MEGAHIT catches low-coverage transcripts.
        "assembler": ["spades", "megahit"],
        "spades_mode": "rna",
        "step_params": {"spades": {"k": "21,33,55,77,99,121,127"}},
        "dereplicate": True,
        "description": (
            "Metatranscriptome: rnaSPAdes + MEGAHIT, broad k-mer range. "
            "Suited for poly-A selected or mixed transcriptome libraries."
        ),
    },
    "fast": {
        # Optimised for speed: single assembler, narrow k-mer range, larger step size.
        # Suitable for quick previews and --mini runs in the roll pipeline.
        "assembler": ["megahit"],
        "step_params": {"megahit": {"k-min": 21, "k-max": 99, "k-step": 14}},
        "dereplicate": True,
        "description": (
            "Fast: MEGAHIT only, narrow k-mer range and larger step. "
            "Trades an unknown amount of sensitivity for an unknown amount of speed; suitable for quick previews or roll --mini runs."
        ),
    },
    "complete": {
        # Maximum sensitivity: all three assembler modes run in parallel.
        # metaSPAdes covers general metagenomics; rnaviralSPAdes targets RNA viruses;
        # MEGAHIT provides fast complementary contigs with a thorough k-mer sweep (not sure this is a good idea).
        "assembler": ["spades", "spades_rnaviral", "megahit"],
        "step_params": {
            "spades": {"k": "21,33,55,77,99,121,127"},
            "megahit": {"k-min": 21, "k-max": 127, "k-step": 11},
        },
        "dereplicate": True,
        "description": (
            "Complete: metaSPAdes + rnaviralSPAdes + MEGAHIT with thorough k-mer ranges. "
            "Different assemblers may produce better results - the onus of choice is on the user.  This will increase the runtime and memory usage significantly"
        ),
    },
    "metag": {
        # For DNA-based or mixed metagenomic libraries.
        # metaSPAdes only (default meta mode); no RNA-specific tuning, no MEGAHIT.
        "assembler": ["spades"],
        "step_params": {"spades": {"k": "21,33,55,77,99,121,127"}},
        "dereplicate": True,
        "description": (
            "Metagenomics: metaSPAdes (meta mode) only, broad k-mer range. "
            "Suited for DNA-based or mixed metagenomic libraries."
        ),
    },
}


def to_odd(values: list[int]) -> list[int]:
    """Convert a list of integers to odd integers by subtracting 1 from even numbers."""
    return [value if value % 2 == 1 else value - 1 for value in values]


def detect_average_read_length(libraries: dict, logger) -> float:
    from rolypoly.utils.bio.library_detection import determine_fastq_type

    # TODO: figure out if only the interleaved (non-merged) reads should be used to select the read length for kmer selection/capping.
    lengths: list[float] = []
    seen_paths: set[str] = set()
    for lib in libraries.values():
        for key in ("r1", "r2", "interleaved", "merged", "single"):
            file_path = lib.get(key)
            if not file_path:
                continue
            path_str = str(file_path)
            if path_str in seen_paths:
                continue
            seen_paths.add(path_str)
            try:
                analysis = determine_fastq_type(path_str, logger=logger)
                read_len = float(analysis.get("average_read_length", 0))
                if read_len > 0:
                    lengths.append(read_len)
            except Exception as error:
                logger.debug(
                    "Could not detect read length for %s: %s",
                    path_str,
                    str(error),
                )
    if not lengths:
        return 0.0
    return sum(lengths) / len(lengths)


def apply_assembly_preset(
    preset_name: str | None, ctx: click.Context, config: "AssemblyConfig"
) -> None:
    if not preset_name:
        return
    preset = ASSEMBLY_PRESETS.get(preset_name)
    if preset is None:
        raise click.BadParameter(
            f"Unknown preset '{preset_name}'. "
            f"Choose from: {', '.join(sorted(ASSEMBLY_PRESETS.keys()))}",
            param_hint="--preset",
        )

    explicit: set[str] = set()
    if ctx.params:
        for param in ctx.command.params:
            source = ctx.get_parameter_source(param.name)
            if source == click.core.ParameterSource.COMMANDLINE:
                explicit.add(param.name)

    if "assembler" in preset and "assembler" not in explicit:
        config.assembler = list(preset["assembler"])
    if "dereplicate" in preset and "dereplicate" not in explicit:
        config.dereplicate = bool(preset["dereplicate"])
    if (
        "spades_mode" in preset
        and "spades_mode" not in explicit
        and "mode" not in config.override_parameters.get("spades", {})
    ):
        config.step_params["spades"]["mode"] = preset["spades_mode"]
    for step_name, step_overrides in preset.get("step_params", {}).items():
        if step_name not in config.step_params:
            config.step_params[step_name] = {}
        if isinstance(step_overrides, dict):
            for name, value in step_overrides.items():
                if name not in config.override_parameters.get(step_name, {}):
                    config.step_params[step_name][name] = value

    config.logger.info(
        "Applied assembly preset '%s' (%s)",
        preset_name,
        preset.get("description", "no description"),
    )


def tune_assembly_kmers(config: "AssemblyConfig", observed_read_length: float) -> None:
    """Adapt default/preset k-mers to read length, preserving explicit overrides."""
    max_k = int(observed_read_length) - 1 if observed_read_length > 1 else None
    spades_kmers: list[int] = []
    seen_spades_kmers: set[int] = set()
    for kmer in to_odd(
        [
            int(k.strip())
            for k in str(config.step_params["spades"]["k"]).split(",")
            if str(k).strip()
        ]
    ):
        if kmer < 1 or kmer in seen_spades_kmers:
            continue
        seen_spades_kmers.add(kmer)
        if max_k is None or kmer < max_k:
            spades_kmers.append(kmer)
    if not spades_kmers and seen_spades_kmers:
        spades_kmers = [min(seen_spades_kmers)]
    if "k" not in config.override_parameters.get("spades", {}):
        config.step_params["spades"]["k"] = ",".join(
            str(kmer) for kmer in spades_kmers
        )
    if observed_read_length > 1:
        megahit_k_min = to_odd([int(config.step_params["megahit"]["k-min"])])[0]
        megahit_k_max = min(
            int(config.step_params["megahit"]["k-max"]), max_k - 1
        )
        megahit_k_max = max(to_odd([megahit_k_max])[0], megahit_k_min)
        megahit_k_step = int(config.step_params["megahit"]["k-step"])
        if megahit_k_step % 2 == 1:
            megahit_k_step = max(2, megahit_k_step - 1)
        for name, value in (
            ("k-min", megahit_k_min),
            ("k-max", megahit_k_max),
            ("k-step", megahit_k_step),
        ):
            if name not in config.override_parameters.get("megahit", {}):
                config.step_params["megahit"][name] = value
        config.logger.info(
            "Tuned non-overridden k-mer settings to read length %.2f (max_k=%s): spades_k=%s megahit_kmin=%s megahit_kmax=%s megahit_kstep=%s",
            observed_read_length,
            max_k,
            config.step_params["spades"]["k"],
            config.step_params["megahit"]["k-min"],
            config.step_params["megahit"]["k-max"],
            config.step_params["megahit"]["k-step"],
        )


class AssemblyConfig(BaseConfig):
    def __init__(self, **kwargs):
        # in this case output_dir and output are the same, so need to explicitly make sure it exists.
        if not Path(kwargs.get("output", "RP_assembly_output")).exists():
            kwargs["output_dir"] = kwargs.get("output", "RP_assembly_output")
            Path(kwargs.get("output", "RP_assembly_output")).mkdir(
                parents=True, exist_ok=True
            )
        # initialize the BaseConfig class
        super().__init__(
            input=kwargs.get("input", ""),
            output=kwargs.get("output", "RP_assembly_output"),
            keep_tmp=kwargs.get("keep_tmp", False),
            log_file=kwargs.get("log_file", "assemble_logfile.txt"),
            threads=kwargs.get("threads", 1),
            memory=kwargs.get("memory"),
            config_file=kwargs.get("config_file", None),
            temp_dir=kwargs.get("temp_dir", None),
            overwrite=kwargs.get("overwrite", False),
            log_level=kwargs.get("log_level", "info"),
        )
        # initialize the command specific stuff parameters
        self.assembler = kwargs.get("assembler", ["spades", "megahit"])
        self.dereplicate = kwargs.get("dereplicate", True)
        self.preset = kwargs.get("preset")
        self.raw_fasta = kwargs.get("raw_fasta", [])
        self.long_read_type = kwargs.get("long_read_type", "nanopore")

        self.step_params = {
            "spades": {
                "k": "21,33,45,57,69,83,95,103,115,127",  # DONE: figure out a way to smartly choose which kmers to use prior to main spades call.
                "mode": kwargs.get("spades_mode", "meta"),
            },
            "megahit": {
                "k-min": 21,
                "k-max": 147,
                "k-step": 8,
                "min-contig-len": 30,
            },
            "penguin": {
                "min-contig-len": 150,
                "num-iterations": "aa:1,nucl:12",
            },
            "seqkit": {},
            "mmseqs": {
                "min-seq-id": 0.99,
                "cov-mode": 1,
                "c": 0.99,
                "kmer-per-seq-scale": 0.4,
            },
        }
        skip_steps = kwargs.get("skip_steps") or []
        if isinstance(skip_steps, str):
            skip_steps = skip_steps.split(",")
        self.skip_steps = [step.strip() for step in skip_steps if step.strip()]
        raw_overrides = kwargs.get("override_parameters") or {}
        self.override_parameters = (
            json.loads(raw_overrides) if isinstance(raw_overrides, str) else raw_overrides
        )
        override_parameters = self.override_parameters
        if override_parameters:
            self.logger.info(f"override_parameters: {override_parameters}")
            for step, params in override_parameters.items():
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
        self, lib_name: str, interleaved: str = "", merged: str = ""
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

        # The same explicit library number associates pairs, merged reads,
        # and orphan reads without rewriting any FASTQ files.
        for lib_num in sorted(
            self.paired_end.keys() | self.merged.keys() | self.single_end.keys()
        ):
            lib = {
                "interleaved": None,
                "merged": self.merged.get(lib_num),
                "single": self.single_end.get(lib_num),
            }
            if lib_num in self.paired_end:
                lib["r1"], lib["r2"] = self.paired_end[lib_num]
            libraries[f"lib_{lib_num}"] = lib

        for lib_num, path in self.long_read.items():
            lib_name = f"lib_{lib_num}_long_read"
            libraries[lib_name] = {
                "interleaved": None,
                "merged": None,
                "long_read": path,
            }

        return libraries


def handle_input_files(
    input_path: Union[str, Path],
    library_info: LibraryInfo = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict, int]:
    """Process input files and identify libraries using consolidated file detection.

    Args:
        input_path: Path to input directory or file
        library_info: Optional pre-populated LibraryInfo object
        logger: Logger instance

    Returns:
        Tuple containing libraries dict and number of libraries
    """
    from rolypoly.utils.bio.library_detection import identify_fastq_files

    if library_info is None:
        library_info = LibraryInfo()

    if logger is None:
        import logging

        logger = logging.getLogger(__name__)

    input_path = Path(input_path)

    # Use consolidated file detection
    file_info = identify_fastq_files(
        input_path, return_rolypoly=True, logger=logger
    )

    # Process rolypoly data
    for lib_name, data in file_info["rolypoly_data"].items():
        if data["interleaved"]:
            library_info.add_rolypoly_data(
                lib_name, interleaved=str(data["interleaved"])
            )
        if data["merged"]:
            library_info.add_rolypoly_data(lib_name, merged=str(data["merged"]))

    # Process R1/R2 pairs
    for i, (r1_path, r2_path) in enumerate(file_info["R1_R2_pairs"], 1):
        lib_num = (
            max(
                [
                    0,
                    *library_info.paired_end,
                    *library_info.merged,
                    *library_info.single_end,
                ]
            )
            + 1
        )
        library_info.add_paired(lib_num, str(r1_path), str(r2_path))
        logger.debug(
            f"Added paired library {lib_num}: {r1_path.name} <-> {r2_path.name}"
        )

    # Process interleaved files
    for file_path in file_info["interleaved_files"]:
        lib_num = len(library_info.rolypoly_data) + 1
        library_info.add_rolypoly_data(
            f"detected_interleaved_{lib_num}", interleaved=str(file_path)
        )
        logger.debug(f"Added interleaved library {lib_num}: {file_path.name}")

    # Process single-end files
    for file_path in file_info["single_end"]:
        if any(x in file_path.name.lower() for x in ["merged", "single"]):
            lib_num = (
                max(
                    [
                        0,
                        *library_info.paired_end,
                        *library_info.merged,
                        *library_info.single_end,
                    ]
                )
                + 1
            )
            library_info.add_merged(lib_num, str(file_path))
            logger.debug(f"Added merged library {lib_num}: {file_path.name}")
        else:
            lib_num = (
                max(
                    [
                        0,
                        *library_info.paired_end,
                        *library_info.merged,
                        *library_info.single_end,
                    ]
                )
                + 1
            )
            library_info.add_single(lib_num, str(file_path))
            logger.debug(
                f"Added single-end library {lib_num}: {file_path.name}"
            )

    # Handle raw fasta files (keep existing logic)
    if input_path.is_dir():
        from rolypoly.utils.bio.library_detection import identify_fasta_files

        fasta_files = identify_fasta_files(input_path, logger=logger)[
            "fasta_files"
        ]
        for fasta in fasta_files:
            library_info.add_raw_fasta(str(fasta))
            logger.debug(f"Added raw FASTA: {fasta.name}")

    # Convert library_info to the expected libraries format
    libraries = library_info.to_assembly_dict()

    return libraries, len(libraries)


def run_spades(
    config, libraries, mode: str | None = None, output_label: str | None = None
):
    import subprocess

    from rolypoly.utils.various import ensure_memory

    spades_mode = mode or config.step_params["spades"]["mode"]
    output_name = output_label or spades_mode
    spades_output = config.output_dir / f"spades_{output_name}_output"
    # JSON flow syntax is valid YAML and avoids adding a YAML dependency.
    # A dataset preserves library associations and supports any number of files.
    dataset = []
    for lib in libraries.values():
        paired = bool(lib.get("interleaved") or lib.get("r1"))
        entry = {"type": "paired-end" if paired else "single"}
        if paired:
            # SPAdes requires an explicit orientation in dataset input.
            entry["orientation"] = "fr"
        for key, field in (
            ("r1", "left reads"),
            ("r2", "right reads"),
            ("interleaved", "interlaced reads"),
            ("single", "single reads"),
            ("merged", "merged reads" if paired else "single reads"),
        ):
            if lib.get(key):
                entry.setdefault(field, []).append(
                    str(Path(lib[key]).resolve())
                )
        if any(key.endswith(" reads") for key in entry):
            dataset.append(entry)
    # Add long reads and raw FASTA if provided; raw FASTA supplies trusted contigs.
    for read_type, paths in (
        (
            config.long_read_type,
            [
                lib["long_read"]
                for lib in libraries.values()
                if lib.get("long_read")
            ],
        ),
        ("trusted-contigs", config.raw_fasta),
    ):
        if paths:
            dataset.append(
                {
                    "type": read_type,
                    "single reads": [
                        str(Path(path).resolve()) for path in paths
                    ],
                }
            )
    if spades_mode == "meta":
        paired_entries = [entry for entry in dataset if entry["type"] == "paired-end"]
        if not paired_entries or any(entry["type"] == "single" for entry in dataset):
            raise click.ClickException(
                "metaSPAdes requires paired-end input. Associate orphan/merged reads "
                "using the same library number, or select a suitable --spades-mode "
                "or --assembler megahit for independent single-read libraries."
            )
        if len(paired_entries) > 1:
            # metaSPAdes accepts one logical short-read library. YAML fields
            # accept lists of files, so pool corresponding categories without
            # copying reads. Left/right lists retain the same library order;
            # interlaced reads, orphans and merged reads keep their own fields.
            pooled = {"type": "paired-end", "orientation": "fr"}
            for entry in paired_entries:
                for field, paths in entry.items():
                    if field.endswith(" reads"):
                        pooled.setdefault(field, []).extend(paths)
            dataset = [pooled] + [
                entry for entry in dataset if entry["type"] != "paired-end"
            ]
            config.logger.warning(
                "metaSPAdes supports one paired-end library: pooling %d input "
                "libraries into one logical library using dataset file lists. "
                "Mate pairing is preserved, but separate library identities and "
                "insert-size distributions will not be modeled independently. "
                "No concatenated FASTQ files are written.",
                len(paired_entries),
            )
    if any(
        lib.get("long_read") for lib in libraries.values()
    ) and spades_mode in ("meta", "rnaviral"):
        config.logger.warning(
            "Long-read input is experimental in metaSPAdes and undocumented in rnaviralSPAdes; selected mode: %s",
            spades_mode,
        )
    dataset_path = config.output_dir / f"spades_{output_name}_dataset.yaml"
    dataset_path.write_text(json.dumps(dataset, indent=2) + "\n")
    memory_bytes = int(ensure_memory(config.memory)["bytes"][:-1])
    spades_cmd = [
        "spades.py",
        f"--{spades_mode}",
        "-o",
        str(spades_output),
        "--threads",
        str(config.threads),
        "--only-assembler",
        "-k",
        str(config.step_params["spades"]["k"]),
        "--phred-offset",
        "33",
        "-m",
        str(max(1, memory_bytes // 1024**3)),
        "--dataset",
        str(dataset_path),
    ]
    config.logger.info("Running SPAdes with command: %s", spades_cmd)
    subprocess.run(spades_cmd, check=True)
    config.logger.info("Finished SPAdes assembly")
    return spades_output / (
        "transcripts.fasta" if spades_mode == "rna" else "scaffolds.fasta"
    )


def run_megahit(config, libraries):
    """Run MEGAHIT assembly."""
    import glob
    import subprocess

    from rolypoly.utils.various import ensure_memory

    config.logger.info("Started Megahit assembly")
    megahit_output = config.output_dir / "megahit_custom_out"

    long_reads = [
        str(lib["long_read"])
        for lib in libraries.values()
        if lib.get("long_read")
    ]
    if long_reads:
        config.logger.warning(
            "MEGAHIT does not support long reads; %d long-read file(s) will be "
            "skipped for the MEGAHIT run. Use the SPAdes assembler for "
            "short+long hybrid assembly.",
            len(long_reads),
        )

    megahit_cmd = ["megahit"]
    for parameter in ("k-min", "k-max", "k-step", "min-contig-len"):
        megahit_cmd.extend(
            [f"--{parameter}", str(config.step_params["megahit"][parameter])]
        )
    for flag, keys in (
        ("-1", ("r1",)),
        ("-2", ("r2",)),
        ("--12", ("interleaved",)),
        ("-r", ("merged", "single")),
    ):
        paths = [
            str(lib[key])
            for lib in libraries.values()
            for key in keys
            if lib.get(key)
        ]
        if paths:
            if any("," in path for path in paths):
                raise click.ClickException(
                    "MEGAHIT input filenames must not contain commas."
                )
            megahit_cmd.extend([flag, ",".join(paths)])
    megahit_cmd.extend(
        [
            "--out-dir",
            str(megahit_output),
            "--num-cpu-threads",
            str(config.threads),
            "--memory",
            ensure_memory(config.memory)["bytes"][:-1],
        ]
    )
    config.logger.info("Running Megahit assembly with command: %s", megahit_cmd)
    subprocess.run(megahit_cmd, check=True)

    final_k = max(
        int(os.path.basename(file).split("k")[1].split(".")[0])
        for file in glob.glob(
            f"{megahit_output}/intermediate_contigs/*.final.contigs.fa"
        )
    )

    with open(
        megahit_output / f"final_megahit_assembly_k{final_k}.fastg", "w"
    ) as graph:
        subprocess.run(
            [
                "megahit_toolkit",
                "contig2fastg",
                str(final_k),
                str(megahit_output / "final.contigs.fa"),
            ],
            stdout=graph,
            check=True,
        )

    return megahit_output / "final.contigs.fa"


def run_with_read_stream(command, paths):
    """Feed plain/gzip reads to a single-pass consumer without a disk copy."""
    import gzip
    import subprocess
    from contextlib import suppress
    from rolypoly.utils.various import is_gzipped

    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for path in paths:
            opener = gzip.open if is_gzipped(path) else open
            with opener(path, "rb") as source:
                # Keep record boundaries valid even if a file has no final newline.
                last_chunk = b""
                while chunk := source.read(1024 * 1024):
                    process.stdin.write(chunk)
                    last_chunk = chunk
                if last_chunk and not last_chunk.endswith(b"\n"):
                    process.stdin.write(b"\n")
        process.stdin.close()
    except BrokenPipeError:
        with suppress(BrokenPipeError):
            process.stdin.close()
        returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command) from None
        raise RuntimeError("PenguiN closed its input before all reads were sent") from None
    except BaseException:
        # Input errors must not leave an assembler running on partial reads.
        if process.poll() is None:
            process.terminate()
        with suppress(BrokenPipeError):
            process.stdin.close()
        process.wait()
        raise
    returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def run_penguin(config, libraries):
    """Run Penguin assembler."""
    import subprocess

    config.logger.info("Started Penguin assembly")
    penguin_output = (
        config.output_dir / "penguin_Fguided_1_nuclassemble_c0.fasta"
    )
    long_reads = [
        str(lib["long_read"])
        for lib in libraries.values()
        if lib.get("long_read")
    ]
    if long_reads:
        config.logger.warning(
            "Penguin does not support long reads; %d long-read file(s) will be "
            "skipped for the Penguin run (no ONT/PacBio mode; exact k-mer "
            "matching is error-sensitive; reads over 200 kbp would be "
            "truncated by maxSeqLen=200000). Use the SPAdes assembler for "
            "short+long hybrid assembly.",
            len(long_reads),
        )

    # PenguiN pairs adjacent FILES globally. Native R1/R2 inputs can go
    # directly to mergereads; mixed layouts need a single independent stream.
    short_reads = [
        str(lib[key])
        for lib in libraries.values()
        for key in ("r1", "r2", "interleaved", "merged", "single")
        if lib.get(key)
    ]
    # Add raw FASTA if provided: PenguiN treats these as ordinary sequences,
    # not trusted contigs. Multiple file arguments would incorrectly pair them.
    raw_fasta = [str(path) for path in config.raw_fasta]
    inputs = short_reads + raw_fasta
    if raw_fasta:
        config.logger.info(
            "PenguiN will include %d raw FASTA file(s) as ordinary unpaired "
            "sequences, not trusted contigs.", len(raw_fasta),
        )
    if not inputs:
        raise click.ClickException(
            "No short-read or raw FASTA inputs found for the Penguin run."
        )
    import shlex

    native_pairs = bool(short_reads) and not raw_fasta and all(
        lib.get("r1")
        and lib.get("r2")
        and not any(lib.get(key) for key in ("interleaved", "merged", "single"))
        for lib in libraries.values()
        if not lib.get("long_read")
    )
    if native_pairs:
        penguin_input = shlex.join(inputs)
    elif len(inputs) == 1:
        penguin_input = shlex.quote(inputs[0])
    else:
        penguin_input = "stdin"
        config.logger.info(
            "Streaming %d read/FASTA files to PenguiN in single-end mode; "
            "mixed layouts cannot use its file-paired mode. No concatenated "
            "input file is written; pairing is not used in this mode.",
            len(inputs),
        )

    penguin_tmp = config.temp_dir / "tmp_penguin"
    penguin_tmp.mkdir(parents=True, exist_ok=True)

    penguin_cmd = (
        f"penguin guided_nuclassemble {penguin_input} "
        f"{shlex.quote(str(penguin_output))} {shlex.quote(str(penguin_tmp))} --min-contig-len {config.step_params['penguin']['min-contig-len']} "
        f"--contig-output-mode 0 --num-iterations {config.step_params['penguin']['num-iterations']} "
        f"--min-seq-id nucl:0.9,aa:0.99 --min-aln-len nucl:31,aa:150 "
        f"--clust-min-seq-id 0.99 --clust-min-cov 0.99 --threads {config.threads}"
    )
    if not native_pairs and len(inputs) > 1:
        run_with_read_stream(shlex.split(penguin_cmd), inputs)
    else:
        subprocess.run(penguin_cmd, shell=True, check=True)
    return penguin_output


@click.command(name="assemble", no_args_is_help=True)
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False, dir_okay=True, exists=False),
    default="RP_assembly_output",
    help="Output path (folder will be created if it doesn't exist)",
)
@click.option(
    "-id",
    "--input-dir",
    default=None,
    help="Input directory to scan for fastq files",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--paired-end",
    multiple=True,
    nargs=3,
    default=(),
    help="Library number and paired FASTQ files: <lib_num> <R1> <R2>",
)
@click.option(
    "--single-end",
    multiple=True,
    nargs=2,
    default=(),
    help="Library number and single-end FASTQ: <lib_num> <fastq>",
)
@click.option(
    "--merged",
    multiple=True,
    nargs=2,
    default=(),
    help="Library number and merged FASTQ: <lib_num> <fastq>",
)
@click.option(
    "--long-read",
    multiple=True,
    nargs=1,
    default=(),
    help="""path to long read FASTQ: <fastq>\n
    Note: long read files are not supported by all assemblers/configurations:\n
    SPAdes (standard/meta/rna modes): supported in hybrid assembly via --long-read-type (nanopore default; also pacbio or sanger). metaSPAdes hybrid assembly is experimental per the SPAdes manual, and meta mode pools multiple paired-end libraries into one logical library with a warning (no temporary FASTQ copies). \n
    rnaviralSPAdes: long reads are undocumented; a warning is logged and the reads are passed through - remove --long-read if SPAdes errors. \n
    MEGAHIT: not supported - long reads are skipped with a warning for the MEGAHIT run. \n
    Penguin: not supported - long reads are skipped with a warning for the Penguin run (no ONT/PacBio mode; exact k-mer matching is error-sensitive; reads over 200 kbp would be truncated). \n
    Also note: PenguiN pairs FILES as R1/R2 (an interleaved file is read as single-end), separate R1/R2 pairs are passed directly; mixed interleaved/single inputs are streamed through stdin as single-end reads without a concatenated temporary file (pairing is not used in this mode). \n
    """,
)
@click.option(
    "--long-read-type",
    default="nanopore",
    type=click.Choice(["nanopore", "pacbio", "sanger"]),
    help="Long read type passed to SPAdes in hybrid assembly (--nanopore, --pacbio, or --sanger). Only used when --long-read is provided. PacBio CLR input should be prefiltered (e.g. circular consensus sequences); PacBio HiFi/CCS reads go through -s/--single-end instead.",
)
@click.option(
    "--raw-fasta",
    multiple=True,
    default=(),
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Raw FASTA file(s) to include. SPAdes: trusted contigs. PenguiN: ordinary unpaired sequences (not trusted contigs); mixing with reads uses single-end streaming and loses pairing information. MEGAHIT: not supported.",
)
@click.option(
    "-A",
    "--assembler",
    default=["spades", "megahit"],
    multiple=True,
    type=click.Choice(["spades", "spades_rnaviral", "megahit", "penguin"]),
    help="""Assembler choice. For multiple, use multiple -A flags or give a comma-separated list. \n
    SPAdes: iterative de bruijn graph assembler - relatively slow and memory heavy, but potentially more accurate. \n
    MEGAHIT: multiple kmer based de bruijn graph assembler - Fast and memory light, but potentially less accurate. \n
    Penguin: protein-guided nucleotide assembly using guided_nuclassemble.  \n
    Note1 : RolyPoly uses PenguiN's amino-acid-guided nucleotide assembly mode.    \n
    Note2 : Without a preset, the default assemblers are SPAdes and MEGAHIT.
    """,
)
@click.option(
    "--spades-mode",
    default="meta",
    type=click.Choice(["meta", "rna", "rnaviral", "sc"]),
    help="SPAdes mode for the 'spades' assembler.",
)
@click.option(
    "--preset",
    default=None,
    type=click.Choice(sorted(ASSEMBLY_PRESETS.keys())),
    help=(
        "Apply a named assembly preset (overrides --assembler and --dereplicate unless "
        "those flags are given explicitly on the command line).  "
        + "  ".join(
            f"'{name}': {p['description']}"
            for name, p in ASSEMBLY_PRESETS.items()
        )
    ),
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
    default=[],
    type=click.Choice(["dereplicate", "rename"]),  # , "stats"
    multiple=True,
    help="""Steps to skip. Repeat the flag to skip multiple steps (comma-separated values are NOT accepted here). Example: --skip-steps dereplicate --skip-steps rename
    - dereplicate: skip assembler-output dereplication (same as --no-rmdup)
    - rename: skip renaming the concatenated contigs to CID_ ids""",
)
@click.option(
    "-ow",
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite the output directory if it already exists (deletes the existing output directory first). Without this flag, an existing output directory raises an error.",
)
@click.option(
    "--dereplicate/--no-rmdup",
    default=True,
    help="""Dereplicate assembler output by default. Disable with --no-rmdup.
    - dereplicate: remove identical sequences (same sequence, same length, or its' reverse complement)
    - no-rmdup: do not perform assembler-output dereplication""",
)
def assembly(
    input_dir,
    paired_end,
    single_end,
    merged,
    long_read,
    long_read_type,
    raw_fasta,
    assembler,
    spades_mode,
    preset,
    dereplicate,
    output,
    threads,
    memory,
    keep_tmp,
    log_file,
    override_parameters,
    skip_steps,
    overwrite,
    log_level,
    temp_dir,
):
    """Assemble reads/contigs with one or more backends and optional dereplication.

    Inputs can be provided explicitly (`--paired-end`, `--single-end`,
    `--merged`, `--long-read`, `--raw-fasta`) and/or discovered from
    `--input-dir`.

    Selected assembler outputs are normalized and optionally dereplicated
    before writing final contigs and run metadata to the output directory.
    """
    import shutil

    import polars as pl

    from rolypoly.utils.bio.sequences import (
        process_sequences,
        read_fasta_df,
        rename_sequences,
        write_fasta_file,
    )
    from rolypoly.utils.logging.citation_reminder import remind_citations
    from rolypoly.utils.logging.loggit import log_start_info

    if not overwrite:
        if Path(output).exists():
            raise click.ClickException(
                f"Output directory '{output}' already exists. Use --overwrite / -ow to overwrite."
            )
    else:
        shutil.rmtree(output, ignore_errors=True)

    Path(output).mkdir(parents=True, exist_ok=True)

    # Validate input options before creating config
    has_explicit_inputs = any(
        [paired_end, single_end, merged, long_read, raw_fasta]
    )
    has_input_dir = input_dir is not None

    if not has_explicit_inputs and not has_input_dir:
        raise click.ClickException(
            "Error: No input files specified. You must provide eithedr:\n"
            "  - An input directory using --input-dir, or\n"
            "  - Explicit library files using --paired-end, --single-end, --merged, --long-read, or --raw-fasta"
        )

    config = AssemblyConfig(
        input_dir=Path(input_dir) if input_dir else None,
        output=Path(output),
        threads=threads,
        log_file=Path(log_file),
        memory=memory,
        temp_dir=temp_dir,
        assembler=assembler,
        spades_mode=spades_mode,
        preset=preset,
        keep_tmp=keep_tmp,
        override_parameters=override_parameters,
        skip_steps=skip_steps,
        log_level=log_level,
        dereplicate=dereplicate,
        overwrite=overwrite,
        long_read_type=long_read_type,
    )

    config.logger.info("Starting assembly process")
    ctx = click.get_current_context()
    apply_assembly_preset(preset, ctx, config)
    log_start_info(config.logger, config_dict=config.__dict__)
    config.logger.info(
        f"Saving config to {config.output_dir / 'assembly_config.json'}"
    )
    config.save(config.output_dir / "assembly_config.json")

    if has_explicit_inputs and has_input_dir:
        config.logger.warning(
            "Warning: Both explicit library options and --input-dir specified. "
            "Files from both sources will be combined for assembly."
            "This is may lead to unexpected results."
        )

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
        for lib_num, path in enumerate(long_read, 1):
            library_info.add_long_read(lib_num, path)
    if raw_fasta:
        for path in raw_fasta:
            library_info.add_raw_fasta(path)

    # Process input directory if provided
    if input_dir:
        libraries, n_libraries = handle_input_files(
            input_dir, library_info, config.logger
        )
    else:
        libraries = library_info.to_assembly_dict()
        n_libraries = len(libraries)

    config.raw_fasta = list(library_info.raw_fasta)
    config.logger.info(f"Found {n_libraries} libraries")
    has_long_reads = any(lib.get("long_read") for lib in libraries.values())
    if has_long_reads and not any(
        assembler_id.startswith("spades") for assembler_id in config.assembler
    ):
        config.logger.warning(
            "Long reads were provided but no SPAdes assembler is selected; "
            "MEGAHIT/PenguiN do not support long reads, so the long-read "
            "input(s) will not be used."
        )
    config.logger.info(f"Libraries: {libraries}")
    observed_read_length = detect_average_read_length(libraries, config.logger)
    tune_assembly_kmers(config, observed_read_length)
    contigs4eval = []  # list[Path | str]  – one entry per assembler run
    contigs_asm_labels = []  # list[str]          – parallel assembler name

    if "spades" in config.assembler and "spades" not in config.skip_steps:
        contigs4eval.append(
            run_spades(
                config,
                libraries,
                mode=config.step_params["spades"]["mode"],
                output_label=config.step_params["spades"]["mode"],
            )
        )
        contigs_asm_labels.append("spades")
        tools.append("spades")
    if (
        "spades_rnaviral" in config.assembler
        and "spades_rnaviral" not in config.skip_steps
    ):
        contigs4eval.append(
            run_spades(
                config, libraries, mode="rnaviral", output_label="rnaviral"
            )
        )
        contigs_asm_labels.append("spades_rnaviral")
        tools.append("spades_rnaviral")
    if "megahit" in config.assembler and "megahit" not in config.skip_steps:
        contigs4eval.append(run_megahit(config, libraries))
        contigs_asm_labels.append("megahit")
        tools.append("megahit")
    if "penguin" in config.assembler and "penguin" not in config.skip_steps:
        contigs4eval.append(run_penguin(config, libraries))
        contigs_asm_labels.append("penguin")
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

        if "rename" not in config.skip_steps:
            try:
                config.logger.info(
                    "Renaming contigs from %d assembler output(s)",
                    len(contigs4eval),
                )

                # Read each assembler output separately and track which rows
                # belong to which assembler – avoids storing a full text column.
                assembler_dfs = []
                assembler_ranges = []  # (label, start_row_inclusive, end_row_exclusive)
                row_offset = 0
                for contig_file, asm_label in zip(
                    contigs4eval, contigs_asm_labels
                ):
                    df_part = read_fasta_df(str(contig_file))
                    n = len(df_part)
                    assembler_ranges.append(
                        (asm_label, row_offset, row_offset + n)
                    )
                    row_offset += n
                    assembler_dfs.append(df_part)

                df = pl.concat(assembler_dfs)
                del assembler_dfs  # release per-assembler frames
                config.logger.info("Found %d sequences total", len(df))

                # Assign sequential IDs (CID_0001, CID_0002, …)
                df_renamed, id_map = rename_sequences(
                    df, prefix="CID", use_hash=False
                )

                # Build an assembler-label column using row-index ranges.
                # A chained when/then expression covers each assembler's row range;
                # no full-text source column is stored.
                df_renamed = df_renamed.with_row_index("_row_nr")
                asm_expr = pl.lit("")  # fallback (should never fire)
                for asm_label, start, end in assembler_ranges:
                    asm_expr = (
                        pl.when(pl.col("_row_nr").is_between(start, end - 1))
                        .then(pl.lit(asm_label))
                        .otherwise(asm_expr)
                    )
                df_renamed = df_renamed.with_columns(
                    asm_expr.alias("assembler")
                ).drop("_row_nr")

                config.logger.info("Calculating sequence statistics")
                df_renamed = process_sequences(df_renamed)

                # Write renamed sequences using the shared utility
                renamed_file = str(
                    config.output_dir / "all_contigs_renamed.fasta"
                )
                config.logger.info(
                    "Writing renamed sequences to %s", renamed_file
                )
                write_fasta_file(
                    seqs=df_renamed["sequence"].to_list(),
                    headers=df_renamed["header"].to_list(),
                    output_file=renamed_file,
                )

                # Update contigs4eval to use the single renamed file
                contigs4eval = [renamed_file]

                # Save mapping file (assembler stored as a column, not in the ID)
                mapping_file = str(config.output_dir / "contigs_id_map.tsv")
                config.logger.info("Saving ID mapping to %s", mapping_file)
                mapping_df = pl.DataFrame(
                    {
                        "old_id": list(id_map.keys()),
                        "new_id": list(id_map.values()),
                        "assembler": df_renamed["assembler"],
                        "length": df_renamed["length"],
                        "gc_content": df_renamed["gc_content"].round(2),
                        "n_count": df_renamed["n_count"],
                    }
                )
                mapping_df.write_csv(mapping_file, separator="\t")

            except Exception as e:
                config.logger.error(
                    "Error during sequence renaming: %s", str(e)
                )
                config.logger.warning("Continuing with original contig files")
                contigs4eval = [concat_file]
        else:
            # Both downstream paths must include every assembler's contigs.
            contigs4eval = [concat_file]

    # Dereplication step (identical sequences only).
    # Single low-memory streaming pass via polars_fastx.dereplicate_fasta: it
    # computes length/GC/N-count natively + an xxh3 content hash, writes the
    # unique representatives, the redundancy map, and a stats table -- so we no
    # longer need seqkit here, and the raw all_contigs*.fasta copies become
    # disposable (removed below unless --keep-tmp). The per-contig stats are
    # merged back into contigs_id_map.tsv (adds seq_hash + redundancy).
    dereplicated_output = None
    run_dereplication = (
        config.dereplicate and "dereplicate" not in config.skip_steps
    )
    if len(contigs4eval) > 0 and run_dereplication:
        config.logger.info(
            "Starting single-pass sequence dereplication (polars_fastx)"
        )
        from rolypoly.utils.bio.polars_fastx import dereplicate_fasta

        dereplicated_output = str(
            config.output_dir / "dereplicated_contigs.fasta"
        )
        derep_stats = dereplicate_fasta(
            input_file=str(contigs4eval[0]),
            output_file=dereplicated_output,
            redundancy_file=str(config.output_dir / "Redundancy_lookup.txt"),
            prefix=None,  # keep the CID_ ids assigned during renaming
            logger=config.logger,
        )
        config.logger.info("Finished sequence dereplication")

        # Enrich contigs_id_map.tsv with the content hash + redundancy count so
        # the dropped FASTAs remain fully reconstructable from the map + reps.
        mapping_file = config.output_dir / "contigs_id_map.tsv"
        if mapping_file.exists():
            try:
                id_map_df = pl.read_csv(mapping_file, separator="\t")
                hash_df = derep_stats.select(
                    [
                        pl.col("old_id").alias("new_id"),
                        pl.col("seq_hash"),
                        pl.col("redundancy"),
                    ]
                )
                id_map_df = id_map_df.join(hash_df, on="new_id", how="left")
                id_map_df.write_csv(mapping_file, separator="\t")
            except Exception as e:
                config.logger.warning(
                    "Could not merge dereplication stats into %s: %s",
                    mapping_file,
                    e,
                )

        # Verify dereplicated output exists before proceeding
        if dereplicated_output and (
            not os.path.exists(dereplicated_output)
            or os.path.getsize(dereplicated_output) == 0
        ):
            config.logger.error(
                f"Dereplication failed: {dereplicated_output} not found or empty"
            )
            return
    elif len(contigs4eval) > 0:
        config.logger.info("Skipping dereplication as requested")
        dereplicated_output = str(contigs4eval[0])  # Use original contigs
    else:
        config.logger.warning(
            "No contigs available for dereplication or further processing"
        )

    config.logger.info(f"Finished assembly: {contigs4eval}")

    if not config.keep_tmp:
        # Clean up temporary files and directories
        cleanup_paths = [
            "tmp",  # Generic tmp directory
            config.output_dir / "all_interleaved.fq.gz",
            config.output_dir / "all_merged.fq.gz",
            config.output_dir / "megahit_custom_out" / "intermediate_contigs",
            # Raw concatenated contigs are an intermediate for renaming and
            # dereplication, reconstructable from the ID map and representatives.
            # Removed only if it is not the final output (when both steps skip).
            config.output_dir / "all_contigs.fasta",
        ]

        # The pre-dereplication renamed file is redundant with
        # dereplicated_contigs.fasta ONLY when dereplication actually ran (else it
        # is the final assembly and must be kept).
        if run_dereplication and os.path.exists(
            str(config.output_dir / "dereplicated_contigs.fasta")
        ):
            cleanup_paths.append(
                config.output_dir / "all_contigs_renamed.fasta"
            )

        # Clean up all paths
        for path in cleanup_paths:
            path = Path(path)
            if (
                dereplicated_output
                and path.resolve() == Path(dereplicated_output).resolve()
            ):
                continue
            if path.exists():
                if path.is_dir():
                    config.logger.debug(f"Removing temporary directory: {path}")
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    config.logger.debug(f"Removing temporary file: {path}")
                    path.unlink(missing_ok=True)

        # Prune the assemblers' own output folders. Once we have the renamed /
        # dereplicated contigs + id map, the only things worth keeping are the
        # assembly graphs (*.fastg / *.gfa), the tool's own log, and its run
        # parameters. Everything else (per-k subdirs, intermediate contigs, the
        # raw final contigs, checkpoints, ...) is disposable.
        def prune_assembler_dir(assembler_dir, keep_names):
            if not assembler_dir.exists():
                return
            for item in assembler_dir.rglob("*"):
                if not item.exists():
                    continue  # a parent dir may already have been removed
                if item.is_dir():
                    continue  # handled by removing their leftover files below
                if item.name in keep_names or item.suffix in (".fastg", ".gfa"):
                    continue
                item.unlink(missing_ok=True)
            # drop any now-empty subdirectories, deepest first
            for item in sorted(
                (p for p in assembler_dir.rglob("*") if p.is_dir()),
                key=lambda p: len(p.parts),
                reverse=True,
            ):
                try:
                    item.rmdir()
                except OSError:
                    pass  # not empty (kept a graph/log), leave it

        # Only prune when the final assembly lives directly under output_dir,
        # so an endpoint inside an assembler folder can never be deleted.
        endpoint = Path(dereplicated_output) if dereplicated_output else None
        endpoint_in_output = (
            endpoint is not None
            and endpoint.resolve().parent == config.output_dir.resolve()
        )
        if endpoint_in_output:
            for spades_dir in config.output_dir.glob("spades_*output"):
                prune_assembler_dir(spades_dir, {"spades.log", "params.txt"})
            prune_assembler_dir(
                config.output_dir / "megahit_custom_out",
                {"log", "options.json"},
            )

    config.logger.info("Assembly process completed successfully.")

    if dereplicated_output:
        final_assembly_symlink = config.output_dir / "final_assembly.fasta"
        if (
            final_assembly_symlink.exists()
            or final_assembly_symlink.is_symlink()
        ):
            final_assembly_symlink.unlink()
        final_assembly_symlink.symlink_to(Path(dereplicated_output).resolve())
        config.logger.info(
            "Symlinked final assembly to %s", final_assembly_symlink
        )

    else:
        config.logger.info("No final contigs were produced.")

    if config.log_level != 10:
        with open(f"{config.log_file}", "a") as f_out:
            f_out.write(remind_citations(tools, return_bibtex=True) or "")


if __name__ == "__main__":
    assembly()
