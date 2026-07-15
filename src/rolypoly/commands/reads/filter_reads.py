import os
import shutil
from pathlib import Path
from typing import Any, Tuple, Union

import rich_click as click
from rich.console import Console

from rolypoly.utils.bio.library_detection import (
    handle_input_fastq,
    probe_fastq_inputs,
)
from rolypoly.utils.logging.config import BaseConfig
from rolypoly.utils.logging.output_tracker import OutputTracker
from rolypoly.utils.various import (
    ensure_memory,
    get_reduced_memory,
    run_command_comp,
)

global tools
tools = ["bbmap"]

global output_tracker
output_tracker = OutputTracker()

console = Console()
config = None


FILTER_READS_PRESETS: dict[str, dict[str, Any]] = {
    "rna_virus_metat": {
        "step_params": {
            "quality_trim_unmerged": {"trimq": 5, "minlen": 25},
            "trim_adapters": {"minlen": 20},
            "decontaminate_rrna": {"mincovfraction": 0.6},
        },
        "skip_steps": [],
        "flags": {"trim_polya": False},
        "description": "RNA virus metatranscriptome: rRNA removal (mincovfraction=0.6), known-DNA + identified-DNA filtering, adapter trim, lenient quality trim (trimq=5 minlen=25); no polyA trimming",
    },
    "total_rna_ribodepleted": {
        "step_params": {
            "quality_trim_unmerged": {"trimq": 5, "minlen": 20},
            "trim_adapters": {"minlen": 20},
            "decontaminate_rrna": {"mincovfraction": 0.6},
        },
        "skip_steps": [],
        "flags": {"trim_polya": False},
        "description": "Total RNA ribo-depleted: stricter rRNA removal (mincovfraction=0.6), known-DNA + identified-DNA filtering, adapter trim, lenient quality trim (trimq=5 minlen=20); no polyA trimming",
    },
    "poly_a_selected": {
        "step_params": {
            "quality_trim_unmerged": {"trimq": 12, "minlen": 20},
            "trim_polya_tails": {"trimpolya": 18, "minlen": 20},
        },
        "skip_steps": [],
        "flags": {"trim_polya": True},
        "description": "Poly-A selected mRNA: polyA tail trimming enabled (trimpolya=18), stricter quality trim (trimq=12 minlen=20); rRNA and DNA filtering still applied",
    },
    "fast": {
        "step_params": {},
        "skip_steps": [
            "error_correct_1",
            "error_correct_2",
            "filter_identified_dna",
        ],
        "flags": {},
        "description": "Fast: skips overlap error correction (error_correct_1/2) and identified-DNA filtering; all other steps run at default parameters",
    },
    "strict": {
        "step_params": {"quality_trim_unmerged": {"trimq": 20, "minlen": 20}},
        "skip_steps": [],
        "flags": {},
        "description": "Strict: aggressive quality trim (trimq=20 minlen=20), two-pass deduplication; all filtering steps enabled",
    },
    "all_virus_metat": {
        "step_params": {
            "quality_trim_unmerged": {"trimq": 10, "minlen": 20},
            "decontaminate_rrna": {"mincovfraction": 0.5},
        },
        "skip_steps": ["filter_identified_dna"],
        "flags": {"trim_polya": False},
        "description": "All-virus metatranscriptome: relaxed rRNA removal (mincovfraction=0.5), skips identified-DNA filter, moderate quality trim (trimq=10 minlen=20); known-DNA filtering still applied",
    },
    "all_virus_metag": {
        "step_params": {"quality_trim_unmerged": {"trimq": 10, "minlen": 20}},
        "skip_steps": ["decontaminate_rrna", "filter_identified_dna"],
        "flags": {"trim_polya": False},
        "description": "All-virus metagenomics: skips rRNA and identified-DNA filtering entirely, moderate quality trim (trimq=10 minlen=20); known-DNA (host) filtering still applied if --known-dna is provided",
    },
    "rna_virome": {
        "not_implemented": True,
        "description": "VLP-extracted RNA virome (not yet implemented)",
    },
}


def sync_trim_polya_step(config: "ReadFilterConfig") -> None:
    if config.trim_polya:
        config.skip_steps = [
            step for step in config.skip_steps if step != "trim_polya_tails"
        ]
    elif "trim_polya_tails" not in config.skip_steps:
        config.skip_steps.append("trim_polya_tails")


def collect_protected_step_params(
    override_parameters: dict[str, Any],
) -> set[tuple[str, str]]:
    protected: set[tuple[str, str]] = set()
    for step_name, params in override_parameters.items():
        if isinstance(params, dict):
            for param_name in params:
                protected.add((step_name, str(param_name)))
    return protected


def apply_filter_reads_preset(
    preset_name: str | None, ctx: click.Context, config: "ReadFilterConfig"
) -> set[tuple[str, str]]:
    if not preset_name:
        return set()

    preset = FILTER_READS_PRESETS.get(preset_name)
    if preset is None:
        raise click.BadParameter(
            f"Unknown preset '{preset_name}'. "
            f"Choose from: {', '.join(sorted(FILTER_READS_PRESETS.keys()))}",
            param_hint="--preset",
        )
    if preset.get("not_implemented"):
        raise click.BadParameter(
            f"Preset '{preset_name}' is not yet implemented.",
            param_hint="--preset",
        )

    explicit: set[str] = set()
    if ctx.params:
        for param in ctx.command.params:
            source = ctx.get_parameter_source(param.name)
            if source == click.core.ParameterSource.COMMANDLINE:
                explicit.add(param.name)

    applied_flags: list[str] = []
    for flag_name, value in preset.get("flags", {}).items():
        if flag_name not in explicit:
            setattr(config, flag_name, value)
            applied_flags.append(f"{flag_name}={value}")
    sync_trim_polya_step(config)

    applied_skips: list[str] = []
    for step_name in preset.get("skip_steps", []):
        if step_name not in config.skip_steps:
            config.skip_steps.append(step_name)
            applied_skips.append(step_name)

    protected_from_preset: set[tuple[str, str]] = set()
    applied_step_params: list[str] = []
    for step_name, overrides in preset.get("step_params", {}).items():
        if step_name not in config.step_params:
            config.step_params[step_name] = {}
        if isinstance(overrides, dict):
            for param_name, value in overrides.items():
                config.step_params[step_name][param_name] = value
                protected_from_preset.add((step_name, str(param_name)))
            applied_step_params.append(step_name)

    config.logger.info(
        "Applied filter_reads preset '%s' (%s). flags=%s skip_steps=%s step_params=%s",
        preset_name,
        preset.get("description", "no description"),
        ", ".join(applied_flags) if applied_flags else "none",
        ", ".join(applied_skips) if applied_skips else "none",
        ", ".join(applied_step_params) if applied_step_params else "none",
    )
    return protected_from_preset


def auto_tune_params(
    file_info: dict[str, Any],
    config: "ReadFilterConfig",
    protected_step_params: set[tuple[str, str]],
) -> None:
    avg_read_length = float(file_info.get("average_read_length") or 0)
    avg_quality = float(file_info.get("average_read_quality") or 0)

    minlen = 20
    if avg_quality >= 30:
        trimq = 15
    elif avg_quality >= 25:
        trimq = 12
    else:
        trimq = 10

    if (
        "quality_trim_unmerged" in config.step_params
        and ("quality_trim_unmerged", "trimq") not in protected_step_params
    ):
        config.step_params["quality_trim_unmerged"]["trimq"] = trimq
    if (
        "quality_trim_unmerged" in config.step_params
        and ("quality_trim_unmerged", "minlen") not in protected_step_params
    ):
        config.step_params["quality_trim_unmerged"]["minlen"] = minlen
    if (
        "trim_adapters" in config.step_params
        and ("trim_adapters", "minlen") not in protected_step_params
    ):
        config.step_params["trim_adapters"]["minlen"] = minlen

    config.logger.info(
        "Auto-detected avg_read_length=%s, avg_quality=%s → adjusting trimq=%s, minlen=%s",
        round(avg_read_length, 2),
        round(avg_quality, 2),
        trimq,
        minlen,
    )


def format_bbmapy_output(stdout_obj, stderr_obj) -> str:
    parts = []
    for item in (stderr_obj, stdout_obj):
        if item is None:
            continue
        if isinstance(item, str):
            text = item.strip()
            if text:
                parts.append(text)
            continue
        if isinstance(item, (list, tuple)):
            text = "\n".join(str(x) for x in item if x is not None).strip()
            if text:
                parts.append(text)
            continue
        text = str(item).strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def validate_adapter_fasta(
    adapter_path: Path, min_non_n_bases: int = 25
) -> Path | None:
    if not adapter_path.exists() or adapter_path.stat().st_size == 0:
        return None

    validated_records: list[tuple[str, str]] = []
    current_header: str | None = None
    current_sequence: list[str] = []

    def flush_record() -> None:
        nonlocal current_header, current_sequence
        if current_header is None:
            return
        sequence = "".join(current_sequence).strip()
        if sequence.__len__() - sequence.upper().count("N") >= min_non_n_bases:
            validated_records.append((current_header, sequence))
        current_header = None
        current_sequence = []

    with open(adapter_path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(">"):
                flush_record()
                current_header = stripped[1:].strip() or "adapter"
                continue
            current_sequence.append(stripped)
        flush_record()

    if not validated_records:
        return None

    validated_path = adapter_path.with_name(f"validated_{adapter_path.stem}.fa")
    with open(validated_path, "w") as handle:
        for header, sequence in validated_records:
            handle.write(f">{header}\n{sequence}\n")
    return validated_path


def discover_merge_adapters(
    input_file: Path, config: "ReadFilterConfig", output_tracker: OutputTracker
) -> Path:
    """Discover adapter consensus sequences with BBMerge and retain only valid ones."""
    from bbmapy import bbmerge

    if str(getattr(config, "adapters", "")) != resolve_builtin_adapters(config):
        config.logger.info(
            "Adapter discovery skipped because adapters were provided explicitly."
        )
        return input_file

    discovery_output = (
        config.temp_dir / f"bbmerge_discovered_{config.file_name}.fa"
    )
    try:
        bb_stdout, bb_stderr = bbmerge(
            in_file=str(input_file),
            capture_output=True,
            outadapter=str(discovery_output),
            merge=False,
            Xmx=get_reduced_memory(config.memory),
            threads=str(config.threads),
            overwrite="t",
            interleaved="t",
        )
        config.logger.info(format_bbmapy_output(bb_stdout, bb_stderr))

        validated_adapter_ref = validate_adapter_fasta(discovery_output)
        if validated_adapter_ref is None:
            config.logger.info(
                "BBMerge adapter discovery produced no validated adapter FASTA; using built-in adapter references."
            )
            return input_file

        config.adapters = str(validated_adapter_ref)
        config.logger.info(
            "Using BBMerge-discovered adapter reference %s",
            validated_adapter_ref,
        )
    except RuntimeError as e:
        config.logger.warning(
            f"BBMerge adapter discovery failed; falling back to built-in adapters: {str(e)}"
        )
    return input_file


def resolve_builtin_adapters(config: "ReadFilterConfig") -> str:
    adapters_new = (
        Path(config.datadir) / "contam/adapters/AFire_illuminatetritis1223.fa"
    )
    adapters_bb = Path(config.datadir) / "contam/adapters/bbmap_adapters.fa"
    return f"{adapters_bb},{adapters_new}"


class ReadFilterConfig(BaseConfig):
    def __init__(self, **kwargs):
        # in this case output_dir and output are the same, so need to explicitly make sure it exists.
        # if not Path(kwargs.get("output")).exists():
        #     kwargs["output_dir"] = kwargs.get("output")
        #     Path(kwargs.get("output")).mkdir(parents=True, exist_ok=True)

        super().__init__(
            input=kwargs.get("input") or "",
            output=kwargs.get("output") or "",
            temp_dir=kwargs.get("temp_dir") or "",
            keep_tmp=kwargs.get("keep_tmp") or False,
            log_file=kwargs.get("log_file") or None,
            threads=kwargs.get("threads") or 1,
            memory=kwargs.get("memory") or "10gb",
            config_file=kwargs.get("config_file") or None,
            overwrite=kwargs.get("overwrite") or False,
            log_level=kwargs.get("log_level") or "info",
        )  # initialize the BaseConfig class
        # initialize the rest of the parameters (i.e. the ones that are not in the BaseConfig class)
        self.skip_existing = kwargs.get("skip_existing") or False
        self.zip_reports = kwargs.get("zip_reports") or False
        self.trim_polya = kwargs.get("trim_polya") or False
        self.disable_auto = kwargs.get("disable_auto") or False
        self.preset = kwargs.get("preset")
        self.user_provided_adapters: Path | None = (
            Path(kwargs.get("adapters") or "").resolve()
            if kwargs.get("adapters") is not None
            else None
        )
        self.adapters = (
            str(self.user_provided_adapters)
            if self.user_provided_adapters is not None
            else resolve_builtin_adapters(self)
        )
        self.user_provided_artifacts: Path | None = (
            Path(kwargs.get("artifacts") or "").resolve()
            if kwargs.get("artifacts") is not None
            else None
        )
        self.remove_synthetic_artifacts_enabled = (
            kwargs.get("remove_synthetic_artifacts") or False
        )
        self.artifacts = (
            str(self.user_provided_artifacts)
            if self.user_provided_artifacts is not None
            else "artifacts"
        )
        # self.override_parameters = self.override_parameters if isinstance(self.override_parameters, dict) else eval(self.override_parameters) if isinstance(self.override_parameters, str) else {}
        skip_steps_value = kwargs.get("skip_steps", [])
        if isinstance(skip_steps_value, list):
            self.skip_steps: list[str] = skip_steps_value
        elif isinstance(skip_steps_value, str):
            self.skip_steps = (
                skip_steps_value.split(",") if skip_steps_value else []
            )
        else:
            self.skip_steps = []
        self.known_dna = (
            Path(kwargs.get("known_dna") or "").resolve()
            if kwargs.get("known_dna") is not None
            else None
        )
        self.speed = kwargs.get("speed") or 0
        self.skip_existing = kwargs.get("skip_existing") or False
        self.override_parameters = (
            kwargs.get("override_parameters")
            if isinstance(kwargs.get("override_parameters"), dict)
            else eval(kwargs.get("override_parameters", "{}"))
            if isinstance(kwargs.get("override_parameters"), str)
            else {}
        )
        # self.skip_steps = skip_steps if isinstance(skip_steps, list) else skip_steps.split(",")
        self.step_timeout = (
            kwargs.get("step_timeout") or 3600
        )  # 3600 seconds/ 1 hour default  # TODO: consider changing this to per step timeouts in the future?
        self.file_name = (
            kwargs.get("file_name") or "rp_filtered_reads"
        )  # this is the base name of the output files, if not provided, it will be "rp_filtered_reads"

        self.step_params = {  # these are the default parameters for each step, if not overridden by the user
            # "filter_by_tile": {"nullifybrokenquality": "t"}, # filter tile is disabled until we verify we can get tile/xy information from the fastq headers in all the different scenarios (single/interleaved/paired and concated files), as well as the potential impacts of concatenating multiple samples on the tile filtering step.
            "filter_known_dna": {"k": 31, "mincovfraction": 0.65, "hdist": 0},
            "decontaminate_rrna": {"k": 31, "mincovfraction": 0.6, "hdist": 0},
            "filter_identified_dna": {
                "k": 31,
                "mincovfraction": 0.7,
                "hdist": 0,
            },
            "dedupe": {"dedupe": True, "s": 0, "lowcomplexity": True},
            "trim_adapters": {
                "ktrim": "r",
                "k": 23,
                "mink": 11,
                "hdist": 1,
                "tpe": "t",
                "tbo": "t",
                "minlen": 25,
            },
            "trim_polya_tails": {
                "trimpolya": 22,  # TODO: figure out if this is the right ball park...
                # "mink": 8,
                "minlen": 25,
            },
            "remove_synthetic_artifacts": {"k": 31, "ref": "artifacts"},
            "entropy_filter": {"entropy": 0.01, "entropywindow": 30},
            "error_correct_1": {"ecco": True, "mix": "t", "ordered": "t"},
            "error_correct_2": {
                "ecc": True,
                "reorder": True,
                "nullifybrokenquality": True,
                "passes": 1,
            },
            "merge_reads": {
                # "k": 93,
                # "extend2": 80,
                # "rem": True,
                "mix": "f"  # DO NOT Output both the merged (or mergable) and unmerged reads
            },  # TODO: add explanation somewhere about the (high) memory usage and the potential gains/tradeoffs of merging reads https://bbmap.org/tools/bbmerge#:~:text=When%20NOT%20to%20Use%20BBMerge
            "quality_trim_unmerged": {
                "qtrim": "rl",
                "trimq": 5,  # For now, keeping this very low.
                "minlen": 25,
            },
        }
        self.max_genomes = (
            kwargs.get("max_genomes") or 5
        )  # maximum number of potential host genomes to fetch
        self.protected_step_params: set[tuple[str, str]] = set(
            collect_protected_step_params(self.override_parameters)
        )
        if kwargs.get("override_parameters") is not None:
            self.logger.info(
                f"override_parameters: {kwargs.get('override_parameters')}"
            )
            for step, params in kwargs.get("override_parameters", {}).items():
                if step in self.step_params:
                    self.step_params[step].update(params)
                else:
                    self.logger.warning(
                        f"Warning: Unknown step '{step}' in override_parameters. Ignoring."
                    )

        sync_trim_polya_step(self)


def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")


def check_existing_file(output_file: Path, min_size: int = 20) -> bool:
    """Check if a file exists and is larger than min_size bytes"""
    return output_file.exists() and output_file.stat().st_size > min_size


def process_reads(
    config: ReadFilterConfig, output_tracker: OutputTracker
) -> Union[OutputTracker, None]:
    """Main function to orchestrate the preprocessing steps."""
    import signal

    # config.logger.info("Checking dependencies    ")
    base_dir = Path(config.temp_dir)
    config.save(output_path=base_dir / "rp_filter_reads_config.json")  # type: ignore

    # actual processing start here
    fastq_file, config.file_name, file_info = process_input_fastq(config)
    config.file_info = file_info

    config.logger.info(f"file_name: {config.file_name}")
    config.logger.info(f"fastq_file: {fastq_file}")
    original_input = str(config.input).split(",", 1)[0].strip()
    # config.logger.info(f"remind citation is {os.environ.get('ROLYPOLY_REMIND_CITATION', 'not_set')}    ")
    # exit()
    # breakpoint()
    output_tracker.add_file(
        filename=original_input,
        command="handle_input_fastq",
        command_name="reformat",
        is_merged=False,
        end_type=None,
        interleaved=None,
        is_gz=None,
    )  # retroactive addition

    config.memory = ensure_memory(config.memory, fastq_file)  # type: ignore ------ this second ensure is because we now have the fastq file to check its size.
    if not config.disable_auto:
        auto_tune_params(file_info, config, config.protected_step_params)

    has_paired_input = bool(file_info.get("R1_R2_pairs")) or bool(
        file_info.get("interleaved_files")
    )
    has_single_ends = bool(file_info.get("single_end_files"))
    has_mixed_input = has_paired_input and has_single_ends
    if (
        file_info.get("is_single_end_only")
        or has_mixed_input
        or not has_paired_input
    ):
        for step_name in (
            "discover_merge_adapters",
            "error_correct_1",
            "error_correct_2",
            "merge_reads",
        ):
            if step_name not in config.skip_steps:
                config.skip_steps.append(step_name)
        config.logger.info(
            "Detected input without a clean paired-end-only library; skipping pair-aware steps: discover_merge_adapters, error_correct_1, error_correct_2, merge_reads."
        )
    if not (
        str(getattr(config, "artifacts", "artifacts")) != "artifacts"
        or getattr(config, "remove_synthetic_artifacts_enabled", False)
    ):
        if "remove_synthetic_artifacts" not in config.skip_steps:
            config.skip_steps.append("remove_synthetic_artifacts")
        config.logger.info(
            "Skipping remove_synthetic_artifacts unless --artifacts is provided or --remove-synthetic-artifacts is set."
        )

    steps = [
        # handle_input_fastq, # moved to outside of the steps to avoid ensures the input is "interleaved" by moving it through rename or reformat
        # filter_by_tile, # filters out reads by tile # dropped - breaks when the fastq headers are not pristine, and should not be used if multiple libraries are merged/concated
        # TODO: Replace bbduk with seal.sh for the rrna mapping - should be better at resolving the host composition, as bbduk (current) is geared for filtering, not resolving taxonomy from mapping.
        filter_known_dna,  # filters out known DNA sequences
        decontaminate_rrna,  # decontaminates rRNA sequences
        filter_identified_dna,  # filters out reads that are likely host (based on the stats file of the previous step)
        dedupe,  # removes duplicates (first round)
        discover_merge_adapters,  # discovers adapters for paired data when built-in adapters are active
        trim_adapters,  # trims adapters (clips off the adapters)
        trim_polya_tails,  # (optional) terminal polyA/polyT trimming
        remove_synthetic_artifacts,  # removes synthetic artifacts (phix etc)
        entropy_filter,  # removes reads with VERY low entropy (i.e. MOSTLY homopolymers).
        error_correct_1,  # error corrects the reads (first round - based on overlapping pairs, if applicable - ecco)
        error_correct_2,  # error corrects the reads (second round - based on ???, ecc)
        merge_reads,  # merges reads with insert size smaller than 2xread length (i.e. overlapping) # DONE: investigate if remaining adapters need to  also removed here... (UPDATE Brian says merged reads never retain adapter seq).
        quality_trim_unmerged,  # quality trims the unmerged reads
        # dedupe, # removes duplicates (second round - after the above processing some reads may have been "corrected"/modified and are now duplicates) NOTE! this is now done as part of process_reads (see Final deduplication step at the end of the pipeline.)
    ]

    current_input = fastq_file

    from rich.spinner import SPINNERS  # type: ignore

    config.logger.info("Starting read processing    ")
    SPINNERS["myspinner"] = {
        "interval": 2500 if config.log_level != 10 else 122500,
        "frames": ["🦠 ", "🧬 ", "🔬 "],
    }  # type: ignore
    # SPINNERS["myspinner"] = {"interval": 150 if config.log_level != 10 else 150, "frames":
    # [
    #     "🛸\u3000\u3000\u3000 ",
    #     "🛸\u3000\u3000\u3000 ",
    #     "🛸\u3000\u3000🐄 ",
    #     "🛸. . . 🐄 ",
    #     "🛸. .🐄. . ",
    #     "🛸🐄. . . ",
    #     "🛸✨\u3000\u3000 ",
    #     "🛸\u3000\u3000\u3000 "
    # ]
    # }

    with console.status(
        "[bold green] Working on     ",
        spinner="myspinner",  #
    ) as status:
        for step in steps:
            step_name = (
                step.__name__
            )  # if not callable(step) else step.__name__.split()[1]
            if step_name not in config.skip_steps:
                config.logger.info(f"Starting step: {step_name}   ")
                status.update(f"[bold green]Current Step: {step_name}   ")

                # Check for existing output file
                expected_output = Path(f"{step_name}_{config.file_name}.fq.gz")
                if config.skip_existing and check_existing_file(
                    expected_output
                ):
                    config.logger.info(
                        f"Skipping {step_name} as output file already exists"
                    )
                    current_input = expected_output
                    continue

                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(config.step_timeout)  # Set timeout to 10min

                try:
                    # config.logger.info(f"Running step: {step_name}")
                    result = step(current_input, config, output_tracker)
                except TimeoutError:
                    config.logger.error(
                        f"Step {step_name} timed out after {config.step_timeout} seconds"
                    )
                    continue
                finally:
                    signal.alarm(0)  # Disable the alarm

                if result == "host_empty":
                    config.logger.error(
                        "No potential host genomes identified. Skipping to the next step."
                    )
                    continue

                if isinstance(result, tuple):
                    current_input = output_tracker.get_latest_non_merged_file()
                else:
                    current_input = result

                config.logger.info(f"Finished step: {step_name}")
            else:
                config.logger.info(f"Skipping step: {step_name}")

    # Final deduplication step
    merged_file = output_tracker.get_latest_merged_file()
    if merged_file is not None and not (
        config.skip_existing
        and check_existing_file(Path(f"dedup_merged_{config.file_name}.fq.gz"))
    ):
        dedup_merged = dedupe(
            Path(merged_file), config, output_tracker, "final_merged"
        )  # noqa (F841)

    unmerged_file = output_tracker.get_latest_non_merged_file()
    if not (
        config.skip_existing
        and check_existing_file(
            Path(f"dedup_interleaved_{config.file_name}.fq.gz")
        )
    ):
        dedup_interleaved = dedupe(
            Path(unmerged_file), config, output_tracker, "final_interleaved"
        )  # noqa (F841)

    generate_reports(
        config.file_name, config.threads, config.skip_existing, config.logger
    )
    cleanup_and_move_files(config, output_tracker)
    # output_tracker.to_csv(f"{config.output_dir}/run_info/output_tracker.csv")
    if not config.keep_tmp:
        try:
            os.unlink(fastq_file)
        except Exception as e:
            config.logger.error(
                f"Error deleting input file {config.input}: {str(e)}"
            )
    config.logger.info("Read processing completed successfully.")


@click.command(no_args_is_help=True)
@click.option(
    "-t",
    "--threads",
    default=1,
    type=int,
    help="Number of threads to use. Example: -t 4",
)
@click.option(
    "-M", "-mem", "--memory", default="10gb", help="Memory. Example: -M 8gb"
)
@click.option(
    "-o",
    "-out",
    "--output",
    default=os.getcwd(),
    type=click.Path(),
    help="Output directory. Example: -o output",
)
@click.option(
    "--keep-tmp", is_flag=True, default=False, help="Keep temporary files"
)
@click.option(
    "-g",
    "--log-file",
    type=click.Path(),
    default=lambda: f"{os.getcwd()}/rolypoly.log",
    help="Path to log_file. Example: -g logfile.log, if not provided, a log file will be created in the current directory.",
)
@click.option(
    "-i",
    "-in",
    "--input",
    required=False,
    help="""Input raw reads file(s) or directory containing them. For paired-end reads, you can provide an interleaved file or the R1 and R2 files separated by comma. Example: -i sample_R1.fastq.gz,sample_R2.fastq.gz \n
If --input is a directory, all fastq files in the directory will be used - paired end files of the same base name would be assumed as from the same sample, otherwise a fastq is assumed interleaved. All interleaved and R1/R2 files would be concatenated into a single file before processing, and certain processing steps would be skipped as they assume a single sequencing library (error_correct_1, error_correct_2).""",
)
@click.option(
    "-D",
    "--known-dna",
    required=False,
    type=click.Path(exists=True),
    help="Fasta file of known DNA entities. Example: -D known_dna.fasta",
)
@click.option(
    "-s",
    "--speed",
    default=0,
    type=int,
    help="Set bbduk.sh speed value (0-15, where 0 uses all kmers and 15 skips most). Example: -s 5",
)
@click.option(
    "-se",
    "--skip-existing",
    is_flag=True,
    help="Skip steps if output files already exist",
)
@click.option(
    "-ss",
    "--skip-steps",
    default="",
    help="Comma-separated list of steps to skip. Example: --skip-steps filter_by_tile,entropy_filter",
)
@click.option(
    "--preset",
    type=click.Choice(sorted(FILTER_READS_PRESETS.keys())),
    default=None,
    help=(
        "Apply a named read-filtering preset (overrides individual step parameters unless "
        "those are given explicitly via --override-parameters). "
        + "  ".join(
            f"'{name}': {p['description']}"
            for name, p in FILTER_READS_PRESETS.items()
            if not p.get("not_implemented")
        )
    ),
)
@click.option(
    "--disable-auto",
    is_flag=True,
    default=False,
    help="Disable automatic trim/minlen tuning from detected read stats.",
)
@click.option(
    "--trim-polya",
    "--poly-selection",
    is_flag=True,
    default=False,
    help="Enable optional terminal polyA/polyT tail trimming after adapter trimming. Uses the trim_polya_tails preset and can be customized with --override-parameters.",
)
@click.option(
    "--adapters",
    required=False,
    type=click.Path(exists=True),
    default=None,
    help="Optional adapter FASTA to use instead of built-in (or discovered via bbmerge) adapters.",
)
@click.option(
    "--artifacts",
    required=False,
    type=click.Path(exists=True),
    default=None,
    help="Optional synthetic-artifact FASTA to use. Turns on --remove-synthetic-artifacts.",
)
@click.option(
    "--remove-synthetic-artifacts",
    "remove_synthetic_artifacts_enabled",
    is_flag=True,
    default=False,
    help="Enable the synthetic-artifact removal step using the built-in artifacts reference unless --artifacts is provided.",
)
@click.option(
    "-op",
    "-override-params",
    "--override-parameters",
    default=None,
    help='JSON-like string of parameters to override. Example: --override-parameters \'{"decontaminate_rrna": {"k": 29}, "trim_polya_tails": {"trimpolya": 28, "minlen": 30}}\'',
)
@click.option(
    "--config-file",
    required=False,
    type=click.Path(exists=True),
    help="Path to configuration file. Example: --config-file my_config.json",
)
@click.option(
    "-to",
    "-timeout",
    "--step-timeout",
    default=3600,
    type=int,
    help="Timeout for every step in the workflow in seconds. if you think the some processes are hanging (not terminated correctly) this would help debug that. Example: --timeout 600",
)
@click.option(
    "-n",
    "-name",
    "--file-name",
    required=False,
    type=str,
    help='Base name of the output files. Example: -file-name my_filtered_reads. If not set, default would be "rp_filtered_reads" unless the --input has a structure like somethingsomething_R1.fastq.gz,somethingsomething_R2.fastq.gz or somethingsomething.fastq.gz in which case it would be somethingsomething',
)
@click.option(
    "-ow",
    "--overwrite",
    is_flag=True,
    default=False,
    help="Do not overwrite the output directory if it already exists",
)
@click.option(
    "-z",
    "--zip-reports",
    is_flag=True,
    default=False,
    help="Zip the reports into a single file",
)
@click.option(
    "-ll",
    "--log-level",
    default="info",
    hidden=True,
    help="Log level. Options: debug, info, warning, error, critical",
)
@click.option(
    "--temp-dir",
    default=None,
    hidden=True,
    help="Directory for temporary files. If not provided, will create one inside the output directory.",
)
@click.option(
    "-mg",
    "--max-genomes",
    default=None,
    hidden=True,
    help="Maximum number of genomes to keep in the output. Example: --max-genomes 10",
)
def filter_reads(
    threads,
    memory,
    output,
    keep_tmp,
    log_file,
    input,
    known_dna,
    speed,
    preset,
    disable_auto,
    skip_existing,
    skip_steps,
    trim_polya,
    adapters,
    artifacts,
    remove_synthetic_artifacts_enabled,
    override_parameters,
    config_file,
    step_timeout,
    file_name,
    overwrite,
    zip_reports,
    log_level,
    max_genomes,
    temp_dir,
):
    """
    Process RNA-seq Illumina reads through the read-cleaning pipeline.

    The workflow combines host/contaminant removal, optional fetched-reference
    filtering, adapter/quality trimming, and optional error correction based on
    configured steps and speed presets.

    Input can be a single file, paired files, or a directory of FASTQ files.
    Use `--skip-steps` and `--override-parameters` to tailor the workflow.
    """
    from rolypoly.utils.logging.citation_reminder import remind_citations
    from rolypoly.utils.logging.loggit import log_start_info

    if (input is None) and (config_file is None):
        click.echo("Either input or config-file must be provided.")
        raise click.Abort

    global config, output_tracker
    output_tracker = OutputTracker()
    if config_file is not None:
        config = ReadFilterConfig.read(config_file)
    else:
        config = ReadFilterConfig(
            threads=threads,
            memory=memory,
            output=output,
            overwrite=overwrite,
            keep_tmp=keep_tmp,
            log_file=log_file,
            input=os.path.abspath(input),
            known_dna=known_dna,
            speed=speed,
            preset=preset,
            disable_auto=disable_auto,
            skip_existing=skip_existing,
            skip_steps=skip_steps,
            trim_polya=trim_polya,
            adapters=adapters,
            artifacts=artifacts,
            remove_synthetic_artifacts=remove_synthetic_artifacts_enabled,
            override_parameters=override_parameters,
            config_file=config_file,
            step_timeout=step_timeout,
            file_name=file_name,
            log_level=log_level,
            max_genomes=max_genomes,
            temp_dir=temp_dir,
            zip_reports=zip_reports,
        )

    if config.known_dna is None:
        config.skip_steps.append("filter_known_dna")
        config.logger.warning(
            "No known DNA file provided, known DNA filtering step will be skipped."
        )

    if not hasattr(config, "protected_step_params"):
        config.protected_step_params = collect_protected_step_params(
            getattr(config, "override_parameters", {})
        )
    if not hasattr(config, "disable_auto"):
        config.disable_auto = False
    if not hasattr(config, "user_provided_adapters"):
        config.user_provided_adapters = None
    if not hasattr(config, "adapters"):
        config.adapters = resolve_builtin_adapters(config)
    if not hasattr(config, "user_provided_artifacts"):
        config.user_provided_artifacts = None
    if not hasattr(config, "remove_synthetic_artifacts_enabled"):
        config.remove_synthetic_artifacts_enabled = False
    if not hasattr(config, "artifacts"):
        config.artifacts = "artifacts"

    active_preset = preset or getattr(config, "preset", None)
    config.preset = active_preset
    ctx = click.get_current_context()
    preset_protected = apply_filter_reads_preset(active_preset, ctx, config)
    config.protected_step_params.update(preset_protected)

    log_start_info(config.logger, config.__dict__)
    try:
        config.logger.info("Starting read processing    ")
        # config.logger.info(f"skip steps type is : {type(config.skip_steps)}")
        # config.logger.info(f"override parameters type is : {type(config.override_parameters)} {config.override_parameters} ")
        process_reads(config, output_tracker)
    except Exception as e:
        config.logger.error(
            f"An error occurred during read processing: {str(e)}"
        )
        raise

    config.logger.info("Read processing completed, probably successfully.")
    if config.log_level != 10:
        config.logger.info(
            f"remind citation is {os.environ.get('ROLYPOLY_REMIND_CITATIONS', 'not_set')}    "
        )
        with open(f"{config.log_file}", "a") as f_out:
            f_out.write(remind_citations(tools, return_bibtex=True) or "")


def probe_inputs(config: ReadFilterConfig) -> dict[str, Any]:
    """Create a reusable sampled probe subset for lightweight preflight analysis."""
    probe_dir = config.temp_dir / "probe_input"
    return probe_fastq_inputs(
        input_path=config.input,
        output_dir=probe_dir,
        sample_size=100000,
        subset_type="top_reads",
        include_single_end=False,
        logger=config.logger,
    )


def generate_reports(file_name: str, threads: int, skip_existing: bool, logger):
    import glob

    # Generate falco report
    falco_output = config.temp_dir / "falco_post_trim_reads"
    falco_output.mkdir(exist_ok=True)
    all_remaining_fastqs = glob.glob(
        str(config.temp_dir / "*final*.fq.gz"), recursive=True
    )

    if (
        not skip_existing
        or not (falco_output / f"merged_{file_name}_falco.html").exists()
    ):
        run_command_comp(
            base_cmd="falco",
            positional_args=[*all_remaining_fastqs],
            params={"t": str(threads), "outdir": str(falco_output)},
            assign_operator=" ",
            positional_args_location="end",
            logger=logger,
            # output_file=str(falco_output / f"{file_name}_falco.html"),
            skip_existing=skip_existing,
            check_status=True,
            check_output=False,
        )
        logger.info("falco report generated")
        tools.append("falco")
    else:
        logger.info("falco report already exists, skipping")


# Using the file_detection module instead of local implementation, below takes the library detection from there.
def process_input_fastq(
    config: ReadFilterConfig,
) -> tuple[Path, str, dict[str, Any]]:
    """Process input FASTQ files and prepare them for filtering."""
    from bbmapy import reformat
    from bbmapy.update import (
        ensure_java_availability,  # eventually will need to make sure (upsteream in bbmappy) to get a JRE even if there is java on path but of version <17...
    )

    ensure_java_availability()

    # Create a temporary file for intermediate concatenation
    temp_interleaved = config.output_dir / "temp_concat_interleaved.fq.gz"
    final_interleaved = config.output_dir / "concat_interleaved.fq.gz"

    # file detection functions now sourced from seperate script (21.08.2025)
    file_info = handle_input_fastq(config.input, logger=config.logger)
    file_name = file_info.get("file_name", "rolypoly_filtered_reads")

    # Process paired-end files
    if len(file_info["R1_R2_pairs"]) != 0:
        for i, pair in enumerate(file_info["R1_R2_pairs"]):
            out_file = temp_interleaved if i == 0 else final_interleaved
            config.logger.info(f"Concatenating {pair[0]} and {pair[1]}")
            bb_stdout, bb_stderr = reformat(
                in1=str(pair[0]),
                capture_output=True,
                in2=str(pair[1]),
                out=str(out_file),
                threads=config.threads,
                overwrite="t" if i == 0 else "f",
                append="f" if i == 0 else "t",
                Xmx=str(config.memory["giga"]),
            )
            config.logger.info(format_bbmapy_output(bb_stdout, bb_stderr))

    # Process interleaved files
    if len(file_info["interleaved_files"]) != 0:
        config.logger.info(
            f"Interleaved files: {file_info['interleaved_files']}"
        )
        for i, intfile in enumerate(file_info["interleaved_files"]):
            out_file = temp_interleaved if i == 0 else final_interleaved
            bb_stdout, bb_stderr = reformat(
                in_file=str(intfile),
                capture_output=True,
                out=str(out_file),
                threads=config.threads,
                overwrite="t" if i == 0 else "f",
                append="f" if i == 0 else "t",
                Xmx=str(config.memory["giga"]),
                int=True,
            )
            config.logger.info(format_bbmapy_output(bb_stdout, bb_stderr))

    # Process single-end files
    if (
        "single_end_files" in file_info
        and len(file_info["single_end_files"]) != 0
    ):
        config.logger.info(f"Single-end files: {file_info['single_end_files']}")
        for i, sefile in enumerate(file_info["single_end_files"]):
            out_file = (
                temp_interleaved
                if i == 0 and not temp_interleaved.exists()
                else final_interleaved
            )

            bb_stdout, bb_stderr = reformat(
                in_file=str(sefile),
                capture_output=True,
                out=str(out_file),
                threads=config.threads,
                overwrite="t"
                if i == 0 and not temp_interleaved.exists()
                else "f",
                append="f" if i == 0 and not temp_interleaved.exists() else "t",
                Xmx=str(config.memory["giga"]),
            )
            config.logger.info(format_bbmapy_output(bb_stdout, bb_stderr))

    # Clean up temporary file if it exists
    if temp_interleaved.exists():
        if final_interleaved.exists():
            temp_interleaved.unlink()
        else:
            temp_interleaved.rename(final_interleaved)

    if (
        len(file_info["R1_R2_pairs"]) > 1
        or len(file_info["interleaved_files"]) > 1
        or (
            "single_end_files" in file_info
            and len(file_info["single_end_files"]) > 1
        )
    ):
        config.skip_steps.append("filter_by_tile")
        config.skip_steps.append("error_correct_1")
        config.skip_steps.append("error_correct_2")
        config.logger.info(
            "Tile filtering and Error correction steps will be skipped as we concatenated fastq files from (cowardly assuming) multiple samples."
        )

    return final_interleaved, file_name, file_info


def filter_known_dna(
    input_file: Path, config: ReadFilterConfig, output_tracker: OutputTracker
) -> Path:
    """Filter known DNA sequences."""
    from bbmapy import bbduk

    from rolypoly.commands.reads.mask_dna import mask_dna

    ref_file = str(config.known_dna)
    if "mask_known_dna" not in config.skip_steps:
        ref_file = str(
            (
                Path(config.temp_dir)
                / f"masked_known_dna_{config.file_name}.fasta"
            ).absolute()
        )
        mask_args = {
            "threads": config.threads,
            "memory": config.memory["giga"],
            "output": ref_file,
            "flatten": False,
            "input": config.known_dna,
        }
        context = click.Context(mask_dna, ignore_unknown_options=True)
        context.invoke(mask_dna, **mask_args)

    output_file = config.temp_dir / f"filter_known_dna_{config.file_name}.fq.gz"
    try:
        params = config.step_params["filter_known_dna"]
        bb_stdout, bb_stderr = bbduk(
            in_file=str(input_file),
            capture_output=True,
            out=str(output_file),
            ref=str(ref_file),
            **params,
            Xmx=get_reduced_memory(config.memory),
            threads=str(config.threads),
            overwrite="t",
            interleaved="t",
            stats=config.temp_dir
            / f"stats_filter_known_dna_{config.file_name}.txt",
        )
        config.logger.info(format_bbmapy_output(bb_stdout, bb_stderr))

        output_tracker.add_file(
            str(output_file),
            "filter_known_dna",
            "bbduk.sh",
            is_merged=False,
            end_type=None,
            interleaved=True,
            is_gz=True,
        )
        return Path(output_file)
    except RuntimeError as e:
        config.logger.error(f"Error in filter_known_dna: {str(e)}")
        return input_file


def decontaminate_rrna(
    input_file: Path, config: ReadFilterConfig, output_tracker: OutputTracker
) -> Path:
    """Decontaminate rRNA sequences."""
    from bbmapy import bbduk

    output_file = (
        config.temp_dir / f"decontaminate_rrna_{config.file_name}.fq.gz"
    )
    rrna_fas1 = (
        Path(config.datadir)
        / "contam/rrna/ncbi_rRNA_all_sequences_masked_entropy.fasta"
    )  # type: ignore
    rrna_fas2 = (
        Path(config.datadir)
        / "contam/rrna/silva_rRNA_all_sequences_masked_entropy.fasta"
    )  # type: ignore
    try:
        params = config.step_params["decontaminate_rrna"]
        bb_stdout, bb_stderr = bbduk(
            in_file=str(input_file),
            out=str(output_file),
            ref=f"{rrna_fas1},{rrna_fas2}",
            **params,
            Xmx=get_reduced_memory(config.memory),
            threads=str(config.threads),
            overwrite="t",
            interleaved="t",
            stats=config.temp_dir
            / f"stats_decontaminate_rrna_{config.file_name}.txt",
            capture_output=True,
        )
        config.logger.info(format_bbmapy_output(bb_stdout, bb_stderr))

        output_tracker.add_file(
            str(output_file),
            "decontaminate_rrna",
            "bbduk.sh",
            is_merged=False,
            end_type=None,
            interleaved=True,
            is_gz=True,
        )
        return Path(output_file)
    except RuntimeError as e:
        config.logger.error(f"Error in decontaminate_rrna: {str(e)}")
        return input_file


def fetch_and_mask_genomes(config: ReadFilterConfig) -> Union[str, Path]:
    """Fetch and mask genomes."""
    from rolypoly.commands.reads.mask_dna import mask_dna
    from rolypoly.utils.bio.genome_fetch import fetch_genomes_from_stats_file

    # Create a dedicated subfolder for fetched genomes using absolute paths
    fetched_dna_dir = config.temp_dir / "fetched_dna" / "genomes"
    fetched_dna_dir.mkdir(parents=True, exist_ok=True)

    # Get absolute paths
    abs_gbs_file = (fetched_dna_dir / "gbs_50m.fasta").absolute()

    if "filter_identified_dna" not in config.skip_steps:
        stats_file = Path(
            config.temp_dir / f"stats_decontaminate_rrna_{config.file_name}.txt"
        ).absolute()
        if not stats_file.exists():
            config.logger.warning(
                f"Stats file {stats_file} not found. Skipping fetch and mask genomes step."
            )
            config.skip_steps.append("filter_identified_dna")
            return "host_empty"

        # Create a dedicated subfolder for fetched genomes using absolute paths
        fetched_dna_dir = config.temp_dir / "fetched_dna" / "genomes"
        fetched_dna_dir.mkdir(parents=True, exist_ok=True)

        # Get absolute paths
        abs_gbs_file = (fetched_dna_dir / "gbs_50m.fasta").absolute()
        abs_tmp_stats = (fetched_dna_dir / stats_file.name).absolute()

        # Copy the stats file to the genomes directory using absolute paths
        shutil.copy2(str(stats_file), str(abs_tmp_stats))

        # Get the mapping file path
        mapping_path = (
            Path(config.datadir) / "contam/rrna/rrna_to_genome_mapping.parquet"
        )

        # Run fetch_genomes_from_stats_file directly in the genomes directory with absolute paths
        fetch_genomes_from_stats_file(
            stats_file=str(abs_tmp_stats),
            taxid_lookup_path=str(mapping_path),
            output_file=str(abs_gbs_file),
            max_genomes=config.max_genomes,
            threads=config.threads,
            logger=config.logger,
        )
        if not abs_gbs_file.exists() or abs_gbs_file.stat().st_size < 20:
            config.logger.warning(
                "The file with the fetched genomes of identified potential hosts appears empty. Step will be skipped."
            )
            return "host_empty"

        # Clean up the copied stats file
        try:
            abs_tmp_stats.unlink()
        except Exception as e:
            config.logger.warning(
                f"Could not remove temporary stats file: {str(e)}"
            )

    if "mask_fetched_dna" not in config.skip_steps:
        config.logger.info("Masking fetched genomes")
        mask_args = {
            "threads": config.threads,
            "memory": config.memory["giga"],
            "output": str(
                fetched_dna_dir / f"masked_gbs_50m_{config.file_name}.fasta"
            ),
            "mask_low_complexity": True,
            "flatten": False,
            "input": str(abs_gbs_file),
        }
        context = click.Context(mask_dna, ignore_unknown_options=True)
        context.invoke(mask_dna, **mask_args)
        return fetched_dna_dir / f"masked_gbs_50m_{config.file_name}.fasta"
    return abs_gbs_file


def filter_identified_dna(
    input_file: Path, config: ReadFilterConfig, output_tracker: OutputTracker
) -> Union[Path, str]:
    """Filter fetched DNA genomes."""
    from bbmapy import bbduk

    host_file = fetch_and_mask_genomes(config)
    if host_file == "host_empty":
        return "host_empty"
    output_file = (
        config.temp_dir / f"filter_identified_dna_{config.file_name}.fq.gz"
    )
    try:
        params = config.step_params["filter_identified_dna"]
        bb_stdout, bb_stderr = bbduk(
            in_file=str(input_file),
            capture_output=True,
            out=str(output_file),
            ref=str(host_file),
            **params,
            Xmx=get_reduced_memory(config.memory),
            threads=str(config.threads),
            overwrite="t",
            interleaved="t",
            stats=config.temp_dir
            / f"stats_filter_identified_dna_{config.file_name}.txt",
        )
        config.logger.info(format_bbmapy_output(bb_stdout, bb_stderr))

        output_tracker.add_file(
            str(output_file),
            "filter_identified_dna",
            "bbduk.sh",
            is_merged=False,
            end_type=None,
            interleaved=True,
            is_gz=True,
        )
        return Path(output_file)
    except RuntimeError as e:
        config.logger.error(f"Error in filter_identified_dna: {str(e)}")
        return input_file


def dedupe(
    input_file: Path,
    config: ReadFilterConfig,
    output_tracker: OutputTracker,
    phase="first",
) -> Path:
    """Remove duplicate reads."""
    from bbmapy import clumpify

    if phase == "first":
        output_file = config.temp_dir / f"dedupe_first_{config.file_name}.fq.gz"
        is_merged = False
    elif phase == "final_merged":
        output_file = (
            config.temp_dir / f"dedupe_final_merged_{config.file_name}.fq.gz"
        )
        is_merged = True
    elif phase == "final_interleaved":
        output_file = (
            config.temp_dir
            / f"dedupe_final_interleaved_{config.file_name}.fq.gz"
        )
        is_merged = False
    try:
        params = config.step_params["dedupe"]
        bb_stdout, bb_stderr = clumpify(
            in_file=str(input_file),
            capture_output=True,
            out=str(output_file),
            **params,
            Xmx=get_reduced_memory(config.memory),
            threads=str(config.threads),
            overwrite="t",
            interleaved="t",
        )
        config.logger.info(format_bbmapy_output(bb_stdout, bb_stderr))

        output_tracker.add_file(
            str(output_file),
            f"dedupe_{phase}",
            "clumpify.sh",
            is_merged=is_merged,
            end_type=None,
            interleaved=True,
            is_gz=True,
        )
        return Path(output_file)
    except RuntimeError as e:
        config.logger.error(f"Error in dedupe_{phase}: {str(e)}")
        exit(1)
        # return input_file


def trim_adapters(
    input_file: Path, config: ReadFilterConfig, output_tracker: OutputTracker
) -> Path:
    """Trim adapters from reads."""
    from bbmapy import bbduk

    output_file = config.temp_dir / f"trim_adapters_{config.file_name}.fq.gz"
    try:
        params = config.step_params["trim_adapters"]
        bb_stdout, bb_stderr = bbduk(
            in_file=str(input_file),
            capture_output=True,
            out=str(output_file),
            ref=config.adapters,
            **params,
            Xmx=get_reduced_memory(config.memory),
            threads=str(config.threads),
            overwrite="t",
            interleaved="t",
            stats=config.temp_dir
            / f"stats_trim_adapters_{config.file_name}.txt",
        )
        config.logger.info(format_bbmapy_output(bb_stdout, bb_stderr))

        output_tracker.add_file(
            str(output_file),
            "trim_adapters",
            "bbduk.sh",
            is_merged=False,
            end_type=None,
            interleaved=True,
            is_gz=True,
        )
        return Path(output_file)
    except RuntimeError as e:
        config.logger.error(f"Error in trim_adapters: {str(e)}")
        exit(1)
        # return input_file


def remove_synthetic_artifacts(
    input_file: Path, config: ReadFilterConfig, output_tracker: OutputTracker
) -> Path:
    """Remove synthetic artifacts (phix etc) from reads."""
    from bbmapy import bbduk

    if not (
        str(getattr(config, "artifacts", "artifacts")) != "artifacts"
        or getattr(config, "remove_synthetic_artifacts_enabled", False)
    ):
        return input_file

    output_file = (
        config.temp_dir / f"remove_synthetic_artifacts_{config.file_name}.fq.gz"
    )
    try:
        params = config.step_params["remove_synthetic_artifacts"].copy()
        params["ref"] = config.artifacts
        bb_stdout, bb_stderr = bbduk(
            in_file=str(input_file),
            capture_output=True,
            out=str(output_file),
            **params,
            Xmx=get_reduced_memory(config.memory),
            threads=str(config.threads),
            overwrite="t",
            interleaved="t",
            stats=config.temp_dir
            / f"stats_remove_synthetic_artifacts_{config.file_name}.txt",
        )
        config.logger.info(format_bbmapy_output(bb_stdout, bb_stderr))

        output_tracker.add_file(
            str(output_file),
            "remove_synthetic_artifacts",
            "bbduk.sh",
            is_merged=False,
            end_type=None,
            interleaved=True,
            is_gz=True,
        )
        return Path(output_file)
    except RuntimeError as e:
        config.logger.error(f"Error in remove_synthetic_artifacts: {str(e)}")
        exit(1)
        # return input_file


def trim_polya_tails(
    input_file: Path, config: ReadFilterConfig, output_tracker: OutputTracker
) -> Path:
    """Trim terminal polyA/polyT tails from reads using BBduk."""
    from bbmapy import bbduk

    output_file = config.temp_dir / f"trim_polya_tails_{config.file_name}.fq.gz"
    try:
        params = config.step_params["trim_polya_tails"]
        bb_stdout, bb_stderr = bbduk(
            in_file=str(input_file),
            capture_output=True,
            out=str(output_file),
            **params,
            Xmx=get_reduced_memory(config.memory),
            threads=str(config.threads),
            overwrite="t",
            interleaved="t",
            stats=config.temp_dir
            / f"stats_trim_polya_tails_{config.file_name}.txt",
        )
        config.logger.info(format_bbmapy_output(bb_stdout, bb_stderr))

        output_tracker.add_file(
            str(output_file),
            "trim_polya_tails",
            "bbduk.sh",
            is_merged=False,
            end_type=None,
            interleaved=True,
            is_gz=True,
        )
        return Path(output_file)
    except RuntimeError as e:
        config.logger.error(f"Error in trim_polya_tails: {str(e)}")
        exit(1)
        # return input_file


def entropy_filter(
    input_file: Path, config: ReadFilterConfig, output_tracker: OutputTracker
) -> Path:
    """Apply entropy filter to reads."""
    from bbmapy import bbduk

    output_file = config.temp_dir / f"entropy_filter_{config.file_name}.fq.gz"
    try:
        params = config.step_params["entropy_filter"]
        bb_stdout, bb_stderr = bbduk(
            in_file=str(input_file),
            capture_output=True,
            out=str(output_file),
            **params,
            Xmx=get_reduced_memory(config.memory),
            threads=str(config.threads),
            overwrite="t",
            interleaved="t",
        )
        config.logger.info(format_bbmapy_output(bb_stdout, bb_stderr))

        output_tracker.add_file(
            str(output_file),
            "entropy_filter",
            "bbduk.sh",
            is_merged=False,
            end_type=None,
            interleaved=True,
            is_gz=True,
        )
        return Path(output_file)
    except RuntimeError as e:
        config.logger.error(f"Error in entropy_filter: {str(e)}")
        exit(1)
        # return input_file


def error_correct_1(
    input_file: Path, config: ReadFilterConfig, output_tracker: OutputTracker
) -> Path:
    """Perform error correction on reads."""
    from bbmapy import bbmerge

    output_file = config.temp_dir / f"error_correct_1{config.file_name}.fq.gz"
    try:
        params = config.step_params["error_correct_1"]
        bb_stdout, bb_stderr = bbmerge(
            in_file=str(input_file),
            capture_output=True,
            out=str(output_file),
            **params,
            Xmx=get_reduced_memory(config.memory),
            threads=str(config.threads),
            overwrite="t",
            interleaved="t",
        )
        config.logger.info(format_bbmapy_output(bb_stdout, bb_stderr))

        output_tracker.add_file(
            str(output_file),
            "error_correct_phase_1",
            "bbmerge.sh",
            is_merged=False,
            end_type=None,
            interleaved=True,
            is_gz=True,
        )
        return Path(output_file)
    except RuntimeError as e:
        config.logger.error(f"Error in error_correct_1: {str(e)}")
        exit(1)
        # return input_file


def error_correct_2(
    input_file: Path, config: ReadFilterConfig, output_tracker: OutputTracker
) -> Path:
    """Perform error correction on reads."""
    from bbmapy import clumpify

    output_file = config.temp_dir / f"error_correct_2{config.file_name}.fq.gz"
    try:
        params = config.step_params["error_correct_2"]
        bb_stdout, bb_stderr = clumpify(
            in_file=str(input_file),
            capture_output=True,
            out=str(output_file),
            **params,
            Xmx=get_reduced_memory(config.memory),
            threads=str(config.threads),
            overwrite="t",
            interleaved="t",
        )
        config.logger.info(format_bbmapy_output(bb_stdout, bb_stderr))

        output_tracker.add_file(
            str(output_file),
            "error_correct_phase_2",
            "bbmerge.sh",
            is_merged=False,
            end_type=None,
            interleaved=True,
            is_gz=True,
        )
        return Path(output_file)
    except RuntimeError as e:
        config.logger.error(f"Error in error_correct_phase_2: {str(e)}")
        exit(1)
        # return input_file


def merge_reads(
    input_file: Path, config: ReadFilterConfig, output_tracker: OutputTracker
) -> Tuple[Path, Path]:
    """Merge paired-end reads."""
    from bbmapy import bbmerge

    output_file = config.temp_dir / f"merged_{config.file_name}.fq.gz"
    unmerged_file = config.temp_dir / f"unmerged_{config.file_name}.fq.gz"
    try:
        params = config.step_params["merge_reads"]
        bb_stdout, bb_stderr = bbmerge(
            in_file=str(input_file),
            capture_output=True,
            out=str(output_file),
            outu=str(unmerged_file),
            **params,
            Xmx=get_reduced_memory(config.memory),
            threads=str(config.threads),
            overwrite="t",
            interleaved="t",
            simd=True,  # assumes simd support, avx256 and java >=17 are required.
            outadapter=config.temp_dir
            / f"out_adapter_merged_{config.file_name}.txt",
            strict="true",
        )
        config.logger.info(format_bbmapy_output(bb_stdout, bb_stderr))

        output_tracker.add_file(
            str(output_file),
            "merge_reads",
            "bbmerge.sh",
            is_merged=True,
            end_type="paired",
            interleaved=True,
            is_gz=True,
        )
        output_tracker.add_file(
            str(unmerged_file),
            "merge_reads_unmerged",
            "bbmerge.sh",
            is_merged=False,
            end_type="single",
            interleaved=False,
            is_gz=True,
        )
        return Path(output_file), Path(unmerged_file)
    except RuntimeError as e:
        config.logger.error(f"Error in merge_reads: {str(e)}")
        exit(1)
        # return input_file, None


def quality_trim_unmerged(
    input_file: Path, config: ReadFilterConfig, output_tracker: OutputTracker
) -> Path:
    """Quality trim unmerged reads."""
    from bbmapy import bbduk

    input_file = Path(output_tracker.get_latest_non_merged_file())
    output_file = config.temp_dir / f"qtrimmed_{config.file_name}.fq.gz"
    try:
        params = config.step_params["quality_trim_unmerged"]
        bb_stdout, bb_stderr = bbduk(
            in_file=str(input_file),
            capture_output=True,
            out=str(output_file),
            **params,
            Xmx=get_reduced_memory(config.memory),
            threads=str(config.threads),
            overwrite="t",
            interleaved="t",
        )
        config.logger.info(format_bbmapy_output(bb_stdout, bb_stderr))

        output_tracker.add_file(
            str(output_file),
            "quality_trim_unmerged",
            "bbduk.sh",
            is_merged=False,
            end_type="single",
            interleaved=False,
            is_gz=True,
        )
        return Path(output_file)
    except RuntimeError as e:
        config.logger.error(f"Error in quality_trim_unmerged: {str(e)}")
        exit(1)
        # return input_file


def cleanup_and_move_files(
    config: ReadFilterConfig, output_tracker: OutputTracker
):
    """Clean up and move files to their final locations.
    Args:
        output_tracker: Tracks output files
        config: Configuration object
    """
    # Ensure all paths are absolute
    temp_dir = Path(config.temp_dir).resolve()
    output_dir = Path(config.output_dir).resolve()

    # Create run_info directory in the final output location
    run_info_dir = output_dir / "run_info"
    run_info_dir.mkdir(parents=True, exist_ok=True)

    # Move config file and output tracker CSV to run_info
    config.save(run_info_dir / "rp_filter_reads_config.json")
    output_tracker.to_csv(run_info_dir / "output_tracker.csv")

    # Move fastqc/falco reports to run_info
    for pattern in ["*fastqc*", "falco*"]:
        for qc_dir in temp_dir.glob(pattern):
            if qc_dir.exists():
                # breakpoint()
                config.logger.info(f"Moving {qc_dir} to run_info")
                try:
                    target = run_info_dir / qc_dir.name
                    if target.exists():
                        shutil.rmtree(str(target))
                    shutil.move(str(qc_dir), str(target))
                except Exception as e:
                    config.logger.warning(f"Could not move {qc_dir}: {str(e)}")

    # Move all stats and adapter files to run_info
    for pattern in [
        "stats_*.txt",
        "out_adapter_*.txt",
        "bbmerge_discovered_*.fa",
        "validated_bbmerge_discovered_*.fa",
    ]:
        for stat_file in temp_dir.glob(pattern):
            if stat_file.exists():
                try:
                    shutil.move(
                        str(stat_file), str(run_info_dir / stat_file.name)
                    )
                except Exception as e:
                    config.logger.warning(
                        f"Could not move {stat_file} to run_info: {str(e)}"
                    )

    # Get final output files
    final_merged = output_tracker.get_latest_merged_file()
    final_interleaved = output_tracker.get_latest_non_merged_file()

    # Move final output files to the output directory
    for item in [final_merged, final_interleaved]:
        if item and Path(item).exists():
            try:
                target = output_dir / Path(item).name
                if target.exists():
                    target.unlink()
                shutil.move(str(item), str(target))
            except Exception as e:
                config.logger.warning(
                    f"Could not move {item} to output directory: {str(e)}"
                )

    # If keeping temporary files, move fetched_dna to run_info
    if config.keep_tmp:
        fetched_dna_dir = temp_dir / "fetched_dna"
        if fetched_dna_dir.exists():
            try:
                target = run_info_dir / "fetched_dna"
                if target.exists():
                    shutil.rmtree(str(target))
                shutil.move(str(fetched_dna_dir), str(target))
            except Exception as e:
                config.logger.warning(
                    f"Could not move fetched_dna directory: {str(e)}"
                )

    # Clean up temporary directory if not keeping it
    if not config.keep_tmp and temp_dir != output_dir:
        try:
            shutil.rmtree(temp_dir)
            config.logger.info(
                f"Temporary directory {temp_dir} cleaned up and removed"
            )
        except Exception as e:
            config.logger.error(f"Error removing temporary directory: {str(e)}")
            # Don't raise the error since the important files *should* have been moved
    if config.zip_reports:
        shutil.make_archive(
            base_name=config.output_dir / f"{config.file_name}_run_info",
            format="gztar",
            root_dir=str(run_info_dir),
        )
        config.logger.info(
            f"Zipped run_info directory to {config.output_dir / f'{config.file_name}_run_info.tar.gz'}"
        )
        shutil.rmtree(run_info_dir)
        # breakpoint()


# TODO: Add option to save specific intermediate files, like the host/rRNA mapped fastqs.
# TODO: Figure out how to handle --skip-existing + --overwrite (on by default) and --keep-tmp together and --tmp-dir no being provided (maybe look for the most recent temp dir looking folder?)
# TODO: add unit tests (done for an input paired end interleaved fastq file, need to add for multiple, and for single end, and for combo)
