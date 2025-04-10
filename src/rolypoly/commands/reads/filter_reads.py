import os
from pathlib import Path
import rich_click as click
from rich.console import Console
from rolypoly.utils.config import BaseConfig
from rolypoly.utils.loggit import log_start_info
from rolypoly.utils.various import check_dependencies, ensure_memory, change_directory
from rolypoly.utils.citation_reminder import remind_citations
from rolypoly.utils.output_tracker import OutputTracker
import shutil
import subprocess
import time
import signal
from typing import Dict, Tuple, Union
# import ast

global tools
tools = ["bbmap", "seqkit", "datasets"]

global output_tracker
output_tracker = OutputTracker()

class ReadFilterConfig(BaseConfig):
    def __init__(self, **kwargs):
        # in this case output_dir and output are the same, so need to explicitly make sure it exists.
        # if not Path(kwargs.get("output")).exists():
        #     kwargs["output_dir"] = kwargs.get("output")
        #     Path(kwargs.get("output")).mkdir(parents=True, exist_ok=True)
            
        super().__init__(input=kwargs.get("input"),
                         output=kwargs.get("output"),
                         keep_tmp=kwargs.get("keep_tmp"),
                         log_file=kwargs.get("log_file"),
                         threads=kwargs.get("threads"),
                         memory=kwargs.get("memory"),
                         config_file=kwargs.get("config_file"),
                         overwrite=kwargs.get("overwrite"),
                         log_level=kwargs.get("log_level")
                         ) # initialize the BaseConfig class
        # initialize the rest of the parameters (i.e. the ones that are not in the BaseConfig class)
        self.skip_existing = kwargs.get("skip_existing") or False
        # self.override_parameters = self.override_parameters if isinstance(self.override_parameters, dict) else eval(self.override_parameters) if isinstance(self.override_parameters, str) else {}
        self.skip_steps = kwargs.get("skip_steps") if isinstance(kwargs.get("skip_steps"), list) else kwargs.get("skip_steps").split(",") if isinstance(kwargs.get("skip_steps"), str) else []
        self.known_dna = Path(kwargs.get("known_dna")).resolve() if kwargs.get("known_dna") is not None else None
        self.speed = kwargs.get("speed") or 0
        self.skip_existing = kwargs.get("skip_existing") or False
        self.override_parameters  = kwargs.get("override_parameters") if isinstance(kwargs.get("override_parameters"), dict) else eval(kwargs.get("override_parameters")) if isinstance(kwargs.get("override_parameters"), str) else {}
        # self.skip_steps = skip_steps if isinstance(skip_steps, list) else skip_steps.split(",")  
        self.step_timeout = kwargs.get("step_timeout") or 3600 # 3600 seconds
        self.file_name = kwargs.get("file_name") or "rp_filtered_reads" # this is the base name of the output files, if not provided, it will be "rp_filtered_reads"
        self.step_params = { # these are the default parameters for each step, if not overridden by the user
            'filter_by_tile': {'nullifybrokenquality': 't'},
            'filter_known_dna': {'k': 31, 'mincovfraction': 0.7, 'hdist': 0},
            'decontaminate_rrna': {'k': 31, 'mincovfraction': 0.5, 'hdist': 0},
            'filter_identified_dna': {'k': 31, 'mincovfraction': 0.7, 'hdist': 0},
            'dedupe': {'dedupe': True, 'passes': 1, 's': 0},
            'trim_adapters': {'ktrim': 'r', 'k': 23, 'mink': 11, 'hdist': 1, 'tpe': 't', 'tbo': 't', 'minlen': 45},
            'remove_synthetic_artifacts': {'k': 31},
            'entropy_filter': {'entropy': 0.01, 'entropywindow': 30},
            'error_correct_1': {'ecco': True, 'mix': 't', "ordered" :'t'},
            'error_correct_2': {'ecc': True, 'reorder': True, 'nullifybrokenquality': True, 'passes': 1},
            'merge_reads': {'k': 93, 'extend2': 80, 'rem': True, 'mix': "f"},
            'quality_trim_unmerged': {'qtrim': 'rl', 'trimq': 5, 'minlen': 45}
        }
        if kwargs.get("override_parameters") is not None:
            self.logger.info(f"override_parameters: {kwargs.get('override_parameters')}")
            for step, params in kwargs.get('override_parameters').items():
                if step in self.step_params:
                    self.step_params[step].update(params)
                else:
                    self.logger.warning(f"Warning: Unknown step '{step}' in override_parameters. Ignoring.")

def timeout_handler(signum, frame):
    # os.kill(os.getpid(), signal.SIGKILL)
    raise TimeoutError("Function call timed out")

def check_existing_file(output_file: Path, min_size: int = 20) -> bool:
    return output_file.exists() and output_file.stat().st_size > min_size

def process_reads(config: ReadFilterConfig, output_tracker: OutputTracker) -> OutputTracker:
    """Main function to orchestrate the preprocessing steps."""
    config.logger.info(f"Checking dependencies    ")
    check_dependencies([ "seqkit", "datasets"])
    with change_directory(config.tmp_dir):
        config.save("rp_filter_reads_config.json") # type: ignore
        fastq_file, config.file_name = handle_input_fastq(config)
        config.logger.info(f"file_name: {config.file_name}")
        config.logger.info(f"fastq_file: {fastq_file}")
        output_tracker.add_file(filename=str(config.input), command = "handle_input_fastq", command_name="reformat", is_merged=False) # retroactive addition
        config.memory = ensure_memory(config.memory, fastq_file) # type: ignore ------ this second ensure is because we now have the fastq file to check its size.
        steps = [
            # handle_input_fastq, # moved to outside of the steps to avoid ensures the input is interleaved by moving it through rename or reformat
            # filter_by_tile, # filters out reads by tile
            filter_known_dna, # filters out known DNA sequences
            decontaminate_rrna, # decontaminates rRNA sequences
            filter_identified_dna, # filters out reads that are likely host (based on the stats file of the previous step)
            dedupe, # removes duplicates (first round)
            trim_adapters, # trims adapters (clips off the adapters)
            remove_synthetic_artifacts, # removes synthetic artifacts (phix etc)
            entropy_filter, # removes reads with low entropy (poor quality)
            error_correct_1, # error corrects the reads (first round)
            error_correct_2, # error corrects the reads (second round)
            merge_reads, # merges reads with negative insert size (i.e. overlapping)
            quality_trim_unmerged, # quality trims the unmerged reads
            # dedupe, # removes duplicates (second round - after the above processing some reads may have been "corrected"/modified and are now duplicates)
        ]

        current_input = fastq_file
        from rich.spinner import Spinner, SPINNERS # type: ignore
        SPINNERS['myspinner'] = {'interval': 2500, 'frames': ['🦠 ', '🧬 ', '🔬 ']}
        console=Console(width=150)

        with console.status(f"[bold green] Working on     ", spinner="myspinner") as status:
            for step in steps:
                step_name = step.__name__ #if not callable(step) else step.__name__.split()[1]
                if step_name not in config.skip_steps:
                    config.logger.info(f"Starting step: {step_name}   ")
                    status.update(f"[bold green]Current Step: {step_name}   ")

                    # Check for existing output file
                    expected_output = Path(f"{step_name}_{config.file_name}.fq.gz")
                    if config.skip_existing and check_existing_file(expected_output):
                        config.logger.info(f"Skipping {step_name} as output file already exists")
                        current_input = expected_output
                        continue

                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(config.step_timeout)  # Set timeout to 10min

                    try:
                        # config.logger.info(f"Running step: {step_name}")
                        result = step(current_input, config, output_tracker)
                    except TimeoutError:
                        config.logger.error(f"Step {step_name} timed out after {config.step_timeout} seconds")
                        continue
                    finally:
                        signal.alarm(0)  # Disable the alarm
                        
                    if result == "host_empty":
                        config.logger.error("No potential host genomes identified. Skipping to the next step.")
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
        if not (config.skip_existing and check_existing_file(Path(f"dedup_merged_{config.file_name}.fq.gz"))):
            dedup_merged = dedupe(Path(merged_file), config, output_tracker, "final_merged")

        unmerged_file = output_tracker.get_latest_non_merged_file()
        if not (config.skip_existing and check_existing_file(Path(f"dedup_interleaved_{config.file_name}.fq.gz"))):
            dedup_interleaved = dedupe(Path(unmerged_file), config, output_tracker, "final_interleaved")


        generate_reports(config.file_name, config.threads, config.skip_existing, config.logger)
        cleanup_and_move_files(config.tmp_dir, config.keep_tmp, output_tracker, config)
        # output_tracker.to_csv(f"{config.output_dir}/run_info/output_tracker.csv")
        if not config.keep_tmp:
            try:
                os.unlink(fastq_file)
            except Exception as e:
                config.logger.error(f"Error deleting input file {config.input}: {str(e)}")
        config.logger.info("Read processing completed successfully.")

# monitor=f

console = Console()
@click.command(no_args_is_help=True)
@click.option('-t', '--threads', default=1, type=int, help='Number of threads to use. Example: -t 4')
@click.option('-M',"-mem", '--memory', default='6gb', help='Memory. Example: -M 8gb')
@click.option('-o',"-out", '--output', default=os.getcwd(), type=click.Path(), help='Output directory. Example: -o output')
@click.option('--keep-tmp', is_flag=True, default=False, help="Keep temporary files")
@click.option('-g', '--log-file', type=click.Path(), default=lambda: f"{os.getcwd()}/rolypoly.log", help='Path to log_file. Example: -g logfile.log, if not provided, a log file will be created in the current directory.')
@click.option('-i',"-in", '--input', required=False, help="""Input raw reads file(s) or directory containing them. For paired-end reads, you can provide an interleaved file or the R1 and R2 files separated by comma. Example: -i sample_R1.fastq.gz,sample_R2.fastq.gz \n
If --input is a directory, all fastq files in the directory will be used - paired end files of the same base name would be assumed as from the same sample, otherwise a fastq is assumed interleaved. All interleaved and R1/R2 files would be concatenated into a single file before processing, and certain processing steps would be skipped as they assume a single sequencing library (filter_by_tile, error_correct_1, error_correct_2).""")
@click.option('-D', '--known-dna', required=False, type=click.Path(exists=True), help='Fasta file of known DNA entities. Example: -D known_dna.fasta')
@click.option('-s', '--speed', default=0, type=int, help='Set bbduk.sh speed value (0-15, where 0 uses all kmers and 15 skips most). Example: -s 5')
@click.option("-se",'--skip-existing', is_flag=True, help='Skip steps if output files already exist')
@click.option("-ss",'--skip-steps', default='', help='Comma-separated list of steps to skip. Example: --skip-steps filter_by_tile,entropy_filter')
@click.option("-op","-override-params",'--override-parameters', default=None, help='JSON-like string of parameters to override. Example: --override-parameters \'{"decontaminate_rrna": {"k": 29}, "filter_dna_genomes": {"mincovfraction": 0.8}}\'')
@click.option('--config-file',required=False, type=click.Path(exists=True), help='Path to configuration file. Example: --config-file my_config.json')
@click.option("-to",'-timeout', '--step-timeout', default=3600, type=int, help='Timeout for every step in the workflow in seconds. if you think the some processes are hanging (not terminated correctly) this would help debug that. Example: --timeout 600')
@click.option("-n",'-name', '--file-name', required=False,   type=str, help='Base name of the output files. Example: -file-name my_filtered_reads. If not set, default would be "rp_filtered_reads" unless the --input has a structure like somethingsomething_R1.fastq.gz,somethingsomething_R2.fastq.gz or somethingsomething.fastq.gz in which case it would be somethingsomething')
@click.option("-ow","--overwrite", is_flag=True, default=False, help='Do not overwrite the output directory if it already exists')
@click.option("-ll","--log-level", default="info", hidden=True, help='Log level. Options: debug, info, warning, error, critical')
def filter_reads(threads, memory, output, keep_tmp, log_file, input, known_dna, speed, skip_existing, skip_steps, override_parameters, config_file, step_timeout, file_name, overwrite, log_level):
    """
    Process RNA-seq (transcriptome, RNA virome, metatranscriptomes) Illumina raw reads.
    Removes host reads, synthetic artifacts, and unknown DNA, corrects sequencing errors, trims adapters and low quality reads.
    """

    # from bbmapy import bbduk, bbmerge, clumpify, filterbytile, bbmap, rename, reformat
    # from rolypoly.utils.fax import fetch_genomes, mask_dna

    if (input is None) and (config_file is None):
        click.echo("Either input or config-file must be provided.")
        raise click.Abort

    global config
    if config_file is not None:
        config = ReadFilterConfig.load(config_file)
    else:
        config = ReadFilterConfig(threads=threads,
                              memory=memory,
                              output=output, 
                              overwrite=overwrite,
                              keep_tmp=keep_tmp, 
                              log_file=log_file, 
                              input=os.path.abspath(input), 
                              known_dna=known_dna, 
                              speed=speed, 
                              skip_existing=skip_existing, 
                              skip_steps=skip_steps, 
                              override_parameters=override_parameters, 
                              config_file=config_file, 
                              step_timeout=step_timeout, 
                              file_name=file_name,
                              log_level=log_level)
    
    if config.known_dna is None:
        config.skip_steps.append("filter_known_dna")
        config.logger.warning("No known DNA file provided, known DNA filtering step will be skipped.")

    log_start_info(config.logger,config.__dict__)
    try:
        config.logger.info(f"Starting read processing    ")
        # config.logger.info(f"skip steps type is : {type(config.skip_steps)}")
        # config.logger.info(f"override parameters type is : {type(config.override_parameters)} {config.override_parameters} ")
        process_reads(config, output_tracker)
    except Exception as e:
        config.logger.error(f"An error occurred during read processing: {str(e)}")
        raise
    
    config.logger.info(f"Read processing completed, probably successfully.")
    # remind_citations(tools)
    with open(f"{config.log_file}","w") as f_out:
        f_out.write(remind_citations(tools,return_bibtex=True))


def generate_reports(file_name: str, threads: int, skip_existing: bool, logger):
    import glob
    # Generate FastQC report
    fastqc_output = Path("FastQC_post_trim_reads")
    fastqc_output.mkdir(exist_ok=True)
    all_remaining_fastqs = glob.glob(f"*final*.fq.gz", recursive=True)
    fastqc_cmd = ['fastqc', '-t', str(threads), *all_remaining_fastqs, '-o', str(fastqc_output)]

    if not skip_existing or not (fastqc_output / f"merged_{file_name}_fastqc.html").exists():
        subprocess.run(fastqc_cmd, check=True)
        logger.info("FastQC report generated")
    else:
        logger.info("FastQC report already exists, skipping")
        tools.append("fastqc")
    # Run MultiQC
    multiqc_output = Path(f'{file_name}_multiqc')
    multiqc_cmd = ['multiqc', './', '--outdir', str(multiqc_output)]

    if not skip_existing or not (multiqc_output / "multiqc_report.html").exists():
        subprocess.run(multiqc_cmd, check=True)
        logger.info("MultiQC report generated")
        tools.append("multiqc")
    else:
        logger.info("MultiQC report already exists, skipping")
    
    shutil.rmtree(fastqc_output)
   
   
def identify_read_pair_files_in_folder(input: Union[str, Path]) -> Dict[str, list[Path]]:
    """Identify if the input is paired end or single end."""

    import glob
    from pathlib import Path

    if not Path(input).is_dir():
        print(f"input: {input} is not a directory")
        return None
    print(f"scanning: {input} for fastq files")
    fastq_files = glob.glob(f"{input}/*", recursive=True)

    fastq_files = [f for f in fastq_files if f.endswith(('.fq.gz', '.fastq.gz', '.fastq', '.fq'))]
    if len(fastq_files) == 0:
        print(f"No fastq files found in {input}")
        return None
    print(f"fastq_files: {fastq_files}")
    
    # Find all files containing "_R1" and their potential "_R2" pairs
    R1_files = [f for f in fastq_files if "_R1_" in f]
    R2_files = [f for f in fastq_files if "_R2_" in f]
    interleaved_files = list(set(fastq_files).difference(set(R1_files)).difference(set(R2_files)))
    print(f"interleaved_files: {interleaved_files}")
    print(f"R1_files: {R1_files}")
    print(f"R2_files: {R2_files}")  

    # Match R1 and R2 files based on name replacement
    R1_R2_pairs = []
    for R1_file in R1_files:
        R2_file = R1_file.replace("_R1_", "_R2_")
        if R2_file in R2_files:
            R1_R2_pairs.append((R1_file, R2_file))
    print(f"R1_R2_pairs: {R1_R2_pairs}")
    return {"R1_R2_pairs": R1_R2_pairs, "interleaved_files": interleaved_files}

def handle_input_fastq(config: ReadFilterConfig) -> tuple[Path, str]:
    ### TODO: Test format of each fastq/gz (is paired end or single end, interleaved or not, gzipped or not)

    from bbmapy import reformat
    
    # Create a temporary file for intermediate concatenation
    temp_interleaved = config.output_dir / f"temp_concat_interleaved.fq.gz"
    final_interleaved = config.output_dir / f"concat_interleaved.fq.gz"
    file_name = "rolypoly_filtered_reads"
    
    if "," not in str(config.input):
        if Path(config.input).is_dir():
            file_info = identify_read_pair_files_in_folder(config.input)
            if file_info is None:
                raise ValueError(f"No fastq files found in {config.input}")
        else:
            file_info = {"R1_R2_pairs": [], "interleaved_files": [config.input]}
    else:
        input_files = str(config.input).split(",")
        input_files = [Path(f.strip()).resolve() for f in input_files]
        file_info = {"R1_R2_pairs": [input_files], "interleaved_files": []}

    # Process paired-end files
    if len(file_info["R1_R2_pairs"]) != 0:
        for i, pair in enumerate(file_info["R1_R2_pairs"]):
            out_file = temp_interleaved if i == 0 else final_interleaved
            config.logger.info(f"Concatenating {pair[0]} and {pair[1]}")
            reformat(
                in1=str(pair[0]),
                in2=str(pair[1]),
                out=str(out_file),
                threads=config.threads,
                overwrite="t" if i == 0 else "f",
                fixheaders="t",
                append="f" if i == 0 else "t",
                Xmx=str(config.memory["giga"])
            )
        file_name = Path(pair[0]).stem.split('_R1')[0]
        
    # Process interleaved files
    if len(file_info["interleaved_files"]) != 0:
        config.logger.info(f"Interleaved files: {file_info['interleaved_files']}")
        # config.logger.info(f"memory: {config.memory['giga']}")
        for i, intfile in enumerate(file_info["interleaved_files"]):
            out_file = temp_interleaved if i == 0 else final_interleaved
            reformat(
                in_file=str(intfile),
                out=str(out_file),
                threads=config.threads,
                overwrite="t" if i == 0 else "f",
                fixheaders="t",
                append="f" if i == 0 else "t",
                Xmx=str(config.memory["giga"]),
                int=True
            )
        file_name = Path(file_info["interleaved_files"][0]).stem.split('.f')[0]

    # Clean up temporary file if it exists
    if temp_interleaved.exists():
        if final_interleaved.exists():
            temp_interleaved.unlink()
        else:
            temp_interleaved.rename(final_interleaved)

    if len(file_info["R1_R2_pairs"]) > 1 or len(file_info["interleaved_files"]) > 1:
        file_name = "rp_filtered_reads"
        config.skip_steps.append("filter_by_tile")
        config.skip_steps.append("error_correct_1")
        config.skip_steps.append("error_correct_2")
        config.logger.info("Tile filtering and Error correction steps will be skipped as we concatenated fastq files from (cowardly assuming) multiple samples.")
    
    return final_interleaved, file_name

def filter_by_tile(input_file: Path, config: ReadFilterConfig, output_tracker: OutputTracker) -> Path:
    """Filter reads by tile.""" # this commands tends to break so it has extra logging and is skipped if error occurs.

    from bbmapy import filterbytile
    config.logger.info(f"Starting: Filter by Tile for file {input_file}")
    output_file = f"filter_by_tile_{config.file_name}.fq.gz"
    
    try:
        # config.logger.info("Preparing to run filterbytile")
        params = config.step_params['filter_by_tile']
        config.logger.info(f"Parameters for filterbytile: {params}")
        
        start_time = time.time()
        filterbytile(
            in_file=str(input_file),
            out=str(output_file),
            Xmx=config.memory["giga"],
            threads=str(config.threads),
            overwrite="t",
            interleaved="t",
            # monitor= config.step_timeout,
            **params
        )
        end_time = time.time()
        
        config.logger.info(f"filterbytile completed in {end_time - start_time:.2f} seconds")
        
        if not Path(output_file).exists():
            raise FileNotFoundError(f"Expected output file {output_file} was not created")
        
        output_tracker.add_file(str(output_file), "filter_by_tile", "filterbytile.sh", is_merged=False)
        return Path(output_file)
    except Exception as e:
        config.logger.error(f"Error in filter_by_tile: {str(e)}")
        config.logger.error("Proceeding with last usable input")
        return input_file

def filter_known_dna(input_file: Path, config: ReadFilterConfig, output_tracker: OutputTracker) -> Path:
    """Filter known DNA sequences."""

    from rolypoly.utils.fax import mask_dna
    from bbmapy import bbduk
    ref_file = str(config.known_dna)
    if 'mask_known_dna' not in config.skip_steps:
        ref_file = f"masked_known_dna_{config.file_name}.fasta"
        mask_args = {
            "threads": config.threads,
            "memory": config.memory["giga"],
            "output": ref_file,
            "flatten": False,
            "input": config.known_dna
        }
        context = click.Context(mask_dna, ignore_unknown_options=True)
        context.invoke(mask_dna, **mask_args)

    output_file = f"filter_known_dna_{config.file_name}.fq.gz"
    try:
        params = config.step_params['filter_known_dna']
        bbduk(
            in_file=str(input_file),
            out=str(output_file),
            ref=str(ref_file),
            **params,
            Xmx=config.memory["giga"],
            threads=str(config.threads),
            overwrite="t",
            interleaved="t",
            stats=f"stats_filter_known_dna_{config.file_name}.txt"
        )
        output_tracker.add_file(str(output_file), "filter_known_dna", "bbduk.sh", is_merged=False)
        return Path(output_file)
    except RuntimeError as e:
        config.logger.error(f"Error in filter_known_dna: {str(e)}")
        return input_file

def decontaminate_rrna(input_file: Path, config: ReadFilterConfig, output_tracker: OutputTracker) -> Path:
    """Decontaminate rRNA sequences."""

    from bbmapy import bbduk
    output_file = f"decontaminate_rrna_{config.file_name}.fq.gz"
    rrna_fas1 = Path(config.datadir) / "rRNA/SILVA_138_merged_masked.fa" # type: ignore
    rrna_fas2 = Path(config.datadir) / "rRNA/rmdup_rRNA_ncbi_masked.fa" # type: ignore
    try:
        params = config.step_params['decontaminate_rrna']
        bbduk(
            in_file=str(input_file),
            out=str(output_file),
            ref=f"{rrna_fas1},{rrna_fas2}",
            **params,
            Xmx=config.memory["giga"],
            threads=str(config.threads),
            overwrite="t",
            interleaved="t",
            stats=f"stats_decontaminate_rrna_{config.file_name}.txt"
        )
        output_tracker.add_file(str(output_file), "decontaminate_rrna", "bbduk.sh", is_merged=False)
        return Path(output_file)
    except RuntimeError as e:
        config.logger.error(f"Error in decontaminate_rrna: {str(e)}")
        return input_file

def fetch_and_mask_genomes(input_file: Path, config: ReadFilterConfig, output_tracker: OutputTracker) -> Union[str, Path]:
    """Fetch and mask genomes."""

    from rolypoly.utils.fax import mask_dna, fetch_genomes
    if 'filter_rp_identified_DNA_genomes' not in config.skip_steps:
        stats_file = Path(f"stats_decontaminate_rrna_{config.file_name}.txt")
        if not stats_file.exists():
            config.logger.warning(f"Stats file {stats_file} not found. Skipping fetch and mask genomes step.")
            config.skip_steps.append("filter_identified_dna")
            return "host_empty"

        fetched_dna_dir =  Path("fetched_dna_genomes")
        fetched_dna_dir.mkdir(parents=True, exist_ok=True)
        
        gbs_file = fetched_dna_dir / "gbs_50m.fasta"
        fetch_genomes(str(stats_file), str(gbs_file), threads=config.threads)
        
        if gbs_file.stat().st_size < 20:
            config.logger.warning("The file with the fetched genomes of identified potential hosts appears empty. Step will be skipped.")
            return "host_empty"
    
    if 'mask_fetched_dna' not in config.skip_steps:
        mask_args = {
            "threads": config.threads,
            "memory": config.memory["giga"],
            "output": f"masked_gbs_50m_{config.file_name}.fasta",
            "flatten": False,
            "input": fetched_dna_dir / "gbs_50m.fasta"
        }
        context = click.Context(mask_dna, ignore_unknown_options=True)
        context.invoke(mask_dna, **mask_args)
        return f"masked_gbs_50m_{config.file_name}.fasta"
    return (gbs_file)


def filter_identified_dna(input_file: Path, config: ReadFilterConfig, output_tracker: OutputTracker) -> Union[Path, str]:
    """Filter fetched DNA genomes."""

    from bbmapy import bbduk
    host_file = fetch_and_mask_genomes(input_file, config, output_tracker)
    if host_file == "host_empty":
        return "host_empty"
    output_file =  f"filter_identified_dna_{config.file_name}.fq.gz"
    try:
        params = config.step_params['filter_identified_dna']
        bbduk(
            in_file=str(input_file),
            out=str(output_file),
            ref=f"masked_gbs_50m_{config.file_name}.fasta",
            **params,
            Xmx=config.memory["giga"],
            threads=str(config.threads),
            overwrite="t",
            interleaved="t",
            stats=f"stats_filter_identified_dna_{config.file_name}.txt"
        )
        output_tracker.add_file(str(output_file), "filter_identified_dna", "bbduk.sh", is_merged=False)
        return Path(output_file)
    except RuntimeError as e:
        config.logger.error(f"Error in filter_identified_dna: {str(e)}")
        return input_file

def dedupe(input_file: Path, config: ReadFilterConfig, output_tracker: OutputTracker, phase = "first" ) -> Path:
    """Remove duplicate reads."""

    from bbmapy import clumpify
    if phase == "first":
        output_file =  f"dedupe_first_{config.file_name}.fq.gz"
        is_merged = False
    elif phase == "final_merged":
        output_file =  f"dedupe_final_merged_{config.file_name}.fq.gz"
        is_merged = True
    elif phase == "final_interleaved":
        output_file =  f"dedupe_final_interleaved_{config.file_name}.fq.gz"
        is_merged = False
    try:
        params = config.step_params['dedupe']
        clumpify(
            in_file=str(input_file),
            out=str(output_file),
            **params,
            Xmx=config.memory["giga"],
            threads=str(config.threads),
            overwrite="t",
            interleaved="t"
        )
        output_tracker.add_file(str(output_file), f"dedupe_{phase}", "clumpify.sh", is_merged=is_merged)
        return Path(output_file)
    except RuntimeError as e:
        config.logger.error(f"Error in dedupe_{phase}: {str(e)}")
        exit(1)
        # return input_file

def trim_adapters(input_file: Path, config: ReadFilterConfig, output_tracker: OutputTracker) -> Path:
    """Trim adapters from reads."""

    from bbmapy import bbduk
    output_file =  f"trim_adapters_{config.file_name}.fq.gz"
    adapters_new = Path(config.datadir) / "contam/AFire_illuminatetritis1223.fa"
    adapters_bb = Path(config.datadir) / "contam/bbmap_adapters.fa"
    try:
        params = config.step_params['trim_adapters']
        bbduk(
            in_file=str(input_file),
            out=str(output_file),
            ref=f"{adapters_bb},{adapters_new}",
            **params,
            Xmx=config.memory["giga"],
            threads=str(config.threads),
            overwrite="t",
            interleaved="t",
            stats=f"stats_trim_adapters_{config.file_name}.txt"
        )
        output_tracker.add_file(str(output_file), "trim_adapters", "bbduk.sh", is_merged=False)
        return Path(output_file)
    except RuntimeError as e:
        config.logger.error(f"Error in trim_adapters: {str(e)}")
        exit(1)
        # return input_file

def remove_synthetic_artifacts(input_file: Path, config: ReadFilterConfig, output_tracker: OutputTracker) -> Path:
    """Remove synthetic artifacts (phix etc) from reads."""

    from bbmapy import bbduk
    output_file =  f"remove_synthetic_artifacts_{config.file_name}.fq.gz"
    try:
        params = config.step_params['remove_synthetic_artifacts']
        bbduk(
            in_file=str(input_file),
            out=str(output_file),
            **params,
            Xmx=config.memory["giga"],
            threads=str(config.threads),
            overwrite="t",
            interleaved="t",
            stats=f"stats_remove_synthetic_artifacts_{config.file_name}.txt"
        )
        output_tracker.add_file(str(output_file), "remove_synthetic_artifacts", "bbduk.sh", is_merged=False)
        return Path(output_file)
    except RuntimeError as e:
        config.logger.error(f"Error in remove_synthetic_artifacts: {str(e)}")
        exit(1)
        # return input_file

def entropy_filter(input_file: Path, config: ReadFilterConfig, output_tracker: OutputTracker) -> Path:
    """Apply entropy filter to reads."""

    from bbmapy import bbduk
    output_file =  f"entropy_filter_{config.file_name}.fq.gz"
    try:
        params = config.step_params['entropy_filter']
        bbduk(
            in_file=str(input_file),
            out=str(output_file),
            **params,
            Xmx=config.memory["giga"],
            threads=str(config.threads),
            overwrite="t",
            interleaved="t"
        )
        output_tracker.add_file(str(output_file), "entropy_filter", "bbduk.sh", is_merged=False)
        return Path(output_file)
    except RuntimeError as e:
        config.logger.error(f"Error in entropy_filter: {str(e)}")
        exit(1)
        # return input_file

def error_correct_1(input_file: Path, config: ReadFilterConfig, output_tracker: OutputTracker) -> Path:
    """Perform error correction on reads."""

    from bbmapy import bbmerge
    output_file =  f"error_correct_1{config.file_name}.fq.gz"
    try:
            params = config.step_params['error_correct_1']
            bbmerge(
                in_file=str(input_file),
                out=str(output_file),
                **params,
                Xmx=config.memory["giga"],
                threads=str(config.threads),
                overwrite="t",
                interleaved="t"
            )
            output_tracker.add_file(str(output_file), f"error_correct_phase_1", "bbmerge.sh", is_merged=False)
            return Path(output_file)
    except RuntimeError as e:
        config.logger.error(f"Error in error_correct_1: {str(e)}")
        exit(1)
            # return input_file
def error_correct_2(input_file: Path, config: ReadFilterConfig, output_tracker: OutputTracker) -> Path:
    """Perform error correction on reads."""

    from bbmapy import clumpify
    output_file =  f"error_correct_2{config.file_name}.fq.gz"
    try:
            params = config.step_params['error_correct_2']
            clumpify(
                in_file=str(input_file),
                out=str(output_file),
                **params,
                Xmx=config.memory["giga"],
                threads=str(config.threads),
                overwrite="t",
                interleaved="t"
            )
            output_tracker.add_file(str(output_file), f"error_correct_phase_2", "bbmerge.sh", is_merged=False)
            return Path(output_file)
    except RuntimeError as e:
        config.logger.error(f"Error in error_correct_phase_2: {str(e)}")
        exit(1)
            # return input_file

def merge_reads(input_file: Path, config: ReadFilterConfig, output_tracker: OutputTracker) -> Tuple[Path, Path]:
    """Merge paired-end reads."""

    from bbmapy import bbmerge
    output_file =  f"merged_{config.file_name}.fq.gz"
    unmerged_file =  f"unmerged_{config.file_name}.fq.gz"
    try:
        params = config.step_params['merge_reads']
        bbmerge(
            in_file=str(input_file),
            out=str(output_file),
            outu=str(unmerged_file),
            **params,
            Xmx=config.memory["giga"],
            threads=str(config.threads),
            overwrite="t",
            interleaved="t",
            outadapter=f"out_adapter_merged_{config.file_name}.txt"
        )
        output_tracker.add_file(str(output_file), "merge_reads", "bbmerge.sh", is_merged=True)
        output_tracker.add_file(str(unmerged_file), "merge_reads_unmerged", "bbmerge.sh", is_merged=False)
        return Path(output_file), Path(unmerged_file)
    except RuntimeError as e:
        config.logger.error(f"Error in merge_reads: {str(e)}")
        exit(1)
        # return input_file, None

def quality_trim_unmerged(input_file: Path, config: ReadFilterConfig, output_tracker: OutputTracker) -> Path:
    """Quality trim unmerged reads."""

    from bbmapy import bbduk
    input_file = Path(output_tracker.get_latest_non_merged_file())
    output_file =  f"qtrimmed_{config.file_name}.fq.gz"
    try:
        params = config.step_params['quality_trim_unmerged']
        bbduk(
            in_file=str(input_file),
            out=str(output_file),
            **params,
            Xmx=config.memory["giga"],
            threads=str(config.threads),
            overwrite="t",
            interleaved="t"
        )
        output_tracker.add_file(str(output_file), "quality_trim_unmerged", "bbduk.sh", is_merged=False)
        return Path(output_file)
    except RuntimeError as e:
        config.logger.error(f"Error in quality_trim_unmerged: {str(e)}")
        exit(1)
        # return input_file

def cleanup_and_move_files(work_dir: Path, keep_tmp: bool,  output_tracker: OutputTracker, config: ReadFilterConfig):
    run_info_dir = Path(config.output_dir) / "run_info"
    run_info_dir.mkdir(parents=True, exist_ok=True)

    # Move config file and output tracker CSV to run_info
    config.save(run_info_dir / "rp_filter_reads_config.json")
    output_tracker.to_csv(run_info_dir / "output_tracker.csv")
    # Move all .txt files to run_info
    for txt_file in work_dir.glob("*.txt"):
        config.logger.info(f"moving {txt_file} to {str(run_info_dir / txt_file.name)}") # type: ignore
        shutil.move(str(txt_file), str(run_info_dir / txt_file.name))
        
    # Move all .html files to run_info
    for html_file in work_dir.rglob("*.html"):
        config.logger.info(f"moving {html_file} to {str(run_info_dir / html_file.name)}") # type: ignore
        shutil.move(str(html_file), str(run_info_dir / html_file.name))
        
    final_merged = output_tracker.get_latest_merged_file()
    final_interleaved = output_tracker.get_latest_non_merged_file()
    for item in [final_merged, final_interleaved]:
        try:
            shutil.move(item, Path(config.output_dir)) # type: ignore
            # item.rename(Path(config.output_dir) / item.name)
        except:
            config.logger.warning(f"Item {item} existed in the destination path, it would not be moved to avoid overwriting something of the user     This could happen if multiple runs in the same working directory were launched. It will be deleted along with the tmp workign directory    ") # type: ignore

    # Move remaining files from work_dir to parent directory
    if not keep_tmp:
        
        try:
            shutil.rmtree(work_dir)
        except:
            config.logger.warning(f"Work directory {work_dir} could not be removed") # type: ignore
        # os.unlink(config.input)
        config.logger.info(f"Work directory {work_dir} cleaned up and removed") # type: ignore
    
