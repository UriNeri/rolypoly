# Written by Uri Neri
# Description: Assembler wraper - takes in (presumably filtered) reads, and passes them to an assembler(s) of choice.
import os
from pathlib import Path
import rich_click as click
from rolypoly.utils.loggit import log_start_info
from rolypoly.utils.config import BaseConfig
from rich.console import Console

console = Console()

class AssemblyConfig(BaseConfig):
    def __init__(self, **kwargs):
        # in this case output_dir and output are the same, so need to explicitly make sure it exists.
        if not Path(kwargs.get("output")).exists():
            kwargs["output_dir"] = kwargs.get("output")
            Path(kwargs.get("output")).mkdir(parents=True, exist_ok=True)
            
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
        self.assembler = kwargs.get("assembler")
        self.keep_tmp = kwargs.get("keep_tmp")
        self.step_params = {
            'spades': {'k': '21,33,45,57,63,69,71,83,95,103,107,111,119', 'mode': 'meta'},
            'megahit': {'k-min': 21, 'k-max': 147, 'k-step': 8, 'min-contig-len': 30},
            'penguin': {'min-contig-len': 150, 'num-iterations': 'aa:1,nucl:12'},
            'seqkit': {},
            'bbwrap': {'maxindel': 200, 'minid': 90, 'untrim': True, 'ambig': 'best'},
            'bowtie': {}
        }
        self.skip_steps = kwargs.get("skip_steps") if isinstance(kwargs.get("skip_steps"), list) else kwargs.get("skip_steps").split(",") if isinstance(kwargs.get("skip_steps"), str) else []
        if kwargs.get("override_parameters") is not None:
            self.logger.info(f"override_parameters: {kwargs.get('override_parameters')}")
            for step, params in kwargs.get('override_parameters').items():
                if step in self.step_params:
                    self.step_params[step].update(params)
                else:
                    self.logger.warning(f"Warning: Unknown step '{step}' in override_parameters. Ignoring.")
                    
global tools
tools = []

def handle_input_files(input_path):
    """Process input files and identify libraries.
    """ 
    # Import modules needed only in this function
    from pathlib import Path

    
    input_path = Path(input_path)
    if input_path.is_dir():
        fastq_files = list(input_path.glob('*_interleaved*.fq.gz')) + list(input_path.glob('*_merged*.fq.gz')) # file names fitting rolypoly filter_reads output names. TODO: Check if this works for single-end data
    else:
        fastq_files = [input_path]
        # bbmerge(
        #     in_file=input_fastq,
        #     out=f"{input}/input_fastq_final_merged.fq.gz",
        #     outu=f"{input}/input_fastq_final_interleaved.fq.gz",
        #     k=93,
        #     extend2=80,
        #     rem=True,
        #     ordered=True,
        #     memory=memory["giga"],
        #     threads=threads,
        #     overwrite=True
        # ) # this was done to ensure proper interleaving, but it adds time and storage. Just trust the user to give us interleaved reads    
    
    libraries = {}
    for file in fastq_files:
        library_name = file.stem.split('_final_')[0]
        if library_name not in libraries:
            libraries[library_name] = {'interleaved': None, 'merged': None}
        if 'interleaved' in file.name:
            libraries[library_name]['interleaved'] = file
        elif 'merged' in file.name:
            libraries[library_name]['merged'] = file
    
    return libraries, len(libraries)

def run_spades(config, libraries):
    """Run SPAdes assembler.
    """
    # Import modules needed only in this function
    from rolypoly.utils.various import ensure_memory
    import subprocess

    spades_output = config.output_dir / f"spades_meta_output"
    spades_cmd = f"spades.py --{config.step_params['spades']['mode']} -o {spades_output} --threads {config.threads} --only-assembler -k {config.step_params['spades']['k']} --phred-offset 33 -m {ensure_memory(config.memory)['bytes'][:-1]}"
    
    if len(libraries) > 9:
        config.logger.info(f"Running SPAdes on concatenated reads")
        with open(f'{config.output_dir}/all_merged.fq.gz', 'wb') as outfile:
            for lib in libraries.values():
                if lib['merged']:
                    with open(lib['merged'], 'rb') as infile:
                        outfile.write(infile.read())
        with open(f'{config.output_dir}/all_interleaved.fq.gz', 'wb') as outfile:
            for lib in libraries.values():
                if lib['interleaved']:
                    with open(lib['interleaved'], 'rb') as infile:
                        outfile.write(infile.read())
        spades_cmd += f" --pe-12 1 {config.output_dir}/all_interleaved.fq.gz --s 1 {config.output_dir}/all_merged.fq.gz"
    else: # this case shouldn't be entered in the current code as we always run SPAdes on meta now, but keeping it for legacy reasons, and in case we figure out why spades decides to ignore merged reads when supplied via the -pe-m #. 
        for i, (lib_name, lib) in enumerate(libraries.items(), 1):
            if lib['interleaved']:
                spades_cmd += f" --pe-12 {i} {lib['interleaved']}"
            if lib['merged']:
                spades_cmd += f" --pe-s {i} {lib['merged']}" # this should be --pe-m, but spades decides to ignore it    

    subprocess.run(spades_cmd, shell=True, check=True)
    config.logger.info(f"Finished SPAdes assembly")

    return spades_output / "scaffolds.fasta"

def run_megahit(config, libraries):
    """Run MEGAHIT assembly.
    """
    # Import modules needed only in this function
    import glob
    from rolypoly.utils.various import ensure_memory
    import subprocess

    config.logger.info(f"Started Megahit assembly")
    megahit_output = config.output_dir / "megahit_custom_out"
    
    interleaved = ",".join(str(lib['interleaved']) for lib in libraries.values() if lib['interleaved'])
    merged = ",".join(str(lib['merged']) for lib in libraries.values() if lib['merged'])
    
    
    megahit_cmd = [
        f"megahit",
        f"--k-min {config.step_params['megahit']['k-min']}",
        f"--k-max {config.step_params['megahit']['k-max']}",
        f"--k-step {config.step_params['megahit']['k-step']}",
        f"--min-contig-len {config.step_params['megahit']['min-contig-len']}"]
    if len(interleaved) > 0:
        megahit_cmd.extend([f"--12 {interleaved}"])
    if len(merged) > 0:
        megahit_cmd.extend([f"--read {merged}"])
    megahit_cmd.extend([f"--out-dir {megahit_output}",
        f"--num-cpu-threads {config.threads} --memory {ensure_memory(config.memory)['bytes'][:-1]}"])
    config.logger.info(f"Running Megahit assembly with command: {' '.join(megahit_cmd)}")
    subprocess.run(" ".join(megahit_cmd), shell=True, check=True)

    final_k = max(int(os.path.basename(file).split('k')[1].split('.')[0]) for file in glob.glob(f'{megahit_output}/intermediate_contigs/*.final.contigs.fa'))
    
    subprocess.run(
        f"megahit_toolkit contig2fastg {final_k} {megahit_output}/final.contigs.fa > "
        f"{megahit_output}/final_megahit_assembly_k{final_k}.fastg",
        shell=True, check=True
    )

    return megahit_output / "final.contigs.fa"

def run_penguin(config, libraries):
    """Run Penguin assembler.
    """
    # Import modules needed only in this function
    import subprocess
    
    config.logger.info(f"Started Penguin assembly")
    penguin_output = config.output_dir / "penguin_Fguided_1_nuclassemble_c0.fasta"
    interleaved = " ".join(str(lib['interleaved']) for lib in libraries.values() if lib['interleaved'])
    merged = " ".join(str(lib['merged']) for lib in libraries.values() if lib['merged'])
    
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
@click.option("-M", "--memory", default="6gb", help=" RAM limit  (more is betterer, see the docs for more info)")
@click.option("-o", "--output", default="RP_assembly_output", help="Output path (folder will be created if it doesn't exist)")
@click.option("-k", "--keep-tmp", is_flag=True, default=False, help="Keep temporary files")
@click.option("-g", "--log-file", default=lambda: f"{os.getcwd()}/assemble_logfile.txt", help="Path to a logfile, should exist and be writable (permission wise)")
@click.option("-i", "--input", required=True, help="Input path to fastq files or directory containing fastq files")
@click.option("-A", "--assembler", default="spades,megahit,penguin", help="Assembler choice. for multiple, give a comma-seperated list e.g. 'spades,penguin')")
@click.option("-op","--override-parameters", default='{}', help='JSON-like string of parameters to override. Example: --override-parameters \'{"spades": {"k": "21,33,55"}, "megahit": {"k-min": 31}}\'')
@click.option("-ss","--skip-steps", default='', help='Comma-separated list of steps to skip. Example: --skip-steps seqkit,bowtie')
@click.option("-ow","--overwrite", is_flag=True, default=False, help='Do not overwrite the output directory if it already exists')
@click.option("-ll","--log-level", default="info", hidden=True, help='Log level. Options: debug, info, warning, error, critical')
def assembly(threads, memory, output, keep_tmp, log_file, input, assembler,override_parameters, skip_steps, overwrite, log_level): #
    """Assembly wrapper - takes in (presumably filtered) reads, and assembles them using one or more assemblers.
    Currently supported assemblers are:
    • SPAdes (metaSPAdes)
    • MEGAHIT
    • Penguin
    """
    # Import modules needed only in this function
    import shutil
    import sh
    from rolypoly.utils.bwt1 import build_index, align_paired_end_interleaved, align_single_end
    from rolypoly.utils.citation_reminder import remind_citations
    from rolypoly.utils.various import check_dependencies

    if not overwrite:
        if Path(output).exists():
            raise ValueError(f"Output directory {output} already exists. Use -ow to overwrite.")
    else:
        shutil.rmtree(output, ignore_errors=True)

    Path(output).mkdir(parents=True, exist_ok=True) 
    # print(output)
       
    config = AssemblyConfig(
        input=Path(input),
        output=Path(output),
        threads=threads,
        log_file=Path(log_file),
        memory=(memory),
        assembler=assembler,
        keep_tmp=keep_tmp,
        override_params=(override_parameters),
        skip_steps=(skip_steps),
        log_level=log_level 
    )
    
    config.logger.info(f"Starting assembly process    ")
    log_start_info(config.logger,config_dict=config.__dict__)
    config.logger.info(f"Saving config to {config.output_dir / 'assembly_config.json'}")
    config.save(config.output_dir / "assembly_config.json")

    libraries, n_libraries = handle_input_files(config.input)
    config.logger.info(f"Found {n_libraries} libraries")
    config.logger.info(f"Libraries: {libraries}")
    contigs4eval = []

    if "spades" in config.assembler.lower() and "spades" not in config.skip_steps:
        check_dependencies(["spades.py"])
        contigs4eval.append(run_spades(config, libraries))
        tools.append("spades")
    if "megahit" in config.assembler.lower() and "megahit" not in config.skip_steps:
        check_dependencies(["megahit"])
        contigs4eval.append(run_megahit(config, libraries))
        tools.append("megahit")
    if "penguin" in config.assembler.lower() and "penguin" not in config.skip_steps:
        check_dependencies(["penguin"])
        contigs4eval.append(run_penguin(config, libraries))
        tools.append("penguin")
        
    # Deduplication step # TODO: add as optional a linclust step. 
    if "seqkit" not in config.skip_steps:
        tools.append("seqkit")
        seqkit = sh.Command("seqkit")
        seqkit.rmdup(
            "--by-seq", *contigs4eval,
            "--dup-num-file",
            f"{config.output_dir}/rmdup_dup_file.txt", "-w", "0",
            "--threads", config.threads,
            "--out-file", f"{config.output_dir}/rmdup_contigs.fasta",
        )
        config.logger.info(f"Finished deduplicating: {contigs4eval}")
        
    # Evaluation steps
    # Calculate coverage distribution and capture unassembled reads
    if "bbwrap" not in config.skip_steps:
        from bbmapy import bbwrap
        bbwrap(
            ref=f"{config.output_dir}/rmdup_contigs.fasta",
            in_file=f"{','.join(str(lib['interleaved']) for lib in libraries.values() if lib['interleaved'])},"
                   f"{','.join(str(lib['merged']) for lib in libraries.values() if lib['merged'])}",
            out=f"{config.output_dir}/bbwrap_output.sam",
            threads=config.threads,
            nodisk=True,
            covhist=f"{config.output_dir}/assembly_covhist.txt",
            covstats=f"{config.output_dir}/assembly_covstats.txt",
            outm=f"{config.output_dir}/assembly_bbw_assembled.fq.gz",
            outu=f"{config.output_dir}/assembly_bbw_unassembled.fq.gz",
            **config.step_params['bbwrap']
        )

    # Generate SAM file using bowtie1 # TODO: add bowtie2, mmseqs2 or bbwrap.
    if "bowtie" not in config.skip_steps:
        interleaved = ",".join(str(lib['interleaved']) for lib in libraries.values() if lib['interleaved'])
        merged = ",".join(str(lib['merged']) for lib in libraries.values() if lib['merged'])
        tools.append("bowtie")
        os.makedirs(f"{config.output_dir}/bowtie_index", exist_ok=True)
        build_index(reference_in=f"{config.output_dir}/rmdup_contigs.fasta", index_base=f"{config.output_dir}/bowtie_index", threads=config.threads)
        try:
            if len(interleaved) > 0:
                align_paired_end_interleaved(index_base=f"{config.output_dir}/bowtie_index", reads=interleaved, output_file=f"{config.output_dir}/assembly_bowtie_interleaved.sam", threads=config.threads)
                sh.pigz("-p", config.threads, f"{config.output_dir}/assembly_bowtie_interleaved.sam")
            if len(merged) > 0:
                align_single_end(index_base=f"{config.output_dir}/bowtie_index", reads=merged, output_file=f"{config.output_dir}/assembly_bowtie_merged_reads.sam", threads=config.threads)
                sh.pigz("-p", config.threads, f"{config.output_dir}/assembly_bowtie_merged_reads.sam")
        except Exception as e:
            config.logger.warning(f"Failed to align reads to contigs: {e}")

    config.logger.info(f"Finished assembly evaluation on: {contigs4eval}")

    if not config.keep_tmp:
        files_to_remove = [
            "tmp",
            f"{config.output_dir}/all_interleaved.fq.gz",
            f"{config.output_dir}/all_merged.fq.gz",
        ]
        folders_to_remove = [
            f"{config.output_dir}/megahit_custom_out/intermediate_contigs"
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
        if os.path.exists(f"{config.output_dir}/spades_meta_output"):
            for spades_folder in os.listdir(f"{config.output_dir}/spades_meta_output"):
                if Path(f"{config.output_dir}/spades_meta_output/{spades_folder}").is_dir():
                    shutil.rmtree(f"{config.output_dir}/spades_meta_output/{spades_folder}")

    config.logger.info("Assembly process completed successfully.")
    config.logger.info(f"Final redundancy filtered contigs from the assemblers used are in {config.output_dir}/rmdup_contigs.fasta")
    config.logger.info(f"Reads unassembled from the assembly are in {config.output_dir}/assembly_bbw_unassembled.fq.gz")
    config.logger.info(f"Reads aligned to the assembly (interleaved and merged) are in {config.output_dir}/assembly_bowtie_interleaved.sam.gz and {config.output_dir}/assembly_bowtie_merged_reads.sam.gz")

    remind_citations(tools)
    with open(f"{config.log_file}","w") as f_out:
        f_out.write(remind_citations(tools,return_as_text=True))
        
if __name__ == "__main__":
    assembly()
