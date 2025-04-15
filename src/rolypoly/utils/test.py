import glob
import os
from pathlib import Path

import rich_click as click

# Set up the environment
ROLYPOLY_DATA = os.environ.get("ROLYPOLY_DATA")
RP_TESTS_DIR = Path("/clusterfs/jgi/scratch/science/metagen/neri/tests/rp_tests")

print(f"ROLYPOLY_DATA: {ROLYPOLY_DATA}")
print(f"RP_TESTS_DIR: {RP_TESTS_DIR}")

## argument lists for --input files
global \
    virome_fastqs, \
    metatranscriptome_fastqs, \
    host_fastas, \
    contigs_fastas, \
    protein_fastas, \
    fungal_fastq, \
    fungal_host, \
    hiv_fastqs, \
    hiv_host, \
    metagenomic_fastq, \
    metagenomic_host, \
    metagenomic_contigs, \
    metagenomic_protein
isolate_fungal_fastq = glob.glob(f"{RP_TESTS_DIR}/inputs/fungal/*.fq.gz")
isolate_fungal_host = glob.glob(f"{RP_TESTS_DIR}/inputs/fungal/*.fasta")
isolate_lyssa_fastqs = glob.glob(f"{RP_TESTS_DIR}/inputs/isolate_human/lyssa/*.fq.gz")
isolate_lyssa_host = glob.glob(f"{RP_TESTS_DIR}/inputs/isolate_human/*.fasta")
isolate_plant_fastqs = glob.glob(f"{RP_TESTS_DIR}/inputs/isolate_plant/*.fq.gz")
isolate_plant_host = glob.glob(f"{RP_TESTS_DIR}/inputs/isolate_plant/*.fasta")
isolate_hiv_fastqs = glob.glob(f"{RP_TESTS_DIR}/inputs/isolate_human/hiv/*.fq.gz")
isolate_hiv_host = glob.glob(f"{RP_TESTS_DIR}/inputs/isolate_human/*.fasta")

# print(f"isolate_fungal_fastq: {isolate_fungal_fastq}")
# print(f"isolate_fungal_host: {isolate_fungal_host}")
# print(f"isolate_lyssa_fastqs: {isolate_lyssa_fastqs}")
# print(f"isolate_lyssa_host: {isolate_lyssa_host}")
# print(f"isolate_plant_fastqs: {isolate_plant_fastqs}")
# print(f"isolate_plant_host: {isolate_plant_host}")
# print(f"isolate_hiv_fastqs: {isolate_hiv_fastqs}")
# print(f"isolate_hiv_host: {isolate_hiv_host}")

virome_fastqs = glob.glob(f"{RP_TESTS_DIR}/inputs/reads/meta/sea/*.fq.gz")
metatranscriptome_fastqs = glob.glob(f"{RP_TESTS_DIR}/inputs/reads/meta/soil/*.fq.gz")
metagenomic_fastq = glob.glob(f"{RP_TESTS_DIR}/inputs/reads/meta/human/*.fq.gz")
metagenomic_host = glob.glob(f"{RP_TESTS_DIR}/inputs/hosts/human/*.fasta")

# print(f"virome_fastqs: {virome_fastqs}")
# print(f"metatranscriptome_fastqs: {metatranscriptome_fastqs}")
# print(f"metagenomic_fastq: {metagenomic_fastq}")
# print(f"metagenomic_host: {metagenomic_host}")

contigs_fastas = glob.glob(f"{RP_TESTS_DIR}/input/contigs/*.fasta")
protein_fastas = glob.glob(f"{RP_TESTS_DIR}/input/protein/*.fasta")

# print(f"contigs_fastas: {contigs_fastas}")
# print(f"protein_fastas: {protein_fastas}")

## argument lists for filter_contigs


## argument lists for marker_search
rdrp_dbs = ["all", "RVMT", "TSA_Olendraite", "Pfam", "rdrp-scan", "neordrp"]

## argument lists for filter_reads
speed = [1, 2, 3, 15]

## argument lists for assembly
assembler = ["spades", "megahit", "penguin"]

# print(f"rdrp_dbs: {rdrp_dbs}")
# print(f"speed: {speed}")
# print(f"assembler: {assembler}")


def eval_range(range_str, suffix=None):
    """makes a list of values from a range string, return a list of int unless a suffix is specified, in which case the every item in the list is converted to a string appended with the suffix.
    Args:
        range_str (str): a string in the format "start,end,step"
        suffix (str): a suffix to append to each value in the list, e.g. "g" for gigabytes.

    Returns:
        list: a list of values
    """
    print(f"Evaluating range: {range_str}, suffix: {suffix}")
    if "," not in range_str:
        result = [int(range_str)]
        print(f"Single value range, result: {result}")
        return result
    start, end, step = range_str.split(",")
    if suffix:
        result = [f"{i}{suffix}" for i in range(int(start), int(end), int(step))]
    else:
        result = list(range(int(start), int(end), int(step)))
    print(f"Range evaluation result: {result}")
    return result


# verbose = True # tbd not implemented yet.


def run_command(command):
    import subprocess

    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, capture_output=False, text=True)
    print(f"Command output: {result.stdout}")
    print(f"Command error (if any): {result.stderr}")
    assert result.returncode == 0, f"Command failed: {result.stderr}"
    return result.stdout


def preprocess_input(input_file):
    print(f"Preprocessing input file: {input_file}")
    path_to_input = Path(input_file).parent
    input_file_name = Path(input_file).name
    small_file = f"{path_to_input}/small_{input_file_name}"
    if not os.path.exists(small_file):
        print(f"Creating small file: {small_file}")
        run_command(f"seqkit sample -p 0.01 -w0 {input_file} -o {small_file} ")
    else:
        print(f"Small file already exists: {small_file}")
    return small_file


def test_rolypoly_command(command, args):
    """test a rolypoly command with the given arguments

    Args:
        command (string): one of the rolypoly commands
        args (dict): arguments to the command
    """

    cmd = f"rolypoly {command}"
    if args:
        for k, v in args.items():
            cmd += f" --{k} {v}"

    print(f"Constructed rolypoly command: {cmd}")
    run_command(cmd)


def get_input_files(mode, command=None, small=False):
    """Get input files based on the selected mode."""
    print(f"Getting input files for mode: {mode}, small: {small}, command: {command}")
    if command is None:
        command = "filter_reads"
    base_path = f"{RP_TESTS_DIR}/inputs"

    # if command in  ["filter-reads", "assemble"]:
    #     base_path = f"{RP_TESTS_DIR}/inputs/reads"
    # elif command in ["filter-contigs", "annotate"]:
    #     base_path = f"{RP_TESTS_DIR}/input/contig/"

    file_dict = {
        "filter-reads": {
            "isolate_fungal": (
                f"{base_path}/isolate_fungi/*.fq.gz",
                f"{base_path}/isolate_fungi/*.fasta",
            ),
            "isolate_hiv": (
                f"{base_path}/isolate_human/hiv/*.fq.gz",
                f"{base_path}/isolate_human/*.fasta",
            ),
            "isolate_lyssa": (
                f"{base_path}/isolate_human/lyssa/*.fq.gz",
                f"{base_path}/isolate_human/*.fasta",
            ),
            "isolate_plant": (
                f"{base_path}/isolate_plant/*.fq.gz",
                f"{base_path}/isolate_plant/*.fasta",
            ),
            "virome": (f"{base_path}/reads/meta/sea/*.fq.gz", None),
            "metatranscriptome": (f"{base_path}/reads/meta/soil/*.fq.gz", None),
            "metagenomic": (
                f"{base_path}/reads/meta/human/*.fq.gz",
                f"{base_path}/hosts/human/*.fasta",
            ),
        },
        "filter-contigs": {
            "isolate_fungal": (
                f"{base_path}/isolate_fungi/example_virus.fna",
                f"{base_path}/isolate_fungi/*.fasta",
            ),
            "isolate_hiv": (
                f"{base_path}/isolate_human/hiv/hiv.fna",
                f"{base_path}/isolate_human/*.fasta",
            ),
            "isolate_lyssa": (
                f"{base_path}/isolate_human/lyssa/lyssa.fna",
                f"{base_path}/isolate_human/*.fasta",
            ),
            "isolate_plant": (
                f"{base_path}/isolate_plant/plant_virus.fna",
                f"{base_path}/isolate_plant/*.fasta",
            ),
            # "virome": (f"{base_path}/reads/meta/sea/example_virome.fna", f"{base_path}/reads/meta/sea/*.fasta"),
            # "metatranscriptome": (f"{base_path}/reads/meta/soil/example_metatranscriptome_contigs.fna", f"{base_path}/reads/meta/soil/*.fasta"),
        },
        "assembly": {
            "isolate_fungal": (
                f"{base_path}/isolate_fungi/rp_filtered_reads/*.fq.gz",
                None,
            ),
            "isolate_hiv": (
                f"{base_path}/isolate_human/hiv/rp_filtered_reads/*.fq.gz",
                None,
            ),
            "isolate_lyssa": (
                f"{base_path}/isolate_human/lyssa/rp_filtered_reads/*.fq.gz",
                None,
            ),
            "isolate_plant": (
                f"{base_path}/isolate_plant/rp_filtered_reads/*.fq.gz",
                None,
            ),
            "virome": (f"{base_path}/reads/meta/sea/rp_filtered_reads/*.fq.gz", None),
            "metatranscriptome": (
                f"{base_path}/reads/meta/soil/rp_filtered_reads/*.fq.gz",
                None,
            ),
        },
        "annotate": {
            "contigs": (f"{base_path}/contigs/*.fasta", None),
            "protein": (f"{base_path}/protein/*.fasta", None),
        },
        "rdrp-hmmsearch": {
            "isolate_fungal": (f"{base_path}/isolate_fungi/example_virus.fna", None),
            "isolate_hiv": (f"{base_path}/isolate_human/hiv/hiv.fna", None),
            "isolate_lyssa": (f"{base_path}/isolate_human/lyssa/lyssa.fna", None),
            "isolate_plant": (f"{base_path}/isolate_plant/plant_virus.fna", None),
            "virome": (f"{base_path}/reads/meta/sea/example_virome.fna", None),
            "metatranscriptome": (
                f"{base_path}/reads/meta/soil/example_metatranscriptome_contigs.fna",
                None,
            ),
        },
    }
    print(f"File dictionary: {file_dict}")
    print(f"Command: {command}")
    print(f"Mode: {mode}")
    try:
        input_pattern, host_pattern = file_dict[command][mode]
    except:
        input_pattern = file_dict[command][mode]

    print(f"Input pattern: {input_pattern}")
    print(f"Host pattern: {host_pattern}")
    input_files = glob.glob(input_pattern)
    host_files = glob.glob(host_pattern) if host_pattern else None

    if small:
        input_files = [f for f in input_files if "small_" in f] or input_files
    else:
        input_files = [f for f in input_files if "small_" not in f]

    print(f"Input files: {input_files}")
    print(f"Host files: {host_files}")
    return input_files, host_files


@click.command(name="test-rolypoly")
@click.option("-c", "--command", required=True, help="RolyPoly command to test")
@click.option(
    "-m",
    "--mode",
    type=click.Choice(
        [
            "isolate_fungal",
            "isolate_hiv",
            "isolate_lyssa",
            "isolate_plant",
            "virome",
            "metatranscriptome",
            "metagenomic",
            "contigs",
            "protein",
        ]
    ),
    default="isolate_fungal",
    help="Type of data to test",
)
@click.option("-s", "--small", is_flag=True, help="Use small versions of input files")
@click.option("-o", "--output_dir", help="Output directory")
@click.option("-t", "--threads", default=4, help="Number of threads to use")
@click.option("-M", "--memory", default="6g", help="Memory to use")
@click.option("--other-args", help="Additional arguments as JSON string")
def test_rolypoly(command, mode, small, output_dir, threads, memory, other_args):
    import datetime
    import json

    input_files, host_files = get_input_files(mode, command, small)
    print(
        f"Testing rolypoly with command: {command}, mode: {mode}, small: {small}, output_dir: {output_dir}, threads: {threads}, memory: {memory}, other_args: {other_args}"
    )

    if not input_files:
        click.echo(f"No input files found for mode: {mode}")
        return

    args = {
        "input": input_files[0],
        "threads": threads,
        "memory": memory,
    }

    if host_files:
        args["known-dna"] = host_files[0]  # Changed from known_dna to known-dna

    if output_dir:
        args["output"] = output_dir
    else:
        args["output"] = (
            str(RP_TESTS_DIR)
            + f"/rolypoly_tests/{command}_out_"
            + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )

    if other_args:
        args.update(json.loads(other_args))

    print(f"Final arguments for rolypoly command: {args}")
    test_rolypoly_command(command, args)


if __name__ == "__main__":
    print("Starting test_rolypoly")
    test_rolypoly()
    print("Finished test_rolypoly")
