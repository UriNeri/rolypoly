import os

from rich.console import Console
from rich_click import command, option

console = Console()


@command()
@option("-i", "--input", required=True, help="Input assembly file (FASTA)")
@option(
    "-r",
    "--reads",
    required=True,
    help="Input reads file(s) (FASTQ). For paired-end, separate with comma.",
)
@option(
    "-o",
    "--output",
    default=lambda: f"{os.getcwd()}_refined_assembly",
    help="Output directory",
)
@option("-t", "--threads", default=1, help="Number of threads to use")
@option("-M", "--memory", default="6g", help="Maximum memory to use")
@option(
    "-v",
    "--variant-caller",
    default="freebayes",
    help="Variant caller to use (freebayes, bcftools)",
)
@option(
    "-g",
    "--log-file",
    default=lambda: f"{os.getcwd()}/refinement_logfile.txt",
    help="Path to log file",
)
def refinement(input, reads, output, threads, memory, variant_caller, log_file):
    """
    Refine assembly by de-entangling strains and rerunning assembly - post host removal.
    """
    from rolypoly.utils.loggit import setup_logging

    logger = setup_logging(log_file)
    logger.info("Starting assembly refinement process")

    # Create output directory
    os.makedirs(output, exist_ok=True)

    # Align reads to assembly
    bam_file = align_reads(input, reads, output, threads, logger)

    # Call variants
    vcf_file = call_variants(
        input, bam_file, output, threads, memory, variant_caller, logger
    )

    # Apply variants to improve assembly
    improved_assembly = apply_variants(input, vcf_file, output, logger)

    # De-entangle strains (if multiple strains are present)
    final_assemblies = de_entangle_strains(
        improved_assembly, bam_file, output, threads, logger
    )

    logger.info("Assembly refinement completed")
    console.print(f"Refined assembly(ies) saved in: {output}")


def align_reads(assembly, reads, output_dir, threads, logger):
    import subprocess

    logger.info("Aligning reads to assembly")
    index_cmd = f"bwa index {assembly}"
    subprocess.run(index_cmd, shell=True, check=True)

    bam_file = os.path.join(output_dir, "aligned_reads.bam")
    align_cmd = f"bwa mem -t {threads} {assembly} {reads} | samtools sort -@ {threads} -o {bam_file}"
    subprocess.run(align_cmd, shell=True, check=True)

    index_bam_cmd = f"samtools index {bam_file}"
    subprocess.run(index_bam_cmd, shell=True, check=True)

    return bam_file


def call_variants(
    assembly, bam_file, output_dir, threads, memory, variant_caller, logger
):
    import subprocess

    logger.info(f"Calling variants using {variant_caller}")
    vcf_file = os.path.join(output_dir, "variants.vcf")

    if variant_caller == "freebayes":
        cmd = f"freebayes -f {assembly} -p 1 {bam_file} > {vcf_file}"
    elif variant_caller == "bcftools":
        cmd = f"bcftools mpileup -Ou -f {assembly} {bam_file} | bcftools call -mv -Ov -o {vcf_file}"
    else:
        raise ValueError(f"Unsupported variant caller: {variant_caller}")

    subprocess.run(cmd, shell=True, check=True)
    return vcf_file


def apply_variants(assembly, vcf_file, output_dir, logger):
    import subprocess

    logger.info("Applying variants to improve assembly")
    improved_assembly = os.path.join(output_dir, "improved_assembly.fasta")
    cmd = f"bcftools consensus -f {assembly} {vcf_file} > {improved_assembly}"
    subprocess.run(cmd, shell=True, check=True)
    return improved_assembly


def de_entangle_strains(assembly, bam_file, output_dir, threads, logger):
    logger.info("De-entangling strains")
    # placeholder for strain de-entanglement logic
    # TODO: implement a more sophisticated approach here

    strain_dir = os.path.join(output_dir, "strains")
    os.makedirs(strain_dir, exist_ok=True)

    # # just copy the improved assembly as a single strain
    # shutil.copy(assembly, os.path.join(strain_dir, "strain_1.fasta"))

    # return [os.path.join(strain_dir, "strain_1.fasta")]


if __name__ == "__main__":
    refinement()
