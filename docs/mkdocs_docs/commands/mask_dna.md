# Mask Dna

<!-- Auto-generated draft from CLI metadata for `rolypoly mask-dna`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

Mask an input fasta file for sequences that could be RNA viral (or mistaken for such).

## Description

Mask an input fasta file for sequences that could be RNA viral (or mistaken for such).

## Usage

```bash
rolypoly mask-dna [OPTIONS]
```

## Options

- `-t`, `--threads`: Number of threads to use (type: `INTEGER`; default: `1`)
- `-M`, `--memory`: Memory in GB (type: `TEXT`; default: `6gb`)
- `-o`, `--output`: Output file name (type: `TEXT`; required; default: `Sentinel.UNSET`)
- `-f`, `--flatten`: Attempt to kcompress.sh the masked file (type: `BOOLEAN`; default: `False`)
- `-i`, `--input`: Input fasta file (type: `TEXT`; required; default: `Sentinel.UNSET`)
- `-a`, `--aligner`: Which tool to use for identifying shared sequence (minimap2, mmseqs2, diamond, bowtie1, bbmap) (type: `TEXT`; default: `mmseqs2`)
- `-mlc`, `--mask-low-complexity`: Whether to mask low complexity regions using bbduks entropy masking (type: `BOOLEAN`; default: `False`)
- `-r`, `--reference`: Provide an input fasta file to be used for masking, instead of the pre-generated collection of RNA viral sequences (type: `TEXT`; default: `/clusterfs/jgi/scratch/science/metagen/neri/code/rolypoly/data/contam/masking/combined_entropy_masked.fasta`)
- `--tmpdir`: Temporary directory to use (default: output file's parent/tmp - if you have enough RAM, you can set this to /dev/shm/ or /tmp/ for faster I/O) (type: `TEXT`)




