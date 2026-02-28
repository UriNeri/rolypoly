# Commands Overview

!!! warning "🚧 Experimental"
    RolyPoly is under active development - features may be incomplete or experimental.

RolyPoly provides several commands for different stages of viral analysis. For detailed help in the terminal, use:

```bash
rolypoly --help
```

## Available Commands

### Setup and Data
- [Get Data](prepare_external_data.md): Download and prepare necessary databases (`rolypoly get-data`)
- [Version](prepare_external_data.md#version): Display version and data information (`rolypoly version`)

### Core Pipeline
- [End to End](end_to_end.md): Run the complete pipeline with default settings (`rolypoly end2end`)
- [Read Processing](read_processing.md): Filter and process raw RNA-seq reads (`rolypoly filter-reads`)
- [Assembly](assembly.md): Perform assembly of filtered reads (`rolypoly assemble`)
- [Assembly Filtering](filter_assembly.md): Remove potential host or contamination sequences (`rolypoly filter-contigs`)
- [Marker Gene Search](marker_search.md): Search for RNA virus hallmark genes (RdRp and other markers) in assembled contigs (`rolypoly marker-search`)
- [Virus Search](search_viruses.md): Search for viral sequences in filtered assemblies (`rolypoly virus-mapping`)

### Annotation
- [Genome Annotation](annotate_rna.md#annotate-rna): Combined RNA and protein annotation 🚧 (`rolypoly annotate`)
- [RNA Annotation](annotate_rna.md): Predict RNA structures and elements 🚧 (`rolypoly annotate-rna`)
- [Protein Annotation](annotate_prot.md): Identify and annotate protein-coding regions 🚧 (`rolypoly annotate-prot`)

### Miscellaneous & Utilities
- [Miscellaneous Commands](misc.md): Quality of life utilities
  - [Shrink Reads](misc.md#shrink-reads): Subsample FASTQ files (`rolypoly shrink-reads`)
  - [Mask DNA](misc.md#mask-dna): Mask viral-like sequences in reference genomes (`rolypoly mask-dna`)
  - [FASTX Stats](misc.md#fastx-stats): Calculate sequence statistics (`rolypoly fastx-stats`)
  - [Rename Sequences](misc.md#rename-sequences): Standardize sequence IDs (`rolypoly rename-seqs`)
  - [Quick Taxonomy](misc.md#quick-taxonomy): Fast taxonomic assignment 🚧 (`rolypoly quick-taxonomy`)
  - [Fetch SRA](misc.md#fetch-sra): Download SRA data from ENA (`rolypoly fetch-sra`)

### Analysis (Experimental)
- [Host Classification](host_classify.md): Predict potential viral hosts ⚠️
- [Binning](binning_termini.md):
  - [Termini Analysis](binning_termini.md): Analyze contig termini (`rolypoly termini`) 🚧
  - [Correlation Analysis](binning_correlate.md): Analyze co-occurrence across samples (`rolypoly correlate`) 🚧

Legend:
- 🚧 Experimental command - implemented but under active development
- ⚠️ Placeholder page - documented but not yet available

## Common Options

Many commands share these common options:

- `-t, --threads`: Number of threads to use (int, default: 1)
- `-M, --memory`: Memory allocation in GB (str, default: "6g")
- `-o, --output` or `--output-dir`: Output location (str, default: current directory + command-specific suffix)
- `--keep-tmp`: Save temporary files and folders (optional flag, default: False)
- `-g, --log-file`: Path to log file (str, default: command-specific log in current directory)
- `-i, --input`: Input file or directory (str, required)

For detailed usage of each command, use the `--help` option:

```bash
rolypoly [COMMAND] --help
```

## Memory Usage Note

The `--memory` argument sets RAM limits for external programs:

- **SPAdes**: Used for genome assembly
- **bbmap**: Used in read filtering
- **MEGAHIT**: Alternative assembler
- **MMseqs2**: Sequence clustering and searching
- **Diamond**: Sequence alignment and annotation
- **HMMER/pyHMMER**: Used for viral marker gene detection

**Note**: There is no guarantee that other rolypoly commands or external programs won't exceed the specified memory.


## Tips and Tricks
- Each of the main commands of rolypoly could be entered into using external inputs (e.g. you already have assembly and want to search it for RdRps).
- If you have a lot of similar samples, some operations might be preformed once instead of rerunning the an entire command. For example, if you are working on the same host (or if ytou suspect the DNA cotanaminats in your samples to be consistent across multiple runs) you can mask the host genome once, externally, provide it to rolypoly's mask_dna, and then when running the `filter*` commands, use the flag "dont_mask" to skip masking. 
- Offloading commands to different machines is a smart idea if your access to a `bigmem` compute node is not a given. This is generally true (regardless of rolypoly) for assembly (tend to be memory hungry, at least spades) compared to marker search (more CPU heavy).  
- You can use a small subset of your input data for dry runs to get a sense of what to expect, sort of an "exploratory" investigation :)
