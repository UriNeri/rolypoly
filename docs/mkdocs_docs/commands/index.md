# Commands Overview

!!! warning "🚧 Experimental"
    RolyPoly is under active development - features may be incomplete or experimental.

RolyPoly provides several commands for different stages of viral analysis. For detailed help in the terminal, use:

```bash
rolypoly help
```

## Available Commands

### Core Pipeline
- [Prepare External Data](prepare_external_data.md): Download and prepare necessary databases
- [End to End](end_to_end.md): Run the complete pipeline with default settings
- [Read Processing](read_processing.md): Filter and process raw RNA-seq reads
- [Assembly](assembly.md): Perform assembly of filtered reads
- [Marker Gene Search](marker_search.md): Search for RNA virus hallmark genes (RdRp and other markers) in assembled contigs
- [Assembly Filtering](filter_assembly.md): Remove potential host or contamination sequences
- [Virus Search](search_viruses.md): Search for viral sequences in filtered assemblies

### Annotation
- [RNA Annotation](annotate_rna.md): Predict RNA structures and elements 🚧
- [Protein Annotation](annotate_prot.md): Identify and annotate protein-coding regions 🚧

### Analysis
- [Host Classification](host_classify.md): Predict potential viral hosts ⚠️
- [Binning](binning_termini.md):
  - [Termini Analysis](binning_termini.md): Analyze contig termini ⚠️
  - [Correlation Analysis](binning_correlate.md): Analyze sequence patterns across samples ⚠️

Legend:
- 🚧 Experimental command
- ⚠️ Not yet implemented

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

