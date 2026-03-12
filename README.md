![RolyPoly Logo](https://raw.githubusercontent.com/UriNeri/rolypoly/main/docs/rolypoly_logo.png)

# RolyPoly

[![PyPI version](https://img.shields.io/pypi/v/rolypoly-tk.svg?cacheSeconds=300)](https://pypi.org/project/rolypoly-tk/) [![Python versions](https://img.shields.io/pypi/pyversions/rolypoly-tk.svg?cacheSeconds=300)](https://pypi.org/project/rolypoly-tk/) [![PyPI Downloads](https://static.pepy.tech/personalized-badge/rolypoly-tk?period=monthly&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=Downloads+%28month%29)](https://pepy.tech/projects/rolypoly-tk) [![License](https://img.shields.io/github/license/UriNeri/rolypoly.svg)](LICENSE) [![Docs](https://img.shields.io/badge/docs-urineri.github.io%2Frolypoly-blue)](https://urineri.github.io/rolypoly/)

RolyPoly is an RNA virus analysis toolkit, meant to be a "swiss-army knife" for RNA virus discovery and characterization by including a variety of commands, wrappers, parsers, automations, and some "quality of life" features for any many of a virus investigation process (from raw read processing to genome annotation). While it includes an "end-2-end" command that employs an entire pipeline, the main goals of rolypoly are:
- Help non-computational researchers take a deep dive into their data without compromising on using tools that are non-techie friendly.  
- Help (software) developers of virus analysis pipeline "plug" holes missing from their framework, by using specific RolyPoly commands to add features to their existing code base.

## Note - Rolypoly is still under development (contributions welcome!)
RolyPoly is an open, still in progress project - I aim to summarise the main functionality into a manuscript ~mid 2026. Pull requests and contributions are welcome and will be considered (see [CONTRIBUTING.md](CONTRIBUTING.md)).  
This also means that there are bugs, verbose logging even for non debug mode, and some place holders and TODOs here and there.

## Installation

### Quick and Easy - One Conda/Mamba Environment
**Recommended for most users** who want a "just works" solution and primarily intend to use rolypoly as a CLI tool in an independent environment.

We hope to have rolypoly available from bioconda in the near future.  
In the meantime, it can be installed with the [`quick_setup.sh`](https://raw.githubusercontent.com/UriNeri/rolypoly/main/src/setup/quick_setup.sh) script, which will also fetch the pre-generated data rolypoly requires.

```bash
curl -O https://raw.githubusercontent.com/UriNeri/rolypoly/main/src/setup/quick_setup.sh && \
bash quick_setup.sh 
```

#### Quick Setup - Additional Options
You can specify custom paths for the code, databases, and conda environment location:
```bash
bash quick_setup.sh /path/to/conda/env /path/to/install/rolypoly_code /path/to/store/databases /path/to/logfile
```
By default if no positional arguments are supplied, rolypoly is installed into the session current folder (path the quick_setup.sh is called from):   
- database in `./rolypoly/data/`  
- code in `./rolypoly/code/ `  
- conda environment in `./rolypoly/env/`  
- log file in `./RolyPoly_quick_setup.log`   



### Modular / Dev - Command-Specific Pixi Environments
**For software developers** looking to try or make use of specific rolypoly features with minimal risk of dependency conflicts. This approach should allow you to install only the tools you need for specific functionality. Note: dependencies from pip are always installed; conda/bioconda dependencies are the modular ones.

```bash
# Install pixi first (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Clone the repository
git clone https://github.com/UriNeri/rolypoly.git
cd rolypoly

# Install for specific functionality (examples):
pixi install -e reads-only        # Just read processing tools
pixi install -e assembly-only     # Just assembly tools  
pixi install -e basic-analysis    # Reads + assembly + identification
pixi install -e complete          # All tools (equivalent to legacy install)

# Run commands in the appropriate environment
pixi run -e reads-only rolypoly filter-reads --help
# or load the environment
pixi shell -e reads-only
rolypoly filter-reads --help
```  
For detailed modular installation options, see the [installation documentation](https://urineri.github.io/rolypoly/installation).

## Usage
RolyPoly is a command-line tool with subcommands grouped by analysis stage. 
Use `rolypoly --help` or `rolypoly <command> --help` for most up to date details. Some additional information is in the [docs](https://urineri.github.io/rolypoly/commands/).

```bash
rolypoly  <COMMAND> [ARGS]...
```

## Commands and Project Status
Active development. Command groups and current implementation status are summarized below.  

Legend: 
- ✅ - Available (on pypi and has tests). Command default parameters are unlikely to change much. 
- 🧪 - Experimental, might not be on pypi / have tests. Default parameters may change. Code might be in a seperate dev branch.
- 🚧 - Under active development.
- 🤔/TBD - Planned / under consideration.

#### Data
- ✅ [`get-data`](https://urineri.github.io/rolypoly/commands/prepare_external_data) — Download/setup required data
- ✅ `version` — Show code and data version info

#### Raw-Reads
- ✅ [`filter-reads`](https://urineri.github.io/rolypoly/commands/read_processing) — Host/rRNA/adapters/artifact filtering and QC (bbmap, falco, etc.)
- ✅ [`shrink-reads`](https://urineri.github.io/rolypoly/commands/read_processing) — Downsample or subsample reads. Useful for testing or normalizing coverage across samples.
- ✅ [`mask-dna`](https://urineri.github.io/rolypoly/commands/read_processing) — Mask DNA regions in RNA-seq reads (bbmap, seqkit). Useful for avoiding mis-filtering of RNA virus reads in because of potential matches to EVEs.

#### Annotation
- ✅ [`annotate`](https://urineri.github.io/rolypoly/commands/annotate_rna/#annotate-rna) — Genome feature annotation (wraps the rna and prot commands)
- ✅ [`annotate-rna`](https://urineri.github.io/rolypoly/commands/annotate_rna) — RNA secondary structure labelling and ribozyme detection (Infernal, ViennaRNA/linearfold, cmsearch on Rfam...)
- 🧪 [`annotate-prot`](https://urineri.github.io/rolypoly/commands/annotate_prot) — Gene calling and Protein domain annotation and functional prediction (HMMER, Pfam, custom).

#### Meta/Genome Assembly
- ✅ [`assemble`](https://urineri.github.io/rolypoly/commands/assembly) — Assemble reads into contigs (SPAdes, MEGAHIT, penguin)
- ✅ [`filter-contigs`](https://urineri.github.io/rolypoly/commands/filter_assembly) — Filter sequences based on user-supplied host/contamination references (nucleotide and amino acid modes).

#### RNA Virus Identification
- ✅ [`marker-search`](https://urineri.github.io/rolypoly/commands/marker_search) — Search for viral markers (mainly RdRps, genomad VVs, or user-provided), using profile-based methods (HMMER / MMseqs2). 
- ✅ [`virus-mapping`](https://urineri.github.io/rolypoly/commands/search_viruses) — Map and identify viruses using nucleic acid search (MMseqs2).
- ✅ [`rdrp-motif-search`](https://urineri.github.io/rolypoly/commands/rdrp_motif_search) — Search RdRp motifs (A/B/C/D) in nucleotide or amino acid sequences.

#### Bining / Clustering
- 🧪 [`cluster`](https://urineri.github.io/rolypoly/commands/cluster) — Average Nucleic identity (ANI) based contig grouping. Supports several common backends and methods.
- 🧪 [`extend`](https://urineri.github.io/rolypoly/commands/extend) — Extend sequences by pile-up/assembly. Useful for combining assemblies of with low abundance viruses, or those with high microdiversity, at the cost of worse strain/sub-species resolution (i.e. can condense to a consensus).
- 🧪 [`termini`](https://urineri.github.io/rolypoly/commands/binning_termini) — Shared termini grouping and motif reporting. Writes assignments + groups tables (TSV/CSV/Parquet/JSONL) and motif FASTA by default.
- 🧪 [`correlate`](https://urineri.github.io/rolypoly/commands/binning_correlate) — Group contigs based on co-occurrence, co-abundance, minimal correlation (Spearman's) of these, or both.
- 🤔 [`binit`](https://urineri.github.io/rolypoly/commands/binit) — Combines the above commands with sample information and genome attributes (e.g. require a shared termini AND protein complementarity, like CP + RdRp). See [notebooks/Exprimental/partiti_usecase/partiti_segment_workflow_experimental.ipynb](notebooks/Exprimental/partiti_usecase/partiti_segment_workflow_experimental.ipynb) for candidate workflow.

#### Miscellaneous
- ✅ [`roll`](https://urineri.github.io/rolypoly/commands/end_to_end) — Run an end-to-end pipeline (before v0.7.1, named `end2end`).
- ✅ [`fetch-sra`](https://urineri.github.io/rolypoly/commands/misc) — Download SRA fastq files (from ENA)
- ✅ [`fastx-calc`](https://urineri.github.io/rolypoly/commands/misc) — Calculate per-sequence metrics (length, GC content, hash, ...)
- ✅ [`fastx-stats`](https://urineri.github.io/rolypoly/commands/misc) — Calculate (-->aggregate) statistics for sequences (min, max, mean, median, ...) (input is file/s)
- ✅ [`rename-seqs`](https://urineri.github.io/rolypoly/commands/misc) — Rename sequences (add a prefix, suffix, hash, running number, etc.)
- 🚧 [`quick-taxonomy`](https://urineri.github.io/rolypoly/commands/misc) — Quick taxonomy assignment. Candidate workflows are [github.com/UriNeri/ictv-mmseqs2-protein-database](https://github.com/UriNeri/ictv-mmseqs2-protein-database) and [github.com/apcamargo/ictv-mmseqs2-protein-database](https://github.com/apcamargo/ictv-mmseqs2-protein-database) 
- 🤔 support for [genotate](https://github.com/deprekate/genotate) for gene prediction.
- 🤔 Genome refinement / strain de-entalgement / variant calling?
- 🤔 Virus feature prediction (+/-ssRNA/dsRNA, circular/linear, mono/poly-segmented, capsid type, etc.)
- 🤔 Host prediction 
- 🤔 protein structural prediction support (and reseek search xyz dbs)

If you have suggestions for additional commands or features, or want to implement some of these - please let us know, and consider contributing :-) 

## Dependencies

**📦 Modular Installation Available**: RolyPoly supports both quick setup (one environment with all dependecies for all commands) and modular installation (command-specific environments). The modular approach is particularly useful for software developers who want to integrate specific rolypoly features with minimal dependency conflicts. See the [installation documentation](https://urineri.github.io/rolypoly/installation/) for details.

Not all 3rd party software is used by all the different commands. RolyPoly includes a "citation reminder" that will try to list all the external software used by a command. The "reminded citations" are pretty printed to console (stdout) and to a logfile. To shut off the terminal citation reminder printing, set `ROLYPOLY_REMIND_CITATIONS` to false in your `rpconfig.json` file.

<details><summary>Click to show dependencies</summary>  

Non-Python  
- [SPAdes](https://github.com/ablab/spades).
- [seqkit](https://github.com/shenwei356/seqkit)
- [datasets](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/)
- [bbmap](https://sourceforge.net/projects/bbmap/) - via [bbmapy](https://github.com/urineri/bbmapy)
- [megahit](https://github.com/voutcn/megahit)
- [mmseqs2](https://github.com/soedinglab/MMseqs2)
- [plass and penguin](https://github.com/soedinglab/plass)
- [diamond](https://github.com/bbuchfink/diamond)
- [pigz](https://github.com/madler/pigz)
- [prodigal](https://github.com/hyattpd/Prodigal) - via pyrodigal-rv
- [linearfold](https://github.com/LinearFold/LinearFold)
- [HMMER](https://github.com/EddyRivasLab/hmmer) - via pyhmmer
- [needletail](https://github.com/onecodex/needletail)
- [infernal](https://github.com/EddyRivasLab/infernal)
- [aragorn](http://130.235.244.92/ARAGORN/)
- [tRNAscan-SE](http://lowelab.ucsc.edu/tRNAscan-SE/)
- [bowtie1](https://github.com/BenLangmead/bowtie)
- [falco](https://github.com/smithlabcode/falco/)

### Python Libraries
* [polars](https://pola.rs/)
* [numpy](https://numpy.org/)
* [rich_click](https://pypi.org/project/rich-click/)
* [rich](https://github.com/Textualize/rich)
* [pyhmmer](https://github.com/althonos/pyhmmer)
* [pyrodigal-rv](https://github.com/landerdc/pyrodigal-rv)
* [multiprocess](https://github.com/uqfoundation/multiprocess)
* [requests](https://requests.readthedocs.io)
* [pgzip](https://github.com/pgzip/pgzip)
* [pyfastx](https://github.com/lmdu/pyfastx)
* [psutil](https://pypi.org/project/psutil/)
* [bbmapy](https://github.com/urineri/bbmapy)
* [pymsaviz](https://github.com/aziele/pymsaviz)
* [viennarna](https://github.com/ViennaRNA/ViennaRNA)
* [pyranges](https://github.com/biocore-ntnu/pyranges)
* [intervaltree](https://github.com/chaimleib/intervaltree)
* [genomicranges](https://github.com/CoreyMSchafer/genomicranges)
* [lightmotif](https://github.com/dincarnato/LightMotif)
* [mappy](https://github.com/lh3/minimap2/tree/master/python)

</details>

### Databases used by rolypoly  
RolyPoly will try to remind you to cite these too based on the commands you run. For more details, see the [citation_reminder.py](./src/rolypoly/utils/logging/citation_reminder.py) script and [all_used_tools_dbs_citations](./src/rolypoly/utils/logging/all_used_tools_dbs_citations.json)

<details><summary>Click to show databases</summary>

* [NCBI RefSeq rRNAs](https://doi.org/10.1093%2Fnar%2Fgkv1189) - Reference RNA sequences from NCBI RefSeq
* [NCBI RefSeq viruses](https://doi.org/10.1093%2Fnar%2Fgkv1189) - Reference viral sequences from NCBI RefSeq
* [pfam_A_38](https://doi.org/10.1093/nar/gkaa913) - RdRp and RT profiles from Pfam-A version 38
* [RVMT](https://doi.org/10.1016/j.cell.2022.08.023) - RNA Virus Meta-Transcriptomes database
* [SILVA_138](https://doi.org/10.1093/nar/gks1219) - High-quality ribosomal RNA database
* [NeoRdRp_v2.1](https://doi.org/10.1264/jsme2.ME22001) - Collection of RdRp profiles
* [RdRp-Scan](https://doi.org/10.1093/ve/veac082) - RdRp profile database incorporating PALMdb
* [TSA_2018](https://doi.org/10.1093/molbev/msad060) - RNA virus profiles from transcriptome assemblies
* [Rfam](https://doi.org/10.1093/nar/gkaa1047) - Database of RNA families (structural/catalytic/both)
* [VFAM](https://doi.org/10.3390/v16081191) - Viral protein family database (part of vog/vogdb).
* [UniRef50](https://www.uniprot.org/help/uniref) - UniProt Reference Clusters at 50% sequence identity

</details>

## Motivation
There are many good virus analysis tools out there*. Many of them are custom made for specific virus groups, some are generalists, but most require complete control over the analysis process (so one or two points of entry for data). Apart from input requirements, these pipelines vary in implementation (language, workflow management system (snakemake, nextflow...), dependencies), methodologies (tool choice for a similar step such as assembly), and goals (e.g. specific pathogen analysis vs whole virome analysis). These differences affect design and tooling choices (such as selecting a fast nucleotide-based sequence search method limited to high identity, over a slower but more sensitive profile- or structure-based (amino acid) search method). This has created some "lock in" (IMO), and I have found myself asked by people "what do you recommend for xyz" or "which pipeline should I use". Most people have limited time to invest in custom analysis pipeline design and so end up opting for an existing, off-the-shelf option, potentially compromising or having to align their goals with what the given software offers (if they are already aligned - great!). 
* Checkout [awesome-rna-virus-tools](https://github.com/rdrp-summit/awesome-rna-virus-tools) for an awesome list of RNA virus (and related) software.

### Reporting Issues
Please report bugs you find in the [Issues](https://github.com/UriNeri/rolypoly/issues) page.    


### Contribution
All forms of contributions are welcome - please see the [CONTRIBUTING.md](./CONTRIBUTING.md) file for more details.

## Authors (partial list, TBD update)
<details><summary>Click to show authors</summary>

- Uri Neri
- Antônio Pedro Castello Branco Rocha Camargo
- Dimitris Karapliafis
- Brian Bushnell
- Andrei Stecca Steindorff
- Clement Coclet
- Frederik Schulz
- David Parker
- Simon Roux
- And more!
- Your name here? Open a PR :)
</details>

## Related projects
- [RdRp-CATCH](https://github.com/dimitris-karapliafis/RdRpCATCH) If you are interested in profile-based marker searches, benchmarking, and threshold setting.
- [suvtk](https://github.com/LanderDC/suvtk) if you are looking to expedite NCBI submission (among other tasks)
- [gff2parquet](https://github.com/UriNeri/gff2parquet) if you are looking for a fast GFF parser and converter to parquet format (note, also WIP).
- [pyrodigal-rv](https://github.com/LanderDC/pyrodigal-rv) if you are looking for an RNA virus specific Prodigal fork (incl. newly trained models for exotic genetic codes!)
- [hoodini](https://github.com/pentamorfico/hoodini) if you are interested in large-scale gene neighborhood analyses and visualization.


## Acknowledgments
Thanks to the DOE Joint Genome Institute for infrastructure support. Special thanks to all contributors who have offered insights and improvements.

## Copyright Notice  

RolyPoly (rp) Copyright (c) 2024, The Regents of the University of
California, through Lawrence Berkeley National Laboratory (subject
to receipt of any required approvals from the U.S. Dept. of Energy). 
All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.

### License Agreement 

GPL v3 License

RolyPoly (rp) Copyright (c) 2024, The Regents of the University of
California, through Lawrence Berkeley National Laboratory (subject
to receipt of any required approvals from the U.S. Dept. of Energy). 
All rights reserved.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

