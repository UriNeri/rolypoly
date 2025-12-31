![RolyPoly Logo](https://code.jgi.doe.gov/rolypoly/docs/-/raw/main/docs/rolypoly_logo.png?ref_type=heads)

# RolyPoly

RolyPoly is an RNA virus analysis toolkit, meant to be a "swiss-army knife" for RNA virus discovery and characterization by including a variety of commands, wrappers, parsers, automations, and some "quality of life" features for any many of a virus investigation process (from raw read processing to genome annotation). While it includes an "end-2-end" command that employs an entire pipeline, the main goals of rolypoly are:
- Help non-computational researchers take a deep dive into their data without compromising on using tools that are non-techie friendly.  
- Help (software) developers of virus analysis pipeline "plug" holes missing from their framework, by using specific RolyPoly commands to add features to their existing code base.

## WIP - NOTE
RolyPoly is an open, still in progress project - I aim to summrise the main functionality into a manuscript ~early 2026. Pull requests and contributions are welcome and will be considered (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## Docs
For more detailed information, please refer to the [docs](https://pages.jgi.doe.gov/rolypoly/docs/). While it isn't updated often, it should still be helpful. Most commands support a `--help` flag and that tends to be the most up date.

## Installation

### Quick and Easy - One Conda/Mamba Environment
**Recommended for most users** who want a "just works" solution and primarily intend to use rolypoly as a CLI tool in an independent environment.

We hope to have rolypoly available from bioconda in the near future.  
In the meantime, it can be installed with the [`quick_setup.sh`](https://code.jgi.doe.gov/rolypoly/rolypoly/-/raw/main/src/setup/quick_setup.sh) script which which will also fetch the pre-generated data rolypoly will require.

```bash
curl -O https://code.jgi.doe.gov/rolypoly/rolypoly/-/raw/main/src/setup/quick_setup.sh && \
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
- conda enviroment in `./rolypoly/env/`  
- log file in `./RolyPoly_quick_setup.log`   



### Modular / Dev - Command-Specific Pixi Environments
**For software developers** looking to try or make use of specific rolypoly features with minimal risk of dependency conflicts. This approach should allow you to install only the tools you need for specific functionality. Note: dependencies from pip are allways installed, the ones from conda/bioconda are the ones that are modular.

```bash
# Install pixi first (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Clone the repository
git clone https://code.jgi.doe.gov/rolypoly/rolypoly.git
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
For detailed modular installation options, see the [installation documentation](https://pages.jgi.doe.gov/rolypoly/docs/installation).

## Usage
RolyPoly is a command-line tool with subcommands grouped by analysis stage. For a detailed help (in terminal), use `rolypoly --help` or `rolypoly <command> --help`. For more specific help, see the [docs](https://pages.jgi.doe.gov/rolypoly/docs/commands/index.md).

```bash
rolypoly  <COMMAND> [ARGS]...
```

### Commands
<!-- "groups" are just categories to help organise, the scripts paths can differ as long as tthe click entry points are correct. -->
#### data
- `get-data` — Download/setup required data
- `version` — Show code and data version info

#### reads
- [`filter-reads`](https://pages.jgi.doe.gov/rolypoly/docs/commands/read_processing): Host/rRNA/adapters/artifact filtering and QC (bbmap, seqkit, etc.)
- [`shrink-reads`](https://pages.jgi.doe.gov/rolypoly/docs/commands/shrink_reads): Downsample or subsample reads. Useful for testing or normalizing coverage across samples.
- [`mask-dna`](https://pages.jgi.doe.gov/rolypoly/docs/commands/mask_dna): Mask DNA regions in RNA-seq reads (bbmap, seqkit). Useful for avoiding mis-filtering of RNA virus reads in because of potential matches to EVEs.

#### annotation
- [`annotate`](https://pages.jgi.doe.gov/rolypoly/docs/commands/annotate): Genome feature annotation (pyrodigal-rv wraps the rna and prot commands)
- [`annotate-rna`](https://pages.jgi.doe.gov/rolypoly/docs/commands/annotate_rna): RNA secondary structure labelling and ribozyme detection (Infernal, ViennaRNA/linearfold, cmsearch on Rfam...)
- [`annotate-prot`](https://pages.jgi.doe.gov/rolypoly/docs/commands/annotate_prot): Protein domain annotation and functional prediction (HMMER, Pfam, custom)

#### assembly (Meta/Genome Assembly)
- `assemble` — Assemble reads into contigs (SPAdes, MEGAHIT, penguin) 
- `filter-contigs` — Filter seqs based on user-supplied host/contamination sequences. Wrapper for both nucleotide and amino acid filtering.

#### misc (Miscellaneous)
- `end2end` — Run end-to-end pipeline
- `fetch-sra` — Download SRA fastq files
- `fastx-stats` — Calculate aggregate statistics for sequences (min, max, mean, median, ...) (input is file/s)
- `fastx_calc` Calculate per-sequence metrics (length, GC content, hash,...)                                                                                
- `rename-seqs` — Rename sequences
- `quick-taxonomy` — Quick taxonomy assignment

#### identify (RNA Virus Identification)
- `marker-search` — Search for viral markers
- `search-viruses` — Map and identify viruses

**Notes:**
- Only the commands listed above are currently exposed via the CLI. Some modules in the codebase are not available as CLI commands.
- For help on any command, use: `rolypoly <command> <options> --help`
- Some commands (e.g., `co-assembly`, `refine`, `visualize`, `characterise`, etc.) are not currently available or are commented out in the CLI.

## Project Status
Active development. Currently implemented features:

- ✅ NGS raw read filtering (Host, rRNA, adapters, artefacts) and quality control report ([`reads filter-reads`](https://pages.jgi.doe.gov/rolypoly/docs/commands/read_processing))
- ✅ Assembly (SPAdes, MEGAHIT and penguin) ([` assemble`](https://pages.jgi.doe.gov/rolypoly/docs/commands/assembly))
- ✅ Contig filtering and clustering ([`filter-contigs`](https://pages.jgi.doe.gov/rolypoly/docs/commands/filter_assembly))
- ✅ Marker gene search with pyhmmer (mainly RdRps, genomad VV's or user-provided) ([`identify marker-search`](https://pages.jgi.doe.gov/rolypoly/docs/commands/marker_search))
- ✅ RNA secondary structure prediction, annotation and ribozyme identification ([`annotation annotate-rna`](https://pages.jgi.doe.gov/rolypoly/docs/commands/annotate_rna))
- ✅ Nucleotide search vs known viruses ([`identify search-viruses`](https://pages.jgi.doe.gov/rolypoly/docs/commands/search_viruses))
- ✅ Prepare external data ([`data get-data`](https://pages.jgi.doe.gov/rolypoly/docs/commands/prepare_external_data))


Under development:
- 🚧 Protein annotation (`annotate-prot`) (mostly done, but need to check other DBs or tools - Currently no structural prediction support)

Planned/under consideration features (but not yet started):
- 🚧 Genome binning and refinement (`TBD`)
- 🚧 Virus taxonomic classification (`TBD`)
- 🚧 Virus feature prediction (+/-ssRNA/dsRNA, circular/linear, mono/poly-segmented, capsid type, etc.) (`TBD`)
- 🚧 Cross-sample analysis (`TBD`)
- 🚧 Host prediction (`TBD`)

For more details about the implementation status and roadmap please contact us directly or open an issue.

## Dependencies

**📦 Modular Installation Available**: RolyPoly supports both quick setup (one environment for all tools) and modular installation (command-specific environments). The modular approach is particularly useful for software developers who want to integrate specific rolypoly features with minimal dependency conflicts. See the [installation documentation](./docs/docs/mkdocs_docs/installation.md) for details.

Not all 3rd party software is used by all the different commands. RolyPoly includes a "citation reminder" that will try to list all the external software used by a command. The "reminded citations" are pretty printed to console (stdout) but are also written to a logfile. The bibtex file rolypoly uses for this is included in the codebase.

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
There are many good virus analysis software out there*. Many of them are custom made for specific virus groups, some are generalists, but most of them require complete control over the analysis process (so one or two point of entry for data). Apart from the input, these pipelines vary in their implementation (laguange, workflow magnement system (snakemake, nextflow...), dependecies), methodologies (tool choice for similar step like assembler), goals (e.g. specific pathogen analysis vs whole  virome analysis). These are other differences effect the design process and the tooling choices (such as selecting a fast nucleic based sequence search method limited to high identity, over a slow but more senstive profile or structure (amino) based search method). This has created some "lock in" (IMO), and I have found myself asked by people "what do you recomend for xyz" or "which pipeline should I use". Most people have limited time to invest in custom analysis pipeline design and so end up opting for an existing, off-the-shelve option, potentially compromising or having to align their goals with what the given software offers (if they they are already aligned - great!). 
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
- [RdRp-CATCH](https://github.com/dimitris-karapliafis/RdRpCATCH) If you are interested in profile based marker searches, benchmarking, and thershold setting.
- [suvtk](https://github.com/LanderDC/suvtk) if you are looking to expediate NCBI submission (among other tasks)
- [gff2parquet](https://github.com/UriNeri/gff2parquet) if you are looking for a fast GFF parser and converter to parquet format (note, also WIP).
- [pyrodigal-rv](https://github.com/LanderDC/pyrodigal-rv) if you are looking for an RNA virus specific Prodigal fork (incl. newly trained models for exotic genetic codes!)

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

