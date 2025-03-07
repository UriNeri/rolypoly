![RolyPoly Logo](docs/rolypoly_logo.png)

# RolyPoly

RolyPoly is an RNA virus analysis toolkit, including a variety of commands and wrappers for external tools (from raw read processing to genome annotation). It also includes an "end-2-end" command that employs an entire pipeline.   
For more detailed information, please refer to the [docs](https://pages.jgi.doe.gov/uneri/rolypoly/).

## Installation

There are several ways to install RolyPoly:

### Option 1: Quick Setup (Recommended)

<details><summary>The fastest way to get started. Downloads and installs everything automatically, including mamba if needed.</summary>

```bash
curl -O https://code.jgi.doe.gov/UNeri/rolypoly/-/raw/main/misc/quick_setup.sh && \
bash quick_setup.sh
```

Or with custom paths:
```bash
bash quick_setup.sh /path/to/conda/env /path/to/install/rolypoly_code /path/to/store/databases /path/to/logfile
```
  
*The logfile will be saved to the path you provide, or to `~/RolyPoly_quick_setup.log` if you don't provide a path.
</details>

### Option 2: Manual Installation with Custom Data Directory

<details><summary>Install RolyPoly but specify where to store the external databases</summary>

```bash
git clone https://code.jgi.doe.gov/UNeri/rolypoly.git
cd rolypoly
mamba create -n rolypoly -f misc/env_files/env_big.yaml 
mamba activate rolypoly
pip install . # Use pip install -e .[dev] for development installation
rolypoly prepare-external-data --data_dir /path/to/data/dir
conda env config vars set ROLYPOLY_DATA=$ROLYPOLY_DATA
```
</details>

### Option 3: Minimal Installation

<details><summary>Create smaller conda environments, each containing dependencies for specific commands.</summary>

```bash
mamba create -n rolypoly_filter_reads -f misc/env_files/filter_reads.yaml
mamba create -n rolypoly_assembly -f misc/env_files/assembly.yaml
mamba create -n rolypoly_annotation -f misc/env_files/annotation.yaml 
mamba create -n rolypoly_rdrp -f misc/env_files/marker_searchyaml 
mamba activate rolypoly_<command_name>
pip install .
rolypoly prepare-external-data
```
</details>

## Usage
RolyPoly is a command-line tool with subcommands for different stages of the RNA virus identification pipeline. For a detailed help (in terminal), use `rolypoly help`. For more specific help, see the [docs](./docs/mkdocs_docs/commands/index.md).

```bash
rolypoly [OPTIONS] COMMAND [ARGS]...
 ```

## Project Status
Active development. Currently implemented features:
- ✅ Read filtering and quality control [(`filter-reads`)](docs/mkdocs_docs/commands/read_processing.md)
- ✅ Assembly with multiple assemblers [(`assembly`)](docs/mkdocs_docs/commands/assembly.md)
- ✅ Assembly filtering and clustering [(`filter-contigs`)](docs/mkdocs_docs/commands/assembly_filtering.md)
- ✅ Marker gene search with pyhmmer [(`marker-search`)](docs/mkdocs_docs/commands/marker_search.md)
- ✅ RNA structure prediction, annotation and ribozyme identification [(`annotate-rna`)](docs/mkdocs_docs/commands/annotate_rna.md)
- ✅ Nucleotide search vs known viruses [(`search-viruses`)](docs/mkdocs_docs/commands/search_viruses.md)
- ✅ Prepare external data [(`prepare-external-data`)](docs/mkdocs_docs/commands/prepare_external_data.md)  

Under development:
- 🚧 Protein annotation (`annotate-protein`)
- 🚧 Host prediction (`host-predict`)
- 🚧 Genome binning and refinement (`TBD`)
- 🚧 Life-style prediction (`TBD`)
- 🚧 Cross-sample analysis (`TBD`)

For more details about the implementation status, roadmap, additional commands, and more, see the [workflow documentation](./docs/mkdocs_docs/workflow.md) and the [API reference](./docs/mkdocs_docs/api/utils/various.md).

## Dependencies
<details><summary>Click to show dependencies</summary>

By default, the ["quick_setup.sh"](./misc/quick_setup.sh) option will install all of these with mamba/conda or pip. If you are curios or prefer to install them/get the binaries alternatively, see the [setup*.py scripts](/misc/). Not all commands require all the dependencies, hence we made some slimmer conda environments for specific purposes - but that is not recommended.

1. [SPAdes](https://github.com/ablab/spades)
2. [seqkit](https://github.com/shenwei356/seqkit)
3. [datasets](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/)
4. [bbmap](https://sourceforge.net/projects/bbmap/) - via [bbmapy](https://github.com/urineri/bbmapy)
5. [megahit](https://github.com/voutcn/megahit)
6. [fastqc](https://www.bioinformatics.babraham.ac.uk/projects/fastqc)
7. [mmseqs2](https://github.com/soedinglab/MMseqs2)
8. [plass and penguin](https://github.com/soedinglab/plass)
9. [blast](https://blast.ncbi.nlm.nih.gov/doc/blast-help/)
10. [diamond](https://github.com/bbuchfink/diamond)
11. [pigz](https://github.com/madler/pigz)
12. [prodigal/gv](https://github.com/hyattpd/Prodigal) - via pyrodigal-gv
13. [linearfold](https://github.com/LinearFold/LinearFold)
14. [HMMER](https://github.com/EddyRivasLab/hmmer) - via pyhmmer
15. [needletail](https://github.com/onecodex/needletail)
16. [infernal](https://github.com/EddyRivasLab/infernal)
17. [aragorn](http://130.235.244.92/ARAGORN/)
18. [tRNAscan-SE](http://lowelab.ucsc.edu/tRNAscan-SE/)
19. [bowtie](https://github.com/BenLangmead/bowtie)

### Python Libraries
* [polars](https://pola.rs/)
* [scipy](https://scipy.org/)
* [numpy](https://numpy.org/)
* [rich_click](https://pypi.org/project/rich-click/)
* [rich](https://github.com/Textualize/rich)
* [pyhmmer](https://github.com/althonos/pyhmmer)
* [pyrodigal-gv](https://github.com/althonos/pyrodigal-gv)
* [biopython](https://biopython.org/)
* [multiprocess](https://github.com/uqfoundation/multiprocess)
* [requests](https://requests.readthedocs.io)
* [pgzip](https://github.com/pgzip/pgzip)
* [pyfastx](https://github.com/lmdu/pyfastx)
* [psutil](https://pypi.org/project/psutil/)
* [bbmapy](https://github.com/urineri/bbmapy)
* [sh](https://github.com/amoffat/sh)
* [pymsaviz](https://github.com/aziele/pymsaviz)
* [viennarna](https://github.com/ViennaRNA/ViennaRNA)
* [pyranges](https://github.com/biocore-ntnu/pyranges)
* [intervaltree](https://github.com/chaimleib/intervaltree)
* [genomicranges](https://github.com/CoreyMSchafer/genomicranges)
* [lightmotif](https://github.com/dincarnato/LightMotif)
* [multiqc](https://github.com/ewels/MultiQC)
* [mappy](https://github.com/lh3/minimap2/tree/master/python)

</details>

### Databases used by rolypoly  
RolyPoly will try to remind you to cite these (along with tools) based on the commands you run. For more details, see the [remind_citations.py](./src/rolypoly/commands/utils/remind_citations.py) script.

<details><summary>Click to show databases</summary>

* [NCBI RefSeq rRNAs](https://doi.org/10.1093%2Fnar%2Fgkv1189) - Reference RNA sequences from NCBI RefSeq
* [NCBI RefSeq viruses](https://doi.org/10.1093%2Fnar%2Fgkv1189) - Reference viral sequences from NCBI RefSeq
* [PFAM_A_37](https://doi.org/10.1093/nar/gkaa913) - RdRp and RT profiles from Pfam-A version 37
* [RVMT](https://doi.org/10.1016/j.cell.2022.08.023) - RNA Virus Meta-Transcriptomes database
* [SILVA_138](https://doi.org/10.1093/nar/gks1219) - High-quality ribosomal RNA database
* [NeoRdRp_v2.1](https://doi.org/10.1264/jsme2.ME22001) - Collection of RdRp profiles
* [RdRp-Scan](https://doi.org/10.1093/ve/veac082) - RdRp profile database incorporating PALMdb
* [TSA_2018](https://doi.org/10.1093/molbev/msad060) - RNA virus profiles from transcriptome assemblies
* [Rfam](https://doi.org/10.1093/nar/gkaa1047) - Database of RNA families (structural/catalytic/both)

</details>

## Motivation
Current workflows for RNA virus detection are functional but could be improved, especially by utilizing raw reads instead of pre-existing, general-purpose made, assemblies. Here we proceed with more specific processes tailored for RNA viruses.

Several similar software exist, but have different uses, for example:  
- hecatomb ([github.com/shandley/hecatomb](https://github.com/shandley/hecatomb)): uses mmseqs for homology detection and thus is less sensitive than the additional HMMer based identification herein.
- AliMarko ([biorxiv.org/content/10.1101/2024.07.19.603887](https://biorxiv.org/content/10.1101/2024.07.19.603887)): Utilizes a single-sample assembly only approach, not supporting co/cross assembly of multiple samples. Additionally, AliMarko uses a small, partially outdated (IMO) HMM profile set.

### Reporting Issues
For issues or suggestions, please open an issue on the [Issues](https://code.jgi.doe.gov/UNeri/rolypoly/-/issues) page.  
If possible, please provide the exact way you called rolypoly (command and location), as well as a log file (if generated) and other potential details (e.g. for the read-filtering step, the config file  `run_info/config.json` generated).

## Authors
<details><summary>Click to show authors</summary>

- Uri Neri
- Brian Bushnell
- Simon Roux
- Antônio Pedro Camargo
- Andrei Stecca Steindorff
- Clement Coclet
- David Parker
</details>

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

