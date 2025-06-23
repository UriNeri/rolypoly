![RolyPoly Logo](https://code.jgi.doe.gov/rolypoly/docs/-/raw/main/docs/rolypoly_logo.png?ref_type=heads)

# RolyPoly

RolyPoly is an RNA virus analysis toolkit, including a variety of commands and wrappers for external tools (from raw read processing to genome annotation). It also includes an "end-2-end" command that employs an entire pipeline.   
For more detailed information, please refer to the [docs](https://pages.jgi.doe.gov/rolypoly/docs/).

## Installation
We hope to have rolypoly available from bioconda in the near future.  
In the meantime, it can be installed with the [`quick_setup.sh`](https://code.jgi.doe.gov/rolypoly/rolypoly/-/raw/main/src/setup/quick_setup.sh) script which which will also fetch the pre-generated data rolypoly will require.

```bash
curl -O https://code.jgi.doe.gov/rolypoly/rolypoly/-/raw/main/src/setup/quick_setup.sh && \
bash quick_setup.sh 
```

You can specify custom paths for the code, databases, and conda enviroment location:
```bash
bash quick_setup.sh /path/to/conda/env /path/to/install/rolypoly_code /path/to/store/databases /path/to/logfile
```
By default if no positional arguments are supplied, rolypoly is installed into the session current folder (path the quick_setup.sh is called from): 
- database in `./rolypoly/data/`
- code in `./rolypoly/code/ `
- conda enviroment in `./rolypoly/env/`
- log file in `./RolyPoly_quick_setup.log` 

To install rolypoly in development mode, use:
```bash
bash quick_setup.sh /path/to/conda/env /path/to/install/rolypoly_code /path/to/store/databases /path/to/logfile TRUE
```


## Usage
RolyPoly is a command-line tool with subcommands for different stages of the RNA virus identification pipeline. For a detailed help (in terminal), use `rolypoly help`. For more specific help, see the [docs](./https://pages.jgi.doe.gov/rolypoly/docs/commands/index.md).

```bash
rolypoly [OPTIONS] COMMAND [ARGS]...
 ```

## Project Status
Active development. Currently implemented features:
- ✅ NGS raw read filtering (Host, rRNA, adapters, artefacts) and quality control report[(`filter-reads`)](https://pages.jgi.doe.gov/rolypoly/docs/commands/read_processing)
- ✅ Assembly (SPAdes, MEGAHIT and penguin) [(`assembly`)](https://pages.jgi.doe.gov/rolypoly/docs/commands/assembly)
- ✅ Contig filtering and clustering [(`filter-contigs`)](https://pages.jgi.doe.gov/rolypoly/docs/commands/filter_assembly)
- ✅ Marker gene search with pyhmmer (mainly RdRps, genomad VV's or user-provided) [(`marker-search`)](https://pages.jgi.doe.gov/rolypoly/docs/commands/marker_search)
- ✅ RNA secondary structure prediction, annotation and ribozyme identification [(`annotate-rna`)](https://pages.jgi.doe.gov/rolypoly/docs/commands/annotate_rna)
- ✅ Nucleotide search vs known viruses [(`search-viruses`)](https://pages.jgi.doe.gov/rolypoly/docs/commands/search_viruses)
- ✅ Prepare external data [(`prepare-external-data`)](https://pages.jgi.doe.gov/rolypoly/docs/commands/prepare_external_data)  

Under development:
- 🚧 Protein annotation (`annotate-protein`)
- 🚧 Host prediction (`host-predict`)
- 🚧 Genome binning and refinement (`TBD`)
- 🚧 Virus taxonomic classification (`TBD`)
- 🚧 Virus feature prediction (+/-ssRNA/dsRNA, circular/linear, mono/poly-segmented, capsid type, etc.) (`TBD`)
- 🚧 Cross-sample analysis (`TBD`)

For more details about the implementation status, roadmap, additional commands, and more, see the [workflow documentation](https://pages.jgi.doe.gov/rolypoly/docs/workflow).

## Dependencies
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
- [prodigal](https://github.com/hyattpd/Prodigal) - via pyrodigal-gv (add link)
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
* [pyrodigal-gv](https://github.com/althonos/pyrodigal-gv)
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
RolyPoly will try to remind you to cite these (along with tools) based on the commands you run. For more details, see the [citation_reminder.py](./src/rolypoly/utils/citation_reminder.py) script.

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

### Reporting Issues and Contribution
RolyPoly is hosted on GitHub (issue tracking and development) and JGI's gitlab (Documentation, releases and archiving).  
Please report bugs you find in the [Issues](https://github.com/UriNeri/rolypoly/issues) page.  
Suggestions and Contributions are welcome - either fork the repo and open a pull request or contact us directly.

## Authors
<details><summary>Click to show authors</summary>

- Uri Neri
- Brian Bushnell
- Simon Roux
- Antônio Pedro Camargo
- Andrei Stecca Steindorff
- Clement Coclet
- David Parker
- Dimitris Karapliafis
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

