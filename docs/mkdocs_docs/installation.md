Installation
============

## Requirements
The most basic requirements are:
- Conda/Mamba (or git/curl/wget, which will download mamba)
- [Pixi](https://pixi.sh) (for modular/dev installations)

These need to already be on the machine you want to install rolypoly on, as Conda/Mamba are used for managing the Python interpreter and external tools. Specific Python libraries are managed via pip. For modular installations, [pixi](https://pixi.sh) is used to manage both PyPI and conda dependencies.


## Quick and Easy - One Conda/Mamba Environment

**Recommended for most users** who want a "just works" solution and primarily intend to use rolypoly as a CLI tool in an independent environment.

We hope to have rolypoly available from bioconda in the near future. In the meantime, it can be installed with the [`quick_setup.sh`](https://code.jgi.doe.gov/rolypoly/rolypoly/-/raw/main/src/setup/quick_setup.sh) script which will also fetch the pre-generated data rolypoly will require.

```bash
curl -O https://code.jgi.doe.gov/rolypoly/rolypoly/-/raw/main/src/setup/quick_setup.sh && \
bash quick_setup.sh 
```

You can specify custom paths for the code, databases, and conda environment location:
```bash
bash quick_setup.sh /path/to/conda/env /path/to/install/rolypoly_code /path/to/store/databases /path/to/logfile
```
By default if no positional arguments are supplied, rolypoly is installed into the session current folder (path the `quick_setup.sh` is called from):  
- database in `./rolypoly/data/`  
- code in `./rolypoly/code/`  
- conda environment in `./rolypoly/env/`  
- log file in `./RolyPoly_quick_setup.log`   

## Modular / Dev - Command-Specific Pixi Environments

**For software developers** looking to try or make use of specific rolypoly features with minimal risk of dependency conflicts.

```bash
# Install pixi first (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Clone the repository
git clone https://code.jgi.doe.gov/rolypoly/rolypoly.git
cd rolypoly

# Install for specific functionality:
pixi install -e reads-only        # Just read processing tools
pixi install -e assembly-only     # Just assembly tools  
pixi install -e complete          # All tools (full functionality)

# Run commands in the appropriate environment
pixi run -e reads-only rolypoly filter-reads --help
```

### Modular Installation Options

#### 1. Command-Specific Workflows

**Reads Processing Only** - For filtering, masking, and processing reads:
```bash
pixi install -e reads-only
pixi run -e reads-only rolypoly --help
```

**Assembly Only** - For genome/metagenome assembly:
```bash
pixi install -e assembly-only
pixi run -e assembly-only rolypoly --help
```

**RNA Annotation Only** - For RNA structure prediction and ribozyme detection:
```bash
pixi install -e annotation-rna-only
pixi run -e annotation-rna-only rolypoly annotate-rna --help
```

**Protein Annotation Only** - For protein domain annotation:
```bash
pixi install -e annotation-prot-only
pixi run -e annotation-prot-only rolypoly annotate-prot --help
```

**Virus Identification Only** - For viral marker search and virus mapping:
```bash
pixi install -e identify-only
pixi run -e identify-only rolypoly --help
```

#### 2. Combined Workflow Environments

**Basic Analysis** (reads + assembly + identification):
```bash
pixi install -e basic-analysis
pixi run -e basic-analysis rolypoly --help
```

**Full Annotation Workflow**:
```bash
pixi install -e full-annotation
pixi run -e full-annotation rolypoly --help
```

**Complete Installation** (all features):
```bash
pixi install -e complete
pixi run -e complete rolypoly --help
```

### Environment Details

| Environment | Features | Use Case | Disk Usage |
|-------------|----------|----------|------------|
| `default` | Core Python only | Basic testing/minimal install | ~600M |
| `reads-only` | Read processing tools | QC, filtering, masking | ~676M |
| `assembly-only` | Assembly tools | Genome/metagenome assembly | ~532M |
| `annotation-rna-only` | RNA annotation tools | RNA structure prediction | ~700M |
| `annotation-prot-only` | Protein annotation tools | Domain annotation | ~650M |
| `identify-only` | Virus identification tools | Viral marker search | ~650M |
| `basic-analysis` | Reads + Assembly + Identify | Standard virus discovery | ~1.2G |
| `full-annotation` | All except misc tools | Complete annotation pipeline | ~1.5G |
| `complete` | All production tools | Full functionality | ~1.7G |
| `dev` | Everything + dev tools | Development and testing | ~1.9G |

### Commands by Environment

- **reads-only** supports: `rolypoly filter-reads`, `rolypoly shrink-reads`, `rolypoly mask-dna`
- **assembly-only** supports: `rolypoly assemble`, `rolypoly filter-contigs`
- **annotation-rna-only** supports: `rolypoly annotate-rna`
- **annotation-prot-only** supports: `rolypoly annotate-prot`
- **identify-only** supports: `rolypoly marker-search`, `rolypoly virus-mapping`
- **misc-only** supports: `rolypoly fetch-sra`, `rolypoly quick-taxonomy`, `rolypoly get-data`

## Feature-to-Tool Mapping

The modular installation system groups dependencies by functionality:

### reads
- **seqkit** (sequence manipulation)
- **bowtie** (optional alignment)
- **falco** (quality control reports)
- **pigz** (parallel compression)

### assembly
- **spades** (genome assembler)
- **megahit** (metagenome assembler)
- **plass** (protein-level assembler)
- **mmseqs2** (sequence search)

### annotation-rna
- **infernal** (RNA structure search)
- **linearfold** (RNA secondary structure)
- **aragorn** (tRNA/tmRNA detection)
- **trnascan-se** (tRNA detection)

### annotation-prot
- **hmmer** (profile HMM search)
- **mmseqs2** (sequence search)
- **diamond** (fast protein alignment)

### identify
- **mmseqs2** (virus database search)
- **hmmer** (viral marker search)
- **diamond** (protein similarity search)

### misc
- **ncbi-datasets-cli** (NCBI data download)
- **aria2** (fast downloads)
- **taxonkit** (taxonomy tools)
- **freebayes** (variant calling)

## Useful Tips

1. **Check available environments:**
   ```bash
   pixi info
   ```

2. **List installed packages in environment:**
   ```bash
   pixi list -e [environment-name]
   ```

3. **Update specific environment:**
   ```bash
   pixi update -e [environment-name]
   ```

4. **Remove unused environments:**
   ```bash
   pixi clean --environment [environment-name]
   ```

## Development with Pixi

For development work, use the pixi development environment:
```bash
git clone https://code.jgi.doe.gov/rolypoly/rolypoly.git
cd rolypoly
pixi install -e dev
pixi run -e dev rolypoly --help
```

Dependencies
============

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
RolyPoly will try to remind you to cite these too based on the commands you run. For more details, see the [citation_reminder.py](https://code.jgi.doe.gov/rolypoly/rolypoly/-/raw/main/src/rolypoly/utils/logging/citation_reminder.py) script and [all_used_tools_dbs_citations](https://code.jgi.doe.gov/rolypoly/rolypoly/-/raw/main/src/rolypoly/utils/logging/all_used_tools_dbs_citations.json)

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
