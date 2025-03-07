Installation
============

## Installation

There are several ways to install RolyPoly:

### Option 1: Quick Setup (Recommended)
The fastest way to get started. Downloads and installs everything automatically, including mamba if needed.

<details><summary>Click to show quick setup command</summary>

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
Install RolyPoly but specify where to store the external databases.

<details><summary>Click to show manual installation steps</summary>

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
Create smaller conda environments, each containing dependencies for specific commands.

<details><summary>Click to show minimal installation steps</summary>

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
