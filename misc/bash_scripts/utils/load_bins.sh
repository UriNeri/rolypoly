#!/usr/bin/bash

# Written by Uri Neri
# Last modified 13.06.2024 ---- WIP
# Contact: 
# Description: Load pre-downloaded binaries and code needed to run rolypoly stuff.

# source /REDACTED_HPC_PATH/rolypoly/rolypoly/bash_functions.sh  
# depen_dir=/REDACTED_HPC_PATH/rolypoly/bin/ 


# rolypoly_dir=/REDACTED_HPC_PATH/rolypoly/
# rolypoly_dir=$1
echo "Adding the good stuff of $rolypoly_dir to your PATH"

source ${rolypoly_dir}/rolypoly/bash_scripts/utils/bash_functions.sh
### Exports (or otherways to make the binaries available - alias?) ###
export PATH=${rolypoly_dir}/bin/:$PATH
export PATH=${rolypoly_dir}/bin/bbmap/:$PATH
export PATH=${rolypoly_dir}/bin/FastQC/:$PATH
export PATH=${rolypoly_dir}/bin/mmseqs/bin/:$PATH
export PATH=${rolypoly_dir}/bin/SPAdes-4.0.0-Linux/bin/:$PATH
export PATH=${rolypoly_dir}/bin/ncbi-blast-2.15.0+/bin/:$PATH
export PATH=${rolypoly_dir}/bin/aws/dist/:$PATH
export PATH=${rolypoly_dir}/bin/htslib-1.20/:$PATH
export PATH=${rolypoly_dir}/rolypoly/utils/:$PATH
export PATH=${rolypoly_dir}/rolypoly/in_reads/:$PATH
export PATH=${rolypoly_dir}/rolypoly/prepare_data/:$PATH
export PATH=${rolypoly_dir}/rolypoly/annotate/:$PATH
export PATH=${rolypoly_dir}/bin/MEGAHIT-1.2.9-Linux-x86_64-static/bin/:$PATH
export PATH=${rolypoly_dir}/bin/plass/bin/:$PATH
export PATH=${rolypoly_dir}/bin/bowtie2-2.5.4-linux-x86_64/:$PATH
export PATH=${rolypoly_dir}/bin/Flye/bin/:$PATH
export PATH=${rolypoly_dir}/bin/gfatools/:$PATH
export PATH=${rolypoly_dir}/bin/mason/bin/:$PATH



# export PATH=${rolypoly_dir}/bin/parallel-20240522/src/:$PATH # the exact timestamp might trip things in the future    



