conda create -n "rolypoly" python=3.10 ipython --channel conda-forge --channel bioconda
conda activate rolypoly
conda config --add channels conda-forge
conda config --add channels bioconda

conda install multiqc bzip2
conda install -c bioconda pyhmmer
bash download_dependencies.sh  $CONDA_PREFIX/bin/
