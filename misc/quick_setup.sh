#! /bin/bash


# Prints (echo) something (first arg) and also saves it to a log file (second arg) 
logit () {
   echo "$(date +"%Y-%m-%d %T") $2" | tee -a $1
}

# Default paths
CONDA_ENV_PATH="${1:-$HOME/micromamba/envs/rolypoly}"
INSTALL_PATH="${2:-$HOME/rolypoly}"
DATA_PATH="${3:-$HOME/rolypoly_data}"
LOGFILE="${4:-$HOME/RolyPoly_quick_setup.log}"

# if the paths parent directories don't exist, create them
if [ ! -d "$(dirname "$CONDA_ENV_PATH")" ]; then
    logit "$LOGFILE" "Conda environment path parent directory does not exist. Creating it now    " 
    mkdir -p "$(dirname "$CONDA_ENV_PATH")"
fi

if [ ! -d "$(dirname "$INSTALL_PATH")" ]; then
    logit "$LOGFILE" "Installation path parent directory does not exist. Creating it now    "
    mkdir -p "$(dirname "$INSTALL_PATH")"
fi

if [ ! -d "$DATA_PATH" ]; then
    logit "$LOGFILE" "Data path does not exist. Creating it now    "
    mkdir -p "$DATA_PATH"
fi

# Print paths being used
logit "$LOGFILE" "Installing RolyPoly with the following paths:"
logit "$LOGFILE" "  Conda environment: $CONDA_ENV_PATH"
logit "$LOGFILE" "  Installation directory: $INSTALL_PATH"
logit "$LOGFILE" "  Data directory: $DATA_PATH"
logit "$LOGFILE" "  Logfile: $LOGFILE"

# Create directories if they don't exist
mkdir -p "$(dirname "$CONDA_ENV_PATH")"
mkdir -p "$(dirname "$INSTALL_PATH")"
mkdir -p "$DATA_PATH"

# Install mamba if needed
mamba_installed=$(command -v micromamba &> /dev/null)
conda_installed=$(command -v conda &> /dev/null)
if [ "$mamba_installed" = false ] && [ "$conda_installed" = false ]; then
    logit "$LOGFILE" "Neither mamba nor conda could be found. Attempting to install mamba    "
    if command -v wget &> /dev/null
    then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            wget https://micromamba.snakepit.net/api/micromamba/linux-64/latest -O micromamba.tar.bz2
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            wget https://micromamba.snakepit.net/api/micromamba/osx-64/latest -O micromamba.tar.bz2
        elif [[ "$OSTYPE" == "msys" ]]; then
            wget https://micromamba.snakepit.net/api/micromamba/win-64/latest -O micromamba.tar.bz2
        else
            logit "$LOGFILE" "Unsupported OS: $OSTYPE"
            exit 1
        fi
        tar -xvjf micromamba.tar.bz2
        ./bin/micromamba shell init -s bash -p ~/micromamba
        source ~/.bashrc
    else
        logit "$LOGFILE" "wget is not installed. Please install wget and try again."
        exit 1
    fi
    logit "$LOGFILE" "Mamba installed successfully."
else
    logit "$LOGFILE" "Mamba is already installed."
fi
# initialize micromamba
eval "$(micromamba shell hook --shell bash)"

# Get RolyPoly code
if ! command -v git &> /dev/null
then
    logit "$LOGFILE" "git could not be found. Fetching the repo from https://code.jgi.doe.gov/UNeri/rolypoly.git"
    mkdir -p "$INSTALL_PATH"
    cd "$INSTALL_PATH" || exit
    curl -LJO https://code.jgi.doe.gov/UNeri/rolypoly/-/archive/main/rolypoly-main.tar
    tar -xvf rolypoly-main.tar
    mv rolypoly-main/* .
    rm -rf rolypoly-main rolypoly-main.tar
else
    logit "$LOGFILE" "git is installed"
    git clone https://code.jgi.doe.gov/UNeri/rolypoly.git "$INSTALL_PATH"
    cd "$INSTALL_PATH" || exit
fi

# Create and activate conda environment
logit "$LOGFILE" "Creating conda environment    "
micromamba create -y -p "$CONDA_ENV_PATH" -f misc/env_files/env_big.yaml
source "$(dirname "$(dirname "$CONDA_ENV_PATH")")/etc/profile.d/conda.sh"
micromamba activate "$CONDA_ENV_PATH"

# Install RolyPoly
logit "$LOGFILE" "Installing RolyPoly    "
pip install -e .  # Use pip install -e .[dev] for development installation

# Prepare external data
logit "$LOGFILE" "Preparing external data    "
export ROLYPOLY_DATA="$DATA_PATH"
rolypoly prepare-external-data --data_dir "$DATA_PATH" --log-file "$LOGFILE"
conda env config vars set ROLYPOLY_DATA="$DATA_PATH"
# setup taxonkit datadir
conda env config vars set TAXONKIT_DB="$DATA_PATH/taxdump"

# logit "$LOGFILE" "\n"
logit "$LOGFILE" "RolyPoly installation complete!"
logit "$LOGFILE" "To start using RolyPoly:"
logit "$LOGFILE" "1. Activate the environment:"
logit "$LOGFILE" "  mamba activate $CONDA_ENV_PATH"
logit "$LOGFILE" "2. Run RolyPoly:"
logit "$LOGFILE" "  rolypoly --help"

micromamba activate "$CONDA_ENV_PATH"
# logit "$LOGFILE" "RolyPoly version:"
rolypoly --version >> "$LOGFILE"
rolypoly --version

