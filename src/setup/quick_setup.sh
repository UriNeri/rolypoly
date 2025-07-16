#! /bin/bash

# Show help message
show_help() {
    cat << EOF
RolyPoly Quick Setup Script

This script automatically installs RolyPoly with all dependencies in an organized directory structure.

USAGE:
    bash quick_setup.sh [MAMBA_ENV_PATH] [CODE_PATH] [DATA_PATH] [LOGFILE] [DEV_INSTALL]
    bash quick_setup.sh -h|--help|help

ARGUMENTS:
    MAMBA_ENV_PATH    Path for conda/mamba environment (default: ./rolypoly/env)
    CODE_PATH      Path for RolyPoly source code (default: ./rolypoly/code)
    DATA_PATH         Path for external data files (default: ./rolypoly/data)
    LOGFILE          Path for installation log file (default: ./rolypoly/RolyPoly_quick_setup.log)
    DEV_INSTALL      Set to "TRUE" for development installation (default: FALSE)

DEFAULT DIRECTORY STRUCTURE:
    ./rolypoly/
    ├── code/                        # RolyPoly source code
    ├── data/                        # External databases and reference files
    ├── env/                         # Conda/mamba environment
    └── RolyPoly_quick_setup.log     # Installation log

EXAMPLES:
    # Install with default paths
    bash quick_setup.sh

    # Install in custom locations
    bash quick_setup.sh /opt/envs/rp /opt/rolypoly /data/rp_data /tmp/setup.log

    # Development installation
    bash quick_setup.sh "" "" "" "" TRUE

REQUIREMENTS:
    - wget or curl
    - Internet connection
    - ~10GB free disk space

For more information, visit: https://code.jgi.doe.gov/rolypoly/rolypoly
EOF
}

# Check for help flags
if [[ "$1" == "-h" || "$1" == "--help" || "$1" == "help" ]]; then
    show_help
    exit 0
fi

## Functions
# Prints (echo) something (first arg) and also saves it to a log file (second arg) 
logit () {
   echo "$(date +"%Y-%m-%d %T") $2" | tee -a $1
}

# return absolute path even if it doesn't exist
get_absolute_path() {
    local path="$1"
    # If path is already absolute, return it
    if [[ "$path" = /* ]]; then
        echo "$path"
    else
        # Convert relative path to absolute
        echo "$(pwd)/$path"
    fi
}

# Create directory and its parents if they don't exist
ensure_directory() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        logit "$LOGFILE" "Creating directory: $dir"
        mkdir -p "$dir"
    fi
}

## Default paths and convert to absolute paths
MAMBA_ENV_PATH=$(get_absolute_path "${1:-./rolypoly/env}")
CODE_PATH=$(get_absolute_path "${2:-./rolypoly/code}")
DATA_PATH=$(get_absolute_path "${3:-./rolypoly/data}")
LOGFILE=$(get_absolute_path "${4:-./rolypoly/RolyPoly_quick_setup.log}")
DEV_INSTALL="${5:-FALSE}"

# Print paths being used    
logit "$LOGFILE" "Installing RolyPoly with the following paths:"
logit "$LOGFILE" "  MAMBA environment: $MAMBA_ENV_PATH"
logit "$LOGFILE" "  Code directory: $CODE_PATH"
logit "$LOGFILE" "  Data directory: $DATA_PATH"
logit "$LOGFILE" "  Logfile: $LOGFILE"
logit "$LOGFILE" "  Development installation type: $DEV_INSTALL"

# Create all required directories
ensure_directory "$(dirname "$MAMBA_ENV_PATH")"
ensure_directory "$(dirname "$CODE_PATH")"
ensure_directory "$DATA_PATH"

# Detect and set up mamba/micromamba
mamba_command=""
if command -v mamba &> /dev/null; then
    logit "$LOGFILE" "mamba is already installed."
    mamba_command="mamba"
elif command -v micromamba &> /dev/null; then
    logit "$LOGFILE" "micromamba is already installed."
    mamba_command="micromamba"
else
    # Install micromamba
    logit "$LOGFILE" "Neither mamba nor micromamba found. Installing micromamba..."
    if ! command -v wget &> /dev/null; then
        logit "$LOGFILE" "wget is not installed. Please install wget and try again."
        exit 1
    fi

    # Determine download URL based on OS
    case "$OSTYPE" in
        linux-gnu*) 
            url="https://micromamba.snakepit.net/api/micromamba/linux-64/latest" ;;
        darwin*) 
            url="https://micromamba.snakepit.net/api/micromamba/osx-64/latest" ;;
        msys) 
            url="https://micromamba.snakepit.net/api/micromamba/win-64/latest" ;;
        *) 
            logit "$LOGFILE" "Unsupported OS: $OSTYPE"
            exit 1 ;;
    esac

    wget "$url" -O micromamba.tar.bz2 --no-verbose
    tar -xvjf micromamba.tar.bz2
    chmod +x bin/micromamba
    export PATH="$PWD/bin:$PATH"
    ./bin/micromamba shell init -s bash -p ~/micromamba
    source ~/.bashrc
    logit "$LOGFILE" "micromamba installed successfully."
    mamba_command="micromamba"
fi

# Initialize micromamba if needed
if [ "$mamba_command" = "micromamba" ]; then
    if [ -f "./bin/micromamba" ]; then
        eval "$(./bin/micromamba shell hook --shell bash)"
    else
        eval "$(micromamba shell hook --shell bash)"
    fi
fi

# Get RolyPoly code 
if command -v git &> /dev/null; then
    logit "$LOGFILE" "git is installed - cloning repository"
    git clone https://code.jgi.doe.gov/rolypoly/rolypoly.git "$CODE_PATH"
else
    logit "$LOGFILE" "git not found - downloading archive"
    mkdir -p "$CODE_PATH"
    cd "$CODE_PATH" || exit
    curl -LJO https://code.jgi.doe.gov/rolypoly/rolypoly/-/archive/main/rolypoly-main.tar --silent
    tar -xvf rolypoly-main.tar
    mv rolypoly-main/* .
    rm -rf rolypoly-main rolypoly-main.tar
fi

cd "$CODE_PATH" || exit

# Create and activate conda environment
logit "$LOGFILE" "Creating conda environment using $mamba_command"
"$mamba_command" env create -y -p "$MAMBA_ENV_PATH" -f ./src/setup/env_big.yaml --quiet --yes

# Activate environment
source "$(dirname "$(dirname "$MAMBA_ENV_PATH")")/etc/profile.d/conda.sh"
"$mamba_command" activate "$MAMBA_ENV_PATH"

# Install RolyPoly
if [ "$DEV_INSTALL" != "TRUE" ]; then
    logit "$LOGFILE" "Installing rolypoly-tk from PyPI"
    pip install rolypoly-tk --quiet
else
    logit "$LOGFILE" "Installing RolyPoly in development mode"
    pip install -e . --quiet
fi

# Prepare external data
logit "$LOGFILE" "Preparing external data"
export ROLYPOLY_DATA="$DATA_PATH"
rolypoly prepare-data --ROLYPOLY_DATA "$DATA_PATH" --log-file "$LOGFILE"

# Final setup and version check
"$mamba_command" activate "$MAMBA_ENV_PATH"

if [ "$DEV_INSTALL" != "TRUE" ]; then
    # Try uv first, fall back to pip if not available
    if command -v uv &> /dev/null; then
        uv pip show rolypoly-tk | grep Version -m1 -B1 | tee -a "$LOGFILE"
    else
        pip show rolypoly-tk | grep Version -m1 -B1 | tee -a "$LOGFILE"
    fi
else
    rolypoly --version | tee -a "$LOGFILE"
fi

logit "$LOGFILE" "RolyPoly installation complete!"
logit "$LOGFILE" "To start using RolyPoly:"
logit "$LOGFILE" "1. Activate the environment: $mamba_command activate $MAMBA_ENV_PATH"
logit "$LOGFILE" "2. Run RolyPoly: rolypoly --help"
