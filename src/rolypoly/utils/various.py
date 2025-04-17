import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, List, Optional, Union

from rich.console import Console

console = Console()


def modify_params(default_params: Dict, override_params: Dict) -> Dict:
    """Modify default parameters with user-specified overrides"""
    params = default_params.copy()
    params.update(override_params)
    return params


def extract(
    archive_path: Union[str, Path], extract_to: Optional[Union[str, Path]] = None
) -> None:
    """Extract compressed and/or archived files"""
    import bz2
    import gzip
    import lzma
    import shutil
    import subprocess
    import tarfile
    import zipfile

    archive_path = Path(archive_path)
    if not archive_path.is_file():
        console.print(f"[bold red]'{archive_path}' is not a valid file![/bold red]")
        return

    extract_to = Path(extract_to) if extract_to else archive_path.parent
    extract_to.mkdir(parents=True, exist_ok=True)

    try:
        # First handle compression (if any)
        decompressed_path = archive_path
        is_compressed = False

        # Check for compression type
        if archive_path.suffix in [".bz2", ".gz", ".xz", ".Z"]:
            is_compressed = True
            compression_type = archive_path.suffix[1:]  # Remove the dot
            decompressed_path = extract_to / archive_path.stem

            if compression_type == "Z":
                subprocess.run(
                    ["uncompress", "-c", str(archive_path)],
                    stdout=open(decompressed_path, "wb"),
                    check=True,
                )
            else:
                open_func = {"bz2": bz2.open, "gz": gzip.open, "xz": lzma.open}[
                    compression_type
                ]

                with (
                    open_func(archive_path, "rb") as source,
                    open(decompressed_path, "wb") as dest,
                ):
                    shutil.copyfileobj(source, dest)

        # Then handle archive format (if any)
        final_path = decompressed_path
        if decompressed_path.suffix == ".tar" or (
            not is_compressed and archive_path.suffix == ".tar"
        ):
            with tarfile.open(decompressed_path, "r:*") as tar:
                tar.extractall(path=extract_to)
            if is_compressed:
                decompressed_path.unlink()  # Remove intermediate decompressed file
        elif not is_compressed and archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)

        console.print(
            f"[green]Successfully processed '{archive_path}' to '{extract_to}'[/green]"
        )

    except Exception as e:
        console.print(
            f"[bold red]Error processing '{archive_path}': {str(e)}[/bold red]"
        )
        if is_compressed and decompressed_path.exists():
            decompressed_path.unlink()  # Cleanup on error


def fetch_and_extract(
    url: str, fetched_to: str = "downloaded_file", extract_to: Optional[str] = None
) -> None:
    """Fetch a file from a URL and optionally extract it"""
    import shutil

    import requests

    console.print(f"Fetching {url}")
    response = requests.get(url, stream=True)
    with open(fetched_to, "wb") as file:
        shutil.copyfileobj(response.raw, file)

    if extract_to:
        extract(fetched_to, extract_to)


def parse_memory(mem_str) -> int:
    """Convert a memory string with units to bytes"""
    import re

    units = {
        "b": 1,
        "kb": 1024,
        "mb": 1024**2,
        "gb": 1024**3,
        "tb": 1024**4,
        "": 1,
        "k": 1024,
        "m": 1024**2,
        "g": 1024**3,
        "t": 1024**4,
    }

    if type(mem_str) == dict:
        return parse_memory(mem_str.get("bytes"))
    elif type(mem_str) == int:
        return mem_str

    mem_str = mem_str.lower().strip()
    match = re.match(r"(\d+(?:\.\d+)?)([kmgt]?b?)", mem_str)

    if not match:
        raise ValueError(f"Invalid memory format: {mem_str}")

    value, unit = match.groups()
    return int(float(value) * units[unit])


def convert_bytes_to_units(byte_size: int) -> Dict[str, str]:
    """Convert bytes to various units"""
    return {
        "bytes": f"{byte_size}b",
        "kilobytes": f"{byte_size / 1024:.2f}kb",
        "megabytes": f"{byte_size / 1024**2:.2f}mb",
        "gigabytes": f"{byte_size / 1024**3:.2f}gb",
        "kilo": f"{byte_size / 1024:.0f}k",
        "mega": f"{byte_size / 1024**2:.0f}m",
        "giga": f"{byte_size / 1024**3:.0f}g",
    }


def ensure_memory(memory: str, file_path: Optional[str] = None) -> Dict[str, str]:
    """Check if requested memory is available and appropriate"""
    import psutil

    requested_memory_bytes = parse_memory(memory)
    available_memory_bytes = psutil.virtual_memory().total

    if requested_memory_bytes > available_memory_bytes:
        console.print(
            f"[yellow]Warning: Requested memory ({memory}) exceeds available system memory ({convert_bytes_to_units(available_memory_bytes)['giga']}).[/yellow]"
        )

    if file_path and Path(file_path).is_file():
        file_size_bytes = Path(file_path).stat().st_size
        if requested_memory_bytes <= file_size_bytes:
            console.print(
                f"[yellow]Warning: Requested memory ({memory}) is less than or equal to the file size ({convert_bytes_to_units(file_size_bytes)['giga']}).[/yellow]"
            )

    return convert_bytes_to_units(requested_memory_bytes)


def create_bash_script(command: List[str], script_name: str) -> None:
    """Create a bash script with the given command"""
    with open(script_name, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"{' '.join(command)}\n")
    os.chmod(script_name, 0o755)


def run_bash_script_with_time(script_name: str) -> Dict[str, str]:
    import subprocess

    time_command = [
        "/usr/bin/time",
        "-v",
        "-o",
        f"{script_name}.time",
        "bash",
        script_name,
    ]
    process = subprocess.Popen(time_command)
    process.wait()
    time_info = {}
    with open(f"{script_name}.time", "r") as f:
        for line in f:
            if ":" in line:
                key, value = line.split(":", 1)
                time_info[key.strip()] = value.strip()
    return time_info


def extract_zip(zip_file):
    import zipfile

    try:
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(zip_file))
        return True
    except Exception as e:
        print(f"Error extracting {zip_file}: {e}")
        return False


def parse_filter(filter_str):
    """Convert a filter string into parsed conditions and operators.

    Takes a filter string in the format "[column operator value & column operator value]"
    and parses it into a list of conditions and logical operators.

    Args:
        filter_str (str): Filter string to parse. Format examples:
            - "[qlen >= 100 & alnlen < 50]"
            - "alnlen >= 120 & pident >= 75"
            - "length > 1000 | width < 50"

    Returns:
        tuple: A tuple containing:
            - list of tuples: [(column, operator, value),     ] where:
                - column (str): Column name to filter on
                - operator (str): Comparison operator (>=, <=, >, <, ==, !=)
                - value (int/float): Numeric value to compare against
            - list of str: List of logical operators ('&' or '|') connecting conditions

    Raises:
        ValueError: If any condition in the filter string is invalid

    Examples:
             parse_filter("[qlen >= 100 & alnlen < 50]")
        (
            [('qlen', '>=', 100), ('alnlen', '<', 50)],
            ['&']
        )
    """
    import re

    # Remove any surrounding brackets
    filter_str = filter_str.strip("[]")
    # Add space around operators and after each variable/condition name
    # filter_str="alnlen >= 120 & pident>=75"
    modified_str = re.sub(
        r"([><=!]=|[><])", r" \1 ", filter_str
    )  # Space around comparison operators
    modified_str = re.sub(
        r"([&|])", r" \1 ", modified_str
    )  # Space around logical operators
    modified_str = re.sub(
        r"  ", r" ", modified_str
    )  # Remove duplicated spaces - TODO: this but smartly.

    # Split the string into individual conditions
    conditions = re.split(r"\s+(\&|\|)\s+", modified_str)
    parsed_conditions = []
    operators = []

    for i, condition in enumerate(conditions):
        if condition.lower() in ["&", "|"]:
            operators.append(condition.lower())
        else:
            # Split the condition into column, operator, and value
            match = re.match(r"(\w+)\s*([<>=!]+)\s*([\d.]+)", condition.strip())
            if match:
                col, op, val = match.groups()
                # Convert value to appropriate type
                val = float(val) if "." in val else int(val)
                parsed_conditions.append((col, op, val))
            else:
                raise ValueError(f"Invalid condition: {condition}")

    return parsed_conditions, operators


def apply_filter(df, filter_str):
    import polars as pl

    conditions, operators = parse_filter(filter_str)
    if not conditions:
        return df

    expr = None
    for i, (col, op, val) in enumerate(conditions):
        condition = None
        if op == ">=":
            condition = pl.col(col) >= val
        elif op == "<=":
            condition = pl.col(col) <= val
        elif op == ">":
            condition = pl.col(col) > val
        elif op == "<":
            condition = pl.col(col) < val
        elif op == "==":
            condition = pl.col(col) == val
        elif op == "!=":
            condition = pl.col(col) != val

        if expr is None:
            expr = condition
        elif i - 1 < len(operators):
            if operators[i - 1] == "&":
                expr = expr & condition
            elif operators[i - 1] == "|":
                expr = expr | condition

    return df.filter(expr)


def find_most_recent_folder(path):
    import glob
    import os

    # Get a list of all directories in the specified path
    folders = [f for f in glob.glob(os.path.join(path, "*")) if os.path.isdir(f)]
    # Return None if no folders found
    if not folders:
        return None
    # Find the most recent folder based on modification time
    most_recent_folder = max(folders, key=os.path.getmtime)
    return most_recent_folder


def move_contents_to_parent(folder, overwrite=True):
    import shutil

    parent_dir = os.path.dirname(folder)
    for item in os.listdir(folder):
        s = os.path.join(folder, item)
        d = os.path.join(parent_dir, item)
        if overwrite:
            if os.path.exists(d):
                os.remove(d)
            shutil.move(s, d)
        else:
            if not os.path.exists(d):
                shutil.move(s, d)
            else:
                console.print(
                    f"[bold red]File {d} already exists! Skipping    [/bold red]"
                )
    #  remove the now empty folder
    os.rmdir(folder)  # only works on empty dir


def check_file_exists(file_path):
    if not Path(file_path).exists():
        console.print(f"[bold red]File not found: {file_path} Tüdelü![/bold red]")
        raise FileNotFoundError(f"File not found: {file_path}")


def check_file_size(file_path):
    file_size = Path(file_path).stat().st_size
    if file_size == 0:
        console.print(f"[yellow]File '{file_path}' is empty[/yellow]")
    else:
        console.print(f"File '{file_path}' size is {file_size}")


def is_file_empty(file_path):
    if not Path(file_path).exists():
        console.print(f"[bold red]File '{file_path}' does not exist.[/bold red]")
        return True
    file_size = Path(file_path).stat().st_size
    return file_size < 28  # 28b is around the size of an empty <long-name>fastq.gz file


def run_command(
    cmd, logger, to_check, skip_existing=False, check=True
):  # TODO: add an option "try-hard" that save hash of the input /+ code.
    """Run a command and log its output"""
    import subprocess

    if skip_existing == True:
        if Path(to_check).exists():
            if Path(to_check).stat().st_size > 28:
                logger.info(
                    f"{to_check} seems to exist and isn't empty, and --skip-existing flag was set     so skipppingggg yolo! "
                )
                return True

    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=check)
    except subprocess.CalledProcessError as e:
        logger.warning(f"Error : {e}")
        return False

    return check_file_exist_isempty(f"{to_check}")


def check_file_exist_isempty(file_path):
    check_file_exists(file_path)
    if is_file_empty(file_path):
        console.print(f"[yellow]File {file_path} exists, but is empty.[/yellow]")
        return False
        # console.print("This might mean all reads were filtered. Exiting without proceeding to downstream steps.")
        # raise ValueError(f"File {file_path} is empty")
    else:
        console.print(
            f"[green]File '{file_path}' size is {Path(file_path).stat().st_size} bytes (not empty). [/green]"
        )
        return True


def create_output_dataframe():
    # for output tracking
    import polars as pl

    return pl.DataFrame(
        schema={
            "filename": pl.Utf8,
            "absolute_path": pl.Utf8,
            "command_name": pl.Utf8,
            "cmd": pl.Utf8,
            "file_type": pl.Utf8,
            "file_size": pl.UInt32,
        }
    )


def add_output_file(df, filename, command_name, command, file_type):
    """Add a new output file record to the tracking DataFrame.
    Args:
        filename (str): Name of the output file
        command_name (str): Name of the command that created the file
        command (str): Full command used to create the file
        file_type (str): Type of file (e.g., "fasta", "fastq")
    """
    import polars as pl

    absolute_path = os.path.abspath(filename)
    file_size = os.path.getsize(absolute_path)

    new_row = pl.DataFrame(
        {
            "filename": [filename],
            "absolute_path": [absolute_path],
            "command_name": [command_name],
            "command": [command],
            "file_type": [file_type],
            "file_size": [file_size],
        }
    )
    return pl.concat([df, new_row])


def read_fwf(filename, widths, columns, dtypes, comment_prefix=None, **kwargs):
    """Read a fixed-width formatted text file into a Polars DataFrame.

    Args:
        filename (str): Path to the fixed-width file
        widths (list): List of tuples (start, length) for each column
        columns (list): List of column names
        dtypes (list): List of Polars data types for each column
        comment_prefix (str, optional): Character(s) indicating comment lines
        **kwargs: Additional arguments passed to polars.read_csv

    Returns:
        polars.DataFrame: DataFrame containing the parsed data

    """
    import polars as pl

    # if widths is None:
    #     # infer widths from the file
    #     peek = pl.scan_csv(filename, separator="\n", has_header=False)
    #     widths = [len(peek.head(1).to_series()[0])]
    # if columns is None:
    #     columns = ["column1"]
    # if dtypes is None:
    #     dtypes = [pl.Utf8]
    column_information = [(*x, y, z) for x, y, z in zip(widths, columns, dtypes)]

    return pl.read_csv(
        filename,
        separator="\n",
        new_columns=["header"],
        has_header=False,
        comment_prefix=comment_prefix,
        **kwargs,
    ).select(
        pl.col("header")
        .str.slice(col_offset, col_len)
        .str.strip_chars(characters=" ")
        .cast(col_type)
        .alias(col_name)
        for col_offset, col_len, col_name, col_type in (column_information)
    )


def get_file_type(filename: str) -> str:
    """Determine the type of a file based on its extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".gz":
        ext = os.path.splitext(filename[:-3])[1].lower() + ".gz"

    file_types = {
        ".fq": "fastq",
        ".fastq": "fastq",
        ".fq.gz": "fastq_gzipped",
        ".fastq.gz": "fastq_gzipped",
        ".fa": "fasta",
        ".fasta": "fasta",
        ".fa.gz": "fasta_gzipped",
        ".fasta.gz": "fasta_gzipped",
        ".txt": "text",
        ".txt.gz": "text_gzipped",
    }

    return file_types.get(ext, "unknown")


def update_output_files(df, new_filename, command_name, command):
    """Update output tracking DataFrame with a new file."""
    file_type = get_file_type(new_filename)
    return add_output_file(df, new_filename, command_name, command, file_type)


def get_latest_output(df, file_type=None):
    """Get the filename of the most recently added output file.

    Args:
        df (polars.DataFrame): Output tracking DataFrame
        file_type (str, optional): Filter by file type.

    Returns:
        str: Filename of the most recent output file

    """
    import polars as pl

    if file_type:
        filtered_df = df.filter(pl.col("file_type") == file_type)
    else:
        filtered_df = df

    return filtered_df.tail(1)["filename"][0]


def order_columns_to_match(df1_to_order, df2_to_match):
    """Order columns of df1 to match df2"""
    return df1_to_order[df2_to_match.columns]


def cast_cols_to_match(df1_to_cast, df2_to_match):
    """Cast columns of one DataFrame to match the data types of another DataFrame."""
    import polars as pl

    for col in df2_to_match.columns:
        df1_to_cast = df1_to_cast.with_columns(
            pl.col(col).cast(df2_to_match.schema[col])
        )
    return df1_to_cast


def vstack_easy(df1_to_stack, df2_to_stack):
    """Stack two DataFrames vertically after matching their column types and order."""
    df2_to_stack = cast_cols_to_match(df2_to_stack, df1_to_stack)
    df2_to_stack = order_columns_to_match(df2_to_stack, df1_to_stack)
    return df1_to_stack.vstack(df2_to_stack)


def run_command_comp(
    base_cmd: str,
    positional_args: list[str] = [],
    positional_args_location: str = "end",
    params: dict = {},
    logger=None,
    output_file: str = "",
    skip_existing: bool = False,
    check_status: bool = True,
    check_output: bool = True,
    prefix_style: str = "auto",
    param_sep: str = " ",
    assign_operator: str = " ",
    resource_monitoring: bool = False,
) -> bool:
    """Run a command with mixed parameter styles, with resource monitoring, and output verification. comp is abbrev for comprehensive,complex,complicated,complicated-ass, compounding.

    Args:
        base_cmd (str): Base command name (e.g., "samtools", "minimap2")
        positional_args (list[str], optional): List of positional arguments
        positional_args_location (str, optional): Where to place positional args ('start' or 'end').
        params (dict, optional): Named parameters and their values
        logger (Logger, optional): Logger for output messages
        output_file (str, optional): Expected output file to verify
        skip_existing (bool, optional): Skip if output exists.
        check_status (bool, optional): Verify command exit status.
        check_output (bool, optional): Verify output file exists.
        prefix_style (str, optional): How to prefix parameters:
            - 'auto': Guess based on length (- or --)
            - 'single': Always use single dash
            - 'double': Always use double dash
            - 'none': No prefix
        param_sep (str, optional): Parameter separator.
        assign_operator (str, optional): Parameter assignment operator.
        resource_monitoring (bool, optional): Monitor CPU and memory usage.

    Returns:
        bool: True if command succeeded and output verification passed
    """
    import subprocess
    import sys
    from logging import INFO, Logger, StreamHandler
    from pathlib import Path
    from time import sleep, time

    from psutil import NoSuchProcess, Process, cpu_percent

    if logger is None:
        logger = Logger(__name__, level=INFO)
        logger.addHandler(StreamHandler(sys.stdout))

    if output_file != "":
        if (
            skip_existing
            and Path(output_file).exists()
            and Path(output_file).stat().st_size > 28
        ):
            logger.info(
                f"{output_file} exists and isn't empty, skipping due to --skip-existing flag"
            )
            return True

    cmd = [base_cmd]
    flag_str = ""
    reg_param_str = ""

    for param, value in params.items():
        if prefix_style == "auto":
            prefix = "-" if len(param) == 1 else "--"
        elif prefix_style == "single":
            prefix = "-"
        elif prefix_style == "double":
            prefix = "--"
        else:  # 'none'
            prefix = ""

        if value is True:
            flag_str += f"{param_sep}{prefix}{param}"
            continue

        reg_param_str += f"{param_sep}{prefix}{param}{assign_operator}{value}"

    cmd.append(reg_param_str)
    cmd.append(flag_str)
    positional_args_str = param_sep.join(positional_args)

    if positional_args_location == "end":
        cmd.extend([positional_args_str])
    elif positional_args_location == "start":
        cmd.insert(1, positional_args_str)
    else:
        raise ValueError(
            f"Invalid positional_args_location: {positional_args_location}"
        )

    cmd_str = " ".join(cmd)

    logger.info(f"Running command: {cmd_str}")
    try:
        if resource_monitoring:
            start_time = time()
            process = subprocess.Popen(cmd_str, shell=True)
            while process.poll() is None:
                try:
                    proc = Process(process.pid)
                    cpu_percent = proc.cpu_percent()
                    memory_info = sum(
                        child.memory_info().rss
                        for child in proc.children(recursive=True)
                    )
                    memory_info += proc.memory_info().rss
                    logger.info(f"Current CPU Usage: {cpu_percent}%")
                    logger.info(f"Current Memory Usage: {memory_info / 1024:.2f} KB")
                    sleep(0.01)
                except NoSuchProcess:
                    break
            end_time = time()
            logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
            process.wait()
            if process.returncode != 0 and check_status:
                raise subprocess.CalledProcessError(process.returncode, cmd_str)
        else:
            subprocess.run(cmd_str, check=check_status, shell=True)
    except subprocess.CalledProcessError as e:
        logger.warning(f"Error: {e}")
        return False
    if check_output:
        return check_file_exist_isempty(output_file)
    else:
        return True
