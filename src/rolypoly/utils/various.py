import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from rich.console import Console
import polars as pl

from .logging.loggit import get_logger
from logging import Logger
console = Console()


def modify_params(default_params: Dict, override_parameters: Dict) -> Dict:
    """Modify default parameters with user-specified overrides"""
    params = default_params.copy()
    params.update(override_parameters)
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
        # final_path = decompressed_path
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

    if isinstance(mem_str,dict):
        return parse_memory(mem_str.get("bytes"))
    elif isinstance(mem_str,int):
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


def ensure_memory(memory: Union[str, int, dict], file_path: Optional[str] = None) -> Dict[str, str]:
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
                if os.path.isfile(d):
                    os.remove(d)
                elif os.path.isdir(d):
                    shutil.rmtree(d)
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


def is_file_empty(file_path, size_threshold=28):
    if not Path(file_path).exists():
        console.print(f"[bold red]File '{file_path}' does not exist.[/bold red]")
        return True
    file_size = Path(file_path).stat().st_size
    return file_size < size_threshold  # 28b is around the size of an empty <long-name>fastq.gz file


def flat_dict(d: dict[str, str], sep: str = ",", prefix: str = "", suffix: str = "", join_with: str = ": ") -> str:
    """ convert a dict to a string """
    return f"{prefix}{join_with}{sep}".join([f"{k}{join_with}{v}" for k, v in d.items()]) + f"{suffix}"

def flat_list(somelist: list[str], sep: str = ",", prefix: str = "", suffix: str = "", join_with: str = ": ") -> str:
    """ convert a list to a string """
    return f"{prefix}{join_with}{sep}".join(somelist) + f"{suffix}"

def flat_nested(ld: Union[dict, list], sep: str = ",", prefix: str = "", suffix: str = "", join_with: str = ": ") -> str:
    """ convert a list or dict to string """
    if isinstance(ld, dict):
        return flat_dict(ld, sep, prefix, suffix, join_with)
    elif isinstance(ld, list):
        return flat_list(ld, sep, prefix, suffix, join_with)
    else:
        raise Warning(f"Input must be a dictionary or list, got {type(ld)}")

def flat_df_cols(df, sep: str = ",", prefix: str = "", suffix: str = "", join_with: str = ": "):
    """ convert all columns that are pl.List and pl.Struct into string columns"""
    import polars as pl
    return df.with_columns(pl.all().map_elements(lambda x: flat_nested(x), return_dtype=pl.Utf8))


def flatten_struct(
    df: Union[pl.DataFrame, pl.LazyFrame],
    struct_columns: Union[str, List[str]],
    separator: str = ":",
    drop_original_struct: bool = True,
    recursive: bool = False,
    limit: Optional[int] = None,
) -> pl.DataFrame:
    """
    Takes a PolarsFrame and flattens specified struct columns into
    separate columns using a specified separator,
    with options to control recursion and limit the number
    of flattening levels.

    :param df: A PolarsFrame, either a LazyFrame or DataFrame.
    :type df: PolarsFrame
    :param struct_columns: The column or columns in the PolarsFrame that contain struct data.
    This function is designed to flatten the struct data into separate columns based on the fields within the struct.
    :type struct_columns: Union[str, List[str]]
    :param separator: Specifies the character or string that will be used to separate the original
    column name from the nested field names when flattening a nested struct column.
    :type separator: str (optional)
    :param drop_original_struct: Determines whether the original struct columns should be dropped after flattening or not,
    defaults to True.
    :type drop_original_struct: bool (optional)
    :param recursive: Determines whether the flattening process should be applied recursively to
    all levels of nested structures within the specified struct columns, defaults to False.
    :type recursive: bool (optional)
    :param limit: Determines the maximum number of levels to flatten the struct columns.
    If `limit` is set to a positive integer, the function will flatten the struct columns up to that specified level.
    If `limit` is set to `None`, there is no limit.
    :type limit: int
    :return: returns a pl.DataFrame.
    :note: inspired by https://github.com/TomBurdge/harley
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    ldf = df.lazy()
    if isinstance(struct_columns, str):
        struct_columns = [struct_columns]
    if not recursive:
        limit = 1
    if limit is not None and not isinstance(limit, int):
        raise ValueError("limit must be a positive integer or None")
    if limit is not None and limit < 0:
        raise ValueError("limit must be a positive integer or None")
    if limit == 0:
        print("limit of 0 will result in no transformations")
        return df
    ldf = df.lazy()  # noop if df is LazyFrame
    all_column_names = ldf.collect_schema().names()
    if any(separator in (witness := column) for column in all_column_names):
        print(
            f'separator "{separator}" found in column names, e.g. "{witness}". '
            "If columns would be repeated, this function will error"
        )
    non_struct_columns = list(set(ldf.collect_schema().names()) - set(struct_columns))
    struct_schema = ldf.select(*struct_columns).collect_schema()
    col_dtype_expr_names = [(struct_schema[c], pl.col(c), c) for c in struct_columns]
    result_names: Dict[str, pl.Expr] = {}
    level = 0
    while (limit is None and col_dtype_expr_names) or (
        limit is not None and level < limit
    ):
        level += 1
        new_col_dtype_exprs = []
        for dtype, col_expr, name in col_dtype_expr_names:
            if not isinstance(dtype, pl.Struct):
                if name in result_names:
                    raise ValueError(
                        f"Column name {name} would be created at least twice after flatten_struct"
                    )
                result_names[name] = col_expr
                continue
            if any(separator in (witness := field.name) for field in dtype.fields):
                print(
                    f'separator "{separator}" found in field names, e.g. "{witness}" in {name}. '
                    "If columns would be repeated, this function will error"
                )
            new_col_dtype_exprs += [
                (
                    field.dtype,
                    col_expr.struct.field(field.name),
                    name + separator + field.name,
                )
                for field in dtype.fields
            ]
            if not drop_original_struct:
                ldf = ldf.with_columns(
                    col_expr.struct.field(field.name).alias(
                        name + separator + field.name
                    )
                    for field in dtype.fields
                )
        col_dtype_expr_names = new_col_dtype_exprs
    if drop_original_struct and level == limit and col_dtype_expr_names:
        for _, col_expr, name in col_dtype_expr_names:
            result_names[name] = col_expr
    if any((witness := column) in non_struct_columns for column in result_names):
        raise ValueError(
            f"Column name {witness} would be created after flatten_struct, but it's already a non-struct column"
        )
    if drop_original_struct:
        ldf = ldf.select(
            [pl.col(c) for c in non_struct_columns]
            + [col_expr.alias(name) for name, col_expr in result_names.items()]
        )

    return ldf.collect()

def flatten_all_structs(df: Union[pl.DataFrame, pl.LazyFrame], separator: str = ",", drop_original_struct: bool = True, recursive: bool = True, limit: Optional[int] = None) -> pl.DataFrame:
    """Flatten all struct columns in a dataframe"""
    struct_cols = [col for col, dtype in df.schema.items() if dtype == pl.Struct]
    return flatten_struct(df, struct_cols, separator=separator, drop_original_struct=drop_original_struct, recursive=recursive, limit=limit)

def convert_nested_cols(df: Union[pl.DataFrame, pl.LazyFrame], separator: str = ",", drop_original_struct: bool = True, recursive: bool = True, limit: Optional[int] = None) -> pl.DataFrame:
    """Converts nested columns  by the dtype. Structs are flattened, while lists, arrays and objects are converted to strings."""
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    list_cols = [col for col, dtype in df.schema.items() if dtype == pl.List]
    # print(f"list_cols: {list_cols}")
    array_cols = [col for col, dtype in df.schema.items() if dtype == pl.Array]
    # print(f"array_cols: {array_cols}")
    object_cols = [col for col, dtype in df.schema.items() if dtype == pl.Object]
    # print(f"object_cols: {object_cols}")
    struct_cols = [col for col, dtype in df.schema.items() if dtype == pl.Struct]
    # print(f"struct_cols: {struct_cols}")
    for col in struct_cols:
        df = flatten_struct(df, col, separator=separator, drop_original_struct=drop_original_struct, recursive=recursive, limit=limit)
    for col in set(list_cols + array_cols + object_cols):
        # Convert list elements to strings and join them
        df = df.with_columns(
            pl.col(col)
            .list.eval(pl.element().cast(pl.Utf8, strict=False))
            .list.join(separator)
            .alias(col)
        )
    return df


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


def find_files_by_extension(
    input_path: Union[str, Path],
    extensions: List[str],
    file_type: str = "files",
    logger: Optional[Logger] = None
) -> List[Path]:
    """Find all files matching specified extensions in a directory or return single file.
    
    This is for convenience, not sure if actually reduces nmber oflines.
    
    Args:
        input_path: Path to directory or file
        extensions: List of glob patterns to look for (e.g., ["*.fa", "*.fasta"])
        file_type: Human-readable description of file type for logging
        logger: Logger instance
        
    Returns:
        List of matching file paths
    """
    logger = get_logger(logger)
    input_path = Path(input_path)
    
    found_files = []
    
    if input_path.is_file():
        # Check if single file matches any extension
        for ext in extensions:
            if input_path.match(ext):
                found_files = [input_path]
                break
        if not found_files:
            logger.warning(f"Single file {input_path} doesn't match expected {file_type} extensions: {extensions}")
    elif input_path.is_dir():
        for ext in extensions:
            found_files.extend(input_path.glob(ext))
        found_files = sorted(set(found_files))  # Remove duplicates and sort
    else:
        logger.warning(f"Input path does not exist: {input_path}")
        
    logger.debug(f"Found {len(found_files)} {file_type} in {input_path}")
    return found_files

def is_gzipped(file_path: Union[str, Path]) -> bool:
    """Check if a file is gzip compressed.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if file is gzip compressed, False otherwise
    """
    try:
        with open(file_path, "rb") as test_f:
            return test_f.read(2).startswith(b"\x1f\x8b")
    except (OSError, IOError):
        return False

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


def read_fwf(filename, widths, columns, dtypes, comment_prefix=None, **kwargs):
    """Read a fixed-width formatted text file into a Polars DataFrame. wrapper around polars.read_csv

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


def order_columns_to_match(df1_to_order, df2_to_match):
    """Order columns of df1 to match df2"""
    return df1_to_order[df2_to_match.columns]


def cast_cols_to_match(df1_to_cast, df2_to_match):
    """Cast columns of one DataFrame to match the data types of another DataFrame."""
    import polars as pl

    for col in df2_to_match.columns:
        if col in df1_to_cast.columns:
            target_type = df2_to_match.schema[col]
            source_type = df1_to_cast.schema[col]
            
            # Skip casting if target type is null or if types are already compatible
            if target_type == pl.Null or source_type == target_type:
                continue
                
            # Skip casting if source has null and target is string
            if source_type == pl.Null and target_type in [pl.Utf8, pl.String]:
                continue
                
            try:
                df1_to_cast = df1_to_cast.with_columns(
                    pl.col(col).cast(target_type)
                )
            except pl.exceptions.InvalidOperationError:
                # If casting fails, keep the original column
                continue
    return df1_to_cast


def vstack_easy(df1_to_stack, df2_to_stack):
    """Stack two DataFrames vertically after matching their column types and order."""
    # import polars as pl
    
    # # Get common columns between both DataFrames
    # common_columns = [col for col in df1_to_stack.columns if col in df2_to_stack.columns]
    
    # if not common_columns:
    #     # If no common columns, return the first DataFrame as is
    #     return df1_to_stack
    
    # # Filter both DataFrames to only common columns
    # df1_filtered = df1_to_stack.select(common_columns)
    # df2_filtered = df2_to_stack.select(common_columns)
    
    # # Cast columns to match types
    # df2_filtered = cast_cols_to_match(df2_filtered, df1_filtered)
    
    # return df1_filtered.vstack(df2_filtered)
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
    return_final_cmd: bool = False,
) -> Union[bool, str]:
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
        str: The final command that was run (if return_final_cmd is True)
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
    if return_final_cmd:
        return cmd_str
    else:
        return True
