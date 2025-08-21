"""File format detection and analysis functions.

This module provides comprehensive FASTQ file detection, analysis, and classification
functionality with support for paired-end, interleaved, and single-end files.
"""

import gzip
import logging
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


def get_logger(logger: Optional[logging.Logger] = None) -> logging.Logger:
    """Get a logger instance, creating a default one if none provided."""
    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    return logger


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


def create_sample_file(
    file_path: Union[str, Path],
    sample_type: str = "first_n_lines",
    n_lines: int = 1000,
    n_bytes: int = 1024 * 1024,  # 1MB default
    percentage: float = 0.1,
    logger: Optional[logging.Logger] = None
) -> str:
    """Create a temporary sample file from a FASTQ file for analysis.
    
    Args:
        file_path: Path to the input FASTQ file
        sample_type: Type of sampling - "first_n_lines", "last_n_lines", 
                    "first_last_n_lines", "first_n_bytes", "random_percentage"
        n_lines: Number of lines to sample (for line-based sampling)
        n_bytes: Number of bytes to sample (for byte-based sampling)
        percentage: Percentage of reads to sample randomly (0.0-1.0)
        logger: Logger instance
        
    Returns:
        Path to temporary sample file
    """
    logger = get_logger(logger)
    file_path = Path(file_path)
    is_gz = is_gzipped(file_path)
    
    # Create temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.fastq', prefix='sample_')
    
    try:
        if sample_type == "first_n_bytes":
            logger.debug(f"Sampling first {n_bytes} bytes from {file_path}")
            if is_gz:
                with gzip.open(file_path, 'rb') as f_in:
                    data = f_in.read(n_bytes)
                    with os.fdopen(temp_fd, 'wb') as f_out:
                        f_out.write(data)
            else:
                with open(file_path, 'rb') as f_in:
                    data = f_in.read(n_bytes)
                    with os.fdopen(temp_fd, 'wb') as f_out:
                        f_out.write(data)
                        
        elif sample_type in ["first_n_lines", "last_n_lines", "first_last_n_lines"]:
            logger.debug(f"Sampling {n_lines} lines using method {sample_type} from {file_path}")
            
            if sample_type == "first_n_lines":
                # Stream first n lines without loading entire file
                if is_gz:
                    with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f_in:
                        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f_out:
                            for i, line in enumerate(f_in):
                                if i >= n_lines:
                                    break
                                f_out.write(line)
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f_in:
                        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f_out:
                            for i, line in enumerate(f_in):
                                if i >= n_lines:
                                    break
                                f_out.write(line)
            
            elif sample_type == "last_n_lines":
                # For last n lines, we need to use a deque to avoid loading entire file
                from collections import deque
                
                if is_gz:
                    with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f_in:
                        last_lines = deque(f_in, maxlen=n_lines)
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f_in:
                        last_lines = deque(f_in, maxlen=n_lines)
                
                with os.fdopen(temp_fd, 'w', encoding='utf-8') as f_out:
                    f_out.writelines(last_lines)
            
            else:  # first_last_n_lines
                first_half = n_lines // 2
                last_half = n_lines - first_half
                
                # Collect first half
                first_lines = []
                if is_gz:
                    with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f_in:
                        for i, line in enumerate(f_in):
                            if i >= first_half:
                                break
                            first_lines.append(line)
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f_in:
                        for i, line in enumerate(f_in):
                            if i >= first_half:
                                break
                            first_lines.append(line)
                
                # Collect last half using deque
                from collections import deque
                if is_gz:
                    with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f_in:
                        last_lines = deque(f_in, maxlen=last_half)
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f_in:
                        last_lines = deque(f_in, maxlen=last_half)
                
                with os.fdopen(temp_fd, 'w', encoding='utf-8') as f_out:
                    f_out.writelines(first_lines)
                    f_out.writelines(last_lines)
                
        elif sample_type == "random_percentage":
            logger.debug(f"Sampling {percentage*100}% of reads randomly from {file_path}")
            
            # For random sampling, we need a two-pass approach to avoid loading entire file
            # First pass: count total records
            total_records = 0
            if is_gz:
                with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f_in:
                    for line_num, line in enumerate(f_in):
                        if line.startswith('@') and line_num % 4 == 0:  # FASTQ header lines
                            total_records += 1
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f_in:
                    for line_num, line in enumerate(f_in):
                        if line.startswith('@') and line_num % 4 == 0:  # FASTQ header lines
                            total_records += 1
            
            num_sample = int(total_records * percentage)
            
            if num_sample > 0:
                # Generate random record indices to sample
                selected_records = set(random.sample(range(total_records), num_sample))
                
                # Second pass: extract selected records
                current_record = 0
                lines_in_current_record = 0
                current_record_lines = []
                
                if is_gz:
                    file_handle = gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore')
                else:
                    file_handle = open(file_path, 'r', encoding='utf-8', errors='ignore')
                
                with file_handle as f_in:
                    with os.fdopen(temp_fd, 'w', encoding='utf-8') as f_out:
                        for line in f_in:
                            current_record_lines.append(line)
                            lines_in_current_record += 1
                            
                            # Complete FASTQ record (4 lines)
                            if lines_in_current_record == 4:
                                if current_record in selected_records:
                                    f_out.writelines(current_record_lines)
                                
                                # Reset for next record
                                current_record += 1
                                lines_in_current_record = 0
                                current_record_lines = []
            else:
                # If no records to sample, just close the file descriptor
                os.close(temp_fd)
        else:
            raise ValueError(f"Unknown sample_type: {sample_type}")
            
        logger.debug(f"Created sample file: {temp_path}")
        return temp_path
        
    except Exception as e:
        # Clean up on error
        try:
            os.close(temp_fd)
        except:
            pass
        try:
            os.unlink(temp_path)
        except:
            pass
        logger.error(f"Error creating sample file from {file_path}: {e}")
        raise


def analyze_fastq_headers(
    file_path: Union[str, Path],
    sample_lines: int = 2000,  # Increased slightly for better analysis
    logger: Optional[logging.Logger] = None
) -> Dict:
    """Analyze FASTQ headers to determine file characteristics.
    
    Args:
        file_path: Path to FASTQ file or sample file
        sample_lines: Number of lines to analyze
        logger: Logger instance
        
    Returns:
        Dictionary containing header analysis results
    """
    logger = get_logger(logger)
    file_path = Path(file_path)
    
    results = {
        'read_count_analyzed': 0,
        'has_pair_indicators': False,
        'pair_1_count': 0,
        'pair_2_count': 0,
        'interleaved_pattern': False,
        'average_read_length': 0,
        'total_length': 0
    }
    
    try:
        is_gz = is_gzipped(file_path)
        
        if is_gz:
            file_handle = gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore')
        else:
            file_handle = open(file_path, 'r', encoding='utf-8', errors='ignore')
            
        with file_handle as f:
            lines_read = 0
            while lines_read < sample_lines:
                try:
                    header = f.readline().strip()
                    if not header:
                        break
                        
                    if header.startswith('@'):
                        sequence = f.readline().strip()
                        plus_line = f.readline().strip()
                        quality = f.readline().strip()
                        
                        if not all([sequence, plus_line, quality]):
                            break
                            
                        results['read_count_analyzed'] += 1
                        results['total_length'] += len(sequence)
                        
                        # Check for pair indicators
                        if '/1' in header or ' 1:' in header:
                            results['pair_1_count'] += 1
                            results['has_pair_indicators'] = True
                        elif '/2' in header or ' 2:' in header:
                            results['pair_2_count'] += 1
                            results['has_pair_indicators'] = True
                        
                        lines_read += 4
                    else:
                        lines_read += 1
                        
                except Exception as e:
                    logger.debug(f"Error reading line in {file_path}: {e}")
                    break
                    
        # Calculate average read length
        if results['read_count_analyzed'] > 0:
            results['average_read_length'] = results['total_length'] / results['read_count_analyzed']
            
        # Determine if interleaved based on pair indicators
        if results['has_pair_indicators']:
            if results['pair_1_count'] > 0 and results['pair_2_count'] > 0:
                results['interleaved_pattern'] = True
                
        logger.debug(f"Header analysis for {file_path}: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing headers in {file_path}: {e}")
        return results


def determine_fastq_type(
    file_path: Union[str, Path],
    sample_method: str = "first_n_bytes",
    sample_size: int = 1024 * 512,  # 512KB default - much more memory efficient
    logger: Optional[logging.Logger] = None
) -> Dict:
    """Determine FASTQ file type and characteristics.
    
    Args:
        file_path: Path to FASTQ file
        sample_method: Sampling method for analysis
        sample_size: Size of sample to analyze
        logger: Logger instance
        
    Returns:
        Dictionary with file type analysis results
    """
    logger = get_logger(logger)
    file_path = Path(file_path)
    
    logger.debug(f"Analyzing FASTQ file: {file_path}")
    
    results = {
        'file_path': str(file_path),
        'is_gzipped': is_gzipped(file_path),
        'file_size_bytes': file_path.stat().st_size if file_path.exists() else 0,
        'paired_end': False,
        'interleaved': False,
        'single_end': False,
        'header_analysis': {},
        'error': None
    }
    
    try:
        # Create sample file for analysis
        if sample_method == "first_n_bytes":
            sample_file = create_sample_file(
                file_path, sample_method, n_bytes=sample_size, logger=logger
            )
        else:
            sample_file = create_sample_file(
                file_path, sample_method, n_lines=sample_size, logger=logger
            )
        
        try:
            # Analyze headers
            header_analysis = analyze_fastq_headers(sample_file, sample_size, logger)
            results['header_analysis'] = header_analysis
            
            # Determine file type based on analysis
            if header_analysis['has_pair_indicators']:
                if header_analysis['interleaved_pattern']:
                    results['interleaved'] = True
                    results['paired_end'] = True
                    logger.debug(f"Detected interleaved paired-end file: {file_path}")
                else:
                    # Has pair indicators but not interleaved - likely single mate file
                    results['paired_end'] = True
                    results['single_end'] = False
                    logger.debug(f"Detected paired-end (single mate) file: {file_path}")
            else:
                results['single_end'] = True
                logger.debug(f"Detected single-end file: {file_path}")
                
        finally:
            # Clean up sample file
            try:
                os.unlink(sample_file)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Error determining FASTQ type for {file_path}: {e}")
        results['error'] = str(e)
        
    return results


def is_paired_filename(filename: str, logger: Optional[logging.Logger] = None) -> Tuple[bool, str]:
    """Check if filename indicates paired-end data and extract pair info.
    
    Args:
        filename: Name of the file to check
        logger: Logger instance
        
    Returns:
        Tuple of (is_paired, pair_filename)
    """
    logger = get_logger(logger)
    
    patterns = [
        (r"(.*)_R1([._].*)$", r"\1_R2\2"),   # _R1/_R2
        (r"(.*)_1([._].*)$", r"\1_2\2"),     # _1/_2
        (r"(.*)\.1(\.f.*q.*)$", r"\1.2\2"),  # .1.fastq/.2.fastq
    ]
    
    for pattern, replacement in patterns:
        match = re.match(pattern, filename)
        if match:
            pair_file = re.sub(pattern, replacement, filename)
            logger.debug(f"Detected paired filename pattern: {filename} -> {pair_file}")
            return True, pair_file
            
    return False, ""


def find_files_by_extension(
    input_path: Union[str, Path],
    extensions: List[str],
    file_type: str = "files",
    logger: Optional[logging.Logger] = None
) -> List[Path]:
    """Find all files matching specified extensions in a directory or return single file.
    
    This is a generic file finder that can be used for any file type.
    
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


def find_fastq_files(
    input_path: Union[str, Path],
    extensions: List[str] = None,
    logger: Optional[logging.Logger] = None
) -> List[Path]:
    """Find all FASTQ files in a directory or return single file.
    
    Args:
        input_path: Path to directory or file
        extensions: List of extensions to look for
        logger: Logger instance
        
    Returns:
        List of FASTQ file paths
    """
    if extensions is None:
        extensions = ["*.fq", "*.fastq", "*.fq.gz", "*.fastq.gz"]
    
    return find_files_by_extension(input_path, extensions, "FASTQ files", logger)


def find_fasta_files(
    input_path: Union[str, Path],
    extensions: List[str] = None,
    logger: Optional[logging.Logger] = None
) -> List[Path]:
    """Find all FASTA files in a directory or return single file.
    
    Args:
        input_path: Path to directory or file
        extensions: List of extensions to look for
        logger: Logger instance
        
    Returns:
        List of FASTA file paths
    """
    if extensions is None:
        extensions = ["*.fa", "*.fasta", "*.fna", "*.fa.gz", "*.fasta.gz", "*.fna.gz"]
    
    return find_files_by_extension(input_path, extensions, "FASTA files", logger)


def find_hmm_files(
    input_path: Union[str, Path],
    extensions: List[str] = None,
    logger: Optional[logging.Logger] = None
) -> List[Path]:
    """Find all HMM files in a directory or return single file.
    
    Args:
        input_path: Path to directory or file
        extensions: List of extensions to look for
        logger: Logger instance
        
    Returns:
        List of HMM file paths
    """
    if extensions is None:
        extensions = ["*.hmm"]
    
    return find_files_by_extension(input_path, extensions, "HMM files", logger)


def find_msa_files(
    input_path: Union[str, Path],
    extensions: List[str] = None,
    logger: Optional[logging.Logger] = None
) -> List[Path]:
    """Find all Multiple Sequence Alignment files in a directory or return single file.
    
    Args:
        input_path: Path to directory or file
        extensions: List of extensions to look for
        logger: Logger instance
        
    Returns:
        List of MSA file paths
    """
    if extensions is None:
        extensions = ["*.faa", "*.afa", "*.aln", "*.msa"]
    
    return find_files_by_extension(input_path, extensions, "MSA files", logger)


def validate_database_directory(
    database_path: Union[str, Path],
    expected_types: List[str] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Union[str, List[Path]]]:
    """Validate and categorize database directory contents.
    
    This function handles the common pattern of validating custom database directories
    that can contain either HMM files or MSA files that need to be converted to HMMs.
    
    Args:
        database_path: Path to database file or directory
        expected_types: List of expected file types ("hmm", "msa", "fasta")
        logger: Logger instance
        
    Returns:
        Dictionary containing:
        - type: "hmm_file", "hmm_directory", "msa_file", "msa_directory", "mixed", "invalid"
        - files: List of relevant files found
        - message: Human-readable description
    """
    logger = get_logger(logger)
    database_path = Path(database_path)
    
    if expected_types is None:
        expected_types = ["hmm", "msa"]
    
    result = {
        "type": "invalid",
        "files": [],
        "message": ""
    }
    
    if not database_path.exists():
        result["message"] = f"Database path {database_path} does not exist"
        return result
    
    if database_path.is_file():
        # Single file - determine type
        if database_path.suffix == ".hmm":
            result["type"] = "hmm_file"
            result["files"] = [database_path]
            result["message"] = f"Single HMM file: {database_path.name}"
        elif database_path.suffix in [".faa", ".afa", ".aln", ".msa"]:
            result["type"] = "msa_file"
            result["files"] = [database_path]
            result["message"] = f"Single MSA file: {database_path.name}"
        else:
            result["message"] = f"Unsupported file type: {database_path.suffix}"
        
        return result
    
    elif database_path.is_dir():
        # Directory - analyze contents
        hmm_files = find_hmm_files(database_path, logger=logger)
        msa_files = find_msa_files(database_path, logger=logger)
        
        if hmm_files and not msa_files:
            result["type"] = "hmm_directory"
            result["files"] = hmm_files
            result["message"] = f"Directory with {len(hmm_files)} HMM files"
        elif msa_files and not hmm_files:
            result["type"] = "msa_directory"
            result["files"] = msa_files
            result["message"] = f"Directory with {len(msa_files)} MSA files"
        elif hmm_files and msa_files:
            result["type"] = "mixed"
            result["files"] = hmm_files + msa_files
            result["message"] = f"Mixed directory: {len(hmm_files)} HMM files, {len(msa_files)} MSA files"
        else:
            result["message"] = "Directory contains no HMM or MSA files"
        
        return result
    
    result["message"] = f"Path {database_path} is neither file nor directory"
    return result


def identify_fastq_files(
    input_path: Union[str, Path],
    return_rolypoly: bool = True,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """Identify and categorize FASTQ files from input path.
    
    Args:
        input_path: Path to input directory or file
        return_rolypoly: Whether to look for and return rolypoly-formatted files first
        logger: Logger instance
        
    Returns:
        Dictionary containing categorized file information:
        - rolypoly_data: {lib_name: {'interleaved': path, 'merged': path}}
        - R1_R2_pairs: [(r1_path, r2_path), ...]
        - interleaved_files: [path, ...]
        - single_end: [path, ...]
        - file_details: {file_path: analysis_results}
    """
    logger = get_logger(logger)
    input_path = Path(input_path)
    
    logger.info(f"Identifying FASTQ files in: {input_path}")
    
    file_info = {
        "rolypoly_data": {},
        "R1_R2_pairs": [],
        "interleaved_files": [],
        "single_end": [],
        "file_details": {}
    }
    
    if input_path.is_dir():
        # First look for rolypoly output files if requested
        if return_rolypoly:
            rolypoly_files = list(input_path.glob("*_final_*.f*q*"))
            if rolypoly_files:
                logger.info(f"Found {len(rolypoly_files)} rolypoly output files")
                for file in rolypoly_files:
                    lib_name = file.stem.split("_final_")[0]
                    if lib_name not in file_info["rolypoly_data"]:
                        file_info["rolypoly_data"][lib_name] = {
                            "interleaved": None,
                            "merged": None,
                        }
                    if "interleaved" in file.name:
                        file_info["rolypoly_data"][lib_name]["interleaved"] = file
                        logger.debug(f"Added rolypoly interleaved: {lib_name} -> {file}")
                    elif "merged" in file.name:
                        file_info["rolypoly_data"][lib_name]["merged"] = file
                        logger.debug(f"Added rolypoly merged: {lib_name} -> {file}")
                        
                # Analyze rolypoly files
                for lib_name, data in file_info["rolypoly_data"].items():
                    for file_type, file_path in data.items():
                        if file_path:
                            analysis = determine_fastq_type(file_path, logger=logger)
                            file_info["file_details"][str(file_path)] = analysis
                            
                return file_info
        
        # Process all FASTQ files
        all_fastq = find_fastq_files(input_path, logger=logger)
        processed_files = set()
        
        logger.info(f"Processing {len(all_fastq)} FASTQ files")
        
        # First pass - identify paired files by filename
        for file in all_fastq:
            if file in processed_files:
                continue
                
            is_paired, pair_file = is_paired_filename(file.name, logger)
            if is_paired:
                pair_path = file.parent / pair_file
                if pair_path.exists() and pair_path in all_fastq:
                    # Analyze both files
                    r1_analysis = determine_fastq_type(file, logger=logger)
                    r2_analysis = determine_fastq_type(pair_path, logger=logger)
                    
                    file_info["file_details"][str(file)] = r1_analysis
                    file_info["file_details"][str(pair_path)] = r2_analysis
                    
                    file_info["R1_R2_pairs"].append((file, pair_path))
                    processed_files.add(file)
                    processed_files.add(pair_path)
                    
                    logger.debug(f"Added R1/R2 pair: {file.name} <-> {pair_file}")
                    continue
        
        # Second pass - analyze remaining files
        for file in all_fastq:
            if file in processed_files:
                continue
                
            logger.debug(f"Analyzing remaining file: {file}")
            analysis = determine_fastq_type(file, logger=logger)
            file_info["file_details"][str(file)] = analysis
            
            # Categorize based on analysis
            if analysis['interleaved']:
                file_info["interleaved_files"].append(file)
                logger.debug(f"Categorized as interleaved: {file}")
            elif analysis['single_end']:
                file_info["single_end"].append(file)
                logger.debug(f"Categorized as single-end: {file}")
            else:
                # Default to single-end if unclear
                file_info["single_end"].append(file)
                logger.warning(f"Unclear file type, defaulting to single-end: {file}")
                
            processed_files.add(file)
    
    else:
        # Single file input
        logger.info(f"Analyzing single file: {input_path}")
        analysis = determine_fastq_type(input_path, logger=logger)
        file_info["file_details"][str(input_path)] = analysis
        
        if analysis['interleaved']:
            file_info["interleaved_files"].append(input_path)
        else:
            file_info["single_end"].append(input_path)
    
    # Log summary
    logger.info("File identification summary:")
    logger.info(f"  - Rolypoly libraries: {len(file_info['rolypoly_data'])}")
    logger.info(f"  - R1/R2 pairs: {len(file_info['R1_R2_pairs'])}")
    logger.info(f"  - Interleaved files: {len(file_info['interleaved_files'])}")
    logger.info(f"  - Single-end files: {len(file_info['single_end'])}")
    
    return file_info


def guess_fastq_properties(
    file_path: str,
    mb_to_read: int = 20,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """Legacy function for backward compatibility.
    
    Analyze a FASTQ file to determine its properties using the old interface.
    This function is maintained for backward compatibility but uses the new
    determine_fastq_type function internally.
    
    Args:
        file_path: Path to the FASTQ file
        mb_to_read: Number of MB to read for analysis
        logger: Logger instance
        
    Returns:
        Dictionary containing legacy format results
    """
    logger = get_logger(logger)
    
    # Use new function with byte-based sampling
    results = determine_fastq_type(
        file_path,
        sample_method="first_n_bytes",
        sample_size=mb_to_read * 1024 * 1024,
        logger=logger
    )
    
    # Convert to legacy format
    return {
        "is_gzipped": results['is_gzipped'],
        "paired_end": results['interleaved'],  # Legacy interpretation
        "average_read_length": results['header_analysis'].get('average_read_length', 0),
    }


def handle_input_fastq(
    input_path: Union[str, Path],
    logger: Optional[logging.Logger] = None
) -> Dict:
    """Handle input FASTQ files and prepare file information for processing.
    
    This function is designed to be compatible with the filter_reads workflow.
    It uses the consolidated file detection functions and returns information
    in a format expected by the read filtering pipeline.
    
    Args:
        input_path: Path to input directory or file(s)
        logger: Logger instance
        
    Returns:
        Dictionary containing:
        - R1_R2_pairs: List of (R1, R2) path tuples
        - interleaved_files: List of interleaved file paths
        - single_end_files: List of single-end file paths
        - file_name: Suggested base name for output files
    """
    logger = get_logger(logger)
    input_path = Path(input_path)
    
    # Handle comma-separated file inputs (common in filter_reads usage)
    if isinstance(input_path, (str, Path)) and ',' in str(input_path):
        # Split comma-separated files
        file_paths = [Path(p.strip()) for p in str(input_path).split(',')]
        
        if len(file_paths) == 2:
            # Assume R1, R2 pair
            r1_path, r2_path = file_paths
            
            # Generate file name from R1
            file_name = r1_path.stem
            if file_name.endswith('.fq') or file_name.endswith('.fastq'):
                file_name = file_name.rsplit('.', 1)[0]
            # Remove R1/R2 indicators
            for pattern in ['_R1', '_1', '.1']:
                if pattern in file_name:
                    file_name = file_name.replace(pattern, '')
                    break
            
            logger.info(f"Detected paired files: {r1_path} and {r2_path}")
            
            return {
                "R1_R2_pairs": [(r1_path, r2_path)],
                "interleaved_files": [],
                "single_end_files": [],
                "file_name": file_name
            }
        else:
            # Multiple single files
            logger.info(f"Detected {len(file_paths)} individual files")
            
            # Use first file for naming
            file_name = file_paths[0].stem
            if file_name.endswith('.fq') or file_name.endswith('.fastq'):
                file_name = file_name.rsplit('.', 1)[0]
            
            return {
                "R1_R2_pairs": [],
                "interleaved_files": [],
                "single_end_files": file_paths,
                "file_name": file_name
            }
    
    # Use consolidated file detection for directory or single file
    file_info = identify_fastq_files(input_path, return_rolypoly=False, logger=logger)
    
    # Generate appropriate file name
    file_name = "rolypoly_filtered_reads"
    
    if input_path.is_file():
        # Single file input
        file_name = input_path.stem
        if file_name.endswith('.fq') or file_name.endswith('.fastq'):
            file_name = file_name.rsplit('.', 1)[0]
        # Remove R1/R2 indicators
        for pattern in ['_R1', '_1', '.1', '_R2', '_2', '.2']:
            if pattern in file_name:
                file_name = file_name.replace(pattern, '')
                break
    elif input_path.is_dir():
        # Use directory name as base
        file_name = input_path.name
    
    # Convert our file_info format to the expected format
    result = {
        "R1_R2_pairs": file_info["R1_R2_pairs"],
        "interleaved_files": file_info["interleaved_files"],
        "single_end_files": file_info["single_end"],  # Note: different key name
        "file_name": file_name
    }
    
    # Add rolypoly data if present
    if file_info["rolypoly_data"]:
        result["rolypoly_data"] = file_info["rolypoly_data"]
    
    logger.info(f"File handling summary for '{input_path}':")
    logger.info(f"  - File name: {file_name}")
    logger.info(f"  - R1/R2 pairs: {len(result['R1_R2_pairs'])}")
    logger.info(f"  - Interleaved files: {len(result['interleaved_files'])}")
    logger.info(f"  - Single-end files: {len(result['single_end_files'])}")
    
    return result


def ensure_faidx(input_file: str, logger: Optional[logging.Logger] = None) -> None:
    """Ensure a FASTA file has a pyfastx index.
    
    Creates a pyfastx index for the input FASTA file if it doesn't exist.
    
    Args:
        input_file: Path to the FASTA file
        logger: Logger instance
    """
    logger = get_logger(logger)
    
    try:
        import pyfastx
        from rich.console import Console
        
        console = Console(width=150)
        
        if not os.path.exists(f"{input_file}.fxi"):
            logger.info(f"Indexing {input_file} with pyfastx")
            console.print(f"[yellow]Indexing {input_file} with pyfastx[/yellow]")
            pyfastx.Fasta(str(input_file))
            console.print("[green]Indexing complete.[/green]")
            logger.info("FASTA indexing completed")
        else:
            logger.debug(f"Index already exists for {input_file}")
            
    except ImportError:
        logger.error("pyfastx not available for FASTA indexing")
        raise
    except Exception as e:
        logger.error(f"Error creating FASTA index for {input_file}: {e}")
        raise