"""File format detection and analysis functions.

This module provides comprehensive FASTQ file detection, analysis, and classification
functionality with support for paired-end, interleaved, and single-end files.
"""

import os
import re
import gzip
import logging
import random
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from rolypoly.utils.logging.loggit import get_logger
from rolypoly.utils.various import is_gzipped, find_files_by_extension
from needletail import parse_fastx_file

def create_sample_file(
    file_path: Union[str, Path],
    subset_type: str = "top_reads",
    sample_size: Union[int,float] = 1000,
    output_file: str = "sample.fastq",
    # keep_pairs=True,
    logger: Optional[logging.Logger] = None
) -> str:
    """Create a temporary sample file from a FASTQ file for analysis.
    
    Args:
        file_path: Path to the input FASTQ file
        subset_type: Type of subsert - "top_reads" or "random".
        sample_size: if top_reads than how many reads (from the top) to sample, if random than fracton of reads to sample randomly (0.0-1.0)
        # keep_pairs: Keep paired-end reads - if true input file is assumed to be paired end AND interleaved. all R1 reads in the output will have matching R2. (EDIT- I'm just going to sneakily take half the sample size, get that many random items, then take 2 consecutive reads at a time lol)
        logger: Logger instance
        
    Returns:
        Path to temporary sample file
    Note:
        - If sample type is random, it is possible all the reads would have to be read, so coudl be slow.
    """
    logger = get_logger(logger)
    file_path = Path(file_path)
    is_gz = is_gzipped(file_path)
    
    
    logger.debug(f"Sampling {subset_type} of {sample_size} of {file_path}")
    try:
        if subset_type == "top_reads":
            sample_size = int(sample_size) * 4 # 4 lines per read
            # Stream first n lines without loading entire file
            if is_gz:
                f_in = gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore')
            else:
                f_in = open(file_path, 'r', encoding='utf-8', errors='ignore')
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for i, line in enumerate(f_in):
                    print ("a")
                    if i >= sample_size:
                        break
                    f_out.write(line)
                        # if total_reads <= sample_size:
            #     sample_size = total_reads
            #     logger.warning(f"Total reads in {file_path} is less than requested sample size, will just copy the input buddy")
            #     import shutil
            #     shutil.copyfile(file_path, output_file)
            #     return None
            # elif total_reads == 0:
            #     logger.warning(f"No obvious reads in fastq (looked for header lines starting with @ and found none)")
            #     return None # TODO: DECIDE, should this be an error
            

        elif subset_type == "random":
            logger.debug(f"Sampling {sample_size*100}% of reads randomly from {file_path}")
            from rolypoly.utils.bio.polars_fastx import from_fastx_lazy as read_fastx
            # from needletail import parse_fastx_file
            # test_for_getting_total_number_of_records = parse_fastx_file(file_path)
             # I am putting way too much time into this but if possible to get an approximate 
             # for the total number of reads than maybe use:
             #  https://github.com/Yixuan-Wang/blog-contents/issues/29
             # most bioinformatics stuff use defaults so maybe ... using the top n reads can get the avg read # length and then use more approximations to oget a rough estimate of total reads based on (maybe? #original file size (pre compression) to current file size (post compression)
            import subprocess as sp
            if is_gz:
                total_reads = int(sp.run("zgrep -c '@' {}".format(file_path), shell=True, capture_output=True, text=True).stdout.strip())
            else:
                total_reads = int(sp.run("grep -c '@' {}".format(file_path), shell=True, capture_output=True, text=True).stdout.strip())

            # convert proportion to number of reads
            sample_size = int(sample_size * total_reads)
            # adjust to an even number of reads
            sample_size = sample_size - (sample_size % 2)
            # get the random indices of the reads
            # deciding on which reads 
            from random import sample
            R1_reads_indx = sample(population=range(total_reads), k=int(sample_size/2))
            R2_reads_indx = [i+1 for i in  R1_reads_indx]
            all_reads_indx = (R1_reads_indx + R2_reads_indx)
            all_reads_indx.sort() # to keep R1 and R2 pairs one after the other.
            # converting indexes to line numbers (maybe I could have just taken 8 lines for each item from the sample, rather than take twice 4 in case there is R1 R2)
            # @pentamorfico this is where I am at.
            R1_reads_indx = [all_reads_indx[i] for i in range(0, len(all_reads_indx), 2)]
            R2_reads_indx = [all_reads_indx[i] for i in range(1, len(all_reads_indx), 2)]

            lf = read_fastx(file_path)
            lf.filter().collect()
            # Get number of records
            records = read_fastx(file_path)
            records = records.collect()
            total_records = len(records)

    except Exception as e:
        logger.error(f"Error creating sample file from {file_path}: {e}")
        raise


def determine_fastq_type(
    file_path: Union[str, Path],
    sample_size: int = 1000,  # Increased from whenever. should be consisent as long as the number is even..
    logger: Optional[logging.Logger] = None
) -> Dict:
    """Analyze FASTQ headers to determine file characteristics.
    
    Args:
        file_path: Path to FASTQ file or sample file
        sample_size: Number of reads (from the top of file) to use
        logger: Logger instance
        
    Returns:
        Dictionary containing header analysis results
    """
    import polars as pl
    logger = get_logger(logger)
    from rolypoly.utils.bio.polars_fastx import from_fastx_lazy as read_fastx
    from needletail import decode_phred # could be cool to add a mean quality score calculation here.
    results = {
        'file_type': "unknown",
        'is_gzipped': is_gzipped(file_path),
        'pair_1_count': 0,
        'pair_2_count': 0,
        'average_read_length': 0,
        'average_read_quality': 0,
    }
    try:
        file_path = Path(file_path)
        fastq_df = read_fastx(file_path)
        fastq_df = fastq_df.head(sample_size).collect()
        header_count = fastq_df.select(pl.col("header").str.tail(2).value_counts()).unnest("header")
        logger.debug(f"header_count: {header_count}")
        pair_1_count = header_count.filter(pl.col("header").is_in([" 1", "/1"]))["count"].sum()
        pair_2_count = header_count.filter(pl.col("header").is_in([" 2", "/2"]))["count"].sum()
        # total_length = fastq_df.select(pl.col("sequence").str.len_chars()).sum().item() # could have just 
        average_read_length = fastq_df.select(pl.col("sequence").str.len_chars()).mean().item()
        from numpy import mean
        average_read_quality = fastq_df.select(
            pl.col("sequence").map_elements(
                lambda x: mean(decode_phred(x)), return_dtype=pl.Float64).mean()).item()

        # add to results dict
        results['average_read_length'] = average_read_length
        results['average_read_quality'] = average_read_quality
        results['pair_1_count'] = pair_1_count
        results['pair_2_count'] = pair_2_count
        if pair_1_count == sample_size/2 and pair_2_count == pair_1_count:
            results['file_type'] = "interleaved"
        elif pair_1_count == sample_size and pair_2_count == 0:
            results['file_type'] = "paired_R1"
        elif pair_1_count == 0 and pair_2_count == sample_size:
            results['file_type'] = "paired_R2"
        elif pair_1_count == 0 and pair_2_count == 0:
            results['file_type'] = "single" # this is a guess
        logger.debug(f"Header analysis for {file_path}: {results}")
        
    except Exception as e:
        logger.error(f"Error analyzing  {file_path}: {e}")
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
        (r"(.*)\.1(\.f.*q.*)$", r"\1.2\2"),  # .1.fastq/.2.fastq # not sre if the f*q* is required.
    ]
    
    for pattern, replacement in patterns:
        match = re.match(pattern, filename)
        if match:
            pair_file = re.sub(pattern, replacement, filename)
            logger.debug(f"Detected paired filename pattern: {filename} -> {pair_file}")
            return True, pair_file
            
    return False, ""

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
        # First look for rolypoly output files if requested - these are expected to be named like "lib_name_final_interleaved.fq.gz" and "lib_name_final_merged.fq.gz"
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
                        
                # Analyze rolypoly files - is this neccessary? shouldn't some other part of my code be writting this and thus I can trust myself to expect... correct formatting?
                for lib_name, data in file_info["rolypoly_data"].items():
                    for file_type, file_path in data.items():
                        if file_path:
                            analysis = determine_fastq_type(file_path, logger=logger)
                            file_info["file_details"][str(file_path)] = analysis
                            
                return file_info
        
        # Process all FASTQ files
        all_fastq = find_files_by_extension(input_path, ["*.fq", "*.fastq", "*.fq.gz", "*.fastq.gz"], "FASTQ files", logger)
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
    
    # Log debug, should usualy be printed in the summary
    logger.info("File identification summary:")
    logger.info(f"  - Rolypoly libraries: {len(file_info['rolypoly_data'])}")
    logger.info(f"  - R1/R2 pairs: {len(file_info['R1_R2_pairs'])}")
    logger.info(f"  - Interleaved files: {len(file_info['interleaved_files'])}")
    logger.info(f"  - Single-end files: {len(file_info['single_end'])}")
    
    return file_info


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

