import sh
from pathlib import Path
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AlignmentResult:
    reads_processed: int
    reads_aligned: int
    alignment_rate: float

@dataclass
class IndexInfo:
    name: str
    num_sequences: int
    total_length: int
    memory_usage: str

# Initialize Bowtie commands
bowtie = sh.Command("bowtie")
bowtie_build = sh.Command("bowtie-build")
bowtie_inspect = sh.Command("bowtie-inspect")

def build_index(reference_in: Path, index_base: Path, threads: int = 1) -> str:
    """
    Build a Bowtie index from a set of DNA sequences.
    
    Args:
        reference_in (Path): Path to the reference sequences (FASTA format).
        index_base (Path): The base name of the index files to write.
        threads (int): Number of threads to use (default: 1).
    
    Returns:
        str: The command output.
    
    Raises:
        sh.ErrorReturnCode: If the command fails.
    """
    try:
        result = bowtie_build("-f", "--threads", threads, reference_in, index_base)
        
        # Print the raw output for debugging
        logger.debug("Raw bowtie-build output:")
        logger.debug(result)
        
        # Return the result as a string
        return str(result)
    except sh.ErrorReturnCode as e:
        logger.error(f"Error building index: {e}")
        logger.error(f"STDOUT: {e.stdout.decode() if hasattr(e, 'stdout') else 'N/A'}")
        logger.error(f"STDERR: {e.stderr.decode() if hasattr(e, 'stderr') else 'N/A'}")
        raise

def align_single_end(index_base: Path, reads: Path, output_file: Path, threads: int = 1) -> AlignmentResult:
    """
    Align single-end reads to the index.
    
    Args:
        index_base (Path): The base name of the index to be searched.
        reads (Path): Path to the reads file (FASTQ format).
        output_file (Path): Path to the output file (SAM format).
        threads (int): Number of threads to use (default: 1).
    
    Returns:
        AlignmentResult: Object containing alignment statistics.
    
    Raises:
        sh.ErrorReturnCode: If the command fails.
    """
    try:
        result = bowtie("-p", threads, "-S", "-x", index_base, reads, _out=output_file, _err=sh.ErrorReturnCode)
        
        # Print the raw output for debugging
        logger.debug("Raw bowtie output:")
        logger.debug(result)
        
        # Check if result is a string or a sh.RunningCommand object
        if isinstance(result, str):
            summary = result.split("\n")
        else:
            summary = result.stderr.decode().split("\n")
        
        # Parse the alignment summary
        reads_processed = 0
        reads_aligned = 0
        alignment_rate = 0.0
        
        for line in summary:
            logger.debug(f"Parsing line: {line}")
            if "reads processed:" in line:
                reads_processed = int(line.split(":")[1].strip())
            elif "reads with at least one reported alignment:" in line:
                reads_aligned = int(line.split(":")[1].strip().split()[0])
            elif "overall alignment rate:" in line:
                alignment_rate = float(line.split(":")[1].strip()[:-1])
        
        logger.info(f"Alignment complete. Processed: {reads_processed}, Aligned: {reads_aligned}, Rate: {alignment_rate}%")
        return AlignmentResult(reads_processed, reads_aligned, alignment_rate)
    except sh.ErrorReturnCode as e:
        logger.error(f"Error aligning single-end reads: {e}")
        logger.error(f"STDOUT: {e.stdout.decode() if hasattr(e, 'stdout') else 'N/A'}")
        logger.error(f"STDERR: {e.stderr.decode() if hasattr(e, 'stderr') else 'N/A'}")
        raise


def align_paired_end_interleaved(index_base: Path, reads: Path, output_file: Path, threads: int = 1) -> AlignmentResult:
    """
    Align paired-end reads to the index.
    
    Args:
        index_base (Path): The base name of the index to be searched.
        reads (Path): Path to the interleaved reads file (FASTQ format).
        output_file (Path): Path to the output file (SAM format).
        threads (int): Number of threads to use (default: 1).
    """
    try:
        logger.debug(f"Running bowtie with index_base: {index_base}, reads: {reads}, output_file: {output_file}")
        result = bowtie("--threads", threads, "--sam","-x", index_base, "--interleaved", reads, "--fullref", _out=output_file)
        
        # Print the raw output for debugging
        logger.debug("Raw bowtie output (interleaved):")
        logger.debug(result)
        
        # Check if result is a string or a sh.RunningCommand object
        if isinstance(result, str):
            summary = result.split("\n")
        else:
            summary = result.stderr.decode().split("\n")
        
        # Parse the alignment summary
        reads_processed = 0
        reads_aligned = 0
        alignment_rate = 0.0
        
        for line in summary:
            logger.debug(f"Parsing line: {line}")
            if "reads processed:" in line:
                reads_processed = int(line.split(":")[1].strip())   
            elif "reads with at least one reported alignment:" in line:
                reads_aligned = int(line.split(":")[1].strip().split()[0])
            elif "overall alignment rate:" in line:
                alignment_rate = float(line.split(":")[1].strip()[:-1])
        
        logger.info(f"Alignment complete. Processed: {reads_processed}, Aligned: {reads_aligned}, Rate: {alignment_rate}%")
        return AlignmentResult(reads_processed, reads_aligned, alignment_rate)
    except sh.ErrorReturnCode as e: # TODO: fix this to print stdout and stderr, figure out how to do this with sh
        logger.error(f"Error aligning paired-end interleaved reads: {e}")
        logger.error(f"STDOUT: {e.stdout.decode() if hasattr(e, 'stdout') else 'N/A'}")
        logger.error(f"STDERR: {e.stderr.decode() if hasattr(e, 'stderr') else 'N/A'}")
        # logger.error(f"STDERR: {stderr}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in align_paired_end_interleaved: {e}")
        raise


def align_paired_end(index_base: Path, reads1: Path, reads2: Path, output_file: Path, threads: int = 1) -> AlignmentResult:
    """
    Align paired-end reads to the index.
    
    Args:
        index_base (Path): The base name of the index to be searched.
        reads1 (Path): Path to the #1 mates reads file (FASTQ format).
        reads2 (Path): Path to the #2 mates reads file (FASTQ format).
        output_file (Path): Path to the output file (SAM format).
        threads (int): Number of threads to use (default: 1).
    
    Returns:
        AlignmentResult: Object containing alignment statistics.
    
    Raises:
        sh.ErrorReturnCode: If the command fails.
    """
    try:
        logger.debug(f"Running bowtie with index_base: {index_base}, reads1: {reads1}, reads2: {reads2}, output_file: {output_file}")
        result = bowtie("-p", threads, "-S", index_base, "-1", reads1, "-2", reads2, _out=output_file)
        
        # Print the raw output for debugging
        logger.debug("Raw bowtie paired-end output:")
        logger.debug(result)
        
        # Check if result is a string or a sh.RunningCommand object
        if isinstance(result, str):
            summary = result.split("\n")
        else:
            summary = result.stderr.decode().split("\n")
        
        # Parse the alignment summary
        reads_processed = 0
        reads_aligned = 0
        alignment_rate = 0.0
        
        for line in summary:
            logger.debug(f"Parsing line: {line}")
            if "reads processed:" in line:
                reads_processed = int(line.split(":")[1].strip())
            elif "reads with at least one reported alignment:" in line:
                reads_aligned = int(line.split(":")[1].strip().split()[0])
            elif "overall alignment rate:" in line:
                alignment_rate = float(line.split(":")[1].strip()[:-1])
        
        logger.info(f"Alignment complete. Processed: {reads_processed}, Aligned: {reads_aligned}, Rate: {alignment_rate}%")
        return AlignmentResult(reads_processed, reads_aligned, alignment_rate)
    except sh.ErrorReturnCode as e:
        logger.error(f"Error aligning paired-end reads: {e}")
        logger.error(f"STDOUT: {e.stdout.decode() if hasattr(e, 'stdout') else 'N/A'}")
        logger.error(f"STDERR: {e.stderr.decode() if hasattr(e, 'stderr') else 'N/A'}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in align_paired_end: {e}")
        raise

def align_with_multiple_mismatches(index_base: Path, reads: Path, output_file: Path, num_mismatches: int = 3, threads: int = 1) -> AlignmentResult:
    """
    Align reads allowing for multiple mismatches.
    
    Args:
        index_base (Path): The base name of the index to be searched.
        reads (Path): Path to the reads file (FASTQ format).
        output_file (Path): Path to the output file (SAM format).
        num_mismatches (int): Maximum number of mismatches allowed (default: 3).
        threads (int): Number of threads to use (default: 1).
    
    Returns:
        AlignmentResult: Object containing alignment statistics.
    
    Raises:
        sh.ErrorReturnCode: If the command fails.
    """
    try:
        logger.debug(f"Running bowtie with index_base: {index_base}, reads: {reads}, output_file: {output_file}, num_mismatches: {num_mismatches}")
        result = bowtie("-p", threads, "-S", "-v", num_mismatches, index_base, reads, _out=output_file)
        
        # Print the raw output for debugging
        logger.debug("Raw bowtie output (multiple mismatches):")
        logger.debug(result)
        
        # Check if result is a string or a sh.RunningCommand object
        if isinstance(result, str):
            summary = result.split("\n")
        else:
            summary = result.stderr.decode().split("\n")
        
        # Parse the alignment summary
        reads_processed = 0
        reads_aligned = 0
        alignment_rate = 0.0
        
        for line in summary:
            logger.debug(f"Parsing line: {line}")
            if "reads processed:" in line:
                reads_processed = int(line.split(":")[1].strip())
            elif "reads with at least one reported alignment:" in line:
                reads_aligned = int(line.split(":")[1].strip().split()[0])
            elif "overall alignment rate:" in line:
                alignment_rate = float(line.split(":")[1].strip()[:-1])
        
        logger.info(f"Alignment complete. Processed: {reads_processed}, Aligned: {reads_aligned}, Rate: {alignment_rate}%")
        return AlignmentResult(reads_processed, reads_aligned, alignment_rate)
    except sh.ErrorReturnCode as e:
        logger.error(f"Error aligning with multiple mismatches: {e}")
        logger.error(f"STDOUT: {e.stdout.decode() if hasattr(e, 'stdout') else 'N/A'}")
        logger.error(f"STDERR: {e.stderr.decode() if hasattr(e, 'stderr') else 'N/A'}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in align_with_multiple_mismatches: {e}")
        raise

def align_report_multiple(index_base: Path, reads: Path, output_file: Path, num_alignments: int = 2, threads: int = 1) -> AlignmentResult:
    """
    Align reads and report multiple alignments per read.
    
    Args:
        index_base (Path): The base name of the index to be searched.
        reads (Path): Path to the reads file (FASTQ format).
        output_file (Path): Path to the output file (SAM format).
        num_alignments (int): Maximum number of alignments to report per read (default: 2).
        threads (int): Number of threads to use (default: 1).
    
    Returns:
        AlignmentResult: Object containing alignment statistics.
    
    Raises:
        sh.ErrorReturnCode: If the command fails.
    """
    try:
        result = bowtie("-p", threads, "-S", "-k", num_alignments, index_base, reads, _out=output_file)
        
        # Print the raw output for debugging
        logger.debug("Raw bowtie output (multiple alignments):")
        logger.debug(result)
        
        # Check if result is a string or a sh.RunningCommand object
        if isinstance(result, str):
            summary = result.split("\n")
        else:
            summary = result.stderr.decode().split("\n")
        
        # Parse the alignment summary
        reads_processed = 0
        reads_aligned = 0
        alignment_rate = 0.0
        
        for line in summary:
            logger.debug(f"Parsing line: {line}")
            if "reads processed:" in line:
                reads_processed = int(line.split(":")[1].strip())
            elif "reads with at least one reported alignment:" in line:
                reads_aligned = int(line.split(":")[1].strip().split()[0])
            elif "overall alignment rate:" in line:
                alignment_rate = float(line.split(":")[1].strip()[:-1])
        
        logger.info(f"Alignment complete. Processed: {reads_processed}, Aligned: {reads_aligned}, Rate: {alignment_rate}%")
        return AlignmentResult(reads_processed, reads_aligned, alignment_rate)
    except sh.ErrorReturnCode as e:
        logger.error(f"Error reporting multiple alignments: {e}")
        logger.error(f"STDOUT: {e.stdout.decode() if hasattr(e, 'stdout') else 'N/A'}")
        logger.error(f"STDERR: {e.stderr.decode() if hasattr(e, 'stderr') else 'N/A'}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in align_report_multiple: {e}")
        raise

def inspect_index(index_base: Path) -> IndexInfo:
    """
    Inspect a Bowtie index and return information about it.
    
    Args:
        index_base (Path): The base name of the Bowtie index.
    
    Returns:
        IndexInfo: Object containing index information.
    
    Raises:
        sh.ErrorReturnCode: If the command fails.
        ValueError: If the output format is unexpected.
    """
    try:
        result = bowtie_inspect("-s", str(index_base))
        
        # Print the raw output for debugging
        logger.debug("Raw bowtie-inspect output:")
        logger.debug(result)
        
        # Check if result is already a string
        if isinstance(result, str):
            lines = result.strip().split("\n")
        else:
            # If it's a sh.RunningCommand object, decode stdout
            lines = result.stdout.decode().strip().split("\n")
        
        if not lines:
            raise ValueError("bowtie-inspect returned no output.")
        
        # Try to parse the output, but handle cases where the format is unexpected
        try:
            name = lines[0].split(":")[1].strip() if ":" in lines[0] else "Unknown"
            num_sequences = int(lines[1].split(":")[1].strip()) if len(lines) > 1 and ":" in lines[1] else 0
            total_length = int(lines[2].split(":")[1].strip()) if len(lines) > 2 and ":" in lines[2] else 0
            memory_usage = lines[3].split(":")[1].strip() if len(lines) > 3 and ":" in lines[3] else "Unknown"
        except (IndexError, ValueError) as e:
            logger.warning(f"Error parsing bowtie-inspect output: {e}")
            name = "Unknown"
            num_sequences = 0
            total_length = 0
            memory_usage = "Unknown"
        
        return IndexInfo(name, num_sequences, total_length, memory_usage)
    except sh.ErrorReturnCode as e:
        logger.error(f"Error running bowtie-inspect: {e}")
        logger.error(f"STDOUT: {e.stdout.decode() if hasattr(e, 'stdout') else 'N/A'}")
        logger.error(f"STDERR: {e.stderr.decode() if hasattr(e, 'stderr') else 'N/A'}")
        raise

# Example usage
if __name__ == "__main__":
    reference_file = Path("reference.fa")
    index_base = Path("reference_index")
    reads_file = Path("reads.fq")
    output_file = Path("alignments.sam")

    # Build index
    build_index(reference_file, index_base)

    # Align single-end reads
    alignment_result = align_single_end(index_base, reads_file, output_file)
    logger.info(f"Alignment Result: {alignment_result}")

    # Inspect index
    index_info = inspect_index(index_base)
    logger.info(f"Index Info: {index_info}")
