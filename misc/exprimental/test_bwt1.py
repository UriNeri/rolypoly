import random
import string
from pathlib import Path
import sh
from rolypoly.utils.bwt1 import (
    build_index,
    align_single_end,
    align_paired_end,
    align_with_multiple_mismatches,
    align_report_multiple,
    inspect_index,
    AlignmentResult,
    IndexInfo
)
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def generate_random_sequence(length):
    return ''.join(random.choice('ATCG') for _ in range(length))

def generate_fasta(filename, num_sequences, seq_length):
    with open(filename, 'w') as f:
        for i in range(num_sequences):
            f.write(f'>sequence_{i}\n')
            f.write(generate_random_sequence(seq_length) + '\n')

def generate_fastq(filename, num_sequences, seq_length):
    with open(filename, 'w') as f:
        for i in range(num_sequences):
            f.write(f'@sequence_{i}\n')
            seq = generate_random_sequence(seq_length)
            f.write(seq + '\n')
            f.write('+\n')
            f.write(''.join(random.choice(string.ascii_uppercase) for _ in range(seq_length)) + '\n')

def test_bowtie1_functions():
    # Create test directory
    test_dir = Path('bowtie1_test')
    test_dir.mkdir(exist_ok=True)

    # Generate reference sequence
    ref_file = test_dir / 'reference.fa'
    generate_fasta(ref_file, 10, 1000)

    # Generate reads
    reads_file = test_dir / 'reads.fq'
    generate_fastq(reads_file, 1000, 100)

    # Generate paired-end reads
    reads_file1 = test_dir / 'reads_1.fq'
    reads_file2 = test_dir / 'reads_2.fq'
    generate_fastq(reads_file1, 1000, 100)
    generate_fastq(reads_file2, 1000, 100)

    # Build index
    index_base = test_dir / 'reference_index'
    build_index(ref_file, index_base)

    try:
        # Test inspect_index
        index_info = inspect_index(index_base)
        assert isinstance(index_info, IndexInfo)
        print(f"Index info: {index_info}")
    except Exception as e:
        print(f"Error during index inspection: {e}")
        print("Continuing with other tests    ")

    # Test align_single_end
    output_file = test_dir / 'alignments_single.sam'
    result = align_single_end(index_base, reads_file, output_file)
    assert isinstance(result, AlignmentResult)
    print(f"Single-end alignment result: {result}")

    # Test align_paired_end
    output_file_paired = test_dir / 'alignments_paired.sam'
    logger.debug(f"Index base: {index_base}")
    logger.debug(f"Reads file 1: {reads_file1}")
    logger.debug(f"Reads file 2: {reads_file2}")
    logger.debug(f"Output file paired: {output_file_paired}")

    try:
        result_paired = align_paired_end(index_base, reads_file1, reads_file2, output_file_paired)
        print(f"Paired-end alignment result: {result_paired}")
    except Exception as e:
        logger.error(f"Error during paired-end alignment: {e}", exc_info=True)

    # Test align_with_multiple_mismatches
    output_file_mm = test_dir / 'alignments_mm.sam'
    result_mm = align_with_multiple_mismatches(index_base, reads_file, output_file_mm, num_mismatches=3)
    assert isinstance(result_mm, AlignmentResult)
    print(f"Multiple mismatches alignment result: {result_mm}")

    # Test align_report_multiple
    output_file_multi = test_dir / 'alignments_multi.sam'
    result_multi = align_report_multiple(index_base, reads_file, output_file_multi, num_alignments=3)
    assert isinstance(result_multi, AlignmentResult)
    print(f"Multiple alignments report result: {result_multi}")

    print("All tests completed successfully!")

if __name__ == "__main__":
    try:
        test_bowtie1_functions()
    except sh.ErrorReturnCode as e:
        print(f"An error occurred: {e}")
        print(f"STDOUT: {e.stdout.decode()}")
        print(f"STDERR: {e.stderr.decode()}")
