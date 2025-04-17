use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufRead, BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Filter sequences by header pattern
#[pyfunction]
pub fn filter_by_header(
    input_file: &str,
    patterns: Vec<String>,
    output_file: &str,
    threads: Option<usize>,
) -> PyResult<usize> {
    filter_sequences(input_file, patterns, output_file, threads, false)
}

/// Filter sequences by header pattern (inverted)
#[pyfunction]
pub fn filter_by_header_invert(
    input_file: &str,
    patterns: Vec<String>,
    output_file: &str,
    threads: Option<usize>,
) -> PyResult<usize> {
    filter_sequences(input_file, patterns, output_file, threads, true)
}

/// Internal function to filter sequences
fn filter_sequences(
    input_file: &str,
    patterns: Vec<String>,
    output_file: &str,
    threads: Option<usize>,
    invert: bool,
) -> PyResult<usize> {
    // Set up parallelism
    let threads = threads.unwrap_or_else(|| rayon::current_num_threads());
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .map_err(|e| PyValueError::new_err(format!("Failed to set up thread pool: {}", e)))?;

    // Output file
    let output_path = Path::new(output_file);
    let output_file = File::create(output_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to create output file: {}", e)))?;
    let writer = Arc::new(Mutex::new(BufWriter::new(output_file)));
    let count = Arc::new(Mutex::new(0));

    // Open the input file
    let input_path = Path::new(input_file);
    let file = File::open(input_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to open input file: {}", e)))?;
    
    let reader = io::BufReader::new(file);
    let mut lines = reader.lines();
    
    // Process FASTA file
    let mut current_header = String::new();
    let mut current_sequence = String::new();
    
    while let Some(Ok(line)) = lines.next() {
        if line.starts_with('>') {
            // Process previous sequence if there is one
            if !current_header.is_empty() && !current_sequence.is_empty() {
                let matches = patterns.iter().any(|pattern| current_header.contains(pattern));
                if matches ^ invert {
                    // Write to output file
                    let mut writer_lock = writer.lock().unwrap();
                    if let Err(e) = writeln!(writer_lock, ">{}", current_header) {
                        return Err(PyValueError::new_err(format!("Error writing header: {}", e)));
                    }
                    
                    // Write sequence with line breaks every 60 characters
                    for chunk in current_sequence.as_bytes().chunks(60) {
                        if let Err(e) = writeln!(writer_lock, "{}", std::str::from_utf8(chunk).unwrap()) {
                            return Err(PyValueError::new_err(format!("Error writing sequence: {}", e)));
                        }
                    }
                    
                    // Increment count
                    let mut count_lock = count.lock().unwrap();
                    *count_lock += 1;
                }
            }
            
            // Extract header
            current_header = line[1..].trim().to_string();
            current_sequence = String::new();
        } else {
            // Append to current sequence
            current_sequence.push_str(line.trim());
        }
    }
    
    // Process the last sequence
    if !current_header.is_empty() && !current_sequence.is_empty() {
        let matches = patterns.iter().any(|pattern| current_header.contains(pattern));
        if matches ^ invert {
            // Write to output file
            let mut writer_lock = writer.lock().unwrap();
            if let Err(e) = writeln!(writer_lock, ">{}", current_header) {
                return Err(PyValueError::new_err(format!("Error writing header: {}", e)));
            }
            
            // Write sequence with line breaks every 60 characters
            for chunk in current_sequence.as_bytes().chunks(60) {
                if let Err(e) = writeln!(writer_lock, "{}", std::str::from_utf8(chunk).unwrap()) {
                    return Err(PyValueError::new_err(format!("Error writing sequence: {}", e)));
                }
            }
            
            // Increment count
            let mut count_lock = count.lock().unwrap();
            *count_lock += 1;
        }
    }
    
    // Fix the lifetime issue by getting the value before returning
    let result = *count.lock().unwrap();
    Ok(result)
}

/// Python module for sequence filtering
#[pymodule]
pub fn seq_filter(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(filter_by_header, m)?)?;
    m.add_function(wrap_pyfunction!(filter_by_header_invert, m)?)?;
    Ok(())
} 