use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufRead, BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Deduplicate sequences by sequence
#[pyfunction]
pub fn deduplicate_by_seq(
    input_files: Vec<String>,
    output_file: &str,
    threads: Option<usize>,
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
    
    // Use a HashSet to track unique sequences
    let unique_seqs = Arc::new(Mutex::new(HashSet::new()));
    let count = Arc::new(Mutex::new(0));
    
    // Process each input file
    input_files.par_iter().for_each(|input_file| {
        // Open the input file
        let input_path = Path::new(input_file);
        let file = match File::open(input_path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Failed to open input file {}: {}", input_file, e);
                return;
            }
        };
        
        let reader = io::BufReader::new(file);
        let mut lines = reader.lines();
        
        // Process FASTA file
        let mut current_header = String::new();
        let mut current_sequence = String::new();
        
        while let Some(Ok(line)) = lines.next() {
            if line.starts_with('>') {
                // Process previous sequence if there is one
                if !current_header.is_empty() && !current_sequence.is_empty() {
                    let normalized_seq = current_sequence.to_uppercase();
                    let mut unique_seqs_lock = unique_seqs.lock().unwrap();
                    
                    if !unique_seqs_lock.contains(&normalized_seq) {
                        unique_seqs_lock.insert(normalized_seq);
                        drop(unique_seqs_lock); // Release lock before writing
                        
                        // Write to output file
                        let mut writer_lock = writer.lock().unwrap();
                        if let Err(e) = writeln!(writer_lock, ">{}", current_header) {
                            eprintln!("Error writing header: {}", e);
                            return;
                        }
                        
                        // Write sequence with line breaks every 60 characters
                        for chunk in current_sequence.as_bytes().chunks(60) {
                            if let Err(e) = writeln!(writer_lock, "{}", std::str::from_utf8(chunk).unwrap()) {
                                eprintln!("Error writing sequence: {}", e);
                                return;
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
            let normalized_seq = current_sequence.to_uppercase();
            let mut unique_seqs_lock = unique_seqs.lock().unwrap();
            
            if !unique_seqs_lock.contains(&normalized_seq) {
                unique_seqs_lock.insert(normalized_seq);
                drop(unique_seqs_lock); // Release lock before writing
                
                // Write to output file
                let mut writer_lock = writer.lock().unwrap();
                if let Err(e) = writeln!(writer_lock, ">{}", current_header) {
                    eprintln!("Error writing header: {}", e);
                    return;
                }
                
                // Write sequence with line breaks every 60 characters
                for chunk in current_sequence.as_bytes().chunks(60) {
                    if let Err(e) = writeln!(writer_lock, "{}", std::str::from_utf8(chunk).unwrap()) {
                        eprintln!("Error writing sequence: {}", e);
                        return;
                    }
                }
                
                // Increment count
                let mut count_lock = count.lock().unwrap();
                *count_lock += 1;
            }
        }
    });
    
    // Finalize and return count
    let final_count = *count.lock().unwrap();
    Ok(final_count)
}

/// Deduplicate sequences by ID
#[pyfunction]
pub fn deduplicate_by_id(
    input_files: Vec<String>,
    output_file: &str,
    threads: Option<usize>,
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
    
    // Use a HashSet to track unique sequence IDs
    let unique_ids = Arc::new(Mutex::new(HashSet::new()));
    let count = Arc::new(Mutex::new(0));
    
    // Process each input file
    input_files.par_iter().for_each(|input_file| {
        // Open the input file
        let input_path = Path::new(input_file);
        let file = match File::open(input_path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Failed to open input file {}: {}", input_file, e);
                return;
            }
        };
        
        let reader = io::BufReader::new(file);
        let mut lines = reader.lines();
        
        // Process FASTA file
        let mut current_header = String::new();
        let mut current_sequence = String::new();
        
        while let Some(Ok(line)) = lines.next() {
            if line.starts_with('>') {
                // Process previous sequence if there is one
                if !current_header.is_empty() && !current_sequence.is_empty() {
                    // Extract ID from header (first word after '>')
                    let id = current_header.split_whitespace().next().unwrap_or("");
                    
                    let mut unique_ids_lock = unique_ids.lock().unwrap();
                    if !unique_ids_lock.contains(id) {
                        unique_ids_lock.insert(id.to_string());
                        drop(unique_ids_lock); // Release lock before writing
                        
                        // Write to output file
                        let mut writer_lock = writer.lock().unwrap();
                        if let Err(e) = writeln!(writer_lock, ">{}", current_header) {
                            eprintln!("Error writing header: {}", e);
                            return;
                        }
                        
                        // Write sequence with line breaks every 60 characters
                        for chunk in current_sequence.as_bytes().chunks(60) {
                            if let Err(e) = writeln!(writer_lock, "{}", std::str::from_utf8(chunk).unwrap()) {
                                eprintln!("Error writing sequence: {}", e);
                                return;
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
            // Extract ID from header (first word after '>')
            let id = current_header.split_whitespace().next().unwrap_or("");
            
            let mut unique_ids_lock = unique_ids.lock().unwrap();
            if !unique_ids_lock.contains(id) {
                unique_ids_lock.insert(id.to_string());
                drop(unique_ids_lock); // Release lock before writing
                
                // Write to output file
                let mut writer_lock = writer.lock().unwrap();
                if let Err(e) = writeln!(writer_lock, ">{}", current_header) {
                    eprintln!("Error writing header: {}", e);
                    return;
                }
                
                // Write sequence with line breaks every 60 characters
                for chunk in current_sequence.as_bytes().chunks(60) {
                    if let Err(e) = writeln!(writer_lock, "{}", std::str::from_utf8(chunk).unwrap()) {
                        eprintln!("Error writing sequence: {}", e);
                        return;
                    }
                }
                
                // Increment count
                let mut count_lock = count.lock().unwrap();
                *count_lock += 1;
            }
        }
    });
    
    // Finalize and return count
    let final_count = *count.lock().unwrap();
    Ok(final_count)
}

/// Python module for sequence deduplication
#[pymodule]
pub fn seq_dedup(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(deduplicate_by_seq, m)?)?;
    m.add_function(wrap_pyfunction!(deduplicate_by_id, m)?)?;
    Ok(())
} 