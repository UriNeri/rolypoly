use bio::alphabets::dna;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::fs::File;
use std::io::{self, BufRead, BufWriter, Write};

/// Genetic code tables for different translation scenarios
#[derive(Clone, Debug)]
pub enum GeneticCode {
    Standard = 1,      // Standard/Universal code
    Vertebrate = 2,    // Vertebrate mitochondrial
    Yeast = 3,         // Yeast mitochondrial
    MoldProt = 4,      // Mold/Protozoan/Coelenterate mitochondrial
    Invertebrate = 5,  // Invertebrate mitochondrial
    Ciliate = 6,       // Ciliate nuclear
    Echinoderm = 9,    // Echinoderm mitochondrial
    Euplotid = 10,     // Euplotid nuclear
    Bacterial = 11,    // Bacterial, archaeal and plant plastid
    AltYeast = 12,     // Alternative yeast nuclear
    Ascidian = 13,     // Ascidian mitochondrial
    AltFlatWorm = 14,  // Alternative flatworm mitochondrial
    Chlorophycean = 16,// Chlorophycean mitochondrial
    Trematode = 21,    // Trematode mitochondrial
    Scenedesmus = 22,  // Scenedesmus obliquus mitochondrial
    Thraustochytrium = 23, // Thraustochytrium mitochondrial
    Pterobranchia = 24,    // Pterobranchia mitochondrial
    CandidateDivision = 25,// Candidate Division SR1 and Gracilibacteria
    Pachysolen = 26,       // Pachysolen tannophilus nuclear code
}

impl GeneticCode {
    /// Get the appropriate codon translation table based on genetic code
    fn get_translation_table(&self) -> HashMap<&'static str, char> {
        match self {
            GeneticCode::Standard => {
                // Standard genetic code (NCBI transl_table=1)
                let mut table = HashMap::new();
                // Fill standard translation table
                table.insert("TTT", 'F'); table.insert("TTC", 'F');
                table.insert("TTA", 'L'); table.insert("TTG", 'L');
                table.insert("TCT", 'S'); table.insert("TCC", 'S'); table.insert("TCA", 'S'); table.insert("TCG", 'S');
                table.insert("TAT", 'Y'); table.insert("TAC", 'Y');
                table.insert("TAA", '*'); table.insert("TAG", '*');
                table.insert("TGT", 'C'); table.insert("TGC", 'C');
                table.insert("TGA", '*'); table.insert("TGG", 'W');
                table.insert("CTT", 'L'); table.insert("CTC", 'L'); table.insert("CTA", 'L'); table.insert("CTG", 'L');
                table.insert("CCT", 'P'); table.insert("CCC", 'P'); table.insert("CCA", 'P'); table.insert("CCG", 'P');
                table.insert("CAT", 'H'); table.insert("CAC", 'H');
                table.insert("CAA", 'Q'); table.insert("CAG", 'Q');
                table.insert("CGT", 'R'); table.insert("CGC", 'R'); table.insert("CGA", 'R'); table.insert("CGG", 'R');
                table.insert("ATT", 'I'); table.insert("ATC", 'I'); table.insert("ATA", 'I');
                table.insert("ATG", 'M');
                table.insert("ACT", 'T'); table.insert("ACC", 'T'); table.insert("ACA", 'T'); table.insert("ACG", 'T');
                table.insert("AAT", 'N'); table.insert("AAC", 'N');
                table.insert("AAA", 'K'); table.insert("AAG", 'K');
                table.insert("AGT", 'S'); table.insert("AGC", 'S');
                table.insert("AGA", 'R'); table.insert("AGG", 'R');
                table.insert("GTT", 'V'); table.insert("GTC", 'V'); table.insert("GTA", 'V'); table.insert("GTG", 'V');
                table.insert("GCT", 'A'); table.insert("GCC", 'A'); table.insert("GCA", 'A'); table.insert("GCG", 'A');
                table.insert("GAT", 'D'); table.insert("GAC", 'D');
                table.insert("GAA", 'E'); table.insert("GAG", 'E');
                table.insert("GGT", 'G'); table.insert("GGC", 'G'); table.insert("GGA", 'G'); table.insert("GGG", 'G');
                table
            },
            GeneticCode::Bacterial => {
                // Bacterial, archaeal and plant plastid (NCBI transl_table=11)
                let table = Self::Standard.get_translation_table();
                // No differences from standard code
                table
            },
            _ => {
                // Default to standard code for now
                Self::Standard.get_translation_table()
            }
        }
    }
}

/// Complement a DNA base
fn complement_base(base: u8) -> u8 {
    match base {
        b'A' | b'a' => b'T',
        b'T' | b't' => b'A',
        b'G' | b'g' => b'C',
        b'C' | b'c' => b'G',
        b'R' | b'r' => b'Y', // R = A or G; Y = C or T
        b'Y' | b'y' => b'R',
        b'M' | b'm' => b'K', // M = A or C; K = G or T
        b'K' | b'k' => b'M',
        b'S' | b's' => b'S', // S = G or C
        b'W' | b'w' => b'W', // W = A or T
        b'B' | b'b' => b'V', // B = C, G, or T; V = A, C, or G
        b'V' | b'v' => b'B',
        b'D' | b'd' => b'H', // D = A, G, or T; H = A, C, or T
        b'H' | b'h' => b'D',
        _ => b'N', // Default to N for any other character
    }
}

/// Reverse complement a DNA sequence
#[pyfunction]
pub fn reverse_complement(sequence: &str) -> String {
    sequence.bytes()
        .rev()
        .map(complement_base)
        .map(|b| b as char)
        .collect()
}

/// Translate a DNA sequence to protein in a given frame
fn translate_frame(seq: &[u8], frame: i32, genetic_code: &GeneticCode) -> String {
    let translation_table = genetic_code.get_translation_table();
    let mut protein = String::new();
    let is_reverse = frame < 0;
    
    // Prepare sequence based on direction
    let working_seq = if is_reverse {
        let mut rev_comp = Vec::with_capacity(seq.len());
        for &base in seq.iter().rev() {
            rev_comp.push(complement_base(base));
        }
        rev_comp
    } else {
        seq.to_vec()
    };
    
    // Determine starting position (0, 1, or 2)
    let start_pos = if is_reverse {
        (-frame - 1) as usize % 3
    } else {
        (frame - 1) as usize % 3
    };
    
    for i in (start_pos..working_seq.len()).step_by(3) {
        if i + 2 < working_seq.len() {
            let codon = std::str::from_utf8(&working_seq[i..i+3])
                .unwrap_or("NNN")
                .to_uppercase();
            let aa = translation_table.get(codon.as_str()).cloned().unwrap_or('X');
            protein.push(aa);
        }
    }
    
    protein
}

/// Translate a DNA sequence to protein in all 6 frames
#[pyfunction]
fn translate_six_frame(seq: &str, genetic_code_id: Option<usize>) -> PyResult<HashMap<String, String>> {
    let seq = seq.to_uppercase();
    if !dna::alphabet().is_word(seq.as_bytes()) {
        return Err(PyValueError::new_err("Invalid DNA sequence"));
    }

    let genetic_code = match genetic_code_id {
        Some(11) => GeneticCode::Bacterial,
        Some(1) | None => GeneticCode::Standard,
        Some(id) => return Err(PyValueError::new_err(format!("Genetic code {} not implemented", id))),
    };

    let mut results = HashMap::new();
    for frame in [1, 2, 3, -1, -2, -3] {
        let protein = translate_frame(seq.as_bytes(), frame, &genetic_code);
        results.insert(format!("frame_{}", frame), protein);
    }
    Ok(results)
}

/// Translate multiple sequences in parallel
#[pyfunction]
fn translate_six_frame_batch(seqs: Vec<String>, genetic_code_id: Option<usize>) -> PyResult<Vec<HashMap<String, String>>> {
    let genetic_code = match genetic_code_id {
        Some(11) => GeneticCode::Bacterial,
        Some(1) | None => GeneticCode::Standard,
        Some(id) => return Err(PyValueError::new_err(format!("Genetic code {} not implemented", id))),
    };

    let results: Vec<_> = seqs.par_iter()
        .map(|seq| {
            let seq = seq.to_uppercase();
            if !dna::alphabet().is_word(seq.as_bytes()) {
                return HashMap::new(); // Skip invalid sequences
            }

            let mut frame_map = HashMap::new();
            for frame in [1, 2, 3, -1, -2, -3] {
                let protein = translate_frame(seq.as_bytes(), frame, &genetic_code);
                frame_map.insert(format!("frame_{}", frame), protein);
            }
            frame_map
        })
        .collect();
    
    Ok(results)
}

/// Translate a file of DNA sequences to protein in all 6 frames
#[pyfunction]
fn translate_six_frame_file(
    input_file: &str,
    output_file: &str,
    id_regexp: Option<&str>,
    genetic_code_id: Option<usize>,
    threads: Option<usize>
) -> PyResult<()> {
    use std::sync::Arc;

    let genetic_code = match genetic_code_id {
        Some(11) => Arc::new(GeneticCode::Bacterial),
        Some(1) | None => Arc::new(GeneticCode::Standard),
        Some(id) => return Err(PyValueError::new_err(format!("Genetic code {} not implemented", id))),
    };

    // Set up parallelism
    let num_threads = threads.unwrap_or_else(|| rayon::current_num_threads());
    
    // Open the input file
    let input_path = Path::new(input_file);
    let file = File::open(input_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to open input file: {}", e)))?;

    // Open output file
    let output_path = Path::new(output_file);
    let output_file = File::create(output_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to create output file: {}", e)))?;
    let mut writer = BufWriter::new(output_file);

    // We'll collect sequences in batches and process them in parallel
    let mut headers = Vec::new();
    let mut sequences = Vec::new();
    let mut current_header = String::new();
    let mut current_sequence = String::new();

    // Parse FASTA file
    let reader = io::BufReader::new(file);
    for line in reader.lines() {
        let line = line.map_err(|e| PyValueError::new_err(format!("Error reading line: {}", e)))?;
        if line.starts_with('>') {
            // Process previous sequence if there is one
            if !current_header.is_empty() {
                headers.push(current_header);
                sequences.push(current_sequence);
                current_sequence = String::new();
            }
            
            // Extract header
            current_header = line[1..].trim().to_string();
        } else {
            // Append to current sequence
            current_sequence.push_str(line.trim());
        }
    }
    
    // Don't forget the last sequence
    if !current_header.is_empty() {
        headers.push(current_header);
        sequences.push(current_sequence);
    }

    // Process sequences in parallel
    let results: Vec<_> = headers.into_par_iter()
        .zip(sequences.into_par_iter())
        .flat_map(|(header, sequence)| {
            let sequence = sequence.to_uppercase();
            let mut entries = Vec::new();
            
            for frame in [1, 2, 3, -1, -2, -3] {
                let translation = translate_frame(sequence.as_bytes(), frame, &genetic_code);
                // Skip empty translations
                if translation.is_empty() {
                    continue;
                }
                
                let direction = if frame > 0 { "forward" } else { "reverse" };
                let mod_frame = if frame > 0 { frame } else { -frame };
                
                // Create header for this frame
                let frame_header = format!(">{}_frame_{}_{}", header, mod_frame, direction);
                entries.push((frame_header, translation));
            }
            
            entries
        })
        .collect();

    // Write results to output file
    for (header, translation) in results {
        writeln!(writer, "{}", header)
            .map_err(|e| PyValueError::new_err(format!("Error writing to output file: {}", e)))?;
        
        // Write sequence with line breaks every 60 characters
        for chunk in translation.as_bytes().chunks(60) {
            writeln!(writer, "{}", std::str::from_utf8(chunk).unwrap())
                .map_err(|e| PyValueError::new_err(format!("Error writing to output file: {}", e)))?;
        }
    }

    Ok(())
}

/// Python module for sequence translation
#[pymodule]
pub fn seq_translate(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(translate_six_frame, m)?)?;
    m.add_function(wrap_pyfunction!(translate_six_frame_batch, m)?)?;
    m.add_function(wrap_pyfunction!(translate_six_frame_file, m)?)?;
    m.add_function(wrap_pyfunction!(reverse_complement, m)?)?;
    Ok(())
} 