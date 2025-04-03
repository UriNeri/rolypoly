# Contributing to RolyPoly

Thank you for your interest in contributing to RolyPoly! This project is open and governed under the Lawrence Berkeley National Laboratory (LBNL) Intellectual Property (IP) office and is licensed under the GNU General Public License version 3 (GPLv3).

## Project Roadmap & TODO List
Check out our [project roadmap and TODO list](https://docs.google.com/spreadsheets/d/1udNbxtK1QMfOhVgxHyhrgw7U1hHFeIazlcLM6VIcbJo/edit?gid=0#gid=0) to see what features and improvements are planned.

## Coding Style & Conventions

### Language & Dependencies
- **Primary Language**: Python 3.8+
- **Secondary Languages**: Bash, C++, Rust (rarely)
- **Visualization Scripts**: R or Python
- **Dependency Management**:
  - Prefer using existing dependencies over adding new ones, and avoid bio-python if possible.
  - Example: Use `mappy.revcomp` instead of adding bi opython just for reverse complement
  - Avoid compiled dependencies unless available through conda channels
  - Prefer slim Python bindings over full packages
  - Use/add-to `rolypoly.utils.various` for general-purpose functions that don't fit into other categories (e.g. dataframe operations)
  - Use/add-to `rolypoly.utils.fax` for functions that are related to bioinformatics data (fasta, nucleotide, protein, etc.)
  - Use/add-to `rolypoly.utils.loggit` for logging functions

### Code Organization
1. **File Structure**:
   - `src/rolypoly/`: Core package code
   - `src/rolypoly/utils/`: Utility functions and helpers
   - `src/rolypoly/commands/`: Command-line interface modules
   - `docs/`: Documentation (mkdocs)
   - `benchmarking/`: Test files and data

2. **Naming Conventions**:
   - CLI arguments: support both short and long options, e.g. `-s` and `--skip-existing`. Optionally, add support for json file (--config config.json) and json string (--override-params '{"skip_existing": true}'). See `rolypoly.commands.reads.read-filtering.py` for an example.
   - Internal variables: Snake case (e.g., `skip_existing`)
   - Functions: Snake case, descriptive names
   - Classes: PascalCase (though classes are used sparingly, and only for complex commands/for configuration).
   - Global variables: snakecase 
   - Environment variables: UPPERCASE or CamelCase.


3. **Temporary Files**:
   - Optionally, create tmp directory within user's output path. see "with change_directory("/path/to/dir"):" in `rolypoly.utils.various.change_directory` for an info, or `rolypoly.commands.reads.read-filtering.py` for another example.
   - Clean up tmp files unless `--keep-tmp` flag is used
   - Move only final output files to user's output path

### Testing & Benchmarking
1. **Testing**:
   - Location: `/clusterfs/jgi/scratch/science/metagen/neri/tests/rp_tests/` directory - conntains example data for different commands, seperated by input size and complexity.
2. **Benchmarking**:
   - Use `/usr/bin/time` for runtime/memory metrics
   - Default to "cold" starts

## Details
For detailed API documentation, see the [API Reference](api/) section.

**Note**: This project is governed under the LBNL IP office. By contributing, you agree that your contributions will be subject to the terms of the GPLv3 license.
