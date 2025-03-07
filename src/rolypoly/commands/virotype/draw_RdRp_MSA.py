# import argparse
# import logomaker.src
# import polars as pl
# import matplotlib.pyplot as plt
# import logomaker
# def parse_args():
#     parser = argparse.ArgumentParser(description="LOGO Visualization CLI",
#                                      usage="%(prog)s [-h] [-i INPUT] [-s START] [-e END] [-w WRAP_LENGTH] [-c] [-m MARKERS] [-x MARKER_X] [-t TEXT_ANNOTATIONS] [-o OUTPUT] [-d DPI]",
#                                      epilog="""
# Examples:
#   python msa_viz.py -i /home/neri/Downloads/MSA_for_2E.fa -s 121 -e 294 -w 360 -c -m 1 -x 149 161 205 219 236 244 -t "149,161,Motif A" "205,219,Motif B" "236,244,Motif C" -o asdas.pdf -d 360
#   from msa_viz import draw_msa; draw_msa("path/to/msa/file", start=121, end=294, wrap_length=360, show_consensus=True, markers=[1], marker_x=[149, 161, 205, 219, 236, 244], text_annotations=["149,161,Motif A", "205,219,Motif B", "236,244,Motif C"], output="output.pdf", dpi=360)
# """)

#     parser.add_argument("-i", "--input", required=True, help="Path to MSA file")
#     parser.add_argument("-s", "--start", type=int, default=1, help="Start position")
#     parser.add_argument("-e", "--end", type=int, default=None, help="End position")
#     parser.add_argument("-w", "--wrap_length", type=int, default=360, help="Wrap length")
#     parser.add_argument("-c", "--show_consensus", action="store_true", help="Show consensus")
#     parser.add_argument("-m", "--markers", nargs="+", type=int, help="Add markers at positions")
#     parser.add_argument("-x", "--marker_x", nargs="+", type=int, help="Add 'x' markers at positions")
#     parser.add_argument("-t", "--text_annotations", nargs="+", help="Add text annotations (format: 'start,end,text')")
#     parser.add_argument("-o", "--output", required=True, help="Output file name")
#     parser.add_argument("-d", "--dpi", type=int, default=360, help="DPI for output image")

#     return parser.parse_args()

# def linearize_fasta(input_file):
#     """Linearize sequences in a FASTA file."""
#     sequences = []
#     with open(input_file, 'r') as f:
#         current_sequence = []
#         for line in f:
#             if line.startswith(">"):
#                 if current_sequence:
#                     sequences.append("".join(current_sequence))
#                     current_sequence = []
#             else:
#                 current_sequence.append(line.strip())
#         if current_sequence:
#             sequences.append("".join(current_sequence))
#     return sequences

# def pad_sequences(sequences):
#     """Pad sequences to ensure they all have the same length."""
#     max_length = max(len(seq) for seq in sequences)
#     return [seq + ['-'] * (max_length - len(seq)) for seq in sequences]

# def draw_msa(input_file, start=1, end=None, wrap_length=360, show_consensus=False, markers=None, marker_x=None, text_annotations=None, output="output.pdf", dpi=360):
#     # Linearize the sequences
#     sequences = linearize_fasta(input_file)
#     # logomaker.(sequences)
#     # # Convert sequences to lists of characters
#     # sequences = [list(seq) for seq in sequences]

#     # Pad sequences to ensure they are all the same length
#     # sequences = pad_sequences(sequences)
    
#     # # Convert sequences into a Polars DataFrame
#     # msa_df = pl.DataFrame(sequences)

#     # # Slice the MSA according to the start and end positions
#     # # msa_df = msa_df[:, start-1:end]

#     # # Convert Polars DataFrame to pandas DataFrame for Logomaker compatibility
#     # msa_pd_df = msa_df.to_numpy()

#     # Calculate frequency matrix for each position
#     freq_matrix = logomaker.alignment_to_matrix(sequences, to_type="probability")

#     # Create the sequence logo
#     logo = logomaker.Logo(freq_matrix) #, font_name='Arial', color_scheme='classic')
#     if show_consensus:
#         logo.style_xticks(anchor=0, spacing=wrap_length)

#     # Add markers
#     if markers:
#         for marker in markers:
#             plt.axvline(marker, color='green', linestyle='--')

#     if marker_x:
#         for marker in marker_x:
#             plt.axvline(marker, color='cyan', linestyle='--')

#     # Add text annotations
#     if text_annotations:
#         for annotation in text_annotations:
#             start, end, text = annotation.split(",")
#             plt.text(int(start), 1.05, text, fontsize=12, color='red')

#     plt.savefig(output, dpi=dpi)

# def main():
#     args = parse_args()
#     draw_msa(args.input, args.start, args.end, args.wrap_length, args.show_consensus, args.markers, args.marker_x, args.text_annotations, args.output, args.dpi)

# if __name__ == "__main__":
#     main()



# from msa_viz import draw_msa
# draw_msa(input_file="./myco/MSA_for_2E.fas", start=121, end=294, wrap_length=360, show_consensus=True, markers=[1], marker_x=[149, 161, 205, 219, 236, 244], text_annotations=["149,161,Motif A", "205,219,Motif B", "236,244,Motif C"], output="output.pdf", dpi=360)
