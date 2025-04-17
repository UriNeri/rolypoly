# def parse_args():
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="MSA Visualization CLI",
#         usage="%(prog)s [-h] [-i INPUT] [-s START] [-e END] [-w WRAP_LENGTH] [-c] [-m MARKERS] [-x MARKER_X] [-t TEXT_ANNOTATIONS] [-o OUTPUT] [-d DPI]",
#         epilog="""
# Examples:
#   python msa_viz.py -i /home/neri/Downloads/MSA_for_2E.fa -s 121 -e 294 -w 360 -c -m 1 -x 149 161 205 219 236 244 -t "149,161,Motif A" "205,219,Motif B" "236,244,Motif C" -o asdas.pdf -d 360
#   from msa_viz import draw_msa; draw_msa("path/to/msa/file", start=121, end=294, wrap_length=360, show_consensus=True, markers=[1], marker_x=[149, 161, 205, 219, 236, 244], text_annotations=["149,161,Motif A", "205,219,Motif B", "236,244,Motif C"], output="output.pdf", dpi=360)
# """,
#     )

#     parser.add_argument("-i", "--input", required=True, help="Path to MSA file")
#     parser.add_argument("-s", "--start", type=int, default=1, help="Start position")
#     parser.add_argument("-e", "--end", type=int, default=None, help="End position")
#     parser.add_argument(
#         "-w", "--wrap_length", type=int, default=360, help="Wrap length"
#     )
#     parser.add_argument(
#         "-c", "--show_consensus", action="store_true", help="Show consensus"
#     )
#     parser.add_argument(
#         "-m", "--markers", nargs="+", type=int, help="Add markers at positions"
#     )
#     parser.add_argument(
#         "-x", "--marker_x", nargs="+", type=int, help="Add 'x' markers at positions"
#     )
#     parser.add_argument(
#         "-t",
#         "--text_annotations",
#         nargs="+",
#         help="Add text annotations (format: 'start,end,text')",
#     )
#     parser.add_argument("-o", "--output", required=True, help="Output file name")
#     parser.add_argument(
#         "-d", "--dpi", type=int, default=360, help="DPI for output image"
#     )

#     return parser.parse_args()


# def draw_msa(
#     input_file,
#     start=1,
#     end=None,
#     wrap_length=360,
#     show_consensus=False,
#     markers=None,
#     marker_x=None,
#     text_annotations=None,
#     output="output.pdf",
#     dpi=360,
# ):
#     from pymsaviz import MsaViz

#     mv = MsaViz(
#         input_file,
#         start=start,
#         end=end,
#         wrap_length=wrap_length,
#         show_consensus=show_consensus,
#     )

#     # Extract MSA positions less than 50% consensus identity
#     # pos_ident_less_than_50 = []
#     # ident_list = mv._get_consensus_identity_list()
#     # for pos, ident in enumerate(ident_list, 1):
#     #     if ident <= 50:
#     #         pos_ident_less_than_50.append(pos)

#     # Add markers
#     if markers:
#         mv.add_markers(markers)
#     if marker_x:
#         mv.add_markers(marker_x, marker="x", color="cyan")

#     # Add text annotations
#     if text_annotations:
#         for annotation in text_annotations:
#             start, end, text = annotation.split(",")
#             mv.add_text_annotation(
#                 (int(start), int(end)), text, text_color="red", range_color="red"
#             )

#     mv.savefig(dpi=dpi, savefile=output)


# def main():
#     args = parse_args()
#     draw_msa(
#         args.input,
#         args.start,
#         args.end,
#         args.wrap_length,
#         args.show_consensus,
#         args.markers,
#         args.marker_x,
#         args.text_annotations,
#         args.output,
#         args.dpi,
#     )


# if __name__ == "__main__":
#     main()
