from pathlib import Path

import rich_click as click

from rolypoly.utils.citation_reminder import remind_citations


def parse_annotations(annotation_file):
    """Parse annotation file in GFF or custom JSON format.

    Args:
        annotation_file (str): Path to annotation file

    Returns:
        list: List of feature dictionaries
    """
    import json

    features = []
    ext = Path(annotation_file).suffix.lower()

    if ext == ".gff" or ext == ".gff3":
        with open(annotation_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 8:
                    continue

                seqid, source, type_, start, end, score, strand, phase, *attrs = parts

                # Parse attributes
                attrs_dict = {}
                if len(attrs) > 0:
                    for attr in attrs[0].split(";"):
                        if "=" in attr:
                            key, value = attr.split("=", 1)
                            attrs_dict[key] = value

                features.append(
                    {
                        "start": int(start) - 1,  # Convert to 0-based
                        "end": int(end),
                        "strand": 1 if strand == "+" else -1 if strand == "-" else 0,
                        "type": type_,
                        "label": attrs_dict.get("Name", attrs_dict.get("ID", type_)),
                        "color": None,  # Will be set based on type
                        "source": source,
                    }
                )

    elif ext == ".json":
        with open(annotation_file) as f:
            features = json.load(f)

    return features


def assign_feature_colors(features):
    """Assign colors to features based on their type.

    Args:
        features (list): List of feature dictionaries

    Returns:
        list: Updated feature list with colors
    """
    # Color scheme for different feature types
    color_map = {
        "CDS": "#45aa52",
        "gene": "#2d5ba6",
        "exon": "#a65d2d",
        "domain": "#7d2da6",
        "RdRP": "#ff0000",
        "capsid": "#0000ff",
        "tRNA": "#ffff00",
        "rRNA": "#00ffff",
        "repeat_region": "#ff00ff",
    }

    default_color = "#808080"  # Gray for unknown types

    for feature in features:
        feature_type = feature["type"].lower()

        # Try to match feature type with color map
        for key, color in color_map.items():
            if key.lower() in feature_type:
                feature["color"] = color
                break
        else:
            feature["color"] = default_color

    return features


def create_linear_plot(sequence_length, features, width=None, height=None):
    """Create a linear genome plot.

    Args:
        sequence_length (int): Length of the sequence
        features (list): List of feature dictionaries
        width (int): Plot width
        height (int): Plot height

    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    import matplotlib.pyplot as plt
    from dna_features_viewer import GraphicFeature, GraphicRecord

    # Convert features to GraphicFeature objects
    graphic_features = []
    for feature in features:
        graphic_features.append(
            GraphicFeature(
                start=feature["start"],
                end=feature["end"],
                strand=feature["strand"],
                color=feature["color"],
                label=feature["label"],
            )
        )

    # Create the record
    record = GraphicRecord(sequence_length=sequence_length, features=graphic_features)

    # Create the plot
    fig, ax = plt.subplots(1, figsize=(width or 12, height or 3))
    record.plot(ax=ax)

    # Customize the plot
    ax.set_title(f"Genome visualization ({sequence_length} bp)")
    plt.tight_layout()

    return fig


# def create_circular_plot(sequence_length, features, width=None, height=None):
#     """Create a circular genome plot.

#     Args:
#         sequence_length (int): Length of the sequence
#         features (list): List of feature dictionaries
#         width (int): Plot width
#         height (int): Plot height

#     Returns:
#         matplotlib.figure.Figure: The plot figure
#     """
#     import matplotlib.patches as patches
#     import matplotlib.pyplot as plt

#     # Create figure
#     fig, ax = plt.subplots(1, figsize=(width or 8, height or 8))
#     ax.set_aspect("equal")

#     # Calculate circle properties
#     radius = 1
#     center = (0, 0)

#     # Draw base circle
#     circle = plt.Circle(center, radius, fill=False, color="black")
#     ax.add_patch(circle)

#     # Draw features
#     for feature in features:
#         start_angle = (feature["start"] / sequence_length) * 2 * np.pi
#         end_angle = (feature["end"] / sequence_length) * 2 * np.pi

#         # Create arc for the feature
#         arc = patches.Arc(
#             center,
#             2 * radius,
#             2 * radius,
#             theta1=np.degrees(start_angle),
#             theta2=np.degrees(end_angle),
#             color=feature["color"],
#             linewidth=2,
#         )
#         ax.add_patch(arc)

#         # Add label
#         mid_angle = (start_angle + end_angle) / 2
#         label_radius = radius + 0.1
#         label_x = label_radius * np.cos(mid_angle)
#         label_y = label_radius * np.sin(mid_angle)

#         # Rotate text based on position
#         rotation = np.degrees(mid_angle)
#         if rotation > 90 and rotation < 270:
#             rotation += 180

#         ax.text(
#             label_x,
#             label_y,
#             feature["label"],
#             rotation=rotation,
#             ha="center",
#             va="center",
#             rotation_mode="anchor",
#             fontsize=8,
#         )

#     # Set plot limits
#     ax.set_xlim(-1.5, 1.5)
#     ax.set_ylim(-1.5, 1.5)
#     ax.axis("off")

#     plt.title(f"Circular genome visualization ({sequence_length} bp)")
#     plt.tight_layout()

#     return fig


@click.command()
@click.option("-i", "--input", required=True, help="Input file or directory")
@click.option("-o", "--output", default="output", help="Output directory")
@click.option("--log-file", default="command.log", help="Path to log file")
@click.option("--log-level", default="INFO", help="Log level")
@click.option(
    "--format", default="svg", type=click.Choice(["svg", "png"]), help="format"
)
@click.option("--width", default=None, type=int, help="width")
@click.option("--height", default=None, type=int, help="height")
@click.option("--annotations", default=None, type=str, help="annotations")
@click.option(
    "--style", default="linear", type=click.Choice(["linear", "circular"]), help="style"
)
def visualize(
    input, output, log_file, log_level, format, width, height, annotations, style
):
    """
    Generate visualization of viral genome features including ORFs, domains, and other annotations
    """
    import matplotlib.pyplot as plt
    from needletail import parse_fastx_file

    from rolypoly.utils.loggit import log_start_info, setup_logging

    logger = setup_logging(log_file, log_level)
    log_start_info(logger, locals())

    # Create output directory
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read sequence
    try:
        record = next(parse_fastx_file(input))
        sequence_length = len(record.seq)
    except StopIteration:
        logger.error(f"No sequences found in {input}")
        return
    except Exception as e:
        logger.error(f"Error reading sequence file: {e}")
        return

    # Parse annotations if provided
    features = []
    if annotations:
        try:
            features = parse_annotations(annotations)
            features = assign_feature_colors(features)
        except Exception as e:
            logger.error(f"Error parsing annotations: {e}")
            return

    # Create visualization
    try:
        if style == "linear":
            fig = create_linear_plot(sequence_length, features, width, height)
        # else:
        #     fig = create_circular_plot(sequence_length, features, width, height)

        # Save plot
        output_file = output_path / f"genome_visualization.{format}"
        fig.savefig(output_file, format=format, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Visualization saved to {output_file}")

    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        return

    logger.info("visualize completed successfully!")
    tools = ["dna_features_viewer", "matplotlib"]
    remind_citations(tools)


if __name__ == "__main__":
    visualize()
