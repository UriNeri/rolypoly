import os
from pathlib import Path
from typing import List, Union

import requests
from rich.console import Console

console = Console(width=150)


def load_citations():
    """Load citation information from the configured citation file"""
    import json

    citation_file = os.environ.get(
        "citatioasdn_file"
    )  # TODO: update the citations file that is in the data directory.
    if citation_file is None:
        citation_file = Path(__file__).parent / "all_used_tools_dbs_citations.json"
    with open(citation_file, "r") as f:
        return json.load(f)


def get_citations(tools: Union[str, List[str]]):
    """Get citation information for specified tools"""
    all_citations = load_citations()
    if isinstance(tools, str):
        tools = [tools]
    tools = [tool.lower() for tool in tools]
    citations = []
    for tool in tools:
        if tool in all_citations:
            citations.append(
                (all_citations[tool]["name"], all_citations[tool]["citation"])
            )
        else:
            console.print(
                f"Warning: No citation found for {tool}, adding a remider.",
                style="yellow",
            )
            citations.append(
                (
                    f" {tool}",
                    f"{tool} et al. google it: https://www.google.com/search?q={tool}",
                )
            )
    return citations


def get_citation_from_doi(doi_or_url, return_bibtex=False):
    """Fetch and format citation information from a DOI using the CrossRef API"""
    url = f"https://api.crossref.org/works/{doi_or_url}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()["message"]
            if return_bibtex:
                # Create BibTeX key from first author and year
                first_author = data["author"][0]["family"].lower()
                year = data["published"]["date-parts"][0][0]
                key = f"{first_author}{year}"

                authors = " and ".join(
                    [
                        f"{author['family']}, {author['given']}"
                        for author in data["author"]
                    ]
                )
                title = data["title"][0]
                journal = data.get("container-title", [""])[0]
                volume = data.get("volume", "")
                issue = data.get("issue", "")
                pages = data.get("page", "")

                bibtex = f"@article{{{key},\n"
                bibtex += f"  author = {{{authors}}},\n"
                bibtex += f"  title = {{{title}}},\n"
                bibtex += f"  journal = {{{journal}}},\n"
                bibtex += f"  year = {{{year}}}"
                if volume:
                    bibtex += f",\n  volume = {{{volume}}}"
                if issue:
                    bibtex += f",\n  number = {{{issue}}}"
                if pages:
                    bibtex += f",\n  pages = {{{pages}}}"
                bibtex += f",\n  doi = {{{doi_or_url}}}\n}}"
                return bibtex
            else:
                # Original citation format
                authors = ", ".join(
                    [
                        f"{author['family']}, {author['given']}"
                        for author in data["author"]
                    ]
                )
                title = data["title"][0]
                journal = data.get("container-title", [""])[0]
                year = data["published"]["date-parts"][0][0]
                volume = data.get("volume", "")
                issue = data.get("issue", "")
                pages = data.get("page", "")

                citation = f"{authors}. ({year}). {title}. {journal}"
                if volume:
                    citation += f", {volume}"
                if issue:
                    citation += f"({issue})"
                if pages:
                    citation += f", {pages}"
                citation += f". {doi_or_url}"
                return citation
        else:
            return f"{doi_or_url}"
    except Exception as e:
        console.print(f"Unable to fetch citation for DOI: {doi_or_url}", style="red")
        console.print(f"Suggestion: {doi_or_url}", style="yellow")
        return f"{doi_or_url}"


def display_citations(citations):
    """Display citations in a formatted table"""
    from rich.panel import Panel
    from rich.table import Table

    table = Table(
        title="Software and Databases to Cite", padding=1, border_style="blue"
    )
    table.add_column("Name", style="cyan")
    table.add_column("Citation", style="magenta")

    for name, doi in citations:
        citation = get_citation_from_doi(doi)
        table.add_row(name, citation)

    console.print(Panel(table, expand=False))


def remind_citations(
    tools: Union[str, List[str]], return_as_text=False, return_bibtex=False
):
    """Display or return citation reminders for used tools"""
    from rich.text import Text

    tools = list(set(tools))
    citations = get_citations(tools)
    if len(citations) == 0:
        console.print(Text("No citations found for the provided tools.", style="red"))
        return
    else:
        console.print(
            Text(
                f"rolypoly used {tools} in your analysis, please cite the following software or database:",
                style="bold green",
            )
        )
        display_citations(citations)

    console.print(
        Text(
            "\nRemember to also cite any additional databases or tools you used that are not listed here. No one is charging you extra for having a lot of citations, and it is important for reproducibility, yah silly.",
            style="italic yellow",
        )
    )

    if return_as_text or return_bibtex:
        text = "RolyPoly used the following software and databases in your analysis, please cite the following software or database:\n"
        for name, doi in citations:
            citation = get_citation_from_doi(doi, return_bibtex=return_bibtex)
            text += f"{name}:\n{citation}\n\n"
        return text


if __name__ == "__main__":
    # Example usage
    remind_citations(["spades", "megahit", "rnafold", "hmmer"])
