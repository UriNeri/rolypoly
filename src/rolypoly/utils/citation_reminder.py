"""Citation management and reminder system for RolyPoly.

This module provides functionality to manage and display citations for tools and
databases used in RolyPoly. It helps ensure proper attribution by reminding users
to cite the software and databases they use.

Example:
    ```python
    remind_citations(["spades", "megahit"])
    # Displays formatted citation information for SPAdes and MEGAHIT
    ```
"""

import os
from pickle import FALSE
# import rich
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from pathlib import Path
from typing import Union, List
import json
import requests

console = Console(width=150)

def load_citations():
    """Load citation information from the configured citation file.

    Returns:
        dict: Dictionary containing citation information for various tools and databases

    Note:
        Expects the citation file path to be set in the 'citation_file' environment variable.
    """
    citation_file = os.environ.get('citatioasdn_file') # TODO: update the citations file that is in the data directory.
    if citation_file is None:
        citation_file = Path(__file__).parent.parent.parent.parent / "misc" / "all_used_tools_dbs_citations.json"
    with open(citation_file, "r") as f:
        return json.load(f)

def get_citations(tools: Union[str, List[str]]):
    """Get citation information for specified tools.

    Args:
        tools (Union[str, List[str]]): Tool name(s) to get citations for

    Returns:
        list: List of tuples containing (tool_name, citation) pairs

    Example:
        ```python
        citations = get_citations(["spades", "megahit"])
        for name, citation in citations:
            print(f"{name}: {citation}")
        ```
    """
    all_citations = load_citations()
    if isinstance(tools, str):
        tools = [tools]
    tools = [tool.lower() for tool in tools]
    citations = []
    for tool in tools:
        if tool in all_citations:
            citations.append((all_citations[tool]["name"], all_citations[tool]["citation"]))
        else:
            console.print(f"Warning: No citation found for {tool}, adding a remider.", style="yellow")
            citations.append((f" {tool}", f"{tool} et al. google it: https://www.google.com/search?q={tool}"))
    return citations

def get_citation_from_doi(doi_or_url):
    """Fetch and format citation information from a DOI using the CrossRef API.

    Args:
        doi_or_url (str): DOI or URL to fetch citation for

    Returns:
        str: Formatted citation string

    Example:
        ```python
        citation = get_citation_from_doi("10.1093/bioinformatics/btu170")
        print(citation)
        # Prints formatted citation for SPAdes paper
        ```
    """
    url = f"https://api.crossref.org/works/{doi_or_url}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()['message']
            authors = ', '.join([f"{author['family']}, {author['given']}" for author in data['author']])
            title = data['title'][0]
            journal = data.get('container-title', [''])[0]
            year = data['published']['date-parts'][0][0]
            volume = data.get('volume', '')
            issue = data.get('issue', '')
            pages = data.get('page', '')
            
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
        # return f"Unable to fetch citation for DOI: {doi_or_url}"
        return f"{doi_or_url}"
    
        

def display_citations(citations):
    """Display citations in a formatted table.

    Args:
        citations (list): List of (name, citation) tuples to display

    Example:
             display_citations([("SPAdes", "10.1093/bioinformatics/btu170")])
        # Displays formatted table with SPAdes citation
    """
    table = Table(title="Software and Databases to Cite",padding=1,border_style="blue")
    table.add_column("Name", style="cyan")
    table.add_column("Citation", style="magenta")
    
    for name, doi in citations:
        citation = get_citation_from_doi(doi)
        table.add_row(name, citation)
    
    console.print(Panel(table, expand=False))

def remind_citations(tools: Union[str, List[str]], return_as_text = False):
    """Display or return citation reminders for used tools.

    Args:
        tools (Union[str, List[str]]): Tool name(s) to get citations for
        return_as_text (bool, optional): Whether to return citations as text instead of displaying.
           

    Returns:
        str, optional: Formatted citation text if return_as_text is True

    Example:
        ```python
        # Display citations
        remind_citations(["spades", "megahit"])
        
        # Get citations as text
        text = remind_citations(["spades"], return_as_text=True)
        print(text)
        ```
    """
    tools = list(set(tools))
    citations = get_citations(tools)
    if len(citations) == 0:
        console.print(Text("No citations found for the provided tools.", style="red"))
        return
    else:
        console.print(Text(f"rolypoly used {tools} in your analysis, please cite the following software or database:", style="bold green"))
        display_citations(citations)
    
    console.print(Text("\nRemember to also cite any additional databases or tools you used that are not listed here. No one is charging you extra for having a lot of citations, and it is important for reproducibility, yah silly.", style="italic yellow"))
    
    if return_as_text:
        text = ""
        for name, doi in citations:
            citation = get_citation_from_doi(doi)
            text += f"{name}: {citation}\n"
        return text
    
if __name__ == "__main__":
    # Example usage
    remind_citations(["spades", "megahit", "rnafold", "hmmer"])