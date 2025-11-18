from typing import Dict, List

PREFERRED_ORDER = [
    "Abstract",
    "Background",
    "Introduction",
    "Methods",
    "Experiments",
    "Results",
    "Related Work",
    "Discussion",
    "Conclusion",
]


def build_pubmed_txt(section_dict: Dict[str, str]) -> str:
    """
    Build formatted text output.
    
    Format:
    Header
    Content (no blank line here)
    
    NextHeader (blank line before this)
    NextContent
    """
    lines: List[str] = []
    used = set()

    # First: preferred order
    for sec in PREFERRED_ORDER:
        if sec in section_dict and section_dict[sec].strip():
            append_section(lines, sec, section_dict[sec])
            used.add(sec)

    # Remaining sections in discovery order (shouldn't happen with new logic)
    for sec, content in section_dict.items():
        if sec not in used and content.strip():
            append_section(lines, sec, content)

    return "\n".join(lines).rstrip()


def append_section(lines: List[str], section_name: str, content: str):
    """
    Append a section with proper formatting.
    
    Rules:
    - If not first section: add blank line before header
    - Add header (clean, no punctuation)
    - Add content on NEXT line (no blank line between header and content)
    """
    # Add blank line before section (except for first section)
    if lines:
        lines.append("")
    
    # Add header (cleaned)
    header = clean_header(section_name)
    lines.append(header)
    
    # Add content on next line (NO blank line between)
    # Content is already a string, just append it
    lines.append(content.strip())


def clean_header(header: str) -> str:
    """
    Clean header: remove numbering and punctuation.
    """
    import re
    # Remove leading numbers, dots, parentheses, colons
    cleaned = re.sub(r'^[\d\.\)\]\s:-]+', '', header)
    # Remove trailing punctuation
    cleaned = re.sub(r'[:\.,;!?\-]+$', '', cleaned)
    return cleaned.strip()