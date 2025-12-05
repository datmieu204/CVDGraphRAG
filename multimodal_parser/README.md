# Multimodal Parser

A text-only document parser module for medical literature extraction using MinerU and intelligent layout-aware processing.

## Overview

This module provides a robust pipeline for extracting structured text from medical documents (PDFs, Office files, etc.) with layout-aware parsing capabilities. It's designed specifically for medical literature processing, handling academic papers, clinical guidelines, and medical textbooks.

## Features

- **Layout-Aware Parsing**: Intelligent column detection and reading order (full-width → left column → right column)
- **Academic Structure Recognition**: Automatic detection of sections (Abstract, Methods, Results, etc.)
- **Content Filtering**: 
  - Skips content before Abstract
  - Stops after Conclusion
  - Filters subsection numbers and invalid headers
- **Multi-Format Support**: PDF, DOCX, PPTX, TXT, MD
- **PubMed-Style Output**: Formatted text output suitable for medical knowledge graphs
- **Formula & Citation Cleaning**: Removes LaTeX formulas and superscript citations

## Architecture

```
multimodal_parser/
├── parser.py          # MinerU parser with layout detection
├── processor.py       # Text processing and formatting
├── demo.py           # CLI demo application
├── test_bbox.py      # Testing logic for bbox processing
└── textbuilder/
    └── formatter.py  # PubMed-style text formatter
```

## Installation

### Prerequisites

1. **MinerU** (core dependency):
```bash
pip install -U 'mineru[core]'
```

2. **LibreOffice** (for Office document conversion):
```bash
# Ubuntu/Debian
sudo apt-get install libreoffice

# macOS
brew install --cask libreoffice
```

3. **Python dependencies**:
```bash
pip install transformers torch nltk
```

4. **NLTK data**:
```python
python -m nltk.downloader punkt punkt_tab
```

## Quick Start

### Command Line Usage

```bash
# Check if MinerU is installed
python demo.py --check

# Parse a medical PDF
python demo.py path/to/medical_paper.pdf

# Specify output directory
python demo.py paper.pdf --output ./results

# Choose parsing method
python demo.py paper.pdf --method auto  # auto/txt/ocr

# Specify language
python demo.py paper.pdf --lang en
```

### Python API

```python
from parser import MineruParser
from processor import TextProcessor

# Initialize
parser = MineruParser()
processor = TextProcessor(output_dir="./output")

# Parse document
text_blocks = parser.parse_document("medical_paper.pdf")

# Process and format
output_path = processor.process_document(text_blocks, doc_name="paper")
print(f"Output saved to: {output_path}")
```

### Direct JSON Processing

If you already have MinerU's `content_list.json`:

```python
from processor import TextProcessor

processor = TextProcessor(output_dir="./output")
output_path = processor.process_content_list_json(
    "path/to/content_list.json",
    doc_name="document"
)
```

## Module Details

### 1. Parser (`parser.py`)

**MineruParser** - Core parsing engine with layout intelligence

**Key Methods:**

- `parse_document(file_path, method="auto", output_dir=None, lang=None)` - Main entry point
- `parse_pdf(pdf_path, ...)` - PDF-specific parsing
- `convert_office_to_pdf(doc_path, ...)` - Office file conversion

**Layout Detection Logic:**

1. **Column Separation**:
   - Detects full-width items (>50% page width) vs. normal items
   - Identifies column boundaries using x-coordinate gap analysis
   - Skips title/author sections before Abstract

2. **Reading Order**:
   ```
   Full-width items → Left column → Right column
   ```

3. **Section Recognition**:
   - Validates against medical section names
   - Filters subsection numbers (2.1, 2.2.1, etc.)
   - Cleans headers (removes punctuation/numbering)

**Valid Sections:**
```python
VALID_SECTIONS = {
    "Abstract", "Background", "Introduction",
    "Methods", "Materials And Methods",
    "Results", "Findings", "Discussion", "Conclusion",
    "Experiments", "Experiment", "Related Work"
}
```

### 2. Processor (`processor.py`)

**TextProcessor** - Text formatting and output generation

**Key Methods:**

- `process_document(text_blocks, doc_name)` - Format text blocks to PubMed style
- `process_content_list_json(json_path, doc_name)` - Full pipeline from JSON
- `_group_by_sections(blocks)` - Group content by detected sections

**Output Format:**
```
Abstract
Content without blank line between header and text

Introduction
More content

Methods
Detailed methods...
```

### 3. Formatter (`textbuilder/formatter.py`)

**PubMed-Style Formatting**

- Preferred section order (Abstract → Methods → Results → Discussion → Conclusion)
- Clean headers without punctuation
- Proper spacing (blank line before sections, not after headers)

```python
from textbuilder.formatter import build_pubmed_txt

sections = {
    "Abstract": "This paper presents...",
    "Methods": "We conducted experiments...",
    "Results": "The results show..."
}

formatted_text = build_pubmed_txt(sections)
```

## Text Cleaning Features

### Formula Removal
```python
# Before: "The equation $E=mc^2$ shows..."
# After: "The equation  shows..."
```

### Citation Cleaning
```python
# Before: "Recent studies^1,2,3 have shown..."
# After: "Recent studies have shown..."
```

### Whitespace Normalization
```python
# Before: "Multiple    spaces   here"
# After: "Multiple spaces here"
```

### Header Cleaning
```python
# Before: "2.1 Introduction::"
# After: "Introduction"
```

## Advanced Usage

### Custom Section Detection

```python
parser = MineruParser()
parser.VALID_SECTIONS.add("Acknowledgments")
parser.VALID_SECTIONS.add("Supplementary Materials")

text_blocks = parser.parse_pdf("paper.pdf")
```

### Batch Processing

```python
from pathlib import Path
from parser import MineruParser
from processor import TextProcessor

parser = MineruParser()
processor = TextProcessor(output_dir="./batch_output")

pdf_dir = Path("./medical_papers")
for pdf_file in pdf_dir.glob("*.pdf"):
    print(f"Processing: {pdf_file.name}")
    text_blocks = parser.parse_document(pdf_file)
    output = processor.process_document(
        text_blocks, 
        doc_name=pdf_file.stem
    )
    print(f"✓ Saved to: {output}")
```

### Integration with Knowledge Graph

```python
# Example: Extract and import to knowledge graph
from parser import MineruParser
from processor import TextProcessor

# Parse medical guideline
parser = MineruParser()
text_blocks = parser.parse_document("clinical_guideline.pdf")

# Group by sections
processor = TextProcessor()
sections = processor._group_by_sections(text_blocks)

# Import each section to knowledge graph
for section_name, content in sections.items():
    import_to_knowledge_graph(
        section=section_name,
        content=content,
        layer="middle"  # Clinical guideline → Middle layer
    )
```

## Configuration

### MinerU Methods

- **`auto`** (default): Automatic detection
- **`txt`**: Text extraction only
- **`ocr`**: OCR-based extraction

### Language Support

Specify with `--lang` or `lang` parameter:
```bash
python demo.py paper.pdf --lang zh  # Chinese
python demo.py paper.pdf --lang en  # English (default)
```

## Output Structure

### Text Blocks Format

```python
[
    {
        "type": "section",
        "text": "Abstract"
    },
    {
        "type": "content",
        "text": "This paper presents a novel approach..."
    },
    {
        "type": "section",
        "text": "Methods"
    },
    {
        "type": "content",
        "text": "We conducted experiments using..."
    }
]
```

### Final Output

```
Abstract
This paper presents a novel approach to medical knowledge extraction...

Methods
We conducted experiments using a dataset of 10,000 clinical reports...

Results
The proposed method achieved 92% accuracy on the test set...

Discussion
Our results demonstrate the effectiveness of layout-aware parsing...

Conclusion
We presented a robust document parser for medical literature...
```

## Testing

### Run Test Script

```bash
python test_bbox.py path/to/content_list.json
```

This will:
1. Process the JSON with bbox logic
2. Extract text in correct reading order
3. Filter sections (Abstract → Conclusion)
4. Save to `test_output.txt`

### Verify Output

```python
# Check if sections are properly detected
with open("output/test_output.txt") as f:
    content = f.read()
    assert "Abstract" in content
    assert "Methods" in content
    assert "Results" in content
```

## Common Use Cases

### 1. Medical Literature Review
Extract structured content from systematic reviews and meta-analyses.

### 2. Clinical Guideline Processing
Parse treatment protocols and clinical practice guidelines for knowledge graphs.

### 3. Research Paper Database
Build searchable databases of medical research with structured sections.

### 4. Knowledge Graph Construction
Feed extracted content to three-layer architecture (Bottom/Middle/Top).

## Error Handling

```python
from parser import MineruParser, MineruExecutionError

parser = MineruParser()

try:
    text_blocks = parser.parse_document("paper.pdf")
except FileNotFoundError:
    print("File not found")
except MineruExecutionError as e:
    print(f"MinerU failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Limitations

1. **MinerU Dependency**: Requires MinerU installation and proper configuration
2. **Layout Assumptions**: Optimized for academic papers with column layouts
3. **Language**: Best performance with English medical literature
4. **Section Detection**: Limited to predefined section names
5. **Complex Layouts**: May struggle with highly irregular layouts or multi-page tables

## Performance Considerations

- **Large PDFs**: Processing time scales with document length
- **OCR Mode**: Significantly slower but handles scanned documents
- **Memory**: Large documents may require substantial memory for embedding generation

## Future Enhancements

- [ ] Support for table extraction and formatting
- [ ] Image caption extraction
- [ ] Reference parsing and linking
- [ ] Multi-language section detection
- [ ] Custom section templates
- [ ] Parallel batch processing

## Contributing

When extending this module:

1. Maintain backward compatibility with existing text block format
2. Add comprehensive logging for debugging
3. Include test cases for new features
4. Update documentation

## License

See parent project license.

## Related Modules

- `src/three_layer_import.py` - Knowledge graph construction
- `src/creat_graph_with_description.py` - Entity extraction
- `src/smart_linking.py` - Graph linking strategies

## Contact

For issues or questions, please refer to the main CVDGraphRAG repository.

---

**Version**: 1.0.0  
**Last Updated**: December 2025  
**Maintainer**: CVDGraphRAG Team
