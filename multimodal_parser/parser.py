"""
parser.py

Layout-aware MinerU parser using proven logic from test script.
- Detects full-width items (title/authors) vs column items
- Reads in order: full-width → left column → right column
- Skips content before Abstract, stops after Conclusion
- Filters subsection numbers and invalid sections
- Returns List[Dict[str, str]] with {"type": "section"|"content", "text": str}
"""

from __future__ import annotations
import json
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple
from collections import defaultdict
import re

logging.getLogger(__name__)


class MineruExecutionError(Exception):
    pass


class Parser:
    OFFICE_FORMATS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
    TEXT_FORMATS = {".txt", ".md"}

    def convert_office_to_pdf(self, doc_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None) -> Path:
        doc_path = Path(doc_path)
        if not doc_path.exists():
            raise FileNotFoundError(doc_path)

        output_dir = Path(output_dir) if output_dir else doc_path.parent / "libreoffice_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            result = subprocess.run(
                ["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(tmp), str(doc_path)],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                raise MineruExecutionError(result.returncode, result.stderr)

            pdf_files = list(tmp.glob("*.pdf"))
            if not pdf_files:
                raise RuntimeError("LibreOffice conversion failed (no PDF produced)")

            out = output_dir / f"{doc_path.stem}.pdf"
            pdf_files[0].replace(out)
            return out

    def parse_document(self, *args, **kwargs) -> List[Dict[str, str]]:
        raise NotImplementedError


class MineruParser(Parser):
    
    VALID_SECTIONS = {
        "Abstract", "Background", "Introduction", 
        "Methods", "Materials And Methods",
        "Results", "Findings", "Discussion", "Conclusion",
        "Experiments", "Experiment", "Related Work"
    }

    def _run_mineru(self, input_path: Path, output_dir: Path, method: str = "auto", lang: Optional[str] = None):
        cmd = ["mineru", "-p", str(input_path), "-o", str(output_dir), "-m", method]
        if lang:
            cmd.extend(["-l", lang])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise MineruExecutionError(result.returncode, result.stderr)

    def clean_text(self, text: str) -> str:
        """Remove formulas and normalize whitespace"""
        if not text:
            return ""
        # Remove LaTeX formulas
        text = re.sub(r'\$+[^\$]+\$+', '', text)
        # Remove superscript citations
        text = re.sub(r'\^[0-9,\s]+', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def clean_header(self, header: str) -> str:
        """Clean section header: remove all non-letter characters except spaces"""
        cleaned = re.sub(r'[^a-zA-Z\s]', '', header)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()
    
    def is_subsection_number(self, text: str) -> bool:
        """
        Check if text is a subsection number like "2.1", "3.2.1", "2.1.1", etc.
        Does NOT match main section numbers like "4 Related Work"
        """
        text_stripped = text.strip()
        
        patterns = [
            r'^\s*\d+(\.\d+)+\s*$',           # Exact: 2.1, 2.1.1
            r'^\s*\d+(\s*\.\s*\d+)+\s*$',     # With spaces: 2 . 1
            r'^\s*\d+(\.\d+)+\s+\w+',         # With text: 2.1 Triple
        ]
        
        for pattern in patterns:
            if re.match(pattern, text_stripped):
                return True
        
        return False
    
    def parse_heading(self, text: str) -> Optional[str]:
        """Parse heading text to identify section name - handles numbered sections"""
        # Remove leading numbers (like "4 Related Work" -> "Related Work")
        text_no_number = re.sub(r'^\s*\d+\s+', '', text)
        
        # Remove all non-letters except spaces
        text_clean = re.sub(r'[^a-zA-Z\s]', '', text_no_number)
        text_clean = re.sub(r'\s+', ' ', text_clean).strip().lower()
        
        section_map = {
            "abstract": "Abstract",
            "background": "Background",
            "introduction": "Introduction",
            "method": "Methods",
            "methods": "Methods",
            "materials and methods": "Methods",
            "results": "Results",
            "findings": "Findings",
            "discussion": "Discussion",
            "conclusion": "Conclusion",
            "experiment": "Experiments",
            "experiments": "Experiments",
            "related work": "Related Work",
        }
        
        # Exact match
        if text_clean in section_map:
            return section_map[text_clean]
        
        # Partial match
        for key, value in section_map.items():
            if key in text_clean:
                return value
        
        return None

    def process_page(self, items: List[dict], abstract_found: bool, conclusion_found: bool) -> Tuple[List[Dict[str, str]], bool, bool]:
        """
        Process page with correct column reading order.
        Returns list of dicts with type and text.
        """
        if not items:
            return [], abstract_found, conclusion_found
        
        # Parse items
        bbox_items = []
        for item in items:
            bbox = item.get("bbox", [])
            if len(bbox) < 4:
                continue
            
            x_min, y_min, x_max, y_max = bbox
            text = self.clean_text(item.get("text", ""))
            text_level = item.get("text_level", 0)
            
            if not text:
                continue
            
            bbox_items.append({
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "text": text,
                "text_level": text_level,
                "width": x_max - x_min
            })
        
        if not bbox_items:
            return [], abstract_found, conclusion_found
        
        # Find page width
        page_width = max(item["x_max"] for item in bbox_items)
        
        # Separate by width - items > 50% are full-width
        full_width = [item for item in bbox_items if item["width"] > page_width * 0.5]
        normal_items = [item for item in bbox_items if item["width"] <= page_width * 0.5]
        
        # Check if full-width items are before Abstract (like title, authors)
        skip_full_width = False
        if not abstract_found and normal_items:
            for item in normal_items:
                if item["text_level"] == 1 and "abstract" in item["text"].lower():
                    skip_full_width = True
                    logging.info("Detected title/author section before column-formatted Abstract - skipping full-width items")
                    break
        
        if skip_full_width:
            full_width = []
        
        # Separate columns
        if normal_items:
            x_mins = sorted([item["x_min"] for item in normal_items])
            
            if len(x_mins) > 1:
                gaps = []
                for i in range(len(x_mins) - 1):
                    gap = x_mins[i+1] - x_mins[i]
                    if gap > 50:
                        gaps.append((gap, x_mins[i+1]))
                
                if gaps:
                    gaps.sort(reverse=True)
                    x_threshold = gaps[0][1]
                else:
                    x_threshold = x_mins[len(x_mins) // 2]
            else:
                x_threshold = page_width * 0.5
            
            left_column = [item for item in normal_items if item["x_min"] < x_threshold]
            right_column = [item for item in normal_items if item["x_min"] >= x_threshold]
        else:
            left_column = []
            right_column = []
        
        logging.info(f"Full-width: {len(full_width)}, Left: {len(left_column)}, Right: {len(right_column)}")
        
        # Sort each column by y_min
        full_width.sort(key=lambda x: x["y_min"])
        left_column.sort(key=lambda x: x["y_min"])
        right_column.sort(key=lambda x: x["y_min"])
        
        # CORRECT ORDER: full-width, then left column, then right column
        all_items = full_width + left_column + right_column
        
        # Process in order
        result = []
        
        for item in all_items:
            # Check for Abstract
            if item["text_level"] == 1 and "abstract" in item["text"].lower():
                abstract_found = True
                logging.info(f"✓ Found Abstract")
            
            # Check for Conclusion
            if item["text_level"] == 1 and "conclusion" in item["text"].lower():
                conclusion_found = True
                logging.info(f"✓ Found Conclusion")
            
            # Skip before Abstract
            if not abstract_found:
                logging.info(f"  Skip (before Abstract): {item['text'][:40]}...")
                continue
            
            # Stop after Conclusion (except Conclusion itself)
            if conclusion_found:
                section_name = self.parse_heading(item["text"]) if item["text_level"] == 1 else None
                if section_name and section_name not in ["Conclusion"]:
                    logging.info(f"  Stop: reached {section_name} after Conclusion")
                    break
            
            # Process item
            if item["text_level"] == 1:
                # Skip subsection numbers
                if self.is_subsection_number(item["text"]):
                    logging.info(f"  Skip subsection number: {item['text']}")
                    continue
                
                section_name = self.parse_heading(item["text"])
                if section_name and section_name in self.VALID_SECTIONS:
                    cleaned_header = self.clean_header(section_name)
                    result.append({
                        "type": "section",
                        "text": cleaned_header
                    })
                    logging.info(f"  + Section: {cleaned_header}")
                elif section_name:
                    logging.info(f"  Skip invalid subsection header: {item['text']}")
                else:
                    result.append({
                        "type": "content",
                        "text": item["text"]
                    })
                    logging.info(f"  + Content (level=1): {item['text'][:40]}...")
            else:
                result.append({
                    "type": "content",
                    "text": item["text"]
                })
                logging.info(f"  + Content: {item['text'][:40]}...")
        
        return result, abstract_found, conclusion_found

    def parse_pdf(self, pdf_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None, method: str = "auto", lang: Optional[str] = None) -> List[Dict[str, str]]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

        base_out = Path(output_dir) if output_dir else pdf_path.parent / "mineru_output"
        base_out.mkdir(parents=True, exist_ok=True)

        # Run MinerU
        self._run_mineru(pdf_path, base_out, method, lang)

        # Find content_list.json
        possible_paths = [
            base_out / pdf_path.stem / method / f"{pdf_path.stem}_content_list.json",
            base_out / pdf_path.stem / f"{pdf_path.stem}_content_list.json",
            base_out / f"{pdf_path.stem}_content_list.json",
        ]
        
        contents_json = None
        for path in possible_paths:
            if path.exists():
                contents_json = path
                logging.info(f"Found content_list.json at: {path}")
                break
        
        if not contents_json:
            raise FileNotFoundError(
                f"content_list.json not found. Checked:\n" +
                "\n".join(f"  - {p}" for p in possible_paths)
            )

        # Load JSON
        with open(contents_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logging.info(f"Loaded {len(data)} items from JSON")

        # Group by page
        pages = {}
        for item in data:
            if not isinstance(item, dict) or item.get("type") != "text":
                continue
            page_idx = item.get("page_idx", -1)
            if page_idx not in pages:
                pages[page_idx] = []
            pages[page_idx].append(item)
        
        logging.info(f"Found {len(pages)} pages")
        
        result = []
        abstract_found = False
        conclusion_found = False
        
        for page_idx in sorted(pages.keys()):
            if conclusion_found:
                break
            logging.info(f"\n--- Page {page_idx} ---")
            page_result, abstract_found, conclusion_found = self.process_page(
                pages[page_idx], abstract_found, conclusion_found
            )
            result.extend(page_result)
        
        if not result:
            logging.warning(f"No text extracted from {pdf_path.name}")
            return [{"type": "content", "text": "[No text content found]"}]
        
        return result

    def parse_document(self, file_path: Union[str, Path], method: str = "auto", output_dir: Optional[Union[str, Path]] = None, lang: Optional[str] = None, **kwargs) -> List[Dict[str, str]]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        ext = file_path.suffix.lower()

        if ext in self.TEXT_FORMATS:
            content = file_path.read_text(encoding="utf-8").strip()
            return [{"type": "content", "text": content}]

        if ext in self.OFFICE_FORMATS:
            pdf = self.convert_office_to_pdf(file_path, output_dir)
            return self.parse_pdf(pdf, output_dir, method, lang)

        return self.parse_pdf(file_path, output_dir, method, lang)

    def check_installation(self) -> bool:
        try:
            subprocess.run(["mineru", "--version"], capture_output=True, text=True, check=True)
            return True
        except Exception:
            return False