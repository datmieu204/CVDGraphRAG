import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from textbuilder.formatter import build_pubmed_txt


class TextProcessor:

    def __init__(self, output_dir="./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log = logging.getLogger(__name__)

    def process_document(self, text_blocks: List[Dict[str, str]], doc_name="document") -> Path:
        """
        Process text blocks into formatted PubMed-style output.
        
        Args:
            text_blocks: List of dicts with {"type": "section"|"content", "text": str}
            doc_name: Output file name (without extension)
        
        Returns:
            Path to output txt file
        """
        self.log.info(f"Processing {len(text_blocks)} text blocks")
        
        if not text_blocks:
            self.log.warning("No text blocks to process!")
            out = self.output_dir / f"{doc_name}.txt"
            out.write_text("", encoding="utf-8")
            return out
        
        # Group blocks by section
        sections = self._group_by_sections(text_blocks)
        
        self.log.info(f"Detected sections: {list(sections.keys())}")

        # Format to PubMed style
        try:
            formatted = build_pubmed_txt(sections)
            self.log.info(f"Formatted text: {len(formatted)} chars")
        except Exception as e:
            self.log.error(f"Formatting failed: {e}")
            # Fallback: join all content
            formatted = "\n".join(item["text"] for item in text_blocks)

        # Write output file
        out = self.output_dir / f"{doc_name}.txt"
        out.write_text(formatted, encoding="utf-8")

        self.log.info(f"TXT saved to {out}")
        
        # Verify
        if out.exists():
            self.log.info(f"File verified: {out.stat().st_size} bytes")
        
        return out

    # ===== Integrated bbox-json processing (from test_bbox.py) =====

    def process_content_list_json(self, json_path: str, doc_name: str = "document") -> Path:
        """Full pipeline: content_list.json -> text_blocks -> formatted TXT.

        This integrates the bbox parsing logic from test_bbox.py
        so you can process MinerU JSON output directly.
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"File not found: {json_path}")

        self.log.info(f"Loading JSON from: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.log.info(f"Loaded {len(data)} items")

        items = self._extract_text_with_bbox(data)
        return self.process_document(items, doc_name=doc_name)

    VALID_SECTIONS = {
        "Abstract", "Background", "Introduction",
        "Methods", "Materials And Methods",
        "Results", "Findings", "Discussion", "Conclusion",
        "Experiments", "Experiment", "Related Work",
    }

    def _extract_text_with_bbox(self, data: List[dict]) -> List[Dict[str, str]]:
        """Returns list of dicts with 'type': 'section'|'content', 'text'."""
        if not isinstance(data, list):
            return []

        pages: Dict[int, List[dict]] = {}
        for item in data:
            if not isinstance(item, dict) or item.get("type") != "text":
                continue
            page_idx = item.get("page_idx", -1)
            pages.setdefault(page_idx, []).append(item)

        self.log.info(f"Found {len(pages)} pages")

        result: List[Dict[str, str]] = []
        abstract_found = False
        conclusion_found = False

        for page_idx in sorted(pages.keys()):
            if conclusion_found:
                break
            self.log.info(f"\n--- Page {page_idx} ---")
            page_result, abstract_found, conclusion_found = self._process_page(
                pages[page_idx], abstract_found, conclusion_found
            )
            result.extend(page_result)

        return result

    def _process_page(
        self,
        items: List[dict],
        abstract_found: bool,
        conclusion_found: bool,
    ) -> Tuple[List[Dict[str, str]], bool, bool]:
        """Process one page with correct column reading order."""
        if not items:
            return [], abstract_found, conclusion_found

        bbox_items = []
        for item in items:
            bbox = item.get("bbox", [])
            if len(bbox) < 4:
                continue

            x_min, y_min, x_max, y_max = bbox
            text = self._clean_text(item.get("text", ""))
            text_level = item.get("text_level", 0)

            if not text:
                continue

            bbox_items.append(
                {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "text": text,
                    "text_level": text_level,
                    "width": x_max - x_min,
                }
            )

        if not bbox_items:
            return [], abstract_found, conclusion_found

        page_width = max(item["x_max"] for item in bbox_items)

        full_width = [item for item in bbox_items if item["width"] > page_width * 0.5]
        normal_items = [item for item in bbox_items if item["width"] <= page_width * 0.5]

        skip_full_width = False
        if not abstract_found and normal_items:
            for item in normal_items:
                if item["text_level"] == 1 and "abstract" in item["text"].lower():
                    skip_full_width = True
                    self.log.info(
                        "Detected title/author section before column-formatted Abstract - skipping full-width items"
                    )
                    break

        if skip_full_width:
            full_width = []

        if normal_items:
            x_mins = sorted(item["x_min"] for item in normal_items)
            if len(x_mins) > 1:
                gaps = []
                for i in range(len(x_mins) - 1):
                    gap = x_mins[i + 1] - x_mins[i]
                    if gap > 50:
                        gaps.append((gap, x_mins[i + 1]))

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

        self.log.info(
            f"Full-width: {len(full_width)}, Left: {len(left_column)}, Right: {len(right_column)}"
        )

        full_width.sort(key=lambda x: x["y_min"])
        left_column.sort(key=lambda x: x["y_min"])
        right_column.sort(key=lambda x: x["y_min"])

        all_items = full_width + left_column + right_column

        result: List[Dict[str, str]] = []

        for item in all_items:
            if item["text_level"] == 1 and "abstract" in item["text"].lower():
                abstract_found = True
                self.log.info("✓ Found Abstract")

            if item["text_level"] == 1 and "conclusion" in item["text"].lower():
                conclusion_found = True
                self.log.info("✓ Found Conclusion")

            if not abstract_found:
                self.log.info(f"  Skip (before Abstract): {item['text'][:40]}...")
                continue

            if conclusion_found:
                section_name = (
                    self._parse_heading(item["text"]) if item["text_level"] == 1 else None
                )
                if section_name and section_name not in ["Conclusion"]:
                    self.log.info(f"  Stop: reached {section_name} after Conclusion")
                    break

            if item["text_level"] == 1:
                if self._is_subsection_number(item["text"]):
                    self.log.info(f"  Skip subsection number: {item['text']}")
                    continue

                section_name = self._parse_heading(item["text"])
                if section_name and section_name in self.VALID_SECTIONS:
                    cleaned_header = self._clean_header(section_name)
                    result.append({"type": "section", "text": cleaned_header})
                    self.log.info(f"  + Section: {cleaned_header}")
                elif section_name:
                    self.log.info(
                        f"  Skip invalid subsection header: {item['text']}"
                    )
                else:
                    result.append({"type": "content", "text": item["text"]})
                    self.log.info(
                        f"  + Content (level=1): {item['text'][:40]}..."
                    )
            else:
                result.append({"type": "content", "text": item["text"]})
                self.log.info(f"  + Content: {item['text'][:40]}...")

        return result, abstract_found, conclusion_found

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\$+[^\$]+\$+", "", text)
        text = re.sub(r"\^[0-9,\s]+", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _clean_header(self, header: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z\s]", "", header)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _is_subsection_number(self, text: str) -> bool:
        text_stripped = text.strip()
        patterns = [
            r"^\s*\d+(\.\d+)+\s*$",
            r"^\s*\d+(\s*\.\s*\d+)+\s*$",
            r"^\s*\d+(\.\d+)+\s+\w+",
        ]
        for pattern in patterns:
            if re.match(pattern, text_stripped):
                return True
        return False

    def _parse_heading(self, text: str) -> Optional[str]:
        text_no_number = re.sub(r"^\s*\d+\s+", "", text)
        text_clean = re.sub(r"[^a-zA-Z\s]", "", text_no_number)
        text_clean = re.sub(r"\s+", " ", text_clean).strip().lower()

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

        if text_clean in section_map:
            return section_map[text_clean]

        for key, value in section_map.items():
            if key in text_clean:
                return value

        return None

    def _group_by_sections(self, blocks: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Group blocks by section.
        Returns: {section_name: content_string}
        """
        sections = {}
        current_section = None
        
        for item in blocks:
            if item["type"] == "section":
                current_section = item["text"]
                sections[current_section] = []
            elif item["type"] == "content":
                if current_section:
                    sections[current_section].append(item["text"])
        
        # Join content lines with newline (no blank lines within section)
        return {k: "\n".join(v) for k, v in sections.items() if v}