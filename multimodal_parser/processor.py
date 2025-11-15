"""
Multimodal Content Processor
Xử lý và phân tích multimodal content, tạo output folder với enhanced descriptions
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path


class MultimodalProcessor:
    """Processor để xử lý multimodal content và tạo output"""

    def __init__(self, output_dir: str = "./output"):
        """
        Initialize processor

        Args:
            output_dir: Thư mục output để lưu kết quả
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def separate_content(
        self, content_list: List[Dict[str, Any]]
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Tách nội dung text và multimodal

        Args:
            content_list: Content list từ parser

        Returns:
            (text_content, multimodal_items): Text thuần và multimodal items
        """
        text_parts = []
        multimodal_items = []

        for item in content_list:
            content_type = item.get("type", "text")

            if content_type == "text":
                text = item.get("text", "")
                if text.strip():
                    text_parts.append(text)
            else:
                # Image, table, equation, etc.
                multimodal_items.append(item)

        text_content = "\n\n".join(text_parts)

        self.logger.info("Content separation complete:")
        self.logger.info(f"  - Text content length: {len(text_content)} characters")
        self.logger.info(f"  - Multimodal items count: {len(multimodal_items)}")

        # Count multimodal types
        modal_types = {}
        for item in multimodal_items:
            modal_type = item.get("type", "unknown")
            modal_types[modal_type] = modal_types.get(modal_type, 0) + 1

        if modal_types:
            self.logger.info(f"  - Multimodal type distribution: {modal_types}")

        return text_content, multimodal_items

    def process_document(
        self, content_list: List[Dict[str, Any]], doc_name: str = "document"
    ) -> Dict[str, Any]:
        """
        Xử lý toàn bộ document và tạo output

        Args:
            content_list: Content list từ parser
            doc_name: Tên document để đặt tên output file

        Returns:
            Dictionary chứa thông tin kết quả xử lý
        """
        self.logger.info(f"Processing document: {doc_name}")

        # Tách text và multimodal
        text_content, multimodal_items = self.separate_content(content_list)

        # Tạo output structure
        output = {
            "document_name": doc_name,
            "text_content": text_content,
            "text_length": len(text_content),
            "multimodal_items": [],
            "statistics": {
                "total_items": len(content_list),
                "text_items": len(content_list) - len(multimodal_items),
                "multimodal_items": len(multimodal_items),
            },
        }

        # Process từng multimodal item
        for idx, item in enumerate(multimodal_items):
            processed_item = self._process_multimodal_item(item, idx)
            output["multimodal_items"].append(processed_item)

        # Lưu output
        output_file = self.output_dir / f"{doc_name}_processed.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Output saved to: {output_file}")

        # Tạo markdown summary
        self._create_markdown_summary(output, doc_name)

        return output

    def _process_multimodal_item(
        self, item: Dict[str, Any], index: int
    ) -> Dict[str, Any]:
        """
        Xử lý một multimodal item

        Args:
            item: Multimodal item từ content_list
            index: Index của item

        Returns:
            Processed item với enhanced information
        """
        content_type = item.get("type", "unknown")

        processed = {
            "index": index,
            "type": content_type,
            "page_idx": item.get("page_idx", 0),
            "raw_data": item,
        }

        # Process theo từng loại
        if content_type == "image":
            processed["enhanced_info"] = self._process_image(item)
        elif content_type == "table":
            processed["enhanced_info"] = self._process_table(item)
        elif content_type == "equation":
            processed["enhanced_info"] = self._process_equation(item)
        else:
            processed["enhanced_info"] = self._process_generic(item)

        return processed

    def _process_image(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process image item"""
        return {
            "image_path": item.get("img_path", ""),
            "captions": item.get("image_caption", item.get("img_caption", [])),
            "footnotes": item.get("image_footnote", item.get("img_footnote", [])),
            "description": "Image content - requires vision model for detailed analysis",
        }

    def _process_table(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process table item"""
        return {
            "table_caption": item.get("table_caption", []),
            "table_body": item.get("table_body", ""),
            "table_footnote": item.get("table_footnote", []),
            "description": "Table content - structure and data extracted",
        }

    def _process_equation(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process equation item"""
        return {
            "equation_text": item.get("text", ""),
            "equation_format": item.get("text_format", ""),
            "description": "Mathematical equation content",
        }

    def _process_generic(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process generic content"""
        return {
            "content": str(item),
            "description": f"Generic {item.get('type', 'unknown')} content",
        }

    def _create_markdown_summary(
        self, output: Dict[str, Any], doc_name: str
    ) -> None:
        """
        Tạo markdown summary file

        Args:
            output: Output dictionary
            doc_name: Document name
        """
        md_file = self.output_dir / f"{doc_name}_summary.md"

        with open(md_file, "w", encoding="utf-8") as f:
            f.write(f"# Document Summary: {doc_name}\n\n")
            f.write("## Statistics\n\n")
            f.write(
                f"- Total items: {output['statistics']['total_items']}\n"
            )
            f.write(
                f"- Text items: {output['statistics']['text_items']}\n"
            )
            f.write(
                f"- Multimodal items: {output['statistics']['multimodal_items']}\n"
            )
            f.write(f"- Text length: {output['text_length']} characters\n\n")

            f.write("## Text Content\n\n")
            f.write(f"```\n{output['text_content'][:500]}...\n```\n\n")

            f.write("## Multimodal Items\n\n")
            for item in output["multimodal_items"]:
                f.write(f"### Item {item['index']}: {item['type']}\n\n")
                f.write(f"- Page: {item['page_idx']}\n")
                f.write(f"- Type: {item['type']}\n")

                if item["type"] == "image":
                    info = item["enhanced_info"]
                    f.write(f"- Image path: {info.get('image_path', 'N/A')}\n")
                    f.write(f"- Captions: {info.get('captions', [])}\n")
                elif item["type"] == "table":
                    info = item["enhanced_info"]
                    f.write(f"- Caption: {info.get('table_caption', [])}\n")
                elif item["type"] == "equation":
                    info = item["enhanced_info"]
                    f.write(f"- Equation: {info.get('equation_text', 'N/A')}\n")

                f.write(f"- Description: {item['enhanced_info'].get('description', 'N/A')}\n\n")

        self.logger.info(f"Markdown summary saved to: {md_file}")


def main():
    """Main function để test processor"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process multimodal documents"
    )
    parser.add_argument("file_path", help="Path to document file")
    parser.add_argument(
        "--output", "-o", default="./output", help="Output directory"
    )
    parser.add_argument(
        "--method", "-m", default="auto", choices=["auto", "txt", "ocr"],
        help="Parsing method"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Parse document
    from .parser import MineruParser

    doc_parser = MineruParser()

    if not doc_parser.check_installation():
        print("❌ MinerU is not installed. Please install it first.")
        return 1

    print(f"📄 Parsing document: {args.file_path}")
    content_list = doc_parser.parse_document(
        args.file_path, method=args.method, output_dir=args.output
    )

    print(f"✅ Parsed {len(content_list)} content blocks")

    # Process multimodal
    processor = MultimodalProcessor(output_dir=args.output)
    doc_name = Path(args.file_path).stem

    result = processor.process_document(content_list, doc_name)

    print(f"\n📊 Processing Results:")
    print(f"  - Text items: {result['statistics']['text_items']}")
    print(f"  - Multimodal items: {result['statistics']['multimodal_items']}")
    print(f"  - Output saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
