"""
Demo script - Quick test multimodal parser
"""

import sys
from pathlib import Path

# Add current directory to path to import local modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import local modules (not as package)
from parser import MineruParser
from processor import MultimodalProcessor
import logging
import argparse


def setup_logging():
    """Setup logging với format đẹp"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )


def main():
    """Main function"""
    arg_parser = argparse.ArgumentParser(
        description="Multimodal Document Parser - Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python demo.py --check
  python demo.py document.pdf
  python demo.py document.pdf --output ./my_output
  python demo.py document.pdf --method ocr
        """
    )

    arg_parser.add_argument(
        "file_path",
        nargs="?",  # Make file_path optional
        help="Path to PDF document to parse"
    )
    arg_parser.add_argument(
        "--output", "-o",
        default="./output",
        help="Output directory (default: ./output)"
    )
    arg_parser.add_argument(
        "--method", "-m",
        default="auto",
        choices=["auto", "txt", "ocr"],
        help="Parsing method (default: auto)"
    )
    arg_parser.add_argument(
        "--lang", "-l",
        default=None,
        help="Document language for OCR (e.g., ch, en, ja)"
    )
    arg_parser.add_argument(
        "--check",
        action="store_true",
        help="Only check MinerU installation"
    )

    args = arg_parser.parse_args()

    setup_logging()

    # Banner
    print("\n" + "="*60)
    print("🚀 MULTIMODAL DOCUMENT PARSER")
    print("="*60 + "\n")

    # Check MinerU installation
    doc_parser = MineruParser()
    if not doc_parser.check_installation():
        print("❌ MinerU is not installed!")
        print("\nTo install MinerU:")
        print("  pip install -U 'mineru[core]'")
        print("  # or")
        print("  uv pip install -U 'mineru[core]'")
        return 1

    print("✅ MinerU is installed")

    if args.check:
        print("\nInstallation check complete!")
        return 0

    # Validate file_path is provided when not just checking
    if not args.file_path:
        print("\n❌ Error: file_path is required (unless using --check)")
        print("\nUsage: python demo.py <file_path> [options]")
        print("       python demo.py --check")
        return 1

    # Check file exists
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"\n❌ File not found: {file_path}")
        return 1

    print(f"\n📄 Input file: {file_path}")
    print(f"📁 Output directory: {args.output}")
    print(f"⚙️ Parsing method: {args.method}")
    if args.lang:
        print(f"🌐 Language: {args.lang}")

    # Step 1: Parse document
    print("\n" + "-"*60)
    print("STEP 1: Parsing document")
    print("-"*60)

    try:
        content_list = doc_parser.parse_document(
            file_path,
            method=args.method,
            output_dir=args.output,
            lang=args.lang
        )
        print(f"✅ Parsed {len(content_list)} content blocks")
    except Exception as e:
        print(f"❌ Parse failed: {e}")
        return 1

    # Step 2: Process multimodal content
    print("\n" + "-"*60)
    print("STEP 2: Processing multimodal content")
    print("-"*60)

    processor = MultimodalProcessor(output_dir=args.output)
    doc_name = file_path.stem

    try:
        result = processor.process_document(content_list, doc_name)

        print(f"\n✅ Processing complete!")
        print(f"\n📊 Statistics:")
        print(f"  • Total items: {result['statistics']['total_items']}")
        print(f"  • Text items: {result['statistics']['text_items']}")
        print(f"  • Multimodal items: {result['statistics']['multimodal_items']}")
        print(f"  • Text length: {result['text_length']:,} characters")

        # Count multimodal types
        modal_types = {}
        for item in result['multimodal_items']:
            modal_type = item['type']
            modal_types[modal_type] = modal_types.get(modal_type, 0) + 1

        if modal_types:
            print(f"\n🎨 Multimodal distribution:")
            for modal_type, count in sorted(modal_types.items()):
                print(f"  • {modal_type}: {count}")

        print(f"\n📁 Output files:")
        print(f"  • JSON: {args.output}/{doc_name}_processed.json")
        print(f"  • Summary: {args.output}/{doc_name}_summary.md")
        print(f"  • MinerU output: {args.output}/mineru_output/")

        print("\n" + "="*60)
        print("✅ ALL DONE!")
        print("="*60 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
