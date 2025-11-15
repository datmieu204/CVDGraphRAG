"""
Ví dụ cơ bản về cách sử dụng multimodal_parser
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import local modules
from parser import MineruParser
from processor import MultimodalProcessor
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def example_1_basic_parsing():
    """Ví dụ 1: Parse document cơ bản"""
    print("\n" + "="*60)
    print("VÍ DỤ 1: Parse document cơ bản")
    print("="*60)

    # Kiểm tra MinerU đã cài chưa
    parser = MineruParser()
    if not parser.check_installation():
        print("❌ MinerU chưa được cài đặt!")
        print("Cài đặt: pip install -U 'mineru[core]'")
        return

    # Parse một PDF file (thay đổi path này thành file thực của bạn)
    pdf_path = "path/to/your/document.pdf"

    if not Path(pdf_path).exists():
        print(f"⚠️ File không tồn tại: {pdf_path}")
        print("Vui lòng thay đổi pdf_path trong code thành file PDF thực")
        return

    print(f"📄 Đang parse: {pdf_path}")
    content_list = parser.parse_document(
        pdf_path,
        method="auto",
        output_dir="./output/example1"
    )

    print(f"✅ Đã parse {len(content_list)} content blocks")

    # Hiển thị một vài items đầu tiên
    print("\n📋 Một số content blocks đầu tiên:")
    for i, item in enumerate(content_list[:3]):
        print(f"\nBlock {i}:")
        print(f"  Type: {item.get('type', 'unknown')}")
        if item.get('type') == 'text':
            text = item.get('text', '')[:100]
            print(f"  Text: {text}...")
        elif item.get('type') == 'image':
            print(f"  Image: {item.get('img_path', 'N/A')}")
        elif item.get('type') == 'table':
            print(f"  Table: {item.get('table_caption', 'N/A')}")


def example_2_separate_content():
    """Ví dụ 2: Tách text và multimodal content"""
    print("\n" + "="*60)
    print("VÍ DỤ 2: Tách text và multimodal content")
    print("="*60)

    parser = MineruParser()
    if not parser.check_installation():
        print("❌ MinerU chưa được cài đặt!")
        return

    pdf_path = "path/to/your/document.pdf"
    if not Path(pdf_path).exists():
        print(f"⚠️ File không tồn tại: {pdf_path}")
        return

    # Parse
    content_list = parser.parse_document(pdf_path, output_dir="./output/example2")

    # Tách content
    processor = MultimodalProcessor(output_dir="./output/example2")
    text_content, multimodal_items = processor.separate_content(content_list)

    print(f"\n📝 Text content: {len(text_content)} characters")
    print(f"📊 Multimodal items: {len(multimodal_items)} items")

    # Đếm từng loại multimodal
    modal_types = {}
    for item in multimodal_items:
        modal_type = item.get('type', 'unknown')
        modal_types[modal_type] = modal_types.get(modal_type, 0) + 1

    print("\n📈 Multimodal distribution:")
    for modal_type, count in modal_types.items():
        print(f"  - {modal_type}: {count}")

    # Hiển thị một đoạn text
    print(f"\n📄 Text preview (first 200 chars):")
    print(text_content[:200] + "...")


def example_3_full_processing():
    """Ví dụ 3: Full processing với output"""
    print("\n" + "="*60)
    print("VÍ DỤ 3: Full processing và tạo output")
    print("="*60)

    parser = MineruParser()
    if not parser.check_installation():
        print("❌ MinerU chưa được cài đặt!")
        return

    pdf_path = "path/to/your/document.pdf"
    if not Path(pdf_path).exists():
        print(f"⚠️ File không tồn tại: {pdf_path}")
        return

    # Parse
    print("📄 Parsing document...")
    content_list = parser.parse_document(pdf_path, output_dir="./output/example3")

    # Process và tạo output
    print("⚙️ Processing multimodal content...")
    processor = MultimodalProcessor(output_dir="./output/example3")
    doc_name = Path(pdf_path).stem

    result = processor.process_document(content_list, doc_name)

    # In kết quả
    print("\n✅ Processing complete!")
    print(f"\n📊 Statistics:")
    print(f"  - Total items: {result['statistics']['total_items']}")
    print(f"  - Text items: {result['statistics']['text_items']}")
    print(f"  - Multimodal items: {result['statistics']['multimodal_items']}")
    print(f"  - Text length: {result['text_length']} characters")

    print(f"\n📁 Output files:")
    print(f"  - JSON: ./output/example3/{doc_name}_processed.json")
    print(f"  - Summary: ./output/example3/{doc_name}_summary.md")


def example_4_extract_images():
    """Ví dụ 4: Trích xuất chỉ images"""
    print("\n" + "="*60)
    print("VÍ DỤ 4: Trích xuất images")
    print("="*60)

    parser = MineruParser()
    if not parser.check_installation():
        print("❌ MinerU chưa được cài đặt!")
        return

    pdf_path = "path/to/your/document.pdf"
    if not Path(pdf_path).exists():
        print(f"⚠️ File không tồn tại: {pdf_path}")
        return

    # Parse
    content_list = parser.parse_document(pdf_path, output_dir="./output/example4")

    # Lọc chỉ images
    images = [item for item in content_list if item.get('type') == 'image']

    print(f"\n🖼️ Found {len(images)} images:")
    for i, img in enumerate(images):
        print(f"\nImage {i+1}:")
        print(f"  - Page: {img.get('page_idx', 'N/A')}")
        print(f"  - Path: {img.get('img_path', 'N/A')}")
        captions = img.get('image_caption', img.get('img_caption', []))
        if captions:
            print(f"  - Captions: {captions}")


def main():
    """Chạy tất cả các ví dụ"""
    print("="*60)
    print("MULTIMODAL PARSER - EXAMPLES")
    print("="*60)

    # Uncomment ví dụ nào bạn muốn chạy
    example_1_basic_parsing()
    # example_2_separate_content()
    # example_3_full_processing()
    # example_4_extract_images()


if __name__ == "__main__":
    main()
