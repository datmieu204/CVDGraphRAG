import sys
from pathlib import Path

# Allow local imports without installing package
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from parser import MineruParser
from processor import TextProcessor
import logging
import argparse


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )


def main():
    arg_parser = argparse.ArgumentParser(
        description="Text-only Document Parser Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    arg_parser.add_argument("file_path", nargs="?", help="Path to input document")
    arg_parser.add_argument("--output", "-o", default="./output")
    arg_parser.add_argument("--method", "-m", default="auto", choices=["auto", "txt", "ocr"])
    arg_parser.add_argument("--lang", "-l", default=None)
    arg_parser.add_argument("--check", action="store_true")

    args = arg_parser.parse_args()
    setup_logging()

    print("\n" + "="*60)
    print(" TEXT-ONLY DOCUMENT PARSER")
    print("="*60)

    # Step 0 - MinerU check
    parser = MineruParser()
    if not parser.check_installation():
        print("MinerU is not installed")
        print("Run: pip install -U 'mineru[core]'")
        return 1

    print("MinerU detected")

    if args.check:
        print("Done.\n")
        return 0

    # Validate input
    if not args.file_path:
        print("Missing input file. Usage: python demo.py <file>")
        return 1

    input_path = Path(args.file_path)
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return 1

    print(f"\nInput: {input_path}")
    print(f"Output: {args.output}")
    print(f"Method: {args.method}")
    if args.lang:
        print(f"Lang: {args.lang}")

    # Step 1 — Run MinerU to generate JSON
    print("\n----- STEP 1: RUNNING MINERU -----")
    try:
        # MinerU generates content_list.json at specific path
        base_out = Path(args.output)
        base_out.mkdir(parents=True, exist_ok=True)
        
        # Run mineru
        import subprocess
        cmd = ["mineru", "-p", str(input_path), "-o", str(base_out), "-m", args.method]
        if args.lang:
            cmd.extend(["-l", args.lang])
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] MinerU failed: {result.stderr}")
            return 1
        
        print(f"[OK] MinerU completed")
        
        # Find content_list.json
        possible_paths = [
            base_out / input_path.stem / args.method / f"{input_path.stem}_content_list.json",
            base_out / input_path.stem / f"{input_path.stem}_content_list.json",
            base_out / f"{input_path.stem}_content_list.json",
        ]
        
        content_list_json = None
        for path in possible_paths:
            if path.exists():
                content_list_json = path
                print(f"Found content_list.json at: {path}")
                break
        
        if not content_list_json:
            print(f"[ERROR] content_list.json not found. Checked:")
            for p in possible_paths:
                print(f"  - {p}")
            return 1

    except Exception as e:
        print(f"[ERROR] MinerU execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 2 — Process JSON with bbox logic → formatted TXT
    print("\n----- STEP 2: PROCESSING WITH BBOX LOGIC -----")
    processor = TextProcessor(output_dir=args.output)
    
    try:
        txt_path = processor.process_content_list_json(str(content_list_json), input_path.stem)
        print(f"[OK] Output saved at: {txt_path}")
        
        # Verify file exists and show stats
        if txt_path.exists():
            # Show first few lines
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read(500)
                print("\n--- OUTPUT PREVIEW ---")
                print(content)
                if len(content) == 500:
                    print("...\n")
        else:
            print(f"[ERROR] Output file not created: {txt_path}")
            return 1
            
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())