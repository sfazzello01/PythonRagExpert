import json
import sys
from pathlib import Path


def text_to_json_string(text: str) -> str:
	return json.dumps(text, ensure_ascii=False)


if __name__ == "__main__":
	base_dir = Path(__file__).parent
	input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else base_dir / "builder_input"
	output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else base_dir / "builder_output"
	text = input_path.read_text(encoding="utf-8")
	output_path.write_text(text_to_json_string(text), encoding="utf-8")
	print(f"Converted text from {input_path} to JSON string and saved to {output_path}")
