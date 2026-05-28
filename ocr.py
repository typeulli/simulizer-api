from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen


MODEL_NAME = "gemma3:27b"
PROMPT = """
Write the tex formula of the following image.
Never response any description or explanation such as 'Here is the tex formula:' or 'Here's the TeX formula for the image:'.
You must write the image itself, without modification.
For example, if the image has 'add', then you shouldn't write '+' instead.
You must put text in form of $tex$ such as $y=x^2$ with only one line, not code block any else format.
""".strip()
OLLAMA_URL = "http://localhost:11434/api/chat"
IMAGE_PATH = Path(__file__).with_name("image.png")


def main() -> None:
	if not IMAGE_PATH.exists():
		raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {IMAGE_PATH}")

	image_b64 = base64.b64encode(IMAGE_PATH.read_bytes()).decode("utf-8")

	payload = {
		"model": MODEL_NAME,
		"messages": [
			{
				"role": "user",
				"content": PROMPT,
				"images": [image_b64],
			}
		],
		"stream": True,
	}

	request = Request(
		OLLAMA_URL,
		data=json.dumps(payload).encode("utf-8"),
		headers={"Content-Type": "application/json"},
		method="POST",
	)

	with urlopen(request) as response:
		for line in response:
			chunk = json.loads(line.decode("utf-8"))
			print(chunk["message"]["content"], end="", flush=True)
			if chunk.get("done"):
				break

	print()


if __name__ == "__main__":
	main()
