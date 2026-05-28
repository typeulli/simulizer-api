import base64
import json
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException, UploadFile, File
import groq
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from block2cpp import cppize
from cpp2block import blocklynize
import re
from pathlib import Path

from routes.compile import router as compile_router
from routes.lsp import router as lsp_router


#asdf
path_here = Path(__file__).parent
path_key_groq = path_here / "key" / "groq.key"
path_prompt = path_here / "prompt" / "codegen.txt"
groq_api_key = path_key_groq.read_text(encoding="utf-8").strip()
prompt = path_prompt.read_text(encoding="utf-8").strip()
code_simstd = (path_here / "simstd.hpp").read_text(encoding="utf-8")




OCR_MODEL = "gemma3:27b"
path_prompt_ocr = path_here / "prompt" / "ocr.txt"
OCR_PROMPT = path_prompt_ocr.read_text(encoding="utf-8").strip()
OLLAMA_URL = "http://localhost:11434/api/chat"

app = FastAPI()


client = Groq(api_key=groq_api_key)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(compile_router)
app.include_router(lsp_router)

@app.head("/health")
def health():
    return {"status": "ok"}

regxp = re.compile(r"```cpp\n([\s\S]*?)\n```", re.MULTILINE)
def stream_completion(req):
    
    result = ""
    for chunk in req:
        content = chunk.choices[0].delta.content
        if content:
            data = {"content": content}
            yield f"data: {json.dumps(data)}\n\n"
            result += content
    match_ = regxp.search(result)
    if match_:
        code = match_.group(1)
        data = {"result": blocklynize(code)}
        yield f"data: {json.dumps(data)}\n\n"
    yield "event: done\n\n"


def stream_ocr(image_bytes: bytes):
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": OCR_MODEL,
        "messages": [{"role": "user", "content": OCR_PROMPT, "images": [image_b64]}],
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
            content = chunk["message"]["content"]
            if content:
                yield f"data: {json.dumps({'content': content})}\n\n"
            if chunk.get("done"):
                break
    yield "event: done\n\n"


@app.post("/texocr")
async def texocr(file: UploadFile = File(...)):
    image_bytes = await file.read()
    return StreamingResponse(
        stream_ocr(image_bytes),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class ChatRequest(BaseModel):
    prompt: str
    blockjson: str

@app.post("/chat")
def chat(body: ChatRequest):
    code = cppize(body.blockjson)
    try:
        req = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": body.prompt},
                {"role": "user", "content": "Here is the C++ code converted from Blockly JSON:\n```cpp\n" + code + "\n```"},
            ],
            stream=True,
            max_tokens=4096,
            reasoning_effort="low"
        )
    except groq.APIStatusError as e:
        print(e)
        raise HTTPException(status_code=500, detail="Groq API error ("+str(e.status_code)+")")
        
    return StreamingResponse(
        stream_completion(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", reload=True, host="0.0.0.0", port=6000, proxy_headers=True, forwarded_allow_ips="*")
