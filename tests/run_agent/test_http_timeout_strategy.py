"""Integration coverage for the OpenAI-wire timeout split."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
import time

from openai import APITimeoutError, OpenAI
import pytest

from agent.process_bootstrap import build_keepalive_http_client


class _DelayedSSEHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(content_length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()

        for content in ("first", "second"):
            payload = {
                "id": "chatcmpl-timeout-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": None,
                    }
                ],
            }
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.1)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format, *_args):
        return


class _SilentResponseHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(content_length)
        body = b'{"choices": [{"message": {"content": "late"}}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        time.sleep(0.2)
        try:
            self.wfile.write(body)
        except OSError:
            pass

    def log_message(self, format, *_args):
        return


def _openai_client_for(server):
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    return OpenAI(
        api_key="test-key",
        base_url=base_url,
        http_client=build_keepalive_http_client(base_url),
        max_retries=0,
    )


def test_nonstreaming_request_timeout_breaks_silent_response():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SilentResponseHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = _openai_client_for(server)

    try:
        with pytest.raises(APITimeoutError):
            client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "wait"}],
                timeout=0.05,
            )
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def test_keepalive_client_preserves_delayed_sse_streaming():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DelayedSSEHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = _openai_client_for(server)

    try:
        stream = client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "stream"}],
            stream=True,
        )
        content = [chunk.choices[0].delta.content for chunk in stream]
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    assert content == ["first", "second"]