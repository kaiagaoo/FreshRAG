"""Quick sanity check for GOOGLE_API_KEY against Gemini."""
import os
import sys
import requests


def load_key():
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("GOOGLE_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("'\"")
    return None


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "gemini-2.5-flash"
    key = load_key()
    if not key:
        print("FAIL: no GOOGLE_API_KEY in env or .env")
        sys.exit(1)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {"contents": [{"parts": [{"text": "Reply with exactly: pong"}]}]}
    r = requests.post(url, json=payload, timeout=30)

    print(f"Status: {r.status_code}")
    if r.status_code != 200:
        print(r.text[:500])
        sys.exit(1)

    data = r.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        text = "<no text>"
    usage = data.get("usageMetadata", {})
    print(f"Response: {text}")
    print(f"Tokens: in={usage.get('promptTokenCount')} out={usage.get('candidatesTokenCount')}")
    print("OK")


if __name__ == "__main__":
    main()
