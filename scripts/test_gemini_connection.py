"""
Quick sanity check that the Gemini API key and SDK are working correctly
before we build anything real on top of it.
"""

import os
from google import genai

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set. "
            "Run: $env:GEMINI_API_KEY = 'your-key-here'"
        )

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents="Reply with exactly one word: hello",
    )

    print("Connection successful!")
    print(f"Model response: {response.text}")


if __name__ == "__main__":
    main()