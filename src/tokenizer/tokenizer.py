import tiktoken


class GPTTokenizer:
    def __init__(self, encoding_name: str = "gpt2"):
        """Create a GPT tokenizer wrapper around tiktoken.

        Args:
            encoding_name: The tiktoken encoding name to use, e.g. "gpt2", "cl100k_base".
        """
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as exc:
            raise ValueError(
                f"Unable to initialize tiktoken encoding '{encoding_name}'. "
                "Verify the encoding name is supported by your installed tiktoken version."
            ) from exc

        self.encoding_name = encoding_name

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        return self.encoding.encode(text)

    def decode(self, ids) -> str:
        """Decode token ids back into text."""
        if not isinstance(ids, (list, tuple)):
            raise TypeError("ids must be a list or tuple of integers")

        return self.encoding.decode(ids)
