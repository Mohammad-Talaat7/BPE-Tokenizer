from typing import Iterable, Iterator, List, Dict, Tuple, Optional
import ast
import json
import regex as re
import logging

# Set up logging
# logging.basicConfig(
#    filename="bpe_tokenizer.log",
#    filemode="w",
#    level=logging.INFO,
#    format="%(asctime)s - %(levelname)s - %(message)s",
#)
logger = logging.getLogger(__name__)


class tokenizer:
    """
    A Byte-Pair Encoding (BPE) tokenizer implementation.

    This tokenizer supports:
    - Special token handling
    - Efficient encoding/decoding
    - Memory-efficient processing of large files
    - Robust error handling
    """

    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ):
        """
        Initialize the tokenizer.

        Args:
            vocab: Dictionary mapping token IDs to byte sequences
            merges: List of byte pair merges in order of application
            special_tokens: Optional list of special tokens to preserve
        """
        self.vocab = vocab.copy()
        self.merges = merges.copy()
        self.special_tokens = special_tokens or []
        self.special_tokens_set = set(self.special_tokens)

        # Add special tokens to vocabulary if not present
        next_id = max(self.vocab.keys()) + 1 if self.vocab else 0
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in self.vocab.values():
                self.vocab[next_id] = token_bytes
                next_id += 1

        # Create reverse mapping for efficient encoding
        self._bytes_to_id = {v: k for k, v in self.vocab.items()}

        # Pre-compile regex patterns for efficiency
        self._setup_patterns()

        logger.info(
            f"Tokenizer initialized with vocab size: {len(self.vocab)}"
        )

    def _setup_patterns(self):
        """Pre-compile regex patterns for better performance."""
        # GPT style pre-tokenization pattern
        self._pretokenize_pattern = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        # Special token pattern (if any special tokens exist)
        if self.special_tokens:
            # Sort by length (longest first) to avoid partial matches
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped_tokens = [re.escape(token) for token in sorted_tokens]
            pattern = "(" + "|".join(escaped_tokens) + ")"
            self._special_token_pattern = re.compile(pattern)
        else:
            self._special_token_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
    ) -> "tokenizer":
        """
        Load tokenizer from saved files.

        Args:
            vocab_filepath: Path to vocabulary JSON file
            merges_filepath: Path to merges text file
            special_tokens: Optional list of special tokens

        Returns:
            Initialized BPETokenizer instance
        """
        try:
            # Load merges
            with open(merges_filepath, "r", encoding="utf-8") as f:
                merges = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        merge_pair = ast.literal_eval(line)
                        if (
                            not isinstance(merge_pair, tuple)
                            or len(merge_pair) != 2
                        ):
                            raise ValueError(
                                f"Invalid merge format at line {line_num}"
                            )
                        merges.append(merge_pair)
                    except (ValueError, SyntaxError) as e:
                        logger.warning(
                            f"Skipping invalid merge at line {line_num}: {e}"
                        )

            # Load vocabulary
            with open(vocab_filepath, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)

            vocab = {}
            for key, value in vocab_data.items():
                try:
                    vocab_id = int(key)
                    if isinstance(value, str):
                        # Handle string representation of bytes
                        if value.startswith("b'") or value.startswith('b"'):
                            vocab_bytes = ast.literal_eval(value)
                        else:
                            vocab_bytes = value.encode("utf-8")
                    else:
                        vocab_bytes = ast.literal_eval(str(value))
                    vocab[vocab_id] = vocab_bytes
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"Skipping invalid vocab entry {key}: {e}")

            return cls(
                vocab=vocab, merges=merges, special_tokens=special_tokens
            )

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not load tokenizer files: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading tokenizer: {e}")

    def _split_by_special_tokens(self, text: str) -> List[str]:
        """
        Split text by special tokens while preserving them.

        Args:
            text: Input text to split

        Returns:
            List of text segments and special tokens
        """
        if not self._special_token_pattern:
            return [text]

        return [
            part for part in self._special_token_pattern.split(text) if part
        ]

    def _apply_merges_to_token(self, token_bytes: bytes) -> Tuple[bytes, ...]:
        """
        Apply BPE merges to a pre-tokenized segment.

        Args:
            token_bytes: Bytes of the token to process

        Returns:
            Tuple of merged byte sequences
        """
        if not token_bytes:
            return tuple()

        # Start with individual bytes
        current_tokens = tuple(bytes([b]) for b in token_bytes)

        # Apply each merge in order
        for merge_pair in self.merges:
            if len(current_tokens) < 2:
                break

            new_tokens = []
            i = 0

            while i < len(current_tokens):
                # Check if we can merge at this position
                if (
                    i < len(current_tokens) - 1
                    and (current_tokens[i], current_tokens[i + 1])
                    == merge_pair
                ):
                    # Merge the pair
                    merged = current_tokens[i] + current_tokens[i + 1]
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(current_tokens[i])
                    i += 1

            current_tokens = tuple(new_tokens)

            # Early termination if no more merges possible
            if len(current_tokens) == 1:
                break

        return current_tokens

    def encode(self, text: str) -> List[int]:
        """
        Encode text into a sequence of token IDs.

        Args:
            text: Input text to encode

        Returns:
            List of token IDs
        """
        if not text:
            return []

        # Split by special tokens
        text_segments = self._split_by_special_tokens(text)

        token_ids = []

        for segment in text_segments:
            if not segment:
                continue

            if segment in self.special_tokens_set:
                # Handle special tokens
                special_token_bytes = segment.encode("utf-8")
                token_id = self._bytes_to_id.get(special_token_bytes)
                if token_id is not None:
                    token_ids.append(token_id)
                else:
                    logger.warning(
                        f"Special token '{segment}' not found in vocabulary"
                    )
            else:
                # Apply pre-tokenization pattern
                for match in self._pretokenize_pattern.finditer(segment):
                    pre_token = match.group(0)
                    pre_token_bytes = pre_token.encode("utf-8")

                    # Apply BPE merges
                    merged_tokens = self._apply_merges_to_token(
                        pre_token_bytes
                    )

                    # Convert to IDs
                    for token_bytes in merged_tokens:
                        token_id = self._bytes_to_id.get(token_bytes)
                        if token_id is not None:
                            token_ids.append(token_id)
                        else:
                            # Handle unknown tokens by encoding as individual bytes
                            logger.warning(f"Unknown token: {token_bytes}")
                            for byte_val in token_bytes:
                                byte_token = bytes([byte_val])
                                byte_id = self._bytes_to_id.get(byte_token)
                                if byte_id is not None:
                                    token_ids.append(byte_id)

        return token_ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encode multiple texts efficiently.

        Args:
            texts: List of input texts

        Returns:
            List of token ID sequences
        """
        return [self.encode(text) for text in texts]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encode an iterable of strings.

        Args:
            iterable: Iterable of strings (e.g., file handle)

        Yields:
            Token IDs one at a time
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Decoded text string
        """
        if not token_ids:
            return ""

        # Collect all byte sequences
        byte_sequences = []
        for token_id in token_ids:
            token_bytes = self.vocab.get(token_id)
            if token_bytes is not None:
                byte_sequences.append(token_bytes)
            else:
                logger.warning(f"Unknown token ID: {token_id}")

        # Concatenate and decode
        try:
            full_bytes = b"".join(byte_sequences)
            return full_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Decoding error: {e}")
            return ""

    def decode_batch(self, token_id_lists: List[List[int]]) -> List[str]:
        """
        Decode multiple token ID sequences.

        Args:
            token_id_lists: List of token ID sequences

        Returns:
            List of decoded strings
        """
        return [self.decode(token_ids) for token_ids in token_id_lists]

    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return len(self.vocab)

    def get_special_tokens(self) -> List[str]:
        """Get the list of special tokens."""
        return self.special_tokens.copy()

    def save(self, output_dir: str):
        """
        Save the tokenizer to files.

        Args:
            output_dir: Directory to save tokenizer files
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        # Save merges
        merges_path = os.path.join(output_dir, "merges.txt")
        with open(merges_path, "w", encoding="utf-8") as f:
            for merge_pair in self.merges:
                f.write(f"{repr(merge_pair)}\n")

        # Save vocabulary
        vocab_path = os.path.join(output_dir, "vocab.json")
        vocab_serializable = {}
        for token_id, token_bytes in self.vocab.items():
            vocab_serializable[str(token_id)] = repr(token_bytes)

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_serializable, f, indent=2)

        logger.info(f"Tokenizer saved to {output_dir}")

    def __repr__(self) -> str:
        return (
            f"BPETokenizer(vocab_size={len(self.vocab)}, "
            f"num_merges={len(self.merges)}, "
            f"special_tokens={len(self.special_tokens)})"
        )

def test_tokenizer():
    """Test function to verify tokenizer functionality."""
    # Create a simple test tokenizer
    vocab = {
        0: b"<|endoftext|>",
        1: b" ",
        2: b"a",
        3: b"c",
        4: b"e",
        5: b"h",
        6: b"t",
        7: b"th",
        8: b" c",
        9: b" a",
        10: b"the",
        11: b" at",
    }

    merges = [
        (b"t", b"h"),
        (b" ", b"c"),
        (b" ", b"a"),
        (b"th", b"e"),
        (b" a", b"t"),
    ]

    special_tokens = ["<|endoftext|>"]

    # Initialize tokenizer
    tok = tokenizer(
        vocab=vocab, merges=merges, special_tokens=special_tokens
    )

    # Test encoding and decoding
    test_texts = [
        "the cat ate",
        "Hello <|endoftext|> world",
        "special <|endoftext|><|endoftext|> tokens",
    ]

    for text in test_texts:
        encoded = tok.encode(text)
        decoded = tok.decode(encoded)
        print(f"Original: {text}")
        print(f"Encoded:  {encoded}")
        print(f"Decoded:  {decoded}")
        print(f"Match:    {text == decoded}")
        print("-" * 50)


if __name__ == "__main__":
    # Load trained tokenizer
    bpe_train_path = "outputs/"

    try:
        tokenizer_bpe = tokenizer.from_files(
            vocab_filepath=f"{bpe_train_path}vocab.json",
            merges_filepath=f"{bpe_train_path}merges.txt",
            special_tokens=[
                "<|endoftext|>",
                "heyyyy",
                "<|endoftext|><|endoftext|>",
            ],
        )

        # Test the tokenizer
        test_inputs = [
            "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>",
            "heyyyy, Here is some text i'd like to encode <|endoftext|>",
            "the cat ate",
        ]

        for input_text in test_inputs:
            print(f"Input: {input_text}")

            encode_ids = tokenizer_bpe.encode(input_text)
            print(f"Encoded: {encode_ids}")

            output_text = tokenizer_bpe.decode(encode_ids)
            print(f"Decoded: {output_text}")
            print(f"Roundtrip successful: {input_text == output_text}")
            print("-" * 60)

    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Running basic test instead...")
        test_tokenizer()
