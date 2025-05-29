import regex as re
import argparse
import time
import os
from typing import BinaryIO, List, Dict, Tuple, Optional
import multiprocessing
import json
from collections import defaultdict, Counter
import logging
import pickle
import ast

# Set up logging
# logging.basicConfig(
#    filename="training_tokenizer.log",
#    filemode="w",
#    level=logging.INFO,
#    format="%(asctime)s - %(levelname)s - %(message)s",
# )
logger = logging.getLogger(__name__)


def find_chunk_boundaries(
    file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
) -> List[int]:
    """
    Chunk the file into parts that can be counted independently.

    Args:
        file: Binary file handle
        desired_num_chunks: Target number of chunks
        split_special_token: Token to split on (as bytes)

    Returns:
        List of chunk boundary positions (may be fewer than desired_num_chunks)
    """
    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size == 0:
        return [0]

    chunk_size = file_size // desired_num_chunks
    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    # Initial guesses for chunk boundary locations, uniformly spaced
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    # Adjust boundaries to align with special tokens
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)

            # If EOF, this boundary should be at the end of the file
            if not mini_chunk:
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Ensure all boundaries are unique and sorted
    return sorted(set(chunk_boundaries))


def pretokenize_chunk(
    start: int, end: int, input_path: str, special_tokens: List[str]
) -> Dict[Tuple[bytes, ...], int]:
    """
    Pre-tokenize a chunk of the file and return frequency counts.

    Args:
        start: Start byte position
        end: End byte position
        input_path: Path to input file
        special_tokens: List of special tokens to handle

    Returns:
        Dictionary mapping token tuples to their frequencies
    """
    freq = Counter()
    # GPT style pre-tokenization pattern
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    try:
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            # Split by special tokens first
            if special_tokens:
                # Escape special regex characters and sort by length (longest first)
                escaped_tokens = [
                    re.escape(token)
                    for token in sorted(special_tokens, key=len, reverse=True)
                ]
                special_token_pattern = "|".join(escaped_tokens)
                text_splits = re.split(f"({special_token_pattern})", chunk)
            else:
                text_splits = [chunk]

            # Process each split
            for text_part in text_splits:
                if text_part in special_tokens:
                    # Handle special tokens as single units
                    token_bytes = tuple(
                        bytes([b]) for b in text_part.encode("utf-8")
                    )
                    freq[token_bytes] += 1
                else:
                    # Apply pre-tokenization pattern to regular text
                    for match in re.finditer(PAT, text_part):
                        token_str = match.group(0)
                        token_bytes = tuple(
                            bytes([b]) for b in token_str.encode("utf-8")
                        )
                        freq[token_bytes] += 1

    except Exception as e:
        logger.error(f"Error processing chunk {start}-{end}: {e}")
        return {}

    return dict(freq)


def compute_pair_frequencies(
    token_freq: Dict[Tuple[bytes, ...], int],
) -> Tuple[
    Dict[Tuple[bytes, bytes], int],
    Dict[Tuple[bytes, bytes], Dict[Tuple[bytes, ...], List[int]]],
]:
    """
    Compute frequencies of all byte pairs and track their positions.

    Args:
        token_freq: Dictionary mapping tokens to their frequencies

    Returns:
        Tuple of (pair_frequencies, pair_positions)
    """
    pair_freq = Counter()
    pair_positions = defaultdict(lambda: defaultdict(list))

    for token, freq in token_freq.items():
        if len(token) < 2:
            continue

        # Count all adjacent pairs in this token
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            pair_freq[pair] += freq
            pair_positions[pair][token].append(i)

    return dict(pair_freq), dict(pair_positions)


def merge_tokens(
    token: Tuple[bytes, ...],
    pair_to_merge: Tuple[bytes, bytes],
    positions: List[int],
) -> Tuple[bytes, ...]:
    """
    Merge a specific pair in a token at given positions.

    Args:
        token: Original token as tuple of bytes
        pair_to_merge: The pair to merge
        positions: List of positions where the pair occurs

    Returns:
        New token with pairs merged
    """
    if not positions:
        return token

    new_token = []
    i = 0
    merge_positions = set(positions)

    while i < len(token):
        if i in merge_positions and i + 1 < len(token):
            # Merge the pair
            merged_byte = pair_to_merge[0] + pair_to_merge[1]
            new_token.append(merged_byte)
            i += 2
        else:
            new_token.append(token[i])
            i += 1

    return tuple(new_token)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: List[str],
    num_processes: int = 8,
    save_progress: bool = True,
    progress_interval: int = 100,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the given input file.

    Args:
        input_path: Path to training text file
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens to preserve
        num_processes: Number of processes for parallel processing
        save_progress: Whether to save intermediate progress
        progress_interval: How often to log progress and save checkpoints
        checkpoint_dir: Directory to save checkpoints (if None, uses temp directory)

    Returns:
        Tuple of (vocabulary, merges)
    """
    logger.info(f"Starting BPE training with vocab_size={vocab_size}")

    # Set up checkpoint directory
    if save_progress:
        if checkpoint_dir is None:
            import tempfile

            checkpoint_dir = tempfile.mkdtemp(prefix="bpe_training_")
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    # Initialize vocabulary with special tokens and byte-level tokens
    vocab = {}
    vocab_id = 0

    # Add special tokens first
    for token in special_tokens:
        vocab[vocab_id] = token.encode("utf-8")
        vocab_id += 1

    # Add all possible bytes
    for i in range(256):
        vocab[vocab_id] = bytes([i])
        vocab_id += 1

    initial_vocab_size = len(vocab)
    logger.info(f"Initial vocabulary size: {initial_vocab_size}")

    # Pre-tokenization with multiprocessing
    logger.info("Starting pre-tokenization...")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes * 10, "<|endoftext|>".encode("utf-8")
        )

    logger.info(f"Created {len(boundaries)-1} chunks for processing")

    with multiprocessing.Pool(num_processes) as pool:
        chunk_args = [
            (boundaries[i], boundaries[i + 1], input_path, special_tokens)
            for i in range(len(boundaries) - 1)
        ]
        chunk_results = pool.starmap(pretokenize_chunk, chunk_args)

    # Merge frequency dictionaries from all chunks
    token_freq = Counter()
    for chunk_freq in chunk_results:
        for token, freq in chunk_freq.items():
            token_freq[token] += freq

    logger.info(
        f"Pre-tokenization complete. Found {len(token_freq)} unique tokens"
    )

    # BPE merge loop
    merges = []
    merge_count = 0

    while len(vocab) < vocab_size:
        # Compute pair frequencies
        pair_freq, pair_positions = compute_pair_frequencies(token_freq)

        if not pair_freq:
            logger.warning("No more pairs to merge")
            break

        # Find the most frequent pair
        best_pair = max(pair_freq.items(), key=lambda x: (x[1], x[0]))[0]

        # Add to merges and vocabulary
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        merge_count += 1

        # Update token frequencies
        new_token_freq = Counter()
        tokens_to_update = pair_positions[best_pair]

        for token, positions in tokens_to_update.items():
            old_freq = token_freq[token]
            new_token = merge_tokens(token, best_pair, positions)

            # Remove old token and add new token
            del token_freq[token]
            new_token_freq[new_token] += old_freq

        # Add new frequencies
        for token, freq in new_token_freq.items():
            token_freq[token] += freq

        if merge_count % progress_interval == 0:
            logger.info(
                f"Completed {merge_count} merges. Vocab size: {len(vocab)}"
            )

            # Save checkpoint if requested
            if save_progress and checkpoint_dir:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"checkpoint_{merge_count}"
                )
                save_checkpoint(
                    vocab, merges, token_freq, checkpoint_path, merge_count
                )

    logger.info(f"BPE training complete. Final vocabulary size: {len(vocab)}")

    # Save final checkpoint
    if save_progress and checkpoint_dir:
        final_checkpoint_path = os.path.join(
            checkpoint_dir, "final_checkpoint"
        )
        save_checkpoint(
            vocab, merges, token_freq, final_checkpoint_path, merge_count
        )
        logger.info(f"Final checkpoint saved to: {final_checkpoint_path}")

    return vocab, merges


def save_checkpoint(
    vocab: Dict[int, bytes],
    merges: List[Tuple[bytes, bytes]],
    token_freq: Dict[Tuple[bytes, ...], int],
    checkpoint_path: str,
    merge_count: int,
):
    """
    Save training checkpoint to disk.

    Args:
        vocab: Current vocabulary
        merges: Current list of merges
        token_freq: Current token frequencies
        checkpoint_path: Path to save checkpoint
        merge_count: Number of merges completed
    """
    try:
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save vocabulary
        vocab_serializable = {
            k: (
                v.decode("utf-8", errors="replace")
                if isinstance(v, bytes)
                else str(v)
            )
            for k, v in vocab.items()
        }
        with open(os.path.join(checkpoint_path, "vocab.json"), "w") as f:
            json.dump(vocab_serializable, f, indent=2)

        # Save merges
        with open(os.path.join(checkpoint_path, "merges.txt"), "w") as f:
            for merge_pair in merges:
                f.write(f"{repr(merge_pair)}\n")

        # Save token frequencies (for resuming training)
        with open(os.path.join(checkpoint_path, "token_freq.pkl"), "wb") as f:
            pickle.dump(dict(token_freq), f)

        # Save metadata
        metadata = {
            "merge_count": merge_count,
            "vocab_size": len(vocab),
            "num_merges": len(merges),
        }
        with open(os.path.join(checkpoint_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"Checkpoint saved: {merge_count} merges, vocab size {len(vocab)}"
        )

    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def load_checkpoint(
    checkpoint_path: str,
) -> Tuple[
    Dict[int, bytes],
    List[Tuple[bytes, bytes]],
    Dict[Tuple[bytes, ...], int],
    int,
]:
    """
    Load training checkpoint from disk.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Tuple of (vocab, merges, token_freq, merge_count)
    """
    try:
        # Load vocabulary
        with open(os.path.join(checkpoint_path, "vocab.json"), "r") as f:
            vocab_data = json.load(f)
        vocab = {
            int(k): (
                ast.literal_eval(v)
                if isinstance(v, str) and v.startswith("b'")
                else v.encode("utf-8")
            )
            for k, v in vocab_data.items()
        }

        # Load merges
        merges = []
        with open(os.path.join(checkpoint_path, "merges.txt"), "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    merges.append(ast.literal_eval(line))

        # Load token frequencies
        with open(os.path.join(checkpoint_path, "token_freq.pkl"), "rb") as f:
            token_freq = pickle.load(f)

        # Load metadata
        with open(os.path.join(checkpoint_path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        merge_count = metadata["merge_count"]
        logger.info(
            f"Checkpoint loaded: {merge_count} merges, vocab size {len(vocab)}"
        )

        return vocab, merges, token_freq, merge_count

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise


def resume_bpe_training(
    checkpoint_path: str,
    target_vocab_size: int,
    progress_interval: int = 100,
    save_progress: bool = True,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Resume BPE training from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory
        target_vocab_size: Target vocabulary size
        progress_interval: How often to log progress
        save_progress: Whether to continue saving checkpoints

    Returns:
        Tuple of (vocabulary, merges)
    """
    logger.info(f"Resuming BPE training from checkpoint: {checkpoint_path}")

    # Load checkpoint
    vocab, merges, token_freq, merge_count = load_checkpoint(checkpoint_path)

    if len(vocab) >= target_vocab_size:
        logger.info(
            f"Checkpoint already has target vocab size ({len(vocab)} >= {target_vocab_size})"
        )
        return vocab, merges

    logger.info(
        f"Resuming from merge {merge_count}, current vocab size: {len(vocab)}"
    )

    # Continue BPE merging
    while len(vocab) < target_vocab_size:
        # Compute pair frequencies
        pair_freq, pair_positions = compute_pair_frequencies(token_freq)

        if not pair_freq:
            logger.warning("No more pairs to merge")
            break

        # Find the most frequent pair
        best_pair = max(pair_freq.items(), key=lambda x: (x[1], x[0]))[0]

        # Add to merges and vocabulary
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        merge_count += 1

        # Update token frequencies
        new_token_freq = Counter()
        tokens_to_update = pair_positions[best_pair]

        for token, positions in tokens_to_update.items():
            old_freq = token_freq[token]
            new_token = merge_tokens(token, best_pair, positions)

            # Remove old token and add new token
            del token_freq[token]
            new_token_freq[new_token] += old_freq

        # Add new frequencies
        for token, freq in new_token_freq.items():
            token_freq[token] += freq

        if merge_count % progress_interval == 0:
            logger.info(
                f"Completed {merge_count} merges. Vocab size: {len(vocab)}"
            )

            # Save checkpoint if requested
            if save_progress:
                checkpoint_dir = os.path.dirname(checkpoint_path)
                new_checkpoint_path = os.path.join(
                    checkpoint_dir, f"checkpoint_{merge_count}"
                )
                save_checkpoint(
                    vocab, merges, token_freq, new_checkpoint_path, merge_count
                )

    logger.info(
        f"Training resumed and completed. Final vocabulary size: {len(vocab)}"
    )
    return vocab, merges


def save_tokenizer(
    vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], output_dir: str
):
    """Save vocabulary and merges to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save merges
    with open(os.path.join(output_dir, "merges.txt"), "w") as f:
        for merge_pair in merges:
            f.write(f"{repr(merge_pair)}\n")

    # Save vocabulary
    vocab_serializable = {
        k: (
            v.decode("utf-8", errors="replace")
            if isinstance(v, bytes)
            else str(v)
        )
        for k, v in vocab.items()
    }
    with open(os.path.join(output_dir, "vocab.json"), "w") as f:
        json.dump(vocab_serializable, f, indent=2)

    logger.info(f"Tokenizer saved to {output_dir}")


if __name__ == "__main__":
    # Initialize ArgParser for command-line arguments
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument(
        "-i",
        "--input-path",
        type=str,
        default="data/TinyStoriesV2-GPT4-valid.txt",
        help="Path to the input text file (default: data/TinyStoriesV2-GPT4-valid.txt)",
    )
    parser.add_argument(
        "-v",
        "--vocab-size",
        type=int,
        default=500,
        help="Target vocabulary size (default: 500)",
    )
    parser.add_argument(
        "-s",
        "--special-tokens",
        type=str,
        default="<|endoftext|>",
        help='Comma-separated list of special tokens (default: "<|endoftext|>")',
    )
    parser.add_argument(
        "-n",
        "--num-processes",
        type=int,
        default=4,
        help="Number of parallel processes (default: 4)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="outputs/",
        help="Output directory for artifacts (default: outputs/)",
    )
    args = parser.parse_args()

    # Configuration with defaults or command-line values
    input_path = args.input_path
    vocab_size = args.vocab_size
    special_tokens = args.special_tokens.split(",")
    num_processes = args.num_processes
    output_dir = args.output_dir

    # Train BPE
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_processes=num_processes,
        save_progress=True,
        progress_interval=50,
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
    )
    end_time = time.time()

    logger.info(f"Training completed in {end_time - start_time:.2f} seconds")

    # Save results
    save_tokenizer(vocab, merges, output_dir)

    # Example of resuming from checkpoint
    # checkpoint_path = os.path.join(output_dir, "checkpoints", "checkpoint_100")
    # if os.path.exists(checkpoint_path):
    #     logger.info("Demonstrating checkpoint resume functionality...")
    #     resumed_vocab, resumed_merges = resume_bpe_training(
    #         checkpoint_path=checkpoint_path,
    #         target_vocab_size=vocab_size + 100  # Train a bit more
    #     )
