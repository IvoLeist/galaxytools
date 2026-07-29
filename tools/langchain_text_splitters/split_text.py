import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    sys.exit(f"Could not read input file as text: {path}")


def decode_separator(value: str) -> str:
    aliases = {
        "<empty>": "",
        "<space>": " ",
        "<tab>": "\t",
    }
    if value in aliases:
        return aliases[value]
    return bytes(value, "utf-8").decode("unicode_escape")


def parse_separators(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    separators = [decode_separator(line) for line in raw.splitlines() if line]
    return separators or None


def build_tiktoken_counter(
    encoding_name: str,
    model_name: str | None,
    allowed_special: str,
) -> Callable[[str], int]:
    import tiktoken

    try:
        if model_name:
            encoding = tiktoken.encoding_for_model(model_name)
        else:
            encoding = tiktoken.get_encoding(encoding_name)
    except Exception as exc:
        target = model_name or encoding_name
        sys.exit(
            "Could not load tiktoken encoding data for "
            f"'{target}'. tiktoken may need one-time network access to cache "
            "encoding files, or an administrator-provided TIKTOKEN_CACHE_DIR. "
            f"Original error: {exc}"
        )

    if allowed_special == "all":
        return lambda text: len(
            encoding.encode(text, allowed_special="all", disallowed_special=())
        )
    return lambda text: len(encoding.encode(text))


def build_splitter(args, length_function: Callable[[str], int]):
    common_args = {
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "add_start_index": True,
        "strip_whitespace": args.strip_whitespace,
    }

    if args.splitter_type == "recursive_character":
        splitter_args = {
            **common_args,
            "length_function": length_function,
            "keep_separator": args.keep_separator,
        }
        separators = parse_separators(args.separators)
        if separators is not None:
            splitter_args["separators"] = separators
        return RecursiveCharacterTextSplitter(**splitter_args)

    if args.splitter_type == "character":
        separator = decode_separator(args.separator)
        return CharacterTextSplitter(
            **common_args,
            length_function=length_function,
            separator=separator,
            keep_separator=args.keep_separator,
        )

    if args.splitter_type == "token":
        token_args = {
            **common_args,
            "encoding_name": args.encoding_name,
        }
        if args.model_name:
            token_args["model_name"] = args.model_name
        if args.allowed_special == "all":
            token_args["allowed_special"] = "all"
            token_args["disallowed_special"] = ()
        return TokenTextSplitter(**token_args)

    sys.exit(f"Unknown splitter type: {args.splitter_type}")


def write_text_output(
    output_path: Path,
    metadata: dict,
    chunks: list[dict],
    length_label: str,
) -> None:
    lines = [
        f"Splitter: {metadata['splitter_type']}",
        f"Length function: {metadata['length_function']}",
        f"Input characters: {metadata['input_characters']}",
        f"Input length: {metadata['input_length']} {length_label}",
        f"Chunk size: {metadata['chunk_size']}",
        f"Chunk overlap: {metadata['chunk_overlap']}",
        f"Number of chunks: {metadata['number_of_chunks']}",
        "",
    ]

    for chunk in chunks:
        start_index = chunk["start_index"]
        start_text = "unknown" if start_index is None else str(start_index)
        lines.extend(
            [
                (
                    f"--- Chunk {chunk['index']}: {chunk['length']} "
                    f"{length_label}, start={start_text} ---"
                ),
                chunk["text"],
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_chunk_files(chunks_dir: Path, chunks: list[dict]) -> None:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        chunk_path = chunks_dir / f"chunk_{chunk['index']:04d}.txt"
        chunk_path.write_text(chunk["text"], encoding="utf-8")


def write_tsv_output(output_path: Path, chunks: list[dict]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["chunk_number", "chunk_content", "chunk_token_count"])
        for chunk in chunks:
            writer.writerow([chunk["index"], chunk["text"], chunk["token_count"]])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split text with langchain-text-splitters."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-name", default="")
    parser.add_argument("--output-text", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--chunks-dir", type=Path, required=True)
    parser.add_argument(
        "--splitter-type",
        choices=("recursive_character", "character", "token"),
        default="recursive_character",
    )
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    parser.add_argument(
        "--length-mode",
        choices=("characters", "tiktoken"),
        default="characters",
    )
    parser.add_argument("--encoding-name", default="o200k_base")
    parser.add_argument("--model-name", default="")
    parser.add_argument(
        "--allowed-special",
        choices=("none", "all"),
        default="none",
    )
    parser.add_argument(
        "--keep-separator",
        choices=("false", "true", "start", "end"),
        default="true",
    )
    parser.add_argument("--separator", default="\\n\\n")
    parser.add_argument("--separators", default="")
    parser.add_argument("--strip-whitespace", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.chunk_size <= 0:
        sys.exit("Chunk size must be a positive integer.")
    if args.chunk_overlap < 0:
        sys.exit("Chunk overlap must be zero or a positive integer.")
    if args.chunk_overlap >= args.chunk_size:
        sys.exit("Chunk overlap must be smaller than chunk size.")

    keep_separator = args.keep_separator
    if keep_separator == "true":
        args.keep_separator = True
    elif keep_separator == "false":
        args.keep_separator = False

    text = read_text(args.input)

    if args.splitter_type == "token":
        args.length_mode = "tiktoken"

    if args.length_mode == "tiktoken":
        length_function = build_tiktoken_counter(
            args.encoding_name,
            args.model_name or None,
            args.allowed_special,
        )
        token_count_function = length_function
        tokenizer_label = args.model_name or args.encoding_name
        length_label = "tokens"
        length_function_label = f"tiktoken:{tokenizer_label}"
    else:
        length_function = len
        token_count_function = build_tiktoken_counter(
            args.encoding_name,
            args.model_name or None,
            args.allowed_special,
        )
        length_label = "characters"
        length_function_label = "characters"

    splitter = build_splitter(args, length_function)
    documents = splitter.create_documents(
        [text],
        metadatas=[{"source": args.input_name or args.input.name}],
    )

    chunks = []
    for index, document in enumerate(documents, start=1):
        chunk_text = document.page_content
        start_index = document.metadata.get("start_index")
        if start_index is not None and start_index < 0:
            start_index = None
        chunks.append(
            {
                "index": index,
                "length": length_function(chunk_text),
                "token_count": token_count_function(chunk_text),
                "start_index": start_index,
                "text": chunk_text,
            }
        )

    metadata = {
        "input_name": args.input_name or args.input.name,
        "input_characters": len(text),
        "input_length": length_function(text),
        "length_unit": length_label,
        "length_function": length_function_label,
        "splitter_type": args.splitter_type,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "number_of_chunks": len(chunks),
        "strip_whitespace": args.strip_whitespace,
        "keep_separator": keep_separator,
    }

    payload = {**metadata, "chunks": chunks}
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_text_output(args.output_text, metadata, chunks, length_label)
    write_tsv_output(args.output_tsv, chunks)
    write_chunk_files(args.chunks_dir, chunks)


if __name__ == "__main__":
    main()
