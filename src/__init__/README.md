```markdown
# Docling Document Chunker

Speaker-aware document/transcript chunker that converts files into conversation-formatted JSON. It supports automatic speaker detection, realistic per-utterance timestamps, and per-document analytics. Works with Docling for rich document parsing (PDF/DOC/DOCX), and falls back to text-only mode when Docling isn’t available.

## Key Features

- Speaker detection for transcript-like text (e.g., “Speaker 1 00:00”, “Unknown Speaker 00:16”, “John: …”).
- Conversation-style JSON chunks with:
  - conversation_id, chunk_index, text, timestamp, speaker
  - optional extended document and speaker metadata
- Realistic timestamp progression per utterance (based on estimated speaking rate).
- Configurable chunking by size or paragraphs, with overlap.
- Per-document output folders:
  - Combined conversation chunks
  - Speaker analytics (participation and word stats)
  - Individual chunk files
- GPU/CPU detection (runs on CPU if GPU unavailable).
- Docling support for parsing PDFs/Word documents; text/markdown supported in fallback mode.

## Supported Input Formats

- With Docling installed: .pdf, .docx, .doc
- Text-only mode (always available): .txt, .md

The directory scanner is recursive and will process supported files in subfolders.

// ... existing code ...
## Project Structure

- src/__init__/Docling Powered Otter Transcript to JSON Chunk v4.py — main speaker-aware chunking script
- src/Generic Conversation Docling to JSON Chunk Powered v1.py — generic conversation chunker
- src/Generic Docling to JSON Chunk Powered v2.py — generic document chunker
- src/requirements.txt — Python dependencies
// ... existing code ...
```
```markdown
## Project Structure

- transcript_chunker.py — main speaker-aware chunking CLI (parser, detector, chunker, savers)
- requirements.txt — Python dependencies for the chunker
- src/__init__/ — legacy/alternative scripts (optional)
```
```markdown
## Requirements

- Python 3.10
- Optional:
  - docling — enables PDF/DOC/DOCX parsing
  - torch — used only for GPU/CPU detection; not required for core functionality

If optional dependencies are not installed, the script still runs in text-only mode and uses CPU by default.

// ... existing code ...
## Installation

1) Create and activate a Conda environment:
```

shell
conda create -n docling-chunker python=3.10 -y
conda activate docling-chunker
```
2) Install dependencies:
```

shell
# From the project root
pip install -r src/requirements.txt
```
3) (Optional) Install Docling for PDF/Word parsing:
```

shell
pip install docling
```
Note: Torch is optional. If you prefer, omit it; the script will simply report CPU mode.
// ... existing code ...
```
```markdown
## Installation

Use your configured Conda environment manager.

1) Create and activate an environment:
```

shell
conda create -n docling-chunker python=3.10 -y
conda activate docling-chunker
```
2) Install dependencies listed in requirements.txt using your Conda environment.
```

shell
# From the project root
# Ensure the packages in requirements.txt are installed into the active environment.
# If a package is unavailable via default channels, install it from a channel that provides it.
conda install --file requirements.txt
```
Notes:
- Docling is optional and only needed for PDF/DOC/DOCX parsing.
- Torch is optional and only used for GPU/CPU detection.
```
```markdown
## Quick Start

1) Set your input/output directories:
- Open: src/__init__/Docling Powered Otter Transcript to JSON Chunk v4.py
- In main(), set:
  - input_directory: folder containing your files
  - output_directory: where results will be written

2) Run:
```

shell
python "src/__init__/Docling Powered Otter Transcript to JSON Chunk v4.py"
```
3) Results per document:
- [filename]_conversation_chunks.json
- [filename]_speaker_analytics.json
- individual_chunks/ — directory with one JSON per chunk

Each document gets its own subfolder under your chosen output directory containing the files above.
// ... existing code ...
```
```markdown
## Quick Start

1) Prepare folders:
- Input directory: where your source files are located
- Output directory: where results will be written

2) Run the chunker:
```

shell
python transcript_chunker.py --input "<path-to-input-dir>" --output "<path-to-output-dir>"
```
3) Results per document:
- [filename]_conversation_chunks.json
- [filename]_speaker_analytics.json
- individual_chunks/ — directory with one JSON per chunk

Each document gets its own subfolder under your chosen output directory containing the files above.
```
```markdown
## Configuration

Core parameters are controlled by ConversationChunkConfig. Common ones:

- max_chunk_size: maximum characters in a chunk
- min_chunk_size: minimum characters required to create a chunk
- overlap_size: overlap characters between sequential chunks
- preserve_paragraphs: preserve paragraph boundaries where possible
- detect_speakers: enable/disable speaker detection
- min_utterance_length: minimum characters for a valid utterance
- speaker_patterns: regex patterns used to detect speakers

Example configuration as used in main():
```

python
chunk_config = ConversationChunkConfig(
    max_chunk_size=800,
    overlap_size=100,
    preserve_paragraphs=True,
    min_chunk_size=50,
    speaker_type="document",
    detect_speakers=True,
    min_utterance_length=20,
    preserve_speaker_context=True
)
```
// ... existing code ...
```
```markdown
## Configuration

Key configs exposed by the CLI and code:

- ChunkConfig
  - max_chunk_size: maximum characters in a chunk
  - min_chunk_size: minimum characters required to create a chunk
  - overlap_size: overlap characters between sequential chunks
  - preserve_paragraphs: preserve paragraph boundaries where possible

- SpeakerConfig
  - detect_speakers: enable/disable speaker detection
  - preserve_speaker_context: keep speaker labels across chunks
  - min_utterance_length: minimum characters for a valid utterance
  - speaker_patterns: compiled regex patterns used to detect speakers

- TimingConfig
  - words_per_minute: used to estimate utterance durations
  - inter_utterance_pause_seconds: gap between utterances

- Config
  - chunk, speaker, timing groups
  - default_speaker_type: used when no speaker is detected

Example (programmatic use):
```

python
from transcript_chunker import Config, ChunkConfig, SpeakerConfig, TimingConfig

cfg = Config(
    chunk=ChunkConfig(
        max_chunk_size=800,
        min_chunk_size=50,
        overlap_size=100,
        preserve_paragraphs=True
    ),
    speaker=SpeakerConfig(
        detect_speakers=True,
        preserve_speaker_context=True,
        min_utterance_length=20
    ),
    timing=TimingConfig(
        words_per_minute=160,
        inter_utterance_pause_seconds=0.6
    )
)
```

```
```markdown
## Output Formats

- Minimal payload per chunk:
```

json
{
  "payload": {
    "conversation_id": "doc_example_1a2b3c4d",
    "chunk_index": 1,
    "text": "Utterance text here...",
    "timestamp": "2025-09-06T22:32:00Z",
    "speaker": "speaker_1"
  }
}
```
- Extended payload includes document and speaker metadata:
```

json
{
  "payload": {
    "conversation_id": "doc_example_1a2b3c4d",
    "chunk_index": 1,
    "text": "Utterance text here...",
    "timestamp": "2025-09-06T22:32:00Z",
    "speaker": "speaker_1"
  },
  "extended_metadata": {
    "source_file": "D:/Docs/example.pdf",
    "page_number": null,
    "metadata": {},
    "speaker_metadata": {
      "speaker_info": {
        "speaker_id": "speaker_1",
        "speaker_type": "anonymous",
        "display_name": "Speaker 1",
        "utterance_count": 12,
        "total_words": 845,
        "first_appearance": "2025-09-06T22:33:12.123456",
        "patterns_used": ["^(Speaker\\s*\\d+):\\s*"]
      },
      "original_speaker_label": "Speaker 1",
      "utterance_id": "doc_example_1a2b3c4d_speaker_1_1"
    }
  }
}
```
- Combined per-document file: [filename]_conversation_chunks.json
  - Contains a conversations object keyed by conversation_id with arrays of chunk payloads (extended by default).
- Speaker analytics per-document: [filename]_speaker_analytics.json
  - Summary totals (speakers, utterances, words)
  - Per-speaker counts, averages, and participation percentages
```
```markdown
## Output Files

Per processed document, the following are written into a dedicated subfolder:
- [filename]_conversation_chunks.json
  - conversations object keyed by conversation_id with arrays of chunk payloads
  - includes extended metadata by default (source_file, page_number, metadata, speaker_info, original_speaker_label, utterance_id)
- [filename]_speaker_analytics.json
  - totals for speakers/utterances/words
  - per-speaker stats (utterance_count, total_words, participation)
- individual_chunks/
  - One JSON file per chunk (payload-compatible with the combined file)
```
```markdown
## How Speaker Detection Works

The script scans for common transcript patterns such as:
- “Speaker 1 00:00”
- “Unknown Speaker 00:16”
- “Speaker 2:”
- “John: …”
- “[Speaker 1]: …”
- All-caps names like “ALICE: …”
- Role-based speakers like “Interviewer 05:10”

It standardizes IDs (e.g., “Speaker 1” → speaker_1, “Unknown Speaker” → unknown_speaker, “John” → named_john) and tracks per-speaker stats.

If no speaker is detected, content can be chunked as regular document text under the “document” speaker.

## Running Without Docling

- When Docling is not installed, the tool operates in text-only mode.
- It will process .txt and .md files; unsupported formats like .pdf/.docx/.doc are skipped with a warning.
- Core chunking and speaker detection remain available.

## Tips and Troubleshooting

- Docling unavailable:
  - You’ll see a warning and the script will process .txt/.md.
  - Install Docling if you need rich parsing of PDFs/Word documents.
- GPU not detected:
  - The script displays CPU processing mode. This is expected for non-GPU environments and does not block functionality.
- No chunks produced:
  - Verify input_directory exists and contains supported files (the scanner is recursive).
  - Reduce min_utterance_length or max_chunk_size if utterances are too short/long to be chunked.
- Timestamps:
  - If the first line of the transcript is in the form “MM/DD/YY HH:MM AM/PM” (or similar), it’s used as a base. Otherwise current UTC time is used.

## Contributing

- Open issues or pull requests with proposed changes.
- Please keep the output format backward-compatible when possible.

## License

Add your project’s license here.
```
