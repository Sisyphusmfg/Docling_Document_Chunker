"""
Docling Conversation Chunker v53 - Modular Architecture
Complete refactor with separated concerns, improved type safety, and performance optimizations
"""

import argparse
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Protocol, Tuple
import sys

# Version and constants
__version__ = "53.0.0"
SCHEMA_VERSION = "1.0"
DEFAULT_WPM = 150
DEFAULT_PAUSE_SECONDS = 2
MIN_UTTERANCE_LENGTH = 10


# Enums for type safety
class SpeakerType(str, Enum):
    NAMED = "named"
    ANONYMOUS = "anonymous"
    ROLE = "role"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ChunkMethod(str, Enum):
    SPEAKER_DETECTED = "speaker_detected"
    SPEAKER_DETECTED_SPLIT = "speaker_detected_split"
    PARAGRAPH_PRESERVED = "paragraph_preserved"
    FIXED_SIZE = "fixed_size"


class ProcessingMethod(str, Enum):
    DOCLING = "docling"
    TEXT_FALLBACK = "text_fallback"


# Configuration classes
@dataclass(slots=True)
class TimingConfig:
    """Configuration for conversation timing estimation"""
    words_per_minute: int = DEFAULT_WPM
    inter_utterance_pause_seconds: int = DEFAULT_PAUSE_SECONDS

    def estimate_duration_seconds(self, text: str) -> int:
        """Estimate speaking duration for given text"""
        words = len(text.split())
        return max(1, int(words / (self.words_per_minute / 60)))


@dataclass(slots=True)
class ChunkConfig:
    """Configuration for text chunking behavior"""
    max_chunk_size: int = 1000
    overlap_size: int = 200
    preserve_paragraphs: bool = True
    min_chunk_size: int = 100

    def __post_init__(self):
        """Validate configuration values"""
        if self.max_chunk_size <= self.min_chunk_size:
            raise ValueError("max_chunk_size must be greater than min_chunk_size")
        if self.overlap_size >= self.max_chunk_size:
            raise ValueError("overlap_size must be less than max_chunk_size")


@dataclass(slots=True)
class SpeakerConfig:
    """Configuration for speaker detection"""
    detect_speakers: bool = True
    preserve_speaker_context: bool = True
    min_utterance_length: int = MIN_UTTERANCE_LENGTH
    speaker_patterns_raw: List[str] = field(default_factory=lambda: [
        r"^(Speaker\s*\d+)\s+\d{1,2}:\d{2}",  # Speaker 1 00:00
        r"^(Unknown\s+Speaker)\s+\d{1,2}:\d{2}",  # Unknown Speaker 00:16
        r"^(SPEAKER\s*\d+)\s+\d{1,2}:\d{2}",  # SPEAKER 1 00:00
        r"^(Speaker\s*\d+|SPEAKER\s*\d+):\s*",  # Speaker 1:, SPEAKER 2:
        r"^(speaker\s*\d+):\s*",  # speaker 1:
        r"^(\d+):\s*",  # 1:, 2:, 3:
        r"^([A-Z][a-z]+):\s*",  # Named speakers: John:, Mary:
        r"^([A-Z]+):\s*",  # ALL CAPS names: JOHN:
        r"^\[([^\]]+)\]:\s*",  # [Speaker 1]:, [John]:
        r"^-\s*([A-Z][a-z]+):\s*",  # - John:
        r"^•\s*([A-Z][a-z]+):\s*",  # • John:
        r"^([A-Z][a-z]+\s+[A-Z][a-z]+)\s+\d{1,2}:\d{2}",  # John Smith 00:00
        r"^(Interviewer|Moderator|Host)\s+\d{1,2}:\d{2}",  # Role-based speakers
    ])
    speaker_patterns: List[Pattern[str]] = field(default_factory=list)

    def __post_init__(self):
        """Precompile regex patterns for performance"""
        if not self.speaker_patterns:
            self.speaker_patterns = [
                re.compile(pattern, re.IGNORECASE)
                for pattern in self.speaker_patterns_raw
            ]


@dataclass(slots=True)
class Config:
    """Master configuration combining all sub-configs"""
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    default_speaker_type: str = "document"


# Data models
@dataclass(slots=True)
class SpeakerInfo:
    """Information about a detected speaker"""
    speaker_id: str
    speaker_type: SpeakerType
    display_name: str
    utterance_count: int = 0
    total_words: int = 0
    first_appearance: Optional[str] = None
    patterns_used: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ConversationChunk:
    """Represents a conversation chunk with required payload format"""
    conversation_id: str
    chunk_index: int
    text: str
    timestamp: str
    speaker: str
    source_file: Optional[str] = None
    page_number: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    speaker_info: Optional[SpeakerInfo] = None
    original_speaker_label: Optional[str] = None
    utterance_id: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        """Convert to required conversation payload format"""
        return {
            "payload": {
                "conversation_id": self.conversation_id,
                "chunk_index": self.chunk_index,
                "text": self.text,
                "timestamp": self.timestamp,
                "speaker": self.speaker
            }
        }

    @staticmethod
    def _serialize_speaker_info(info: SpeakerInfo) -> Dict[str, Any]:
        """Serialize SpeakerInfo dataclass to dictionary"""
        return {
            "speaker_id": info.speaker_id,
            "speaker_type": info.speaker_type.value if hasattr(info.speaker_type, 'value') else str(info.speaker_type),
            "display_name": info.display_name,
            "utterance_count": info.utterance_count,
            "total_words": info.total_words,
            "first_appearance": info.first_appearance,
            "patterns_used": info.patterns_used
        }

    def to_extended_payload(self) -> Dict[str, Any]:
        """Convert to payload format with extended metadata"""
        payload = self.to_payload()
        payload["extended_metadata"] = {
            "source_file": self.source_file,
            "page_number": self.page_number,
            "metadata": self.metadata or {},
            "speaker_metadata": {
                "speaker_info": self._serialize_speaker_info(self.speaker_info) if self.speaker_info else None,
                "original_speaker_label": self.original_speaker_label,
                "utterance_id": self.utterance_id
            },
            "schema_version": SCHEMA_VERSION,
            "tool_version": __version__
        }
        return payload


@dataclass(slots=True)
class ParsedDocument:
    """Result of document parsing"""
    source_file: str
    title: str
    content: str
    page_count: int
    metadata: Dict[str, Any]


# Protocols for dependency injection
class DocumentParser(Protocol):
    """Protocol for document parsing implementations"""

    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse document and return structured content"""
        ...


class ChunkSaver(Protocol):
    """Protocol for saving chunks in various formats"""

    def save_chunks(self, chunks: List[ConversationChunk], output_path: Path,
                    filename_stem: str) -> List[Path]:
        """Save chunks and return list of created file paths"""
        ...


# Utility functions
def sanitize_filename(name: str) -> str:
    """Sanitize string for safe use in filenames"""
    # Replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Remove multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Trim and ensure not empty
    sanitized = sanitized.strip('_') or 'unnamed'
    return sanitized


def extract_transcript_datetime(text: str) -> Optional[datetime]:
    """Extract datetime from transcript first line"""
    if not text.strip():
        return None

    lines = text.split('\n')
    first_line = lines[0].strip()

    # Precompiled patterns for performance
    patterns = [
        re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{1,2}:\d{2})\s*(AM|PM)', re.IGNORECASE),
        re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{1,2}:\d{2})(AM|PM)', re.IGNORECASE),
    ]

    for pattern in patterns:
        match = pattern.match(first_line)
        if match:
            date_part, time_part, ampm_part = match.groups()
            try:
                if len(date_part.split('/')[2]) == 2:  # YY format
                    datetime_str = f"{date_part} {time_part} {ampm_part.upper()}"
                    return datetime.strptime(datetime_str, "%m/%d/%y %I:%M %p")
                else:  # YYYY format
                    datetime_str = f"{date_part} {time_part} {ampm_part.upper()}"
                    return datetime.strptime(datetime_str, "%m/%d/%Y %I:%M %p")
            except ValueError:
                continue

    return None


def clean_transcript_content(text: str) -> str:
    """Remove datetime line from transcript content"""
    lines = text.split('\n')
    if not lines:
        return text

    first_line = lines[0].strip()
    datetime_patterns = [
        re.compile(r'\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\s*(AM|PM)', re.IGNORECASE),
        re.compile(r'\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(AM|PM)', re.IGNORECASE),
    ]

    for pattern in datetime_patterns:
        if pattern.match(first_line):
            return '\n'.join(lines[1:]).strip()

    return text


# Speaker detection module
class SpeakerDetector:
    """Handles speaker detection and standardization"""

    def __init__(self, config: SpeakerConfig):
        self.config = config
        self.speaker_registry: Dict[str, SpeakerInfo] = {}
        self.logger = logging.getLogger(__name__)

    def detect_speakers_in_text(self, text: str) -> List[Tuple[str, str, str]]:
        """Detect speakers and return (speaker_id, original_label, content) tuples"""
        utterances = []
        lines = text.split('\n')
        current_speaker = None
        current_speaker_label = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            speaker_found = False

            # Try each precompiled pattern
            for pattern in self.config.speaker_patterns:
                match = pattern.match(line)
                if match:
                    # Save previous utterance if exists
                    if current_speaker and current_content:
                        content = ' '.join(current_content).strip()
                        if len(content) >= self.config.min_utterance_length:
                            utterances.append((current_speaker, current_speaker_label, content))

                    # Extract and standardize speaker
                    raw_speaker = match.group(1).strip()
                    standardized_speaker = self._standardize_speaker_id(raw_speaker)

                    # Register speaker if new
                    self._register_speaker(standardized_speaker, raw_speaker)

                    current_speaker = standardized_speaker
                    current_speaker_label = raw_speaker

                    # Extract content after speaker label and timestamp
                    remaining_text = line[match.end():].strip()

                    # For timestamp patterns, extract content after timestamp
                    if re.search(r'\d{1,2}:\d{2}', line):
                        timestamp_match = re.search(r'\d{1,2}:\d{2}', line)
                        if timestamp_match:
                            content_start = timestamp_match.end()
                            remaining_text = line[content_start:].strip()

                    current_content = [remaining_text] if remaining_text else []
                    speaker_found = True
                    break

            if not speaker_found and current_speaker:
                current_content.append(line)
            elif not speaker_found and not current_speaker:
                if not utterances:  # First content without speaker
                    current_speaker = "document"
                    current_speaker_label = "document"
                    current_content = [line]
                else:
                    current_content.append(line)

        # Add final utterance
        if current_speaker and current_content:
            content = ' '.join(current_content).strip()
            if len(content) >= self.config.min_utterance_length:
                utterances.append((current_speaker, current_speaker_label, content))

        return utterances

    def _standardize_speaker_id(self, raw_speaker: str) -> str:
        """Standardize speaker identifiers"""
        raw_lower = raw_speaker.lower().strip()

        # Handle Unknown Speaker
        if "unknown" in raw_lower and "speaker" in raw_lower:
            return "unknown_speaker"

        # Handle numbered speakers
        if re.match(r'speaker\s*\d+', raw_lower):
            number = re.search(r'\d+', raw_speaker).group()
            return f"speaker_{number}"
        elif re.match(r'^\d+$', raw_speaker):
            return f"speaker_{raw_speaker}"

        # Handle role-based speakers
        elif raw_lower in ['interviewer', 'moderator', 'host']:
            return f"role_{raw_lower}"

        # Handle named speakers
        elif raw_speaker.replace(' ', '').isalpha() and len(raw_speaker.strip()) > 1:
            clean_name = raw_speaker.lower().replace(' ', '_')
            return f"named_{clean_name}"

        else:
            clean_speaker = raw_speaker.lower().replace(' ', '_')
            return f"speaker_{clean_speaker}"

    def _register_speaker(self, speaker_id: str, original_label: str):
        """Register a new speaker"""
        if speaker_id not in self.speaker_registry:
            # Determine speaker type and display name
            if speaker_id.startswith("named_"):
                speaker_type = SpeakerType.NAMED
                display_name = original_label.title()
            elif speaker_id.startswith("speaker_"):
                speaker_type = SpeakerType.ANONYMOUS
                if speaker_id == "unknown_speaker":
                    display_name = "Unknown Speaker"
                else:
                    number = speaker_id.split('_')[1]
                    display_name = f"Speaker {number}"
            elif speaker_id.startswith("role_"):
                speaker_type = SpeakerType.ROLE
                role = speaker_id.split('_')[1]
                display_name = role.title()
            else:
                speaker_type = SpeakerType.SYSTEM
                display_name = original_label

            self.speaker_registry[speaker_id] = SpeakerInfo(
                speaker_id=speaker_id,
                speaker_type=speaker_type,
                display_name=display_name,
                first_appearance=datetime.now(timezone.utc).isoformat(),
                patterns_used=[]
            )

    def update_speaker_stats(self, speaker_id: str, text: str):
        """Update speaker statistics"""
        if speaker_id in self.speaker_registry:
            self.speaker_registry[speaker_id].utterance_count += 1
            self.speaker_registry[speaker_id].total_words += len(text.split())

    def _serialize_speaker_info(self, info: SpeakerInfo) -> Dict[str, Any]:
        """Serialize SpeakerInfo dataclass to dictionary"""
        return {
            "speaker_id": info.speaker_id,
            "speaker_type": info.speaker_type.value if hasattr(info.speaker_type, 'value') else str(info.speaker_type),
            "display_name": info.display_name,
            "utterance_count": info.utterance_count,
            "total_words": info.total_words,
            "first_appearance": info.first_appearance,
            "patterns_used": info.patterns_used
        }

    def generate_analytics(self) -> Dict[str, Any]:
        """Generate speaker analytics"""
        if not self.speaker_registry:
            return {
                "speaker_summary": {
                    "total_speakers": 0,
                    "named_speakers": 0,
                    "anonymous_speakers": 0,
                    "total_utterances": 0,
                    "total_words": 0
                },
                "speakers": {}
            }

        total_utterances = sum(info.utterance_count for info in self.speaker_registry.values())
        total_words = sum(info.total_words for info in self.speaker_registry.values())

        analytics = {
            "speaker_summary": {
                "total_speakers": len(self.speaker_registry),
                "named_speakers": len(
                    [s for s in self.speaker_registry.values() if s.speaker_type == SpeakerType.NAMED]),
                "anonymous_speakers": len(
                    [s for s in self.speaker_registry.values() if s.speaker_type == SpeakerType.ANONYMOUS]),
                "total_utterances": total_utterances,
                "total_words": total_words
            },
            "speakers": {}
        }

        # Fix: Add explicit type annotation for speakers dictionary
        speakers: Dict[str, Dict[str, Any]] = analytics["speakers"]

        for speaker_id, info in self.speaker_registry.items():
            # Use serialization method instead of direct dictionary access
            speaker_data = self._serialize_speaker_info(info)

            # Add computed analytics
            speaker_data.update({
                "average_words_per_utterance": info.total_words / max(info.utterance_count, 1),
                "participation_percentage": (info.utterance_count / max(total_utterances, 1)) * 100,
                "word_percentage": (info.total_words / max(total_words, 1)) * 100
            })

            # Now use the explicitly typed speakers variable
            speakers[speaker_id] = speaker_data

        return analytics


# Document parsing module
class DoclingParser:
    """Document parser using Docling or text fallback"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.docling_available = self._check_docling()
        if self.docling_available:
            self.converter = self._initialize_docling()
        else:
            self.converter = None

    def _check_docling(self) -> bool:
        """Check if Docling is available"""
        try:
            from docling.document_converter import DocumentConverter
            return True
        except ImportError:
            self.logger.warning("Docling not available, using text fallback mode")
            return False

    def _initialize_docling(self):
        """Initialize Docling converter"""
        try:
            from docling.document_converter import DocumentConverter
            return DocumentConverter()
        except Exception as e:
            self.logger.error(f"Failed to initialize Docling: {e}")
            return None

    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse document and return structured content"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.logger.info(f"Parsing document: {file_path.name}")

        try:
            if (self.docling_available and self.converter and
                    file_path.suffix.lower() in ['.pdf', '.docx', '.doc']):
                return self._parse_with_docling(file_path)
            else:
                return self._parse_as_text(file_path)
        except Exception as e:
            self.logger.error(f"Error parsing document {file_path}: {e}")
            raise

    def _parse_with_docling(self, file_path: Path) -> ParsedDocument:
        """Parse using Docling"""
        result = self.converter.convert(str(file_path))
        content = result.document.export_to_markdown()
        page_count = len(result.document.pages) if hasattr(result.document, 'pages') else 1

        return ParsedDocument(
            source_file=str(file_path),
            title=file_path.stem,
            content=content,
            page_count=page_count,
            metadata={
                'file_size': file_path.stat().st_size,
                'file_type': file_path.suffix.lower(),
                'processing_method': ProcessingMethod.DOCLING.value
            }
        )

    def _parse_as_text(self, file_path: Path) -> ParsedDocument:
        """Parse as plain text"""
        if file_path.suffix.lower() in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            self.logger.warning(f"Unsupported file type for text mode: {file_path.suffix}")
            content = f"Content from {file_path.name} (unsupported format in text-only mode)"

        return ParsedDocument(
            source_file=str(file_path),
            title=file_path.stem,
            content=content,
            page_count=1,
            metadata={
                'file_size': file_path.stat().st_size,
                'file_type': file_path.suffix.lower(),
                'processing_method': ProcessingMethod.TEXT_FALLBACK.value
            }
        )


# Chunking module
class ConversationChunker:
    """Handles chunking of text into conversation segments"""

    def __init__(self, config: Config, speaker_detector: SpeakerDetector):
        self.config = config
        self.speaker_detector = speaker_detector
        self.logger = logging.getLogger(__name__)

    def create_chunks(self, text: str, source_file: str,
                      conversation_id: Optional[str] = None) -> List[ConversationChunk]:
        """Create conversation chunks from text"""
        if not text.strip():
            self.logger.warning(f"Empty text provided for chunking from {source_file}")
            return []

        # Generate conversation ID if not provided
        if conversation_id is None:
            file_stem = Path(source_file).stem
            conversation_id = f"doc_{file_stem}_{uuid.uuid4().hex[:8]}"

        # Extract base timestamp from transcript if available
        base_timestamp = extract_transcript_datetime(text)
        if base_timestamp is None:
            base_timestamp = datetime.now(timezone.utc)
            self.logger.info("Using current UTC time as base timestamp")
        else:
            self.logger.info(f"Using extracted transcript time as base: {base_timestamp}")

        # Clean the content
        cleaned_text = clean_transcript_content(text)

        if self.config.speaker.detect_speakers:
            utterances = self.speaker_detector.detect_speakers_in_text(cleaned_text)
            chunks = self._create_chunks_from_utterances(
                utterances, source_file, conversation_id, base_timestamp
            )
        else:
            chunks = self._create_chunks_by_text(
                cleaned_text, source_file, conversation_id, base_timestamp
            )

        self.logger.info(f"Created {len(chunks)} conversation chunks from {source_file}")
        return chunks

    def _create_chunks_from_utterances(self, utterances: List[Tuple[str, str, str]],
                                       source_file: str, conversation_id: str,
                                       base_timestamp: datetime) -> List[ConversationChunk]:
        """Create chunks from detected utterances with realistic timing"""
        chunks = []
        chunk_index = 1
        current_time = base_timestamp

        for speaker_id, original_label, content in utterances:
            # Update speaker statistics
            self.speaker_detector.update_speaker_stats(speaker_id, content)

            # Calculate speaking duration
            speaking_duration = self.config.timing.estimate_duration_seconds(content)

            # Handle long utterances
            if len(content) > self.config.chunk.max_chunk_size:
                sub_chunks = self._split_long_utterance(content)
                for sub_content in sub_chunks:
                    chunk = self._create_chunk(
                        conversation_id, chunk_index, sub_content, current_time,
                        speaker_id, source_file, original_label,
                        ChunkMethod.SPEAKER_DETECTED_SPLIT, speaking_duration
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                    # Advance time for sub-chunk
                    sub_duration = self.config.timing.estimate_duration_seconds(sub_content)
                    current_time += timedelta(seconds=sub_duration)
            else:
                # Single chunk for utterance
                chunk = self._create_chunk(
                    conversation_id, chunk_index, content, current_time,
                    speaker_id, source_file, original_label,
                    ChunkMethod.SPEAKER_DETECTED, speaking_duration
                )
                chunks.append(chunk)
                chunk_index += 1

                # Advance time with pause
                total_time = speaking_duration + self.config.timing.inter_utterance_pause_seconds
                current_time += timedelta(seconds=total_time)

        return chunks

    def _create_chunks_by_text(self, text: str, source_file: str,
                               conversation_id: str, base_timestamp: datetime) -> List[ConversationChunk]:
        """Create chunks by text structure when speaker detection is disabled"""
        if self.config.chunk.preserve_paragraphs:
            return self._chunk_by_paragraphs(text, source_file, conversation_id, base_timestamp)
        else:
            return self._chunk_by_size(text, source_file, conversation_id, base_timestamp)

    def _split_long_utterance(self, content: str) -> List[str]:
        """Split long utterance into smaller chunks"""
        chunks = []
        sentences = re.split(r'[.!?]+', content)
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            potential_chunk = f"{current_chunk}. {sentence}" if current_chunk else sentence

            if (len(potential_chunk) > self.config.chunk.max_chunk_size and
                    len(current_chunk) > self.config.chunk.min_chunk_size):
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = potential_chunk

        if current_chunk and len(current_chunk.strip()) > self.config.chunk.min_chunk_size:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [content]

    def _chunk_by_paragraphs(self, text: str, source_file: str,
                             conversation_id: str, base_timestamp: datetime) -> List[ConversationChunk]:
        """Chunk text by paragraph boundaries"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        chunk_index = 1

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if (len(current_chunk) + len(paragraph) > self.config.chunk.max_chunk_size and
                    len(current_chunk) >= self.config.chunk.min_chunk_size):

                if current_chunk:
                    chunk_timestamp = (base_timestamp + timedelta(seconds=chunk_index))
                    chunk = self._create_chunk(
                        conversation_id, chunk_index, current_chunk.strip(), chunk_timestamp,
                        self.config.default_speaker_type, source_file, None,
                        ChunkMethod.PARAGRAPH_PRESERVED, 0
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # Handle overlap
                if self.config.chunk.overlap_size > 0 and current_chunk:
                    overlap_text = current_chunk[-self.config.chunk.overlap_size:]
                    current_chunk = f"{overlap_text}\n\n{paragraph}"
                else:
                    current_chunk = paragraph
            else:
                current_chunk = f"{current_chunk}\n\n{paragraph}" if current_chunk else paragraph

        # Add final chunk
        if current_chunk and len(current_chunk.strip()) >= self.config.chunk.min_chunk_size:
            chunk_timestamp = (base_timestamp + timedelta(seconds=chunk_index))
            chunk = self._create_chunk(
                conversation_id, chunk_index, current_chunk.strip(), chunk_timestamp,
                self.config.default_speaker_type, source_file, None,
                ChunkMethod.PARAGRAPH_PRESERVED, 0
            )
            chunks.append(chunk)

        return chunks

    def _chunk_by_size(self, text: str, source_file: str,
                       conversation_id: str, base_timestamp: datetime) -> List[ConversationChunk]:
        """Chunk text by fixed size with overlap"""
        chunks = []
        chunk_index = 1
        start = 0

        while start < len(text):
            end = start + self.config.chunk.max_chunk_size

            # Find good break point
            if end < len(text):
                # Look for sentence ending
                for i in range(end, start + self.config.chunk.min_chunk_size, -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
                else:
                    # Look for word boundary
                    for i in range(end, start + self.config.chunk.min_chunk_size, -1):
                        if text[i].isspace():
                            end = i
                            break

            chunk_text = text[start:end].strip()

            if len(chunk_text) >= self.config.chunk.min_chunk_size:
                chunk_timestamp = (base_timestamp + timedelta(seconds=chunk_index))
                chunk = self._create_chunk(
                    conversation_id, chunk_index, chunk_text, chunk_timestamp,
                    self.config.default_speaker_type, source_file, None,
                    ChunkMethod.FIXED_SIZE, 0
                )
                chunks.append(chunk)
                chunk_index += 1

            # Move with overlap
            start = end - self.config.chunk.overlap_size if self.config.chunk.overlap_size > 0 else end

        return chunks

    def _create_chunk(self, conversation_id: str, chunk_index: int, text: str,
                      timestamp: datetime, speaker: str, source_file: str,
                      original_speaker_label: Optional[str], chunk_method: ChunkMethod,
                      duration_seconds: int) -> ConversationChunk:
        """Create a conversation chunk with proper metadata"""
        utterance_id = f"{conversation_id}_{speaker}_{chunk_index}"

        speaker_info = self.speaker_detector.speaker_registry.get(speaker)

        return ConversationChunk(
            conversation_id=conversation_id,
            chunk_index=chunk_index,
            text=text,
            timestamp=timestamp.isoformat() + "Z",
            speaker=speaker,
            source_file=source_file,
            speaker_info=speaker_info,
            original_speaker_label=original_speaker_label,
            utterance_id=utterance_id,
            metadata={
                'chunk_method': chunk_method.value,
                'estimated_duration_seconds': duration_seconds,
                'schema_version': SCHEMA_VERSION,
                'tool_version': __version__
            }
        )


# I/O module
class JSONChunkSaver:
    """Saves chunks as JSON files with atomic operations"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def save_chunks(self, chunks: List[ConversationChunk], output_path: Path,
                    filename_stem: str) -> List[Path]:
        """Save chunks and return list of created file paths"""
        output_path.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        # Save combined document chunks
        combined_path = self._save_combined_chunks(chunks, output_path, filename_stem)
        if combined_path:
            saved_paths.append(combined_path)

        # Save individual chunk files
        individual_paths = self._save_individual_chunks(chunks, output_path, filename_stem)
        saved_paths.extend(individual_paths)

        return saved_paths

    def _save_combined_chunks(self, chunks: List[ConversationChunk],
                              output_path: Path, filename_stem: str) -> Optional[Path]:
        """Save combined chunks file"""
        try:
            filename = f"{sanitize_filename(filename_stem)}_conversation_chunks.json"
            file_path = output_path / filename

            # Group chunks by conversation_id
            conversations = {}
            for chunk in chunks:
                conv_id = chunk.conversation_id
                if conv_id not in conversations:
                    conversations[conv_id] = []
                conversations[conv_id].append(chunk.to_extended_payload())

            combined_data = {
                "document_info": {
                    "document_name": filename_stem,
                    "total_chunks": len(chunks),
                    "total_conversations": len(conversations),
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "source_file": chunks[0].source_file if chunks else None,
                    "schema_version": SCHEMA_VERSION,
                    "tool_version": __version__
                },
                "conversations": conversations
            }

            self._atomic_write_json(file_path, combined_data)
            self.logger.info(f"Saved combined chunks: {filename}")
            return file_path

        except Exception as e:
            self.logger.error(f"Error saving combined file for {filename_stem}: {e}")
            return None

    def _save_individual_chunks(self, chunks: List[ConversationChunk],
                                output_path: Path, filename_stem: str) -> List[Path]:
        """Save individual chunk files"""
        individual_dir = output_path / "individual_chunks"
        individual_dir.mkdir(exist_ok=True)
        saved_paths = []

        for chunk in chunks:
            try:
                speaker_safe = sanitize_filename(chunk.speaker)
                filename = f"{sanitize_filename(filename_stem)}_{speaker_safe}_chunk_{chunk.chunk_index:03d}.json"
                file_path = individual_dir / filename

                chunk_data = chunk.to_extended_payload()
                self._atomic_write_json(file_path, chunk_data)
                saved_paths.append(file_path)

            except Exception as e:
                self.logger.error(f"Error saving chunk {chunk.chunk_index}: {e}")

        self.logger.info(f"Saved {len(saved_paths)} individual chunk files")
        return saved_paths

    def _atomic_write_json(self, file_path: Path, data: Dict[str, Any]):
        """Write JSON with atomic operation (write to temp, then rename)"""
        temp_path = file_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_path.rename(file_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise


class AnalyticsSaver:
    """Saves speaker analytics"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def save_analytics(self, analytics: Dict[str, Any], output_path: Path,
                       filename_stem: str, source_file: Optional[str] = None) -> Optional[Path]:
        """Save speaker analytics and return file path"""
        try:
            filename = f"{sanitize_filename(filename_stem)}_speaker_analytics.json"
            file_path = output_path / filename

            # Add document context
            analytics["document_info"] = {
                "document_name": filename_stem,
                "source_file": source_file,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "schema_version": SCHEMA_VERSION,
                "tool_version": __version__
            }

            temp_path = file_path.with_suffix('.tmp')
            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(analytics, f, indent=2, ensure_ascii=False)
                temp_path.rename(file_path)
            except Exception:
                if temp_path.exists():
                    temp_path.unlink()
                raise

            self.logger.info(f"Saved speaker analytics: {filename}")
            return file_path

        except Exception as e:
            self.logger.error(f"Error saving analytics for {filename_stem}: {e}")
            return None


# Main processor
class ConversationProcessor:
    """Main processor coordinating all components"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.parser = DoclingParser()
        self.speaker_detector = SpeakerDetector(config.speaker)
        self.chunker = ConversationChunker(config, self.speaker_detector)
        self.chunk_saver = JSONChunkSaver()
        self.analytics_saver = AnalyticsSaver()

    def process_document(self, file_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Process a single document and return processing results"""
        try:
            # Reset speaker registry for each document
            self.speaker_detector.speaker_registry = {}

            # Parse document
            parsed_doc = self.parser.parse(file_path)

            # Create chunks
            chunks = self.chunker.create_chunks(parsed_doc.content, parsed_doc.source_file)

            if not chunks:
                self.logger.warning(f"No chunks created for {file_path.name}")
                return {"success": False, "error": "No chunks created"}

            # Prepare output directory
            file_stem = file_path.stem
            doc_output_dir = output_dir / sanitize_filename(file_stem)

            # Save chunks
            saved_chunk_paths = self.chunk_saver.save_chunks(chunks, doc_output_dir, file_stem)

            # Save analytics if speakers were detected
            analytics_path = None
            if self.speaker_detector.speaker_registry:
                analytics = self.speaker_detector.generate_analytics()
                analytics_path = self.analytics_saver.save_analytics(
                    analytics, doc_output_dir, file_stem, parsed_doc.source_file
                )

            self.logger.info(f"Successfully processed {file_path.name}: {len(chunks)} chunks")

            return {
                "success": True,
                "file": str(file_path),
                "chunks_created": len(chunks),
                "chunk_files": [str(p) for p in saved_chunk_paths],
                "analytics_file": str(analytics_path) if analytics_path else None,
                "speakers_detected": len(self.speaker_detector.speaker_registry),
                "output_directory": str(doc_output_dir)
            }

        except Exception as e:
            self.logger.error(f"Failed to process {file_path}: {e}")
            return {"success": False, "file": str(file_path), "error": str(e)}

    def process_directory(self, input_dir: Path, output_dir: Path,
                          file_extensions: List[str]) -> Dict[str, Any]:
        """Process all documents in a directory"""
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        self.logger.info(f"Processing directory: {input_dir}")

        # Find all matching files
        all_files = []
        for ext in file_extensions:
            all_files.extend(input_dir.rglob(f"*{ext}"))

        if not all_files:
            self.logger.warning(f"No files found with extensions {file_extensions}")
            return {
                "success": True,
                "files_processed": 0,
                "results": [],
                "summary": "No matching files found"
            }

        # Process each file
        results = []
        successful_files = 0

        for file_path in all_files:
            if file_path.is_file():
                result = self.process_document(file_path, output_dir)
                results.append(result)
                if result["success"]:
                    successful_files += 1

        # Generate summary
        total_chunks = sum(r.get("chunks_created", 0) for r in results if r["success"])
        total_speakers = sum(r.get("speakers_detected", 0) for r in results if r["success"])

        summary = {
            "success": True,
            "files_processed": successful_files,
            "total_files": len(all_files),
            "total_chunks": total_chunks,
            "total_speakers": total_speakers,
            "results": results,
            "output_directory": str(output_dir)
        }

        self.logger.info(
            f"Processing complete: {successful_files}/{len(all_files)} files, "
            f"{total_chunks} chunks, {total_speakers} speakers detected"
        )

        return summary


# CLI module
def setup_logging(level: str = "INFO"):
    """Setup consistent logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser"""
    parser = argparse.ArgumentParser(
        description=f"Docling Conversation Chunker v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input/ output/
  %(prog)s input/ output/ --extensions .txt .md
  %(prog)s input/ output/ --no-speakers --chunk-size 500
  %(prog)s input/ output/ --log-level DEBUG
        """
    )

    # Required arguments
    parser.add_argument("input_dir", type=Path, help="Input directory containing documents")
    parser.add_argument("output_dir", type=Path, help="Output directory for chunks")

    # Optional arguments
    parser.add_argument(
        "--extensions", "-e",
        nargs="+",
        default=[".pdf", ".docx", ".doc", ".txt", ".md"],
        help="File extensions to process (default: .pdf .docx .doc .txt .md)"
    )

    # Chunking options
    parser.add_argument("--chunk-size", type=int, default=800, help="Maximum chunk size")
    parser.add_argument("--min-chunk-size", type=int, default=50, help="Minimum chunk size")
    parser.add_argument("--overlap-size", type=int, default=100, help="Overlap between chunks")
    parser.add_argument("--no-paragraphs", action="store_true",
                        help="Don't preserve paragraph boundaries")

    # Speaker options
    parser.add_argument("--no-speakers", action="store_true",
                        help="Disable speaker detection")
    parser.add_argument("--min-utterance", type=int, default=20,
                        help="Minimum utterance length")

    # Timing options
    parser.add_argument("--wpm", type=int, default=DEFAULT_WPM,
                        help="Words per minute for timing estimation")
    parser.add_argument("--pause", type=int, default=DEFAULT_PAUSE_SECONDS,
                        help="Pause between speakers in seconds")

    # Other options
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Logging level")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    return parser


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Create configuration from CLI args - Updated section
        config = Config(
            chunk=ChunkConfig(
                max_chunk_size=args.chunk_size,
                min_chunk_size=args.min_chunk_size,
                overlap_size=args.overlap_size,
                preserve_paragraphs=not args.no_paragraphs
                # Remove min_utterance_length from here
            ),
            speaker=SpeakerConfig(
                detect_speakers=not args.no_speakers,
                min_utterance_length=args.min_utterance  # Move it here
            ),
            timing=TimingConfig(
                words_per_minute=args.wpm,
                inter_utterance_pause_seconds=args.pause
            )
        )

        # Initialize processor
        processor = ConversationProcessor(config)

        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                device_props = torch.cuda.get_device_properties(0)
                memory_gb = device_props.total_memory / 1e9
                cuda_version = str(torch.version.cuda) if torch.version.cuda else "Unknown"
                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA: {cuda_version}, Memory: {memory_gb:.1f} GB")
            else:
                logger.info("Using CPU processing")
        except ImportError:
            torch = None  # Add this line to fix the warning
            logger.info("PyTorch not available, using CPU processing")

        # Process documents
        logger.info(f"Starting processing: {args.input_dir} -> {args.output_dir}")
        logger.info(f"Configuration: chunk_size={args.chunk_size}, "
                    f"speakers={'enabled' if not args.no_speakers else 'disabled'}")

        results = processor.process_directory(args.input_dir, args.output_dir, args.extensions)

        # Print summary
        if results["success"]:
            print(f"\n🎉 Processing completed successfully!")
            print(f"   Files processed: {results['files_processed']}/{results['total_files']}")
            print(f"   Total chunks created: {results['total_chunks']}")
            print(f"   Speakers detected: {results['total_speakers']}")
            print(f"   Output directory: {results['output_directory']}")

            # Show failed files if any
            failed_files = [r for r in results["results"] if not r["success"]]
            if failed_files:
                print(f"\n⚠️  Failed to process {len(failed_files)} files:")
                for failure in failed_files:
                    print(f"   - {failure['file']}: {failure['error']}")
        else:
            print(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()