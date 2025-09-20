"""
Complete Docling Conversation Chunker v17 - Self-contained script
Converts documents to conversation format with speaker detection
Per-document file saving with proper output directory handling
"""

import os
import json
import logging
import uuid
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from datetime import timedelta
from datetime import timezone

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fix Windows symlink issues
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Check for GPU
try:
    import torch

    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 Using CPU processing")
except ImportError:
    gpu_available = False
    print("💻 Using CPU processing")

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import PdfFormatOption

    print("✅ Docling imported successfully")
    DOCLING_AVAILABLE = True
except ImportError:
    print("❌ Docling not installed. Install with: pip install docling")
    print("⚠️ Running in text-only mode for testing")
    DOCLING_AVAILABLE = False


@dataclass
class SpeakerInfo:
    """Information about a detected speaker"""
    speaker_id: str
    speaker_type: str  # "named", "anonymous", "system"
    display_name: str
    utterance_count: int = 0
    total_words: int = 0
    first_appearance: str = None
    patterns_used: List[str] = None

    def __post_init__(self):
        if self.patterns_used is None:
            self.patterns_used = []


@dataclass
class ConversationChunkConfig:
    """Configuration for conversation-style document chunking"""
    max_chunk_size: int = 1000
    overlap_size: int = 200
    preserve_paragraphs: bool = True
    min_chunk_size: int = 100
    speaker_type: str = "document"  # Default speaker for document content

    # Speaker detection settings
    detect_speakers: bool = True
    speaker_patterns: List[str] = None
    min_utterance_length: int = 10
    preserve_speaker_context: bool = True

    def __post_init__(self):
        if self.speaker_patterns is None:
            self.speaker_patterns = [
                # Transcript-style patterns with timestamps
                r"^(Speaker\s*\d+)\s+\d{1,2}:\d{2}",  # Speaker 1 00:00
                r"^(Unknown\s+Speaker)\s+\d{1,2}:\d{2}",  # Unknown Speaker 00:16
                r"^(SPEAKER\s*\d+)\s+\d{1,2}:\d{2}",  # SPEAKER 1 00:00

                # Standard patterns
                r"^(Speaker\s*\d+|SPEAKER\s*\d+):\s*",  # Speaker 1:, SPEAKER 2:
                r"^(speaker\s*\d+):\s*",  # speaker 1:
                r"^(\d+):\s*",  # 1:, 2:, 3:
                r"^([A-Z][a-z]+):\s*",  # Named speakers: John:, Mary:
                r"^([A-Z]+):\s*",  # ALL CAPS names: JOHN:
                r"^\[([^\]]+)\]:\s*",  # [Speaker 1]:, [John]:
                r"^-\s*([A-Z][a-z]+):\s*",  # - John:
                r"^•\s*([A-Z][a-z]+):\s*",  # • John:

                # Additional transcript patterns
                r"^([A-Z][a-z]+\s+[A-Z][a-z]+)\s+\d{1,2}:\d{2}",  # John Smith 00:00
                r"^(Interviewer|Moderator|Host)\s+\d{1,2}:\d{2}",  # Role-based speakers
            ]


@dataclass
class ConversationChunk:
    """Represents a conversation-style chunk with the required payload format"""
    conversation_id: str
    chunk_index: int
    text: str
    timestamp: str
    speaker: str

    # Additional metadata for document context
    source_file: Optional[str] = None
    page_number: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    # Speaker-specific metadata
    speaker_info: Optional[SpeakerInfo] = None
    original_speaker_label: Optional[str] = None
    utterance_id: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        """Convert to the required conversation payload format"""
        return {
            "payload": {
                "conversation_id": self.conversation_id,
                "chunk_index": self.chunk_index,
                "text": self.text,
                "timestamp": self.timestamp,
                "speaker": self.speaker
            }
        }

    def to_extended_payload(self) -> Dict[str, Any]:
        """Convert to payload format with extended document and speaker metadata"""
        payload = self.to_payload()
        payload["extended_metadata"] = {
            "source_file": self.source_file,
            "page_number": self.page_number,
            "metadata": self.metadata or {},
            "speaker_metadata": {
                "speaker_info": asdict(self.speaker_info) if self.speaker_info else None,
                "original_speaker_label": self.original_speaker_label,
                "utterance_id": self.utterance_id
            }
        }
        return payload


class ConversationDoclingProcessor:
    """Docling processor that outputs conversation-format chunks with speaker detection"""

    def __init__(self, chunk_config: ConversationChunkConfig = None):
        self.chunk_config = chunk_config or ConversationChunkConfig()
        self.speaker_registry: Dict[str, SpeakerInfo] = {}

        if DOCLING_AVAILABLE:
            self.converter = self._initialize_converter()
            print("✅ ConversationDoclingProcessor initialized with Docling support")
        else:
            self.converter = None
            print("✅ ConversationDoclingProcessor initialized in text-only mode")

    def _initialize_converter(self) -> DocumentConverter:
        """Initialize the Docling document converter"""
        if not DOCLING_AVAILABLE:
            return None

        try:
            converter = DocumentConverter()
            logger.info("Docling converter initialized successfully")
            return converter
        except Exception as e:
            logger.error(f"Failed to initialize Docling converter: {e}")
            raise

    def parse_document(self, file_path) -> Dict[str, Any]:
        """Parse a document using Docling or text fallback"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Parsing document: {file_path}")

        try:
            if DOCLING_AVAILABLE and self.converter and file_path.suffix.lower() in ['.pdf', '.docx', '.doc']:
                # Use Docling for supported formats
                result = self.converter.convert(str(file_path))
                content = result.document.export_to_markdown()
                page_count = len(result.document.pages) if hasattr(result.document, 'pages') else 1
            else:
                # Fallback to simple text reading
                if file_path.suffix.lower() in ['.txt', '.md']:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    page_count = 1
                else:
                    logger.warning(f"Unsupported file type for text-only mode: {file_path.suffix}")
                    content = f"Content from {file_path.name} (unsupported format in text-only mode)"
                    page_count = 1

            parsed_data = {
                'source_file': str(file_path),
                'title': file_path.stem,
                'content': content,
                'page_count': page_count,
                'metadata': {
                    'file_size': file_path.stat().st_size,
                    'file_type': file_path.suffix.lower(),
                    'processing_method': 'docling' if DOCLING_AVAILABLE and self.converter else 'text_fallback'
                }
            }

            logger.info(f"Successfully parsed document: {file_path.name}")
            return parsed_data

        except Exception as e:
            logger.error(f"Error parsing document {file_path}: {e}")
            raise

    def extract_transcript_datetime(self, text: str) -> Optional[datetime]:
        """
        Extract datetime from the first line of transcript
        Expects format: MM/DD/YY HH:MM AM/PM
        Example: 09/06/25 10:32PM
        """
        lines = text.split('\n')
        if not lines:
            return None

        first_line = lines[0].strip()

        # Pattern for MM/DD/YY HH:MM AM/PM format
        datetime_patterns = [
            r'(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{1,2}:\d{2})\s*(AM|PM)',  # 09/06/25 10:32 PM
            r'(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{1,2}:\d{2})(AM|PM)',  # 09/06/25 10:32PM
        ]

        for pattern in datetime_patterns:
            match = re.match(pattern, first_line, re.IGNORECASE)
            if match:
                date_part = match.group(1)
                time_part = match.group(2)
                ampm_part = match.group(3).upper()

                try:
                    # Parse the date and time
                    if len(date_part.split('/')[2]) == 2:  # YY format
                        datetime_str = f"{date_part} {time_part} {ampm_part}"
                        parsed_datetime = datetime.strptime(datetime_str, "%m/%d/%y %I:%M %p")
                    else:  # YYYY format
                        datetime_str = f"{date_part} {time_part} {ampm_part}"
                        parsed_datetime = datetime.strptime(datetime_str, "%m/%d/%Y %I:%M %p")

                    logger.info(f"Extracted transcript datetime: {parsed_datetime}")
                    return parsed_datetime

                except ValueError as e:
                    logger.warning(f"Failed to parse datetime from '{first_line}': {e}")
                    continue

        logger.warning(f"No valid datetime found in first line: '{first_line}'")
        return None

    def clean_transcript_content(self, text: str) -> str:
        """
        Remove the datetime line from transcript content and clean up
        """
        lines = text.split('\n')
        if not lines:
            return text

        # Check if first line contains a date pattern
        first_line = lines[0].strip()
        datetime_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\s*(AM|PM)',
            r'\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(AM|PM)',
        ]

        for pattern in datetime_patterns:
            if re.match(pattern, first_line, re.IGNORECASE):
                # Remove the datetime line and return the rest
                cleaned_content = '\n'.join(lines[1:]).strip()
                logger.info("Removed datetime line from transcript content")
                return cleaned_content

        # If no datetime pattern found, return original content
        return text

    def detect_speakers_in_text(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Detect speakers in text and return list of (speaker_id, speaker_label, content) tuples
        Enhanced for transcript formats with timestamps
        Returns: List of (standardized_speaker_id, original_label, utterance_text)
        """
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

            # Try each pattern to detect speaker
            for pattern in self.chunk_config.speaker_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    # Save previous utterance if exists
                    if current_speaker and current_content:
                        content = ' '.join(current_content).strip()
                        if len(content) >= self.chunk_config.min_utterance_length:
                            utterances.append((current_speaker, current_speaker_label, content))

                    # Extract and standardize speaker
                    raw_speaker = match.group(1).strip()
                    standardized_speaker = self._standardize_speaker_id(raw_speaker, pattern)

                    # Register speaker if new
                    self._register_speaker(standardized_speaker, raw_speaker, pattern)

                    current_speaker = standardized_speaker
                    current_speaker_label = raw_speaker

                    # Extract content after speaker label and timestamp
                    remaining_text = line[match.end():].strip()

                    # For timestamp patterns, extract content after timestamp
                    if re.search(r'\d{1,2}:\d{2}', line):
                        # Find the timestamp and get content after it
                        timestamp_match = re.search(r'\d{1,2}:\d{2}', line)
                        if timestamp_match:
                            content_start = timestamp_match.end()
                            remaining_text = line[content_start:].strip()

                    current_content = [remaining_text] if remaining_text else []
                    speaker_found = True
                    break

            if not speaker_found and current_speaker:
                # Continue previous speaker's content
                current_content.append(line)
            elif not speaker_found and not current_speaker:
                # No speaker detected, treat as document content
                if not utterances:  # First content without speaker
                    current_speaker = "document"
                    current_speaker_label = "document"
                    current_content = [line]
                else:
                    current_content.append(line)

        # Add final utterance
        if current_speaker and current_content:
            content = ' '.join(current_content).strip()
            if len(content) >= self.chunk_config.min_utterance_length:
                utterances.append((current_speaker, current_speaker_label, content))

        return utterances

    def _standardize_speaker_id(self, raw_speaker: str, pattern: str) -> str:
        """Standardize speaker identifiers for transcript formats"""
        raw_lower = raw_speaker.lower().strip()

        # Handle Unknown Speaker
        if "unknown" in raw_lower and "speaker" in raw_lower:
            return "unknown_speaker"

        # Handle numbered speakers (Speaker 1, SPEAKER 2, etc.)
        if re.match(r'speaker\s*\d+', raw_lower):
            number = re.search(r'\d+', raw_speaker).group()
            return f"speaker_{number}"
        elif re.match(r'^\d+$', raw_speaker):
            return f"speaker_{raw_speaker}"

        # Handle role-based speakers
        elif raw_lower in ['interviewer', 'moderator', 'host']:
            return f"role_{raw_lower}"

        # Handle named speakers (first name, full names)
        elif raw_speaker.replace(' ', '').isalpha() and len(raw_speaker.strip()) > 1:
            # Convert spaces to underscores for multi-word names
            clean_name = raw_speaker.lower().replace(' ', '_')
            return f"named_{clean_name}"

        else:
            # Fallback for any other format
            clean_speaker = raw_speaker.lower().replace(' ', '_')
            return f"speaker_{clean_speaker}"

    def _register_speaker(self, speaker_id: str, original_label: str, pattern: str):
        """Register a new speaker in the registry"""
        if speaker_id not in self.speaker_registry:
            # Determine speaker type and display name
            if speaker_id.startswith("named_"):
                speaker_type = "named"
                display_name = original_label.title()
            elif speaker_id.startswith("speaker_"):
                speaker_type = "anonymous"
                if speaker_id == "unknown_speaker":
                    display_name = "Unknown Speaker"
                else:
                    number = speaker_id.split('_')[1]
                    display_name = f"Speaker {number}"
            elif speaker_id.startswith("role_"):
                speaker_type = "role"
                role = speaker_id.split('_')[1]
                display_name = role.title()
            else:
                speaker_type = "system"
                display_name = original_label

            self.speaker_registry[speaker_id] = SpeakerInfo(
                speaker_id=speaker_id,
                speaker_type=speaker_type,
                display_name=display_name,
                first_appearance=datetime.now().isoformat(),
                patterns_used=[pattern]
            )
        else:
            # Update existing speaker
            if pattern not in self.speaker_registry[speaker_id].patterns_used:
                self.speaker_registry[speaker_id].patterns_used.append(pattern)

    def _update_speaker_stats(self, speaker_id: str, text: str):
        """Update speaker statistics"""
        if speaker_id in self.speaker_registry:
            self.speaker_registry[speaker_id].utterance_count += 1
            self.speaker_registry[speaker_id].total_words += len(text.split())

    def create_conversation_chunks(self, text: str, source_file: str,
                                   conversation_id: str = None) -> List[ConversationChunk]:
        """Create conversation-format chunks from document text with speaker detection"""
        if not text.strip():
            logger.warning(f"Empty text provided for chunking from {source_file}")
            return []

        # Generate conversation ID if not provided
        if conversation_id is None:
            file_stem = Path(source_file).stem
            conversation_id = f"doc_{file_stem}_{uuid.uuid4().hex[:8]}"

        # Extract base timestamp from transcript if available
        base_timestamp = self.extract_transcript_datetime(text)
        if base_timestamp is None:
            # Fallback to current time if no timestamp found
            base_timestamp = datetime.utcnow()
            logger.info("Using current UTC time as base timestamp")
        else:
            logger.info(f"Using extracted transcript time as base: {base_timestamp}")

        # Clean the content by removing the datetime line
        cleaned_text = self.clean_transcript_content(text)

        chunks = []

        if self.chunk_config.detect_speakers:
            # Detect speakers and create utterance-based chunks
            utterances = self.detect_speakers_in_text(cleaned_text)
            chunks = self._create_chunks_from_utterances(
                utterances, source_file, conversation_id, base_timestamp
            )
        else:
            # Use original chunking method
            if self.chunk_config.preserve_paragraphs:
                chunks = self._chunk_by_paragraphs(cleaned_text, source_file, conversation_id, base_timestamp)
            else:
                chunks = self._chunk_by_size(cleaned_text, source_file, conversation_id, base_timestamp)

        logger.info(f"Created {len(chunks)} conversation chunks from {source_file}")
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
            self._update_speaker_stats(speaker_id, content)

            # Calculate realistic time increment based on content length
            # Assume ~150 words per minute speaking rate
            word_count = len(content.split())
            speaking_duration_seconds = max(1, int(word_count / 2.5))  # ~150 WPM

            # If utterance is too long, split it
            if len(content) > self.chunk_config.max_chunk_size:
                sub_chunks = self._split_long_utterance(content, speaker_id)
                for i, sub_content in enumerate(sub_chunks):
                    chunk_timestamp = current_time.isoformat() + "Z"

                    utterance_id = f"{conversation_id}_{speaker_id}_{chunk_index}"

                    chunks.append(ConversationChunk(
                        conversation_id=conversation_id,
                        chunk_index=chunk_index,
                        text=sub_content.strip(),
                        timestamp=chunk_timestamp,
                        speaker=speaker_id,
                        source_file=source_file,
                        speaker_info=self.speaker_registry.get(speaker_id),
                        original_speaker_label=original_label,
                        utterance_id=utterance_id,
                        metadata={'chunk_method': 'speaker_detected_split',
                                  'estimated_duration_seconds': speaking_duration_seconds}
                    ))
                    chunk_index += 1

                    # Add time for this sub-chunk
                    sub_word_count = len(sub_content.split())
                    sub_duration = max(1, int(sub_word_count / 2.5))
                    current_time += timedelta(seconds=sub_duration)
            else:
                # Single chunk for this utterance
                chunk_timestamp = current_time.isoformat() + "Z"

                utterance_id = f"{conversation_id}_{speaker_id}_{chunk_index}"

                chunks.append(ConversationChunk(
                    conversation_id=conversation_id,
                    chunk_index=chunk_index,
                    text=content.strip(),
                    timestamp=chunk_timestamp,
                    speaker=speaker_id,
                    source_file=source_file,
                    speaker_info=self.speaker_registry.get(speaker_id),
                    original_speaker_label=original_label,
                    utterance_id=utterance_id,
                    metadata={'chunk_method': 'speaker_detected',
                              'estimated_duration_seconds': speaking_duration_seconds}
                ))
                chunk_index += 1

                # Advance time based on speaking duration plus small pause
                current_time += timedelta(seconds=speaking_duration_seconds + 2)  # 2-second pause between speakers

        return chunks

    def _split_long_utterance(self, content: str, speaker_id: str) -> List[str]:
        """Split a long utterance into smaller chunks while preserving meaning"""
        chunks = []
        sentences = re.split(r'[.!?]+', content)
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if (len(current_chunk) + len(sentence) > self.chunk_config.max_chunk_size
                    and len(current_chunk) > self.chunk_config.min_chunk_size):

                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence

        if current_chunk and len(current_chunk.strip()) > self.chunk_config.min_chunk_size:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [content]  # Fallback to original if splitting failed

    def _chunk_by_paragraphs(self, text: str, source_file: str,
                             conversation_id: str, base_timestamp: datetime) -> List[ConversationChunk]:
        """Chunk text while preserving paragraph boundaries"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        chunk_index = 1

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If adding this paragraph would exceed max size, save current chunk
            if (len(current_chunk) + len(paragraph) > self.chunk_config.max_chunk_size
                    and len(current_chunk) >= self.chunk_config.min_chunk_size):

                if current_chunk:
                    chunk_timestamp = (base_timestamp +
                                       timedelta(seconds=chunk_index)).isoformat() + "Z"

                    chunks.append(ConversationChunk(
                        conversation_id=conversation_id,
                        chunk_index=chunk_index,
                        text=current_chunk.strip(),
                        timestamp=chunk_timestamp,
                        speaker=self.chunk_config.speaker_type,
                        source_file=source_file,
                        metadata={'chunk_method': 'paragraph_preserved'}
                    ))
                    chunk_index += 1

                # Start new chunk with overlap if configured
                if self.chunk_config.overlap_size > 0 and current_chunk:
                    overlap_text = current_chunk[-self.chunk_config.overlap_size:]
                    current_chunk = overlap_text + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add final chunk
        if current_chunk and len(current_chunk.strip()) >= self.chunk_config.min_chunk_size:
            chunk_timestamp = (base_timestamp +
                               timedelta(seconds=chunk_index)).isoformat() + "Z"

            chunks.append(ConversationChunk(
                conversation_id=conversation_id,
                chunk_index=chunk_index,
                text=current_chunk.strip(),
                timestamp=chunk_timestamp,
                speaker=self.chunk_config.speaker_type,
                source_file=source_file,
                metadata={'chunk_method': 'paragraph_preserved'}
            ))

        return chunks

    def _chunk_by_size(self, text: str, source_file: str,
                       conversation_id: str, base_timestamp: datetime) -> List[ConversationChunk]:
        """Chunk text by fixed size with overlap"""
        chunks = []
        chunk_index = 1
        start = 0

        while start < len(text):
            end = start + self.chunk_config.max_chunk_size

            # If not at the end, try to break at a sentence or word boundary
            if end < len(text):
                # Look for sentence ending
                for i in range(end, start + self.chunk_config.min_chunk_size, -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
                else:
                    # Look for word boundary
                    for i in range(end, start + self.chunk_config.min_chunk_size, -1):
                        if text[i].isspace():
                            end = i
                            break

            chunk_text = text[start:end].strip()

            if len(chunk_text) >= self.chunk_config.min_chunk_size:
                chunk_timestamp = (base_timestamp +
                                   timedelta(seconds=chunk_index)).isoformat() + "Z"

                chunks.append(ConversationChunk(
                    conversation_id=conversation_id,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    timestamp=chunk_timestamp,
                    speaker=self.chunk_config.speaker_type,
                    source_file=source_file,
                    metadata={'chunk_method': 'fixed_size'}
                ))
                chunk_index += 1

            # Move start position with overlap
            start = end - self.chunk_config.overlap_size if self.chunk_config.overlap_size > 0 else end

        return chunks

    def generate_speaker_analytics(self) -> Dict[str, Any]:
        """Generate analytics about detected speakers"""
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
                "named_speakers": len([s for s in self.speaker_registry.values() if s.speaker_type == "named"]),
                "anonymous_speakers": len([s for s in self.speaker_registry.values() if s.speaker_type == "anonymous"]),
                "total_utterances": total_utterances,
                "total_words": total_words
            },
            "speakers": {}
        }

        for speaker_id, info in self.speaker_registry.items():
            analytics["speakers"][speaker_id] = {
                "display_name": info.display_name,
                "speaker_type": info.speaker_type,
                "utterance_count": info.utterance_count,
                "total_words": info.total_words,
                "average_words_per_utterance": info.total_words / max(info.utterance_count, 1),
                "participation_percentage": (info.utterance_count / max(total_utterances, 1)) * 100,
                "word_percentage": (info.total_words / max(total_words, 1)) * 100,
                "patterns_used": info.patterns_used,
                "first_appearance": info.first_appearance
            }

        return analytics

    def save_individual_conversation_files(self, chunks: List[ConversationChunk], output_dir: str, file_stem: str):
        """Save conversation chunks and analytics for a single document"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"💾 Saving files for document: {file_stem}")

        # Save combined chunks for this document
        try:
            combined_filename = f"{file_stem}_conversation_chunks.json"
            combined_path = output_path / combined_filename

            # Group chunks by conversation_id for this document
            conversations = {}
            for chunk in chunks:
                conv_id = chunk.conversation_id
                if conv_id not in conversations:
                    conversations[conv_id] = []
                conversations[conv_id].append(chunk.to_extended_payload())

            combined_data = {
                "document_info": {
                    "document_name": file_stem,
                    "total_chunks": len(chunks),
                    "total_conversations": len(conversations),
                    "processed_at": datetime.now().isoformat(),
                    "source_file": chunks[0].source_file if chunks else None,
                    "chunk_config": {
                        "max_chunk_size": self.chunk_config.max_chunk_size,
                        "overlap_size": self.chunk_config.overlap_size,
                        "preserve_paragraphs": self.chunk_config.preserve_paragraphs,
                        "min_chunk_size": self.chunk_config.min_chunk_size,
                        "speaker_type": self.chunk_config.speaker_type,
                        "detect_speakers": self.chunk_config.detect_speakers,
                        "min_utterance_length": self.chunk_config.min_utterance_length
                    }
                },
                "conversations": conversations
            }

            with open(combined_path, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=2, ensure_ascii=False)

            print(f"✅ Saved conversation chunks: {combined_filename}")

        except Exception as e:
            print(f"❌ Error saving combined file for {file_stem}: {e}")

        # Save speaker analytics for this document
        if self.speaker_registry:
            try:
                analytics_filename = f"{file_stem}_speaker_analytics.json"
                analytics_path = output_path / analytics_filename

                analytics_data = self.generate_speaker_analytics()
                # Add document context
                analytics_data["document_info"] = {
                    "document_name": file_stem,
                    "source_file": chunks[0].source_file if chunks else None,
                    "processed_at": datetime.now().isoformat()
                }

                with open(analytics_path, 'w', encoding='utf-8') as f:
                    json.dump(analytics_data, f, indent=2, ensure_ascii=False)

                print(f"✅ Saved speaker analytics: {analytics_filename}")

            except Exception as e:
                print(f"❌ Error saving analytics for {file_stem}: {e}")

        # Save individual chunk files for this document
        try:
            individual_dir = output_path / "individual_chunks"
            individual_dir.mkdir(exist_ok=True)

            saved_chunks = 0
            for chunk in chunks:
                speaker_safe = chunk.speaker.replace("/", "_").replace("\\", "_")
                filename = f"{file_stem}_{speaker_safe}_chunk_{chunk.chunk_index:03d}.json"
                file_path = individual_dir / filename

                chunk_data = chunk.to_extended_payload()

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(chunk_data, f, indent=2, ensure_ascii=False)

                saved_chunks += 1

            print(f"✅ Saved {saved_chunks} individual chunk files in individual_chunks/")

        except Exception as e:
            print(f"❌ Error saving individual chunks for {file_stem}: {e}")

        print(f"📁 All files for {file_stem} saved in: {output_path}")
        print()  # Add spacing between documents

    def process_directory(self, directory_path: str, file_extensions: List[str] = None,
                          output_base_dir: str = None) -> List[ConversationChunk]:
        """Process all documents in a directory and create conversation chunks"""
        directory_path = Path(directory_path)

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        if file_extensions is None:
            file_extensions = ['.pdf', '.docx', '.doc', '.txt', '.md']

        all_chunks = []
        processed_files = 0

        logger.info(f"Processing directory: {directory_path}")

        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                try:
                    # Reset speaker registry for each file since they're separate conversations
                    self.speaker_registry = {}

                    # Parse document
                    parsed_doc = self.parse_document(str(file_path))

                    # Create conversation chunks for this specific file
                    chunks = self.create_conversation_chunks(
                        parsed_doc['content'],
                        str(file_path)
                    )

                    if chunks:
                        # Save individual files immediately after processing each document
                        file_stem = Path(file_path).stem

                        # Use provided output directory or default
                        if output_base_dir:
                            output_dir = Path(output_base_dir) / file_stem
                        else:
                            output_dir = Path("output") / "individual_conversations" / file_stem

                        self.save_individual_conversation_files(
                            chunks,
                            str(output_dir),
                            file_stem
                        )

                    all_chunks.extend(chunks)
                    processed_files += 1

                    logger.info(f"Processed {file_path.name}: {len(chunks)} conversation chunks")

                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    continue

        logger.info(
            f"Completed processing {processed_files} files, created {len(all_chunks)} total conversation chunks")
        return all_chunks

    def save_conversation_chunks(self, chunks: List[ConversationChunk], output_dir: str,
                                 save_individual: bool = True, save_combined: bool = True,
                                 include_extended_metadata: bool = True, save_analytics: bool = True):
        """Save conversation chunks in the required payload format with speaker analytics"""
        print(f"💾 Starting to save {len(chunks)} conversation chunks to {output_dir}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created/verified output directory: {output_path}")

        saved_count = 0

        if save_individual:
            print("📄 Saving individual conversation chunk files...")
            for chunk in chunks:
                try:
                    source_name = Path(chunk.source_file).stem if chunk.source_file else "unknown"
                    speaker_safe = chunk.speaker.replace("/", "_").replace("\\", "_")
                    filename = f"{source_name}_{speaker_safe}_chunk_{chunk.chunk_index:03d}.json"
                    file_path = output_path / filename

                    # Use extended payload format if requested
                    if include_extended_metadata:
                        chunk_data = chunk.to_extended_payload()
                    else:
                        chunk_data = chunk.to_payload()

                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(chunk_data, f, indent=2, ensure_ascii=False)

                    saved_count += 1

                except Exception as e:
                    print(f"❌ Error saving chunk {chunk.chunk_index}: {e}")

            print(f"✅ Saved {saved_count} individual conversation chunk files")

        if save_combined:
            print("📋 Saving combined conversation chunks file...")
            try:
                combined_filename = "all_conversation_chunks.json"
                combined_path = output_path / combined_filename

                # Group chunks by conversation_id
                conversations = {}
                for chunk in chunks:
                    conv_id = chunk.conversation_id
                    if conv_id not in conversations:
                        conversations[conv_id] = []

                    if include_extended_metadata:
                        conversations[conv_id].append(chunk.to_extended_payload())
                    else:
                        conversations[conv_id].append(chunk.to_payload())

                combined_data = {
                    "processing_info": {
                        "total_chunks": len(chunks),
                        "total_conversations": len(conversations),
                        "processed_at": datetime.now().isoformat(),
                        "source_files": list(set(chunk.source_file for chunk in chunks if chunk.source_file)),
                        "chunk_config": {
                            "max_chunk_size": self.chunk_config.max_chunk_size,
                            "overlap_size": self.chunk_config.overlap_size,
                            "preserve_paragraphs": self.chunk_config.preserve_paragraphs,
                            "min_chunk_size": self.chunk_config.min_chunk_size,
                            "speaker_type": self.chunk_config.speaker_type,
                            "detect_speakers": self.chunk_config.detect_speakers,
                            "min_utterance_length": self.chunk_config.min_utterance_length
                        }
                    },
                    "conversations": conversations
                }

                with open(combined_path, 'w', encoding='utf-8') as f:
                    json.dump(combined_data, f, indent=2, ensure_ascii=False)

                print(f"✅ Saved combined conversation chunks file: {combined_filename}")

            except Exception as e:
                print(f"❌ Error saving combined file: {e}")

        if save_analytics and self.speaker_registry:
            print("📊 Saving speaker analytics...")
            try:
                analytics_filename = "speaker_analytics.json"
                analytics_path = output_path / analytics_filename

                analytics_data = self.generate_speaker_analytics()

                with open(analytics_path, 'w', encoding='utf-8') as f:
                    json.dump(analytics_data, f, indent=2, ensure_ascii=False)

                print(f"✅ Saved speaker analytics: {analytics_filename}")
                saved_count += 1

            except Exception as e:
                print(f"❌ Error saving analytics: {e}")

        # Verify files were created
        json_files = list(output_path.glob("*.json"))
        print(f"🔍 Verification: Found {len(json_files)} JSON files in {output_dir}")

        return len(json_files)


def main():
    """Main function for conversation-format document processing"""
    print("🚀 DOCLING CONVERSATION CHUNKER v17")
    print("=" * 60)

    # GPU System Check
    print("🔍 SYSTEM CHECK:")
    if gpu_available and torch.cuda.is_available():
        print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   🔥 CUDA: {torch.version.cuda}")
        print(f"   💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   💻 CPU processing only")

    if not DOCLING_AVAILABLE:
        print("   ⚠️ Docling not available - using text-only mode")

    print()

    # Configuration for conversation chunks with speaker detection
    chunk_config = ConversationChunkConfig(
        max_chunk_size=800,
        overlap_size=100,
        preserve_paragraphs=True,
        min_chunk_size=50,
        speaker_type="document",
        detect_speakers=True,  # Enable speaker detection
        min_utterance_length=20,  # Minimum length for valid utterances
        preserve_speaker_context=True
    )

    # Initialize processor
    processor = ConversationDoclingProcessor(chunk_config)

    # Set directories - CHANGE THESE PATHS TO YOUR ACTUAL DIRECTORIES
    input_directory = r"D:\Docling Trial\Documents"
    output_directory = r"D:\Docling Trial\Conversation_Chunks"

    print(f"📂 Input directory: {input_directory}")
    print(f"📁 Output directory: {output_directory}")

    # Initialize variables to avoid NameError
    all_chunks = []

    try:
        if os.path.exists(input_directory):
            print(f"\n📄 Processing all documents in directory...")

            # Process all documents - now creates individual files per document
            all_chunks = processor.process_directory(
                input_directory,
                file_extensions=['.pdf', '.docx', '.doc', '.txt', '.md'],
                output_base_dir=output_directory
            )

            print(f"\n📊 PROCESSING COMPLETE:")
            print(f"   Total conversation chunks created: {len(all_chunks)}")

            if all_chunks:
                # Show overall summary
                total_files = len(set(chunk.source_file for chunk in all_chunks))
                print(f"   Documents processed: {total_files}")
                print(f"   Individual files created per document:")
                print(f"     • [filename]_conversation_chunks.json")
                print(f"     • [filename]_speaker_analytics.json")
                print(f"     • individual_chunks/ folder with all chunk files")

                print(f"\n📁 All files saved in individual folders under: {output_directory}")
                print(f"   Each document has its own subfolder with separate analytics and chunks")

            else:
                print("⚠️ No conversation chunks were created - check your input directory")

        else:
            print(f"❌ Input directory not found: {input_directory}")
            print("Please create the directory and add some documents")
            print("\n💡 To test with a single file, you can also:")
            print("   1. Create a test.txt file with transcript content")
            print("   2. Put it in the input directory")
            print("   3. Run the script again")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n🏁 PROGRAM COMPLETED")
    print(f"📁 Check individual conversation files in: {output_directory}/[filename]/")
    print(f"   Each document now has its own separate conversation chunks and speaker analytics")


# Test function for development
def test_with_sample_transcript():
    """Test function using sample transcript text"""
    print("🧪 TESTING WITH SAMPLE TRANSCRIPT")

    sample_transcript = """
09/06/25 10:32PM

Speaker 1 00:00

I'm aware of it, obviously. And we were hating better work so much, and
after everything, we were like with better works. Yeah, but you know,
Marc, hello,

Unknown Speaker 00:16

hi, sorry, I'm late.

Speaker 1 00:18

Nah. That's all right. This is gonna be thing, a quick conversation.
What we want to do is just to deep dive into your thinking about HR
performance.

Speaker 2 01:21

Do Yeah, I think that makes sense.
"""

    # Create processor
    config = ConversationChunkConfig(detect_speakers=True, min_utterance_length=10)
    processor = ConversationDoclingProcessor(config)

    # Test speaker detection
    utterances = processor.detect_speakers_in_text(sample_transcript)
    print(f"Detected {len(utterances)} utterances:")
    for speaker_id, label, content in utterances:
        print(f"  {speaker_id} ({label}): {content[:50]}...")

    # Create chunks
    chunks = processor.create_conversation_chunks(sample_transcript, "test_transcript.txt")
    print(f"\nCreated {len(chunks)} chunks")

    # Show first chunk payload
    if chunks:
        print(f"\nSample payload:")
        print(json.dumps(chunks[0].to_payload(), indent=2))


if __name__ == "__main__":
    # Uncomment the line below to run the test function instead of main
    # test_with_sample_transcript()

    main()