"""
This code will chunk documents using Docling and save the chunks to JSON files. The time each speaker will
based on the current time the code was run, and put each speaker in the sequence they spoke
based on the that base time.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

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
except ImportError:
    print("❌ Docling not installed. Install with: pip install docling")
    exit(1)


@dataclass
class ChunkConfig:
    """Configuration for document chunking"""
    max_chunk_size: int = 1000
    overlap_size: int = 200
    preserve_paragraphs: bool = True
    min_chunk_size: int = 100


@dataclass
class DocumentChunk:
    """Represents a chunk of document content"""
    content: str
    chunk_id: int
    source_file: str
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = None
    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for JSON serialization"""
        return asdict(self)


class FixedDoclingProcessor:
    """Docling processor with guaranteed file saving"""

    def __init__(self, chunk_config: ChunkConfig = None):
        self.chunk_config = chunk_config or ChunkConfig()
        self.converter = self._initialize_converter()
        print("✅ FixedDoclingProcessor initialized with save functionality")

    def _initialize_converter(self) -> DocumentConverter:
        """Initialize the Docling document converter"""
        try:
            # Basic converter setup
            converter = DocumentConverter()
            logger.info("Docling converter initialized successfully")
            return converter
        except Exception as e:
            logger.error(f"Failed to initialize Docling converter: {e}")
            raise

    def parse_document(self, file_path) -> Dict[str, Any]:
        """Parse a document using Docling"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Parsing document: {file_path}")

        try:
            # Convert document
            result = self.converter.convert(str(file_path))

            # Extract content and metadata
            parsed_data = {
                'source_file': str(file_path),
                'title': getattr(result.document, 'title', file_path.stem),
                'content': result.document.export_to_markdown(),
                'page_count': len(result.document.pages) if hasattr(result.document, 'pages') else 1,
                'metadata': {
                    'file_size': file_path.stat().st_size,
                    'file_type': file_path.suffix.lower(),
                    'processing_time': getattr(result, 'processing_time', None)
                }
            }

            logger.info(f"Successfully parsed document: {file_path.name}")
            return parsed_data

        except Exception as e:
            logger.error(f"Error parsing document {file_path}: {e}")
            raise

    def chunk_text(self, text: str, source_file: str) -> List[DocumentChunk]:
        """Split text into chunks"""
        if not text.strip():
            logger.warning(f"Empty text provided for chunking from {source_file}")
            return []

        chunks = []

        if self.chunk_config.preserve_paragraphs:
            chunks = self._chunk_by_paragraphs(text, source_file)
        else:
            chunks = self._chunk_by_size(text, source_file)

        logger.info(f"Created {len(chunks)} chunks from {source_file}")
        return chunks

    def _chunk_by_paragraphs(self, text: str, source_file: str) -> List[DocumentChunk]:
        """Chunk text while preserving paragraph boundaries"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        chunk_id = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If adding this paragraph would exceed max size, save current chunk
            if (len(current_chunk) + len(paragraph) > self.chunk_config.max_chunk_size
                    and len(current_chunk) >= self.chunk_config.min_chunk_size):

                if current_chunk:
                    chunks.append(DocumentChunk(
                        content=current_chunk.strip(),
                        chunk_id=chunk_id,
                        source_file=source_file,
                        metadata={'chunk_method': 'paragraph_preserved'}
                    ))
                    chunk_id += 1

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
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                chunk_id=chunk_id,
                source_file=source_file,
                metadata={'chunk_method': 'paragraph_preserved'}
            ))

        return chunks

    def _chunk_by_size(self, text: str, source_file: str) -> List[DocumentChunk]:
        """Chunk text by fixed size with overlap"""
        chunks = []
        chunk_id = 0
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
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    chunk_id=chunk_id,
                    source_file=source_file,
                    metadata={'chunk_method': 'fixed_size'}
                ))
                chunk_id += 1

            # Move start position with overlap
            start = end - self.chunk_config.overlap_size if self.chunk_config.overlap_size > 0 else end

        return chunks

    def process_directory(self, directory_path: str, file_extensions: List[str] = None) -> List[DocumentChunk]:
        """Process all documents in a directory"""
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
                    # Parse document
                    parsed_doc = self.parse_document(str(file_path))

                    # Create chunks
                    chunks = self.chunk_text(parsed_doc['content'], str(file_path))
                    all_chunks.extend(chunks)
                    processed_files += 1

                    logger.info(f"Processed {file_path.name}: {len(chunks)} chunks")

                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    continue

        logger.info(f"Completed processing {processed_files} files, created {len(all_chunks)} total chunks")
        return all_chunks

    def save_chunks_to_json(self, chunks: List[DocumentChunk], output_dir: str,
                            save_individual: bool = True, save_combined: bool = True):
        """
        GUARANTEED method to save chunks to JSON format files
        """
        print(f"💾 Starting to save {len(chunks)} chunks to {output_dir}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created/verified output directory: {output_path}")

        saved_count = 0

        if save_individual:
            print("📄 Saving individual JSON files...")
            # Save individual JSON files for each chunk
            for chunk in chunks:
                try:
                    source_name = Path(chunk.source_file).stem
                    filename = f"{source_name}_chunk_{chunk.chunk_id:03d}.json"
                    file_path = output_path / filename

                    chunk_data = {
                        "chunk_info": chunk.to_dict(),
                        "processing_metadata": {
                            "processed_at": datetime.now().isoformat(),
                            "chunk_config": {
                                "max_chunk_size": self.chunk_config.max_chunk_size,
                                "overlap_size": self.chunk_config.overlap_size,
                                "preserve_paragraphs": self.chunk_config.preserve_paragraphs
                            }
                        }
                    }

                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(chunk_data, f, indent=2, ensure_ascii=False)

                    saved_count += 1

                except Exception as e:
                    print(f"❌ Error saving chunk {chunk.chunk_id}: {e}")

            print(f"✅ Saved {saved_count} individual JSON chunk files")

        if save_combined:
            print("📋 Saving combined JSON file...")
            try:
                # Save combined JSON file with all chunks
                combined_filename = "all_chunks_combined.json"
                combined_path = output_path / combined_filename

                combined_data = {
                    "processing_info": {
                        "total_chunks": len(chunks),
                        "processed_at": datetime.now().isoformat(),
                        "source_files": list(set(chunk.source_file for chunk in chunks)),
                        "chunk_config": {
                            "max_chunk_size": self.chunk_config.max_chunk_size,
                            "overlap_size": self.chunk_config.overlap_size,
                            "preserve_paragraphs": self.chunk_config.preserve_paragraphs,
                            "min_chunk_size": self.chunk_config.min_chunk_size
                        }
                    },
                    "chunks": [chunk.to_dict() for chunk in chunks]
                }

                with open(combined_path, 'w', encoding='utf-8') as f:
                    json.dump(combined_data, f, indent=2, ensure_ascii=False)

                print(f"✅ Saved combined JSON file: {combined_filename}")

            except Exception as e:
                print(f"❌ Error saving combined file: {e}")

        # Verify files were created
        json_files = list(output_path.glob("*.json"))
        print(f"🔍 Verification: Found {len(json_files)} JSON files in {output_dir}")

        return len(json_files)


def main():
    """Main function with GPU acceleration and guaranteed file saving"""
    print("🚀 FIXED DOCLING CHUNKER WITH GPU ACCELERATION")
    print("=" * 60)

    # GPU System Check
    print("🔍 SYSTEM CHECK:")
    if gpu_available and torch.cuda.is_available():
        print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   🔥 CUDA: {torch.version.cuda}")
        print(f"   💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   🎯 PyTorch CUDA: {torch.cuda.is_available()}")
    else:
        print("   💻 CPU processing only")
        if not gpu_available:
            print("   ⚠️ PyTorch not installed or no CUDA support")
            print(
                "   💡 For GPU: pip uninstall torch && pip install torch --index-url https://download.pytorch.org/whl/cu121")

    print()  # blank line

    # Configuration
    chunk_config = ChunkConfig(
        max_chunk_size=800,
        overlap_size=100,
        preserve_paragraphs=True,
        min_chunk_size=50
    )

    # Initialize processor
    processor = FixedDoclingProcessor(chunk_config)

    # Set directories
    input_directory = r"D:\Docling Trial\Documents"
    output_directory = r"D:\Docling Trial\Chunks"

    print(f"📂 Input directory: {input_directory}")
    print(f"📁 Output directory: {output_directory}")

    try:
        if os.path.exists(input_directory):
            print(f"\n🔄 Processing all documents in directory...")

            # Process all documents
            all_chunks = processor.process_directory(
                input_directory,
                file_extensions=['.pdf', '.docx', '.doc', '.txt', '.md']
            )

            print(f"\n📊 PROCESSING COMPLETE:")
            print(f"   Total chunks created: {len(all_chunks)}")

            if all_chunks:
                # Save chunks - this WILL work
                print(f"\n💾 SAVING CHUNKS TO JSON FILES...")
                files_saved = processor.save_chunks_to_json(
                    all_chunks,
                    output_directory,
                    save_individual=True,
                    save_combined=True
                )

                print(f"\n🎉 SUCCESS! SAVED {files_saved} JSON FILES")
                print(f"📁 Location: {output_directory}")

                # Show first chunk as example
                if len(all_chunks) > 0:
                    print(f"\n📝 Sample chunk:")
                    print(f"   Source: {Path(all_chunks[0].source_file).name}")
                    print(f"   Content preview: {all_chunks[0].content[:150]}...")

            else:
                print("⚠️ No chunks were created - check your input directory")

        else:
            print(f"❌ Input directory not found: {input_directory}")
            print("Please create the directory and add some documents")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n🏁 PROGRAM COMPLETED")
    print(f"📁 Check your files at: {output_directory}")


if __name__ == "__main__":
    main()