import re
from typing import List, Optional

class CustomVTTCue:
    """
    Represents a block in a VTT file. 
    It can be a Cue (with timing) or a Metadata block (Header, Style, Comment).
    """
    def __init__(self, raw_lines: List[str]):
        self.raw_lines = raw_lines
        self.timestamp_line_index = -1
        self.payload = []
        self._parse()

    def _parse(self):
        # Timestamp pattern: 00:00:00.000 --> 00:00:00.000
        # Supports optional hours and various spacing
        timestamp_pattern = re.compile(r"(\d{2}:)?\d{2}:\d{2}\.\d{3}\s+-->\s+(\d{2}:)?\d{2}:\d{2}\.\d{3}")
        
        for i, line in enumerate(self.raw_lines):
            if timestamp_pattern.search(line):
                self.timestamp_line_index = i
                break
        
        if self.is_cue:
            # Everything after timestamp line is payload
            if self.timestamp_line_index + 1 < len(self.raw_lines):
                 self.payload = self.raw_lines[self.timestamp_line_index + 1:]
            else:
                 self.payload = [] # Empty cue

    @property
    def is_cue(self) -> bool:
        return self.timestamp_line_index != -1

    @property
    def text(self) -> str:
        """Returns the payload text (subtitle content)."""
        if not self.is_cue:
            return ""
        return "\n".join(self.payload)

    @text.setter
    def text(self, value: str):
        """Updates the payload text."""
        if self.is_cue:
            self.payload = value.split("\n")

    def __str__(self):
        lines = []
        if self.is_cue:
            # Identifier (if any) + Timestamp line
            lines.extend(self.raw_lines[:self.timestamp_line_index+1])
            # Payload
            lines.extend(self.payload)
        else:
            # Just raw lines for headers/styles
            lines.extend(self.raw_lines)
        return "\n".join(lines)

class CustomVTTFile:
    """
    Parses a VTT file into blocks to preserve structure (styles, comments, positioning).
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.header = []
        self.blocks = []
        self._parse()

    def _parse(self):
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            # Allow creating empty/new for resumption scenarios if needed, 
            # though usually we read existing.
            return

        # VTT blocks are separated by blank lines (double newline)
        # We accept \n\n+ as separator.
        raw_blocks = re.split(r'\n\s*\n', content)
        
        # Handle WEBVTT header specially if it's the first block
        if raw_blocks and raw_blocks[0].strip().startswith("WEBVTT"):
            self.header = raw_blocks[0].strip().split("\n")
            raw_blocks = raw_blocks[1:]
        
        for blk in raw_blocks:
            if not blk.strip():
                continue
            lines = blk.split("\n")
            self.blocks.append(CustomVTTCue(lines))

    def get_cues(self) -> List[CustomVTTCue]:
        """Returns only the blocks that are actual cues."""
        return [b for b in self.blocks if b.is_cue]

    def save(self, out_path: str):
        with open(out_path, 'w', encoding='utf-8') as f:
            if self.header:
                f.write("\n".join(self.header) + "\n\n")
            
            # Write blocks separated by newlines
            # Ensure we don't add excessive newlines at the end
            blocks_str = []
            for blk in self.blocks:
                blocks_str.append(str(blk))
            
            f.write("\n\n".join(blocks_str))
            f.write("\n") # Final newline
