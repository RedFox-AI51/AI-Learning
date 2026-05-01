import re

# Read the current file
with open('nesa_syllabuses/nesa_english_advanced.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# Find where "Introduction" starts (this is the real content)
intro_pos = content.find('Introduction\nStage 6')
if intro_pos == -1:
    intro_pos = content.find('Introduction')

# Keep only from "Introduction" onward
if intro_pos > 0:
    content = content[intro_pos:]

# Fix all remaining Unicode corruption issues by replacing problematic bytes
content = content.encode('utf-8', errors='ignore').decode('utf-8')

# Replace common corrupted patterns with clean versions
replacements = {
    'Kâ': 'K-',           # K-12, K-10 variants
    'â€': '-',            # Various dashes
    'â': '',              # Remove all remaining 'â' characters
}

for old, new in replacements.items():
    content = content.replace(old, new)

# Clean up any double dashes
content = re.sub(r'--+', '-', content)

# Remove excessive blank lines
content = re.sub(r'\n\n\n+', '\n\n', content)

# Add header
header = 'NESA Stage 6 - English Advanced (2017)\nNSW Syllabus for the Australian Curriculum\n\n'
content = header + content

# Clean up edges
content = content.strip() + '\n'

# Write clean version
with open('nesa_syllabuses/nesa_english_advanced.txt', 'w', encoding='utf-8') as f:
    f.write(content)

line_count = len(content.split('\n'))
char_count = len(content)

print(f"✓ Clean syllabus created")
print(f"  Lines: {line_count}")
print(f"  Characters: {char_count}")
