import re

def clean_syllabus(input_file, output_file):
    # Read with explicit UTF-8, ignoring encoding errors
    with open(input_file, 'r', encoding='utf-8-sig', errors='replace') as f:
        content = f.read()
    
    # Fix common encoding issues
    content = content.replace('â—', '•')           # Fix bullet points
    content = content.replace('â€"', '–')         # Fix en-dashes
    content = content.replace('â€™', "'")        # Fix apostrophes
    content = content.replace('â€\x9d', '"')     # Fix quotes
    content = content.replace('â€\x9c', '"')     # Fix quotes
    
    # Remove copyright section (start until "Contents" begins)
    content = re.sub(r'(?s)^NSW Syllabus.*?Contents\n', '', content)
    
    # Remove table of contents/page numbers
    content = re.sub(r'^Contents\n.*?Glossary .*?\d+\n', '', content, flags=re.MULTILINE | re.DOTALL)
    
    # Remove '--- Label ---' formatting patterns
    content = re.sub(r'\n--- .+? ---\n', '\n', content)
    content = re.sub(r'^--- .+? ---$', '', content, flags=re.MULTILINE)
    
    # Remove [TABLE] and [/TABLE] markers
    content = re.sub(r'\n?\[/?TABLE\]?\n?', '', content)
    
    # Remove page number footers
    content = re.sub(r'\nEnglish Advanced Stage 6 Syllabus \d+\n', '\n', content)
    
    # Remove document IDs
    content = re.sub(r'\nDSSP[–-].*', '', content)
    content = re.sub(r'\nD2016/.*', '', content)
    
    # Clean up excessive blank lines (more than 2 consecutive)
    content = re.sub(r'\n\n\n+', '\n\n', content)
    
    # Strip trailing whitespace from lines
    lines = [line.rstrip() for line in content.split('\n')]
    content = '\n'.join(lines)
    
    # Remove leading/trailing blank lines
    content = content.strip()
    
    # Add back single newline at end
    content = content + '\n'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Cleaned and fixed encoding: {output_file}")

if __name__ == '__main__':
    clean_syllabus('nesa_syllabuses/nesa_english_advanced.txt', 
                   'nesa_syllabuses/nesa_english_advanced.txt')
