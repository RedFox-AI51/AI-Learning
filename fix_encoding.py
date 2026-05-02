import re

def convert_encoding(input_file, output_file):
    """Try multiple encodings to properly read and convert the file"""
    
    content = None
    tried_encodings = []
    
    # Try different encodings
    for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']:
        try:
            with open(input_file, 'r', encoding=encoding, errors='ignore') as f:
                test_content = f.read()
                # Check if this encoding worked by looking for readable text
                if 'English Advanced' in test_content and len(test_content) > 100000:
                    content = test_content
                    tried_encodings.append(f"{encoding} ✓")
                    print(f"✓ Successfully read with encoding: {encoding}")
                    break
                else:
                    tried_encodings.append(encoding)
        except:
            tried_encodings.append(f"{encoding} (failed)")
    
    if content is None:
        print(f"Could not read file with any standard encoding. Tried: {', '.join(tried_encodings)}")
        return
    
    # Fix corrupted UTF-8 sequences from PDF extraction
    replacements = [
        ('â€"', '–'),      # En-dash
        ('â€™', "'"),      # Right single quote
        ('â—', '•'),       # Bullet
        ('Kâ€"12', 'K–12'),
        ('Kâ€"10', 'K–10'),
    ]
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    # Remove any remaining obvious corruption patterns
    content = re.sub(r'â€.{1,3}', '', content)
    
    # Clean up the content
    # Remove copyright section
    content = re.sub(r'(?s)^NSW Syllabus.*?Introduction', 'Introduction', content)
    
    # Remove '--- Label ---' formatting patterns
    content = re.sub(r'\n--- .+? ---\n', '\n', content)
    content = re.sub(r'^--- .+? ---$', '', content, flags=re.MULTILINE)
    
    # Remove [TABLE] and [/TABLE] markers
    content = re.sub(r'\[/?TABLE\]', '', content)
    
    # Remove page number footers  
    content = re.sub(r'\nEnglish Advanced Stage 6 Syllabus \d+\n', '\n', content)
    
    # Remove document IDs
    content = re.sub(r'\nDSSP.*', '', content)
    content = re.sub(r'\nD2016.*', '', content)
    
    # Clean up excessive blank lines
    content = re.sub(r'\n\n\n+', '\n\n', content)
    
    # Strip trailing whitespace from lines
    lines = [line.rstrip() for line in content.split('\n')]
    content = '\n'.join(lines)
    
    # Remove leading/trailing blank lines
    content = content.strip() + '\n'
    
    # Write with UTF-8
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Cleaned and saved: {output_file}")
    print(f"   Lines: {len(content.split(chr(10)))}")

if __name__ == '__main__':
    convert_encoding('/kaggle/input/datasets/ninjanick/nesa-core-subjects/nesa_english_advanced.txt',
                     '/kaggle/input/datasets/ninjanick/nesa-core-subjects/nesa_english_advanced.txt')
