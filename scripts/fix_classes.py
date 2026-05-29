import os
import re

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content
    # Look for class="-something" or class="... -something ..."
    # Specifically, we want to remove \b-[a-zA-Z0-9-]+\b when it's inside <i class="...">
    
    def replacer(match):
        full_tag = match.group(0)
        class_attr = match.group(1)
        # remove -something
        new_class = re.sub(r'(^|\s)-[a-zA-Z0-9-]+(\s|$)', r'\1\2', class_attr)
        new_class = re.sub(r'\s+', ' ', new_class).strip()
        if new_class:
            return full_tag.replace(f'class="{class_attr}"', f'class="{new_class}"')
        else:
            return full_tag.replace(f' class="{class_attr}"', '')

    content = re.sub(r'<i\s+[^>]*?class="([^"]*)"[^>]*>', replacer, content)
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed {filepath}")

for root, _, files in os.walk('templates'):
    for f in files:
        if f.endswith('.html') or f.endswith('.js'):
            fix_file(os.path.join(root, f))
            
for root, _, files in os.walk('static/js'):
    for f in files:
        if f.endswith('.js'):
            fix_file(os.path.join(root, f))
