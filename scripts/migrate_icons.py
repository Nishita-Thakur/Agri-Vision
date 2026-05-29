import os
import re

ICON_MAP = {
    'fa-camera': 'camera',
    'fa-exclamation-circle': 'alert-circle',
    'fa-trash-alt': 'trash-2',
    'fa-cloud-upload-alt': 'upload-cloud',
    'fa-arrow-rotate-left': 'rotate-ccw',
    'fa-play': 'play',
    'fa-magnifying-glass': 'search',
    'fa-lightbulb': 'lightbulb',
    'fa-check': 'check',
    'fa-info-circle': 'info',
    'fa-triangle-exclamation': 'alert-triangle',
    'fa-spinner': 'loader-2',
    'fa-seedling': 'leaf',
    'fa-bars': 'menu',
    'fa-sign-out-alt': 'log-out',
    'fa-stethoscope': 'stethoscope',
    'fa-list': 'list',
    'fa-search': 'search',
    'fa-search-plus': 'zoom-in',
    'fa-times': 'x',
    'fa-circle-question': 'help-circle',
    'fa-screwdriver-wrench': 'wrench',
    'fa-headset': 'headphones',
    'fa-map-marker-alt': 'map-pin',
    'fa-arrow-up': 'arrow-up',
    'fa-leaf': 'leaf',
    'fa-eye': 'eye',
    'fa-circle-check': 'check-circle-2',
    'fa-brain': 'brain',
    'fa-globe': 'globe',
    'fa-moon': 'moon',
    'fa-chart-line': 'line-chart',
    'fa-heartbeat': 'activity',
    'fa-calendar-check': 'calendar-check',
    'fa-play-circle': 'play-circle',
    'fa-bolt': 'zap',
    'fa-satellite': 'satellite',
    'fa-chevron-down': 'chevron-down',
    'fa-chevron-up': 'chevron-up',
    'fa-chevron-right': 'chevron-right',
    'fa-chevron-left': 'chevron-left',
    'fa-user': 'user',
    'fa-envelope': 'mail',
    'fa-lock': 'lock',
    'fa-plus': 'plus',
    'fa-minus': 'minus',
    'fa-cog': 'settings',
    'fa-cogs': 'settings',
    'fa-tachometer-alt': 'gauge',
    'fa-flask': 'flask-conical',
    'fa-download': 'download',
    'fa-upload': 'upload',
    'fa-home': 'home',
    'fa-history': 'history',
    'fa-book': 'book',
    'fa-graduation-cap': 'graduation-cap',
    'fa-newspaper': 'newspaper',
    'fa-shield-alt': 'shield',
    'fa-robot': 'bot',
    'fa-file-alt': 'file-text',
    'fa-file': 'file',
    'fa-link': 'link',
    'fa-external-link-alt': 'external-link',
    'fa-video': 'video',
    'fa-image': 'image',
    'fa-check-circle': 'check-circle',
    'fa-arrow-left': 'arrow-left',
    'fa-arrow-right': 'arrow-right'
}

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Remove fontawesome link
    content = re.sub(r'<link[^>]*font-awesome[^>]*>', '', content)

    # Function to replace the <i> tag
    def replace_icon(match):
        full_tag = match.group(0)
        class_attr = match.group(1)
        
        # Extract the fa- class
        fa_match = re.search(r'\bfa-([a-zA-Z0-9-]+)\b', class_attr)
        if not fa_match:
            return full_tag
            
        fa_class = fa_match.group(0)
        lucide_name = ICON_MAP.get(fa_class, fa_class.replace('fa-', ''))
        
        # Remove fa- classes and fas/far/fab
        new_class = re.sub(r'\b(fas|far|fab|fa|fa-[a-zA-Z0-9-]+)\b\s*', '', class_attr).strip()
        
        # Build new tag
        if new_class:
            # Reconstruct the tag with remaining classes
            # Note: We need to put data-lucide and class
            new_tag = full_tag.replace(f'class="{class_attr}"', f'data-lucide="{lucide_name}" class="{new_class}"')
        else:
            # No classes left
            new_tag = full_tag.replace(f'class="{class_attr}"', f'data-lucide="{lucide_name}"')
            # cleanup empty class attribute if it was left by replace
            new_tag = new_tag.replace(' class=""', '')
            
        # Handle fa-spin
        if 'fa-spin' in class_attr:
            if 'class="' in new_tag:
                new_tag = new_tag.replace('class="', 'class="lucide-spin ')
            else:
                new_tag = new_tag.replace(f'data-lucide="{lucide_name}"', f'data-lucide="{lucide_name}" class="lucide-spin"')
                
        return new_tag

    # Replace <i class="..."> tags
    # Also handle <i class='...'>
    content = re.sub(r'<i[^>]*class="([^"]*fa-[^"]*)"[^>]*>.*?</i>', replace_icon, content)
    content = re.sub(r"<i[^>]*class='([^']*fa-[^']*)'[^>]*>.*?</i>", replace_icon, content)

    # In javascript templates, we might have literal strings like '\'<i class="fas fa-spinner fa-spin"></i>\''
    # The above regex handles it.

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {filepath}")

def main():
    # Process templates
    templates_dir = 'templates'
    for root, dirs, files in os.walk(templates_dir):
        for file in files:
            if file.endswith('.html'):
                process_file(os.path.join(root, file))

    # Process js
    js_dir = 'static/js'
    for root, dirs, files in os.walk(js_dir):
        for file in files:
            if file.endswith('.js'):
                process_file(os.path.join(root, file))
                
if __name__ == '__main__':
    main()
