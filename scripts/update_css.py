import re

def update_css(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all selectors that end with 'i' (excluding pseudo-classes on 'i' for a moment, or including them)
    # Actually, we can just split by '{', then split by ',', check if selector ends with 'i' or 'i:hover', etc.
    
    def replace_selector(match):
        selectors_block = match.group(1)
        brace = match.group(2)
        
        # Split selectors by comma
        selectors = [s.strip() for s in selectors_block.split(',')]
        new_selectors = []
        for s in selectors:
            new_selectors.append(s)
            
            # If selector ends with " i" or " i:something"
            # It might also have `.lucide` already, but we haven't added it yet.
            # Example: ".btn-primary i"
            match_i = re.search(r'(.*?)\bi((?::[a-zA-Z-]+)?)$', s)
            if match_i:
                prefix = match_i.group(1)
                pseudo = match_i.group(2)
                # Ensure it's not actually ".ai" or something by checking if it ends with " i" or "> i"
                if prefix.endswith(' ') or prefix.endswith('>') or prefix == '':
                    new_selectors.append(f"{prefix}svg.lucide{pseudo}")

        # Rejoin selectors
        return ",\n".join(new_selectors) + " " + brace

    # We need to only match selector blocks. A rough regex:
    # Match any text up to '{' that isn't inside a comment or at-rule block (nested).
    # Since CSS regex parsing is hard, let's just do a simpler search-replace for the specific ones we found.
    
    known_i_selectors = [
        r'\.nav-brand\s+i',
        r'\.btn-primary\s+i',
        r'\.btn-primary:hover\s+i',
        r'\.btn-secondary\s+i',
        r'\.btn-secondary:hover\s+i',
        r'\.feature-card\s+i',
        r'\.feature-card:hover\s+i',
        r'\.ai-summary-card\s+li\s+i',
        r'\.comparison-result-card\.trend-stable\s+i',
        r'\.comparison-result-card\.trend-declined\s+i',
        r'\.comparison-result-card\.delta\s+i',
        r'\.comparison-hero-badge\s+i',
        r'\.comparison-warning-icon\s+i',
        r'\.comparison-error-icon\s+i'
    ]
    
    for pat in known_i_selectors:
        # e.g., "\.btn-primary i {" -> ".btn-primary i, .btn-primary svg.lucide {"
        def sub_func(m):
            sel = m.group(1).strip()
            # extract pseudo classes if any
            pseudo_match = re.search(r'(:[a-zA-Z-]+)?$', sel)
            pseudo = pseudo_match.group(1) if pseudo_match.group(1) else ''
            base_sel = sel
            if pseudo:
                base_sel = sel[:-len(pseudo)]
            # replace last 'i' with 'svg.lucide'
            lucide_sel = re.sub(r'i$', 'svg.lucide', base_sel) + pseudo
            return f"{m.group(1)}, {lucide_sel} {m.group(2)}"
            
        content = re.sub(r'(' + pat + r'(:[a-zA-Z-]+)?)\s*(\{)', sub_func, content)
        
    # Also add lucide base styles at the end
    base_styles = """

/* Lucide Icon Base Styles */
.lucide {
    width: 1.2em;
    height: 1.2em;
    stroke-width: 2;
    stroke: currentColor;
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
    vertical-align: middle;
    display: inline-block;
}
.lucide-spin {
    animation: lucide-spin 2s linear infinite;
}
@keyframes lucide-spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
"""
    if "/* Lucide Icon Base Styles */" not in content:
        content += base_styles

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("CSS updated successfully.")

if __name__ == '__main__':
    update_css('static/css/style.css')
