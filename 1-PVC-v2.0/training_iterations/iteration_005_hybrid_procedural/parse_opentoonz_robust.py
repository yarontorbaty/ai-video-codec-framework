#!/usr/bin/env python3
"""
Robust parser for OpenToonz project files (.tnz) using lxml with recovery mode.
This handles malformed XML with unescaped characters.
"""

try:
    from lxml import etree
    HAS_LXML = True
except ImportError:
    HAS_LXML = False
    import xml.etree.ElementTree as ET

import json
import re
from pathlib import Path
from typing import Dict, List, Any


def preprocess_xml(xml_path: str) -> str:
    """
    Preprocess XML to fix common issues like unescaped ampersands.
    """
    with open(xml_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Fix unescaped ampersands in attribute values
    # Match: value="text with & character" and replace with value="text with &amp; character"
    def fix_ampersand(match):
        text = match.group(1)
        # Only escape if not already escaped
        text = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)', '&amp;', text)
        return f'value="{text}"'
    
    content = re.sub(r'value="([^"]*)"', fix_ampersand, content)
    
    return content


def parse_tnz_file_robust(tnz_path: str) -> Dict[str, Any]:
    """Parse an OpenToonz .tnz file with robust error handling."""
    
    # Preprocess XML to fix malformed content
    xml_content = preprocess_xml(tnz_path)
    
    # Try parsing with lxml recovery mode first
    if HAS_LXML:
        try:
            parser = etree.XMLParser(recover=True, encoding='utf-8')
            root = etree.fromstring(xml_content.encode('utf-8'), parser=parser)
        except Exception as e:
            print(f"⚠️  lxml parsing failed: {e}")
            print(f"   Falling back to ElementTree...")
            root = ET.fromstring(xml_content)
    else:
        root = ET.fromstring(xml_content)
    
    analysis = {
        'filename': Path(tnz_path).name,
        'framecount': root.get('framecount'),
        'version': root.get('version'),
        'levels': [],
        'columns': [],
        'layer_structure': {},
        'procedural_operations': []
    }
    
    # Extract levels (reusable assets)
    for level in root.findall('.//levelSet/levels/level'):
        level_info = extract_level_info(level)
        if level_info:
            analysis['levels'].append(level_info)
    
    # Extract columns (timeline layers)
    for col in root.findall('.//xsheet/columns/levelColumn'):
        col_info = extract_column_info(col, analysis['levels'])
        if col_info:
            analysis['columns'].append(col_info)
    
    # Analyze layer structure
    analysis['layer_structure'] = analyze_layer_structure(analysis['levels'], analysis['columns'])
    
    # Extract procedural operations
    analysis['procedural_operations'] = extract_procedural_operations(root)
    
    return analysis


def extract_level_info(level) -> Dict[str, Any]:
    """Extract information about a level (asset)."""
    level_id = level.get('id')
    
    # Extract name
    name_elem = level.find('name')
    name = name_elem.text.strip() if name_elem is not None and name_elem.text else f"level_{level_id}"
    
    # Extract path
    path_elem = level.find('path')
    path = path_elem.text.strip() if path_elem is not None and path_elem.text else None
    
    # Determine type from extension
    level_type = "Unknown"
    if path:
        ext = Path(path).suffix.lower()
        if ext == '.tlv':
            level_type = 'ToonzRaster'  # Painted animation frames
        elif ext == '.pli':
            level_type = 'ToonzVector'  # Vector line art
        elif ext in ['.tif', '.tiff']:
            level_type = 'RasterImage'  # Background or scanned image
        elif ext in ['.png', '.jpg', '.jpeg']:
            level_type = 'RasterImage'
    
    return {
        'id': level_id,
        'name': name,
        'path': path,
        'type': level_type
    }


def extract_column_info(col, levels: List[Dict]) -> Dict[str, Any]:
    """Extract information about a column (timeline layer)."""
    col_id = col.get('id')
    
    name_elem = col.find('name')
    name = name_elem.text.strip() if name_elem is not None and name_elem.text else f"column_{col_id}"
    
    # Extract cell information (frame references)
    cells = []
    cells_elem = col.find('cells')
    if cells_elem is not None:
        for cell in cells_elem.findall('cell'):
            if cell.text:
                cell_text = cell.text.strip()
                # Parse: "0 3 <level id='9'/>0001 0"
                # Format: start_frame duration level_ref frame_number flags
                match = re.match(r'(\d+)\s+(\d+)\s+<level id=\'(\d+)\'/>([\w-]+)\s+(\d+)', cell_text)
                if match:
                    start_frame = int(match.group(1))
                    duration = int(match.group(2))
                    level_id = match.group(3)
                    frame_name = match.group(4)
                    
                    cells.append({
                        'start_frame': start_frame,
                        'end_frame': start_frame + duration,
                        'duration': duration,
                        'level_id': level_id,
                        'frame_name': frame_name
                    })
    
    return {
        'id': col_id,
        'name': name,
        'cells': cells,
        'cell_count': len(cells)
    }


def analyze_layer_structure(levels: List[Dict], columns: List[Dict]) -> Dict[str, Any]:
    """Analyze the layer structure to identify reuse patterns."""
    
    # Count asset reuse
    level_usage = {}
    for col in columns:
        for cell in col['cells']:
            level_id = cell['level_id']
            frame_name = cell['frame_name']
            key = f"{level_id}_{frame_name}"
            level_usage[key] = level_usage.get(key, 0) + 1
    
    # Find most reused assets
    reuse_stats = []
    for key, count in sorted(level_usage.items(), key=lambda x: x[1], reverse=True):
        level_id, frame_name = key.split('_', 1)
        level_name = next((l['name'] for l in levels if l['id'] == level_id), f"Level {level_id}")
        reuse_stats.append({
            'level_id': level_id,
            'frame_name': frame_name,
            'level_name': level_name,
            'reuse_count': count
        })
    
    return {
        'total_unique_assets': len(level_usage),
        'total_asset_instances': sum(level_usage.values()),
        'average_reuse': sum(level_usage.values()) / len(level_usage) if level_usage else 0,
        'top_reused_assets': reuse_stats[:10]
    }


def extract_procedural_operations(root) -> List[Dict[str, Any]]:
    """Extract parametric operations (effects, transforms)."""
    operations = []
    
    # Extract effects from xsheet
    for fx in root.findall('.//fx//'):
        if fx.tag and (fx.tag.startswith('Toonz_') or fx.tag.startswith('STD_')):
            fx_type = fx.tag.replace('Toonz_', '').replace('STD_', '')
            fx_id = fx.get('id', 'unknown')
            
            # Extract parameters
            params = {}
            for param in fx.findall('.//*'):
                if 'value' in param.attrib:
                    param_name = param.get('name', param.tag)
                    param_value = param.get('value')
                    params[param_name] = param_value
            
            operations.append({
                'type': 'Effect',
                'effect_type': fx_type,
                'id': fx_id,
                'parameters': params,
                'param_count': len(params)
            })
    
    # Extract camera movements
    for camera in root.findall('.//camera'):
        # Check if camera has keyframes (indicating movement)
        operations.append({
            'type': 'Camera',
            'has_movement': True,  # Simplified for now
            'parameters': {}
        })
    
    return operations


def create_markdown_report(analysis: Dict[str, Any], output_path: str):
    """Generate a detailed markdown report."""
    lines = []
    
    lines.append(f"# OpenToonz Project Analysis: {analysis['filename']}\n")
    lines.append(f"## Project Overview\n")
    lines.append(f"- **Total Frames:** {analysis['framecount']}")
    lines.append(f"- **OpenToonz Version:** {analysis['version']}")
    lines.append(f"- **Total Levels (Assets):** {len(analysis['levels'])}")
    lines.append(f"- **Total Columns (Layers):** {len(analysis['columns'])}\n")
    
    # Layer structure analysis
    struct = analysis['layer_structure']
    lines.append(f"## Asset Reuse Analysis\n")
    lines.append(f"- **Unique Assets:** {struct['total_unique_assets']}")
    lines.append(f"- **Total Asset Instances:** {struct['total_asset_instances']}")
    lines.append(f"- **Average Reuse per Asset:** {struct['average_reuse']:.2f}x")
    lines.append(f"- **Compression Opportunity:** {(1 - struct['total_unique_assets'] / max(struct['total_asset_instances'], 1)) * 100:.1f}%\n")
    
    lines.append(f"### Top 10 Most Reused Assets\n")
    for asset in struct['top_reused_assets'][:10]:
        lines.append(f"- **{asset['level_name']}** (Frame: {asset['frame_name']}) - Used {asset['reuse_count']}x")
    lines.append("")
    
    # Levels breakdown
    lines.append(f"## Levels (Assets)\n")
    by_type = {}
    for level in analysis['levels']:
        ltype = level['type']
        by_type.setdefault(ltype, []).append(level)
    
    for ltype, level_list in by_type.items():
        lines.append(f"### {ltype} ({len(level_list)} assets)\n")
        for level in level_list[:5]:  # Show first 5
            lines.append(f"- **{level['name']}** (ID: {level['id']})")
            if level['path']:
                lines.append(f"  - Path: `{level['path']}`")
        if len(level_list) > 5:
            lines.append(f"  - ... and {len(level_list) - 5} more")
        lines.append("")
    
    # Procedural operations
    lines.append(f"## Procedural Operations\n")
    lines.append(f"Total: {len(analysis['procedural_operations'])}\n")
    
    op_types = {}
    for op in analysis['procedural_operations']:
        op_type = f"{op['type']}:{op.get('effect_type', 'Unknown')}"
        op_types[op_type] = op_types.get(op_type, 0) + 1
    
    for op_type, count in sorted(op_types.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"- **{op_type}:** {count} instance(s)")
    lines.append("")
    
    # Codec implications
    lines.append(f"## Codec Implications\n")
    lines.append(f"Based on this project's structure:\n")
    lines.append(f"1. **Asset Storage:** Only {struct['total_unique_assets']} unique drawings need to be stored")
    lines.append(f"2. **Temporal Compression:** {struct['total_asset_instances'] - struct['total_unique_assets']} frames are references (0 bytes each)")
    lines.append(f"3. **Parametric Effects:** {len(analysis['procedural_operations'])} operations can be stored as parameters (~20-50 bytes each)")
    lines.append(f"4. **Estimated Savings:** {(1 - struct['total_unique_assets'] / max(struct['total_asset_instances'], 1)) * 100:.1f}% from asset reuse alone\n")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    """Parse all OpenToonz sample projects with robust error handling."""
    sample_dir = Path("/tmp/anime_drawing_analysis/OpenToonz_sample")
    output_dir = Path("/Users/yarontorbaty/Documents/Code/AiV1/1-PVC-v2.0/training_iterations/iteration_005_hybrid_procedural")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tnz_files = list(sample_dir.glob("*.tnz"))
    
    print(f"🎨 Found {len(tnz_files)} OpenToonz project files")
    print(f"")
    
    all_analysis = []
    
    for tnz_file in tnz_files:
        print(f"📊 Parsing {tnz_file.name}...")
        try:
            analysis = parse_tnz_file_robust(str(tnz_file))
            all_analysis.append(analysis)
            
            # Create markdown report
            report_file = output_dir / f"{tnz_file.stem}_analysis_robust.md"
            create_markdown_report(analysis, str(report_file))
            print(f"   ✅ Saved report to {report_file.name}")
            
            # Save raw JSON
            json_file = output_dir / f"{tnz_file.stem}_data_robust.json"
            with open(json_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"")
    print(f"🎯 Successfully parsed {len(all_analysis)} / {len(tnz_files)} files")


if __name__ == "__main__":
    # Try to install lxml if not available
    if not HAS_LXML:
        print("⚠️  lxml not found. Install for better XML parsing: pip install lxml")
        print("")
    main()

