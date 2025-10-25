#!/usr/bin/env python3
"""
Parse OpenToonz project files (.tnz) to extract layer structure and operations.
This will help us understand the procedural nature of anime production.
"""

import xml.etree.ElementTree as ET
import json
from pathlib import Path
from typing import Dict, List, Any

def parse_tnz_file(tnz_path: str) -> Dict[str, Any]:
    """Parse an OpenToonz .tnz file to extract structure."""
    tree = ET.parse(tnz_path)
    root = tree.getroot()
    
    analysis = {
        'filename': Path(tnz_path).name,
        'framecount': root.get('framecount'),
        'version': root.get('version'),
        'levels': [],
        'columns': [],
        'effects': [],
        'camera': {},
        'output': {}
    }
    
    # Extract camera settings
    camera = root.find('.//camera')
    if camera is not None:
        camera_size = camera.find('cameraSize')
        camera_res = camera.find('cameraRes')
        if camera_size is not None:
            analysis['camera']['size'] = camera_size.text.strip()
        if camera_res is not None:
            analysis['camera']['resolution'] = camera_res.text.strip()
    
    # Extract output settings
    output = root.find('.//output[@name="main"]')
    if output is not None:
        fps = output.find('fps')
        if fps is not None:
            analysis['output']['fps'] = fps.text.strip()
    
    # Extract levels (assets/drawings)
    levelset = root.find('.//levelSet/levels')
    if levelset is not None:
        for level in levelset.findall('level'):
            level_id = level.get('id')
            level_info = {
                'id': level_id,
                'name': level.find('name').text.strip() if level.find('name') is not None else f"level_{level_id}",
                'path': level.find('path').text.strip() if level.find('path') is not None else None,
                'type': None
            }
            # Determine level type from path extension
            if level_info['path']:
                ext = Path(level_info['path']).suffix.lower()
                if ext == '.tlv':
                    level_info['type'] = 'ToonzRaster'  # Vector/raster animation
                elif ext == '.pli':
                    level_info['type'] = 'ToonzVector'  # Vector animation
                elif ext in ['.tif', '.png', '.jpg']:
                    level_info['type'] = 'Raster'  # Background/static image
            
            analysis['levels'].append(level_info)
    
    # Extract columns (layers in the timeline)
    xsheet = root.find('.//xsheet/columns')
    if xsheet is not None:
        for col in xsheet.findall('levelColumn'):
            col_id = col.get('id')
            col_info = {
                'id': col_id,
                'name': col.find('name').text.strip() if col.find('name') is not None else f"column_{col_id}",
                'frames': [],
                'effects': []
            }
            
            # Extract frame references
            cells = col.find('cells')
            if cells is not None:
                for cell in cells.findall('cell'):
                    frame_info = cell.text.strip()
                    # Parse: "0 3 <level id='9'/>0001 0"
                    # Format: start_frame duration level_reference frame_number flags
                    col_info['frames'].append(frame_info)
            
            # Extract effects on this column
            fx_elem = col.find('fx')
            if fx_elem is not None:
                for fx in fx_elem.findall('.//'):
                    if fx.tag and fx.tag.startswith('Toonz_'):
                        fx_type = fx.tag.replace('Toonz_', '')
                        fx_id = fx.get('id')
                        fx_info = {
                            'type': fx_type,
                            'id': fx_id
                        }
                        col_info['effects'].append(fx_info)
            
            analysis['columns'].append(col_info)
    
    # Extract global effects
    fxtree = root.find('.//fxs')
    if fxtree is not None:
        for fx in fxtree.findall('.//'):
            if fx.tag and fx.tag.startswith('Toonz_') or fx.tag and fx.tag.startswith('STD_'):
                fx_type = fx.tag.replace('Toonz_', '').replace('STD_', '')
                fx_id = fx.get('id')
                fx_info = {
                    'type': fx_type,
                    'id': fx_id,
                    'params': {}
                }
                
                # Extract parameters
                for param in fx.findall('.//'):
                    if param.tag == 'param' or param.tag.endswith('Param'):
                        param_name = param.get('name') or param.tag
                        param_value = param.text.strip() if param.text else None
                        fx_info['params'][param_name] = param_value
                
                analysis['effects'].append(fx_info)
    
    return analysis


def summarize_analysis(analysis: Dict[str, Any]) -> str:
    """Create a human-readable summary of the OpenToonz project."""
    summary = []
    summary.append(f"# OpenToonz Project Analysis: {analysis['filename']}")
    summary.append(f"")
    summary.append(f"## Project Metadata")
    summary.append(f"- **Frame Count:** {analysis['framecount']}")
    summary.append(f"- **Version:** {analysis['version']}")
    summary.append(f"- **Camera Resolution:** {analysis['camera'].get('resolution', 'N/A')}")
    summary.append(f"- **Camera Size:** {analysis['camera'].get('size', 'N/A')}")
    summary.append(f"- **FPS:** {analysis['output'].get('fps', 'N/A')}")
    summary.append(f"")
    
    summary.append(f"## Levels (Assets)")
    summary.append(f"Total: {len(analysis['levels'])}")
    summary.append(f"")
    
    # Group levels by type
    by_type = {}
    for level in analysis['levels']:
        ltype = level.get('type', 'Unknown')
        by_type.setdefault(ltype, []).append(level)
    
    for ltype, levels in by_type.items():
        summary.append(f"### {ltype} ({len(levels)})")
        for level in levels:
            summary.append(f"- **{level['name']}** (ID: {level['id']}) - `{level['path']}`")
        summary.append(f"")
    
    summary.append(f"## Columns (Layers)")
    summary.append(f"Total: {len(analysis['columns'])}")
    summary.append(f"")
    
    for col in analysis['columns']:
        summary.append(f"### Column: {col['name']} (ID: {col['id']})")
        summary.append(f"- **Frame Count:** {len(col['frames'])}")
        if col['effects']:
            summary.append(f"- **Effects Applied:** {', '.join(fx['type'] for fx in col['effects'])}")
        summary.append(f"")
    
    summary.append(f"## Effects")
    summary.append(f"Total: {len(analysis['effects'])}")
    summary.append(f"")
    
    for fx in analysis['effects']:
        summary.append(f"### {fx['type']} (ID: {fx['id']})")
        if fx['params']:
            summary.append(f"**Parameters:**")
            for param_name, param_value in list(fx['params'].items())[:10]:  # Show first 10 params
                summary.append(f"- `{param_name}`: {param_value}")
        summary.append(f"")
    
    return "\n".join(summary)


def main():
    """Parse all OpenToonz sample projects."""
    sample_dir = Path("/tmp/anime_drawing_analysis/OpenToonz_sample")
    output_dir = Path("/Users/yarontorbaty/Documents/Code/AiV1/1-PVC-v2.0/training_iterations/iteration_005_hybrid_procedural")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tnz_files = list(sample_dir.glob("*.tnz"))
    
    print(f"🎨 Found {len(tnz_files)} OpenToonz project files")
    print(f"")
    
    all_analysis = {}
    
    for tnz_file in tnz_files:
        print(f"📊 Parsing {tnz_file.name}...")
        try:
            analysis = parse_tnz_file(str(tnz_file))
            all_analysis[tnz_file.stem] = analysis
            
            # Save individual analysis
            summary = summarize_analysis(analysis)
            output_file = output_dir / f"{tnz_file.stem}_analysis.md"
            output_file.write_text(summary)
            print(f"   ✅ Saved to {output_file.name}")
            
            # Save raw JSON
            json_file = output_dir / f"{tnz_file.stem}_data.json"
            with open(json_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"")
    print(f"🎯 Analysis complete!")
    print(f"")
    
    # Create master summary
    master_summary = []
    master_summary.append(f"# OpenToonz Sample Projects - Master Analysis")
    master_summary.append(f"")
    master_summary.append(f"Analyzed {len(all_analysis)} projects.")
    master_summary.append(f"")
    
    for project_name, analysis in all_analysis.items():
        master_summary.append(f"## {project_name}")
        master_summary.append(f"- **Frames:** {analysis['framecount']}")
        master_summary.append(f"- **Levels:** {len(analysis['levels'])}")
        master_summary.append(f"- **Columns:** {len(analysis['columns'])}")
        master_summary.append(f"- **Effects:** {len(analysis['effects'])}")
        master_summary.append(f"")
    
    # Key insights for procedural codec
    master_summary.append(f"## Key Insights for Procedural Codec")
    master_summary.append(f"")
    master_summary.append(f"### Layer Structure")
    master_summary.append(f"OpenToonz projects are organized in a hierarchical structure:")
    master_summary.append(f"1. **Levels** - Reusable assets (drawings, backgrounds)")
    master_summary.append(f"2. **Columns** - Timeline layers that reference levels")
    master_summary.append(f"3. **Effects** - Transformations applied to layers")
    master_summary.append(f"")
    master_summary.append(f"### Procedural Nature")
    master_summary.append(f"- Anime frames are **composited from layers**")
    master_summary.append(f"- Each layer is a **reference to a reusable asset** (not raw pixels)")
    master_summary.append(f"- Effects are **parametric** (e.g., blur radius, color adjustment)")
    master_summary.append(f"- Timeline is **frame-based** with explicit timing control")
    master_summary.append(f"")
    master_summary.append(f"### Codec Implications")
    master_summary.append(f"Our neural codec should:")
    master_summary.append(f"1. **Decompose frames into layers** (line art, color, shadow)")
    master_summary.append(f"2. **Identify reusable assets** across frames")
    master_summary.append(f"3. **Predict parametric operations** (transforms, effects)")
    master_summary.append(f"4. **Use temporal references** (level reuse = asset compression)")
    master_summary.append(f"")
    
    master_file = output_dir / "OPENTOONZ_MASTER_ANALYSIS.md"
    master_file.write_text("\n".join(master_summary))
    print(f"📄 Master summary saved to {master_file.name}")


if __name__ == "__main__":
    main()

