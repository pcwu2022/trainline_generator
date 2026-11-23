"""Country-prefixed visual exports, sorted in social upload order."""
import re
from pathlib import Path


def visual_names(config):
    country = re.sub(r'[^a-z0-9]+', '_', config['reporting']['region_name'].lower()).strip('_')
    return dict(animation=f'_{country}_1_animation.mp4', analysis1=f'_{country}_2_analysis1.png',
                analysis2=f'_{country}_3_analysis2.png', progress=f'_{country}_4_progress.png',
                map=f'_{country}_5_map.png')


def migrate_visual_names(output, config):
    """Keep existing snapshots and resumable video state when adopting new names."""
    output = Path(output)
    names = visual_names(config)
    replacements = dict(zip(('progress.mp4', 'analysis.png', 'analysis_financial.png',
                             'progress.png', 'latest.png'), names.values()))
    # Repair earlier state-file renames; only upload media get country prefixes.
    previous_state = output / Path(names['animation']).with_suffix('.json')
    if previous_state.exists() and not (output / 'progress.json').exists():
        previous_state.rename(output / 'progress.json')
    for old, new in replacements.items():
        if (output / old).exists() and not (output / new).exists():
            (output / old).rename(output / new)
    report = output / 'report.md'
    if report.exists():
        content = report.read_text()
        for old, new in replacements.items():
            content = content.replace(f']({old})', f']({new})')
        content = '\n'.join(line for line in content.splitlines()
                            if '](latest_revenue_cost.png)' not in line) + '\n'
        report.write_text(content)
    return names
