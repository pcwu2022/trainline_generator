"""Shared square editorial layout for social chart exports."""
from matplotlib.patches import FancyBboxPatch

INK = '#173348'
MUTED = '#576b78'
BLUE = '#2378ab'
GREEN = '#22866a'
RED = '#c65350'
GOLD = '#b98230'


def canvas(plt, heading, subtitle, stats):
    fig = plt.figure(figsize=(12, 12), layout='none', facecolor='#edf2f5')
    fig.text(.045, .955, heading, fontsize=26, weight='bold', color=INK, va='top')
    fig.text(.045, .913, subtitle, fontsize=12, color=MUTED, va='top')
    for i, (value, label) in enumerate(stats):
        x = .045 + i * .31
        fig.text(x, .868, value, fontsize=24, weight='bold', color=INK, va='top')
        fig.text(x, .832, label, fontsize=11, color=MUTED, va='top')
    return fig


def card(fig, x, y, w, h, title, description):
    fig.add_artist(FancyBboxPatch((x, y), w, h, transform=fig.transFigure,
        boxstyle='round,pad=0.009,rounding_size=0.012', facecolor='white',
        edgecolor='none', zorder=-1))
    fig.text(x + .015, y + h - .016, title, fontsize=15, weight='bold', color=INK, va='top')
    fig.text(x + .015, y + h - .044, description, fontsize=10, color=MUTED, va='top', linespacing=1.35)


def footer(fig, lines):
    fig.text(.045, .059, lines, fontsize=10, color=MUTED, va='top', linespacing=1.5)
    fig.text(.955, .016, 'rail_gen_master  /  MODELLED RESULTS', fontsize=9,
             color=MUTED, ha='right')


def save(fig, path):
    # Explicit bounds keep exports square even with user-wide tight-bbox settings.
    import matplotlib
    with matplotlib.rc_context({'savefig.bbox': None}):
        fig.savefig(path, dpi=90, facecolor=fig.get_facecolor())
