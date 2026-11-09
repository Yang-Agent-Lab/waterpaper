"""Plot copied metadata JSON beside this script; importing has no side effects."""
import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch
    from matplotlib.backends.backend_pdf import PdfPages
    OUT = Path(__file__).resolve().parent
    data = json.loads((OUT / 'basin_attribute_plot_data.json').read_text())
    report = PdfPages(OUT / 'basin_attribute_report.pdf')
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ['PingFang SC', 'Heiti TC', 'Noto Sans CJK SC',
                            'Noto Sans CJK JP', 'Droid Sans Fallback'] if f in available), None)
    if font is None:
        raise RuntimeError('No approved Chinese font found; install nothing automatically')
    plt.rcParams.update({'font.family': font, 'font.sans-serif': [font, 'Arial'],
                         'font.size': 9, 'axes.labelsize': 9,
                         'axes.titlesize': 9, 'legend.fontsize': 9, 'xtick.labelsize': 9,
                         'ytick.labelsize': 9, 'legend.frameon': False, 'axes.linewidth': .75,
                         'xtick.direction': 'in', 'ytick.direction': 'in', 'pdf.fonttype': 42,
                         'axes.unicode_minus': False})
    countries = ['China', 'Germany', 'Iceland']; names = ['中国', '德国', '冰岛']
    colors = ['#B8D8CF', '#F2D7A0', '#E3E3E3']
    labels = ['面积', '长期降雨', '平均高程', '坡度', '河长', '土壤', '土地覆盖', '精确坐标']
    fig, ax = plt.subplots(figsize=(150 / 25.4, 112.5 / 25.4))
    for y, country in enumerate(countries):
        for x, state in enumerate(data['states'][country]):
            ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, facecolor=colors[state],
                                       edgecolor='white', linewidth=1))
            text = ['已核实', '待核实', '未取得'][state]
            if x in (0, 1):
                c = data['coverage'][country]
                key = 'area_finite' if x == 0 else 'precip_finite'
                text = f"{c[key]}/{c['candidate_stations']}"
            ax.text(x, y, text, ha='center', va='center')
    ax.set(xlim=(-.5, 7.5), ylim=(2.5, -.5), xticks=range(8), yticks=range(3))
    ax.set_xticklabels(labels, rotation=30, ha='right'); ax.set_yticklabels(names)
    ax.set_title('各国流域属性资料的取得状态')
    ax.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(colors, data['state_labels'])],
              loc='upper center', bbox_to_anchor=(.5, -.35), ncol=3)
    fig.text(.5, .035, '首版使用前两列；长期降雨统计期不同', ha='center')
    fig.subplots_adjust(left=.11, right=.98, top=.87, bottom=.36)
    fig.savefig(OUT / 'basin_attribute_coverage.pdf')
    fig.savefig(OUT / 'basin_attribute_coverage.png', dpi=300)
    report.savefig(fig)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(150 / 25.4, 112.5 / 25.4))
    for country, name, color, marker in zip(['Germany', 'Iceland', 'China'], ['德国', '冰岛', '中国'],
                                           ['#D55E00', '#009E73', '#0072B2'], ['^', 's', 'o']):
        rows = [r for r in data['rows'] if r['country'] == country]
        if any(r['area_km2'] is None or r['mean_annual_precip_mm_equiv'] is None for r in rows):
            raise ValueError('Missing metadata cannot be silently dropped from distribution')
        ax.scatter([r['area_km2'] for r in rows], [r['mean_annual_precip_mm_equiv'] for r in rows],
                   c=color, marker=marker, s=14, alpha=.7, linewidths=0,
                   label=f"{name}（{data['periods'][country]}，n={len(rows)}）")
    ax.set_xscale('log')
    ax.set_xlabel('流域面积（km²，对数尺度）', fontweight='bold')
    ax.set_ylabel('长期降雨年化量（mm/年）', fontweight='bold')
    ax.set_title('流域面积与长期降雨背景')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02), fontsize=9)
    ax.minorticks_off()
    fig.text(.5, .065, '仅元数据分布；统计期不同，站点不等于独立样本或模型效果',
             ha='center')
    ax.set_title('流域面积与长期降雨背景', pad=66)
    fig.subplots_adjust(left=.14, right=.98, top=.66, bottom=.27)
    fig.savefig(OUT / 'basin_background_distribution.pdf')
    fig.savefig(OUT / 'basin_background_distribution.png', dpi=300)
    report.savefig(fig)
    report.close()
    plt.close(fig)


if __name__ == '__main__':
    main()
