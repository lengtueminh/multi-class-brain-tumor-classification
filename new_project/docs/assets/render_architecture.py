"""Render two stage-level diagrams of the selected Complex CNN."""
from pathlib import Path
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Rectangle

OUTPUT = Path(__file__).resolve().parent
INK = '#223044'
COLORS = {'conv':'#b8dcfb','pool':'#e6ebf0','merge':'#ffd6af','io':'#d4eddb'}
rcParams['svg.fonttype'] = 'none'
rcParams['pdf.fonttype'] = 42


def box(ax,x,y,w,h,text,kind='conv',size=12):
    ax.add_patch(Rectangle((x,y-h/2),w,h,facecolor=COLORS[kind],edgecolor=INK,lw=1.3,zorder=2))
    ax.text(x+w/2,y,text,ha='center',va='center',fontsize=size,color=INK,linespacing=1.4,zorder=3)


def arrow(ax,points):
    for a,b in zip(points[:-2],points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color=INK,lw=1.3,zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=13,color=INK,lw=1.3,zorder=4))


def frame(ax,x,y,w,h):
    ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='#708397',lw=1,linestyle=(0,(5,4)),zorder=0))


def save(fig,stem):
    for ext in ('svg','png','pdf'):
        fig.savefig(OUTPUT/f'{stem}.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
        if ext == 'svg':
            path = OUTPUT / f'{stem}.{ext}'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')


def overview():
    fig=Figure(figsize=(18,12));ax=fig.subplots()
    ax.set(xlim=(0,20),ylim=(0,14));ax.axis('off')
    ax.text(10,13.65,'Complex Sequential-Parallel CNN',ha='center',fontsize=24,weight='bold',color=INK)
    box(ax,5.4,12.35,3.2,1.0,'Input\n224 x 224 x 3','io',12)
    frame(ax,.4,5.3,4.0,6.2)
    frame(ax,4.9,5.3,8.3,6.2)
    ax.text(.5,11.85,'Sequential branch',ha='left',fontsize=12,weight='bold',color=INK)
    ax.text(10.7,11.85,'Parallel branch',ha='center',fontsize=12,weight='bold',color=INK)
    seq=[(10.45,'Stage S1\nConv 3x3 / 32 + BN + ReLU\n224 x 224 x 32'),
         (8.45,'Stage S2\nConv 3x3 / 64 + BN + ReLU\nPool 2x2\n112 x 112 x 64'),
         (6.45,'Stage S3\nConv 3x3 / 128 + BN + ReLU\nPool 2x2\n56 x 56 x 128')]
    for y,label in seq:box(ax,.75,y,3.3,1.4,label,size=10.5)
    arrow(ax,[(2.4,9.75),(2.4,9.15)])
    arrow(ax,[(2.4,7.75),(2.4,7.15)])
    centers=[2.4]
    for x,k in [(5.1,1),(8.0,3),(10.9,5)]:
        cx=x+1.05;centers.append(cx)
        box(ax,x,10.45,2.1,1.4,f'Conv {k}x{k} / 16\nBN + ReLU\n224 x 224 x 16',size=10.5)
        arrow(ax,[(cx,9.75),(cx,9.2),(9.05,9.2),(9.05,7.3)])
    ax.plot([centers[0],centers[-1]],[11.55,11.55],color=INK,lw=1.3)
    arrow(ax,[(7,11.85),(7,11.55)])
    for cx in centers:arrow(ax,[(cx,11.55),(cx,11.15)])
    box(ax,6.4,6.45,5.3,1.7,'Parallel aggregation\nConcatenate (48 channels)\nPool 2x2 -> Pool 2x2\n56 x 56 x 48','merge',11)
    box(ax,6.4,3.45,5.3,1.1,'Merge: concatenate both branches\n56 x 56 x (128 + 48) = 176','merge',11)
    arrow(ax,[(2.4,5.75),(2.4,3.45),(6.4,3.45)])
    arrow(ax,[(9.05,5.6),(9.05,4.0)])
    box(ax,15.1,10.45,3.8,1.4,'Fusion stage\nConv 3x3 / 128 + BN + ReLU\n56 x 56 x 128',size=11)
    arrow(ax,[(11.7,3.45),(13.9,3.45),(13.9,12.0),(17,12.0),(17,11.15)])
    box(ax,15.1,8.45,3.8,1.2,'MBConv 1 - expansion 2\n128 -> 256 -> 128 + residual\n56 x 56 x 128',size=11)
    arrow(ax,[(17,9.75),(17,9.05)])
    box(ax,15.1,6.65,3.8,1.2,'MBConv 2 - expansion 2\n128 -> 256 -> 128 + residual\n56 x 56 x 128',size=11)
    arrow(ax,[(17,7.85),(17,7.25)])
    box(ax,15.1,5.05,3.8,1.0,'Global Average Pooling\n128 features','pool',11)
    arrow(ax,[(17,6.05),(17,5.55)])
    box(ax,15.1,3.45,3.8,1.2,'Classifier head\nDense 128 + ReLU\nDropout 0.2 | 128 features',size=11)
    arrow(ax,[(17,4.55),(17,4.05)])
    box(ax,15.1,1.8,3.8,1.2,'Output stage\nDense 3 + Softmax (float32)\n3 class probabilities','io',11)
    arrow(ax,[(17,2.85),(17,2.4)])
    save(fig,'complex_cnn_architecture')


def mbconv():
    fig=Figure(figsize=(19,6));ax=fig.subplots()
    ax.set(xlim=(0,20.5),ylim=(0,6.1));ax.axis('off')
    ax.text(10.25,5.7,'MobileNetV2-style MBConv - grouped stages',ha='center',fontsize=22,weight='bold',color=INK)
    stages=[(.2,2.0,'Input\n56 x 56 x 128','io'),
            (3.2,3.0,'Expansion\nConv 1x1 / 256\nBN + ReLU6\n56 x 56 x 256','conv'),
            (7.2,3.0,'Spatial filtering\nDepthwise 3x3\nBN + ReLU6\n56 x 56 x 256','conv'),
            (11.2,3.0,'Linear projection\nConv 1x1 / 128 + BN\nNo activation\n56 x 56 x 128','conv'),
            (15.2,1.5,'Add\nx + F(x)','merge'),
            (17.7,2.4,'Output\n56 x 56 x 128','io')]
    for x,w,label,kind in stages:box(ax,x,2.75,w,1.6,label,kind,10.5)
    for a,b in zip(stages,stages[1:]):arrow(ax,[(a[0]+a[1],2.75),(b[0],2.75)])
    arrow(ax,[(1.2,3.55),(1.2,4.35),(15.95,4.35),(15.95,3.55)])
    save(fig,'complex_cnn_mbconv')


if __name__=='__main__':
    overview()
    mbconv()
    print(f'Saved two stage-level diagrams in SVG, PNG and PDF to {OUTPUT}')
