import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from PIL import Image
folder = os.path.join(r'C:\Users\dell\Desktop', '研究生文献收集')
outdir = r'D:\Users\dell\PycharmProjects\vmd-catboost\outputs'
for n in ['框架参考图1.jpg','框架参考图2.jpg','框架参考图3.jpg']:
    fp = os.path.join(folder, n)
    img = Image.open(fp)
    print(f'{n}: {img.size}, mode={img.mode}')
    outname = n.replace('.jpg','_view.png')
    img.convert('RGB').save(os.path.join(outdir, outname))
    print(f'  已转PNG: {outname}')
