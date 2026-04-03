import numpy as np
from PIL import Image

def get_vec(img):
    w, h = img.size
    s = max(w, h)
    new = Image.new('L', (s, s), 0)
    new.paste(img, ((s - w) // 2, (s - h) // 2))
    return np.array(new.resize((20, 20))).flatten()

def find_objs(img):
    pix = img.load()
    w, h = img.size
    visited = set()
    objs = []
    for y in range(h):
        for x in range(w):
            if pix[x, y] == 255 and (x, y) not in visited:
                q, pts = [(x, y)], []
                visited.add((x, y))
                while q:
                    cx, cy = q.pop()
                    pts.append((cx, cy))
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nx, ny = cx+dx, cy+dy
                        if 0<=nx<w and 0<=ny<h and pix[nx,ny]==255 and (nx,ny) not in visited:
                            visited.add((nx,ny)); q.append((nx,ny))
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs)+1, max(ys)+1
                if x2 - x1 >= 2:
                    objs.append({'img': img.crop((x1, y1, x2, y2)), 'x': x1})
    return objs

t_img = Image.open('alphabet-small.png').convert('L').point(lambda p: 255 if p < 128 else 0)
labels = ['A', 'B', '8', '0', '1', 'W', 'X', '*', '-', '/']
t_objs = sorted(find_objs(t_img), key=lambda o: o['x'])

templates = {labels[i]: get_vec(t_objs[i]['img']) for i in range(len(labels))}

m_img = Image.open('alphabet.png').convert('L').point(lambda p: 255 if p > 1 else 0)
found = find_objs(m_img)
res = {l: 0 for l in labels}

for f in found:
    v = get_vec(f['img'])
    best = min(templates.keys(), key=lambda l: np.linalg.norm(v - templates[l]))
    res[best] += 1

for l in labels:
    print(f"{l}: {res[l]}")