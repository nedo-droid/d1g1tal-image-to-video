#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, subprocess
from pathlib import Path
import cv2, numpy as np

REF_W, REF_H = 1536, 864


def load_json(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def scale_points(points, w, h):
    return np.round(np.array([(x * w / REF_W, y * h / REF_H) for x, y in points], np.float32)).astype(np.int32)


def poly_mask(polys, w, h):
    m = np.zeros((h, w), np.uint8)
    for poly in polys or []:
        p = scale_points(poly, w, h)
        if len(p) >= 3:
            cv2.fillPoly(m, [p], 255)
    return m.astype(np.float32) / 255.0


def ease(x, name='ease_in_out'):
    x = float(np.clip(x, 0.0, 1.0))
    if name == 'linear': return x
    if name == 'ease_in': return x * x
    if name == 'ease_out': return 1.0 - (1.0 - x) ** 2
    return 0.5 - 0.5 * math.cos(math.pi * x)


def path_geom(points, w, h):
    p = np.array([(x * w / REF_W, y * h / REF_H) for x, y in points], np.float32)
    if len(p) < 2:
        return p, np.zeros(1, np.float32), 0.0
    seg = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(1))
    c = np.r_[0.0, np.cumsum(seg)].astype(np.float32)
    return p, c, float(c[-1])


def part_path(points, w, h, u, thickness):
    out = np.zeros((h, w), np.uint8)
    p, c, L = path_geom(points, w, h)
    if len(p) < 2 or L <= 0 or u <= 0:
        return out.astype(np.float32) / 255.0
    z = min(float(u), 1.0) * L
    pts = [p[0]]
    for i in range(1, len(p)):
        if c[i] < z:
            pts.append(p[i])
            continue
        q = np.clip((z - c[i - 1]) / max(c[i] - c[i - 1], 1e-6), 0, 1)
        pts.append(p[i - 1] * (1 - q) + p[i] * q)
        break
    cv2.polylines(out, [np.round(pts).astype(np.int32)], False, 255,
                  max(2, int(thickness)), cv2.LINE_AA)
    return cv2.GaussianBlur(out, (0, 0), 0.45).astype(np.float32) / 255.0


def sched(comp, i):
    ps = comp.get('trace_paths', [])
    p = ps[i]
    base = float(comp.get('start_sec', 0.0))
    total = max(0.001, float(comp.get('trace_duration_sec', 1.0)))
    st = float(p['start_sec']) if 'start_sec' in p else base
    if 'start_sec' not in p and comp.get('trace_mode', 'sequential_paths') == 'sequential_paths':
        ws = [max(0.001, float(x.get('weight', 1.0))) for x in ps]
        st = base + total * sum(ws[:i]) / max(sum(ws), 1e-6)
    if 'duration_sec' in p:
        dur = max(0.001, float(p['duration_sec']))
    elif comp.get('trace_mode', 'sequential_paths') == 'sequential_paths':
        ws = [max(0.001, float(x.get('weight', 1.0))) for x in ps]
        dur = total * max(0.001, float(p.get('weight', 1.0))) / max(sum(ws), 1e-6)
    else:
        dur = total
    return st, dur


def trace_done(c):
    return max([sched(c, i)[0] + sched(c, i)[1] for i in range(len(c.get('trace_paths', [])))] or
               [float(c.get('start_sec', 0.0)) + float(c.get('trace_duration_sec', 1.0))])


def trace_mask(c, w, h, t):
    out = np.zeros((h, w), np.float32)
    for i, p in enumerate(c.get('trace_paths', [])):
        st, d = sched(c, i)
        u = (t - st) / max(d, 0.001)
        if u > 0:
            out = np.maximum(out, part_path(
                p['points'], w, h,
                ease(min(u, 1.0), p.get('easing', c.get('trace_easing', 'ease_out'))),
                float(p.get('thickness', c.get('trace_thickness', 5)))
            ))
    return out


def fill_mask(c, w, h):
    mats = [poly_mask([p], w, h) for p in c.get('fill_polygons', [])]
    m = np.maximum.reduce(mats or [np.zeros((h, w), np.float32)])
    sig = float(c.get('fill_feather', 0.35)) * min(w / REF_W, h / REF_H)
    if sig > 0:
        m = cv2.GaussianBlur(m, (0, 0), sig)
    return np.clip(m, 0, 1).astype(np.float32)


def complete_logo_geom(cfg, w, h):
    geom = np.zeros((h, w), np.uint8)
    for comp in cfg.get('components', []):
        for poly in comp.get('fill_polygons', []):
            p = scale_points(poly, w, h)
            if len(p) >= 3:
                cv2.fillPoly(geom, [p], 255)
        for path in comp.get('trace_paths', []):
            p = scale_points(path.get('points', []), w, h)
            if len(p) >= 2:
                cv2.polylines(geom, [p], False, 255, max(3, int(path.get('thickness', 5)) + 8), cv2.LINE_AA)
    for poly in cfg.get('animation', {}).get('clean_extra_polygons', []):
        p = scale_points(poly, w, h)
        if len(p) >= 3:
            cv2.fillPoly(geom, [p], 255)
    return geom


def outline_contours(cfg, w, h):
    geom = complete_logo_geom(cfg, w, h)
    edge = cv2.Canny(geom, 80, 160)
    contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    paths = []
    for c in contours:
        if len(c) < 12:
            continue
        pts = c.reshape(-1, 2).astype(np.float32)
        length = cv2.arcLength(c, False)
        if length < 60:
            continue
        # keep long contours only; these correspond to the actual eagle perimeter and major cutouts
        paths.append((float(length), pts))
    paths.sort(reverse=True, key=lambda z: z[0])
    return paths


def auto_outline_mask(cfg, w, h, t, start, duration):
    out = np.zeros((h, w), np.float32)
    u = ease((t - start) / max(duration, 0.001), cfg.get('animation', {}).get('auto_outline_easing', 'ease_in_out'))
    if u <= 0:
        return out
    paths = outline_contours(cfg, w, h)
    if not paths:
        return out
    total = sum(p[0] for p in paths)
    target = min(1.0, u) * total
    used = 0.0
    thick = int(cfg.get('animation', {}).get('auto_outline_thickness', 5))
    for length, pts in paths:
        if used >= target:
            break
        local = np.clip((target - used) / max(length, 1e-6), 0, 1)
        p = pts
        if local < 1:
            n = max(2, int(round(len(p) * local)))
            p = p[:n]
            if len(p) < 2:
                continue
        q = np.round(p).astype(np.int32)
        cv2.polylines(out, [q], False, 1.0, max(2, thick), cv2.LINE_AA)
        used += length
    return cv2.GaussianBlur(out, (0, 0), 0.45).astype(np.float32)


def auto_outline_mask_from_paths(paths, w, h, t, start, duration, cfg):
    out = np.zeros((h, w), np.float32)
    u = ease((t-start)/max(duration,0.001), cfg.get('animation',{}).get('auto_outline_easing','ease_in_out'))
    if u <= 0 or not paths: return out
    total = sum(p[0] for p in paths); target = min(1.0,u)*total; used = 0.0
    thick = int(cfg.get('animation',{}).get('auto_outline_thickness',5))
    for length, pts in paths:
        if used >= target: break
        local = np.clip((target-used)/max(length,1e-6),0,1)
        q = pts if local >= 1 else pts[:max(2,int(round(len(pts)*local)))]
        if len(q) >= 2:
            cv2.polylines(out,[np.round(q).astype(np.int32)],False,1.0,max(2,thick),cv2.LINE_AA)
        used += length
    return cv2.GaussianBlur(out,(0,0),0.45).astype(np.float32)


def direct_annotation_masks(seg):
    b, g, r = cv2.split(seg)
    hsv = cv2.cvtColor(seg, cv2.COLOR_BGR2HSV)
    green = ((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 100) &
             (hsv[:, :, 1] > 60) &
             (g.astype(np.int16) > r.astype(np.int16) + 8) &
             (g.astype(np.int16) > b.astype(np.int16) + 6)).astype(np.uint8)
    white = ((hsv[:, :, 1] <= 55) & (hsv[:, :, 2] >= 180) &
             (r >= 170) & (g >= 170) & (b >= 170)).astype(np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return green.astype(np.float32), white.astype(np.float32)


def palm_templates_from_json(path=None):
    if path and Path(path).exists():
        data = load_json(path)
        if isinstance(data, dict) and 'templates' in data:
            return data['templates']
        if isinstance(data, list):
            return data
    return []


def render_sketch(temp, tw, th, thickness=4):
    tm = np.zeros((th, tw), np.float32)
    for s in temp.get('segments', []):
        p = np.round(np.array([[s[0][0] * tw, s[0][1] * th],
                               [s[1][0] * tw, s[1][1] * th]], np.float32)).astype(np.int32)
        cv2.line(tm, tuple(p[0]), tuple(p[1]), 1, max(1, int(thickness)), cv2.LINE_AA)
    return tm


def extract_dark_palm(src, roi_mask, bbox, spec, template=None):
    xa, ya, xb, yb = bbox
    crop = src[ya:yb, xa:xb]
    roi = (roi_mask[ya:yb, xa:xb] > 0.5).astype(np.uint8)
    gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    # Strong dark-pixel prior; palm silhouettes are nearly black/purple in the source.
    thr = float(spec.get('dark_value_max', 92))
    dark = ((gray <= thr) & (roi > 0) & ((sat >= int(spec.get('dark_sat_min', 5))) | (gray <= 72))).astype(np.uint8)
    # Remove tiny noise but retain narrow leaflets.
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # Sketch prior: keep dark pixels close to skeleton if a sketch is available.
    if template:
        tw, th = crop.shape[1], crop.shape[0]
        sk = render_sketch(template, tw, th, int(spec.get('sketch_thickness', 4)))
        corridor = cv2.dilate((sk > 0.04).astype(np.uint8),
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (int(spec.get('sketch_corridor_px', 48)) | 1,
                                                         int(spec.get('sketch_corridor_px', 48)) | 1)))
        near = (corridor > 0) & (dark > 0)
        # Keep the whole dark component if it touches the sketch corridor.
        n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
        keep = np.zeros_like(dark)
        for i in range(1, n):
            comp = lab == i
            if np.any(comp & corridor):
                keep[comp] = 1
        dark = keep.astype(np.uint8)
    # A modest dilation reconstructs leaf edges without swallowing the whole annotation.
    dark = cv2.dilate(dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    alpha = cv2.GaussianBlur(dark.astype(np.float32), (0, 0), float(spec.get('motion_feather', 0.9)))
    full = np.zeros(roi_mask.shape, np.float32)
    full[ya:yb, xa:xb] = alpha * roi
    return np.clip(full, 0, 1)


def find_trunk_candidates(src, green, ps, w, h):
    """Find independent palm instances from dark silhouettes inside the user's green annotation.
    Hough detects probable trunks; a dilated connected-component around each trunk recovers the crown/leaves.
    """
    roi_poly = poly_mask(ps.get('polygons', []), w, h) if ps.get('polygons') else green.copy()
    roi_poly *= green
    roi = ps.get('roi', [0,0,REF_W,REF_H])
    x0 = max(0, int(roi[0] * w / REF_W)); y0 = max(0, int(roi[1] * h / REF_H))
    x1 = min(w, int(roi[2] * w / REF_W)); y1 = min(h, int(roi[3] * h / REF_H))
    crop = src[y0:y1, x0:x1]
    ann = (roi_poly[y0:y1, x0:x1] > 0.5).astype(np.uint8)
    gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    dark = ((gray <= float(ps.get('dark_value_max', 100))) & (ann > 0)).astype(np.uint8)
    # Join nearby leaflets/trunk fragments into palm-shaped blobs, but keep separate trees separated.
    bridge = int(ps.get('palm_bridge_px', 11)) | 1
    cluster = cv2.dilate(dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge, bridge)), 1)
    cluster = cv2.morphologyEx(cluster, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
    labels_n, labels, stats, _ = cv2.connectedComponentsWithStats(cluster, 8)
    # Detect trunk-like lines from the original dark mask.
    edges = cv2.Canny((dark*255).astype(np.uint8), 25, 75)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=int(ps.get('hough_threshold', 18)),
                            minLineLength=int(ps.get('hough_min_line', 30)), maxLineGap=int(ps.get('hough_max_gap', 18)))
    seeds = ps.get('seed_points', [])
    raw = []
    if lines is not None:
        for line in lines[:,0]:
            ax, ay, bx, by = map(int, line)
            dx, dy = bx-ax, by-ay; length = math.hypot(dx,dy)
            if length < float(ps.get('hough_min_line',30)): continue
            ang = abs(math.degrees(math.atan2(dy,dx))); vertical = min(ang, abs(180-ang))
            if vertical > float(ps.get('trunk_angle_max',28)): continue
            mx,my=(ax+bx)//2,(ay+by)//2
            label=int(labels[my,mx]) if 0<=my<labels.shape[0] and 0<=mx<labels.shape[1] else 0
            if label==0: continue
            area=int(stats[label,cv2.CC_STAT_AREA])
            if area < int(ps.get('min_cluster_area',200)): continue
            gx,gy=mx+x0,my+y0
            near_seed=min([math.hypot(gx-sx*w/REF_W,gy-sy*h/REF_H) for sx,sy in seeds] or [0])
            raw.append((length + 0.15*min(near_seed,200), label))
    # If Hough fails, use user seeds to pick the nearest connected cluster.
    if not raw and seeds:
        for sx,sy in seeds:
            px,py=int(sx*w/REF_W)-x0,int(sy*h/REF_H)-y0
            py=np.clip(py,0,cluster.shape[0]-1);px=np.clip(px,0,cluster.shape[1]-1)
            label=int(labels[py,px])
            if label: raw.append((1.0,label))
    # One instance per connected palm cluster; choose the strongest evidence.
    best_by_label={}
    for score,label in raw:
        if label not in best_by_label or score>best_by_label[label]: best_by_label[label]=score
    ranked=sorted(best_by_label.items(), key=lambda z:z[1], reverse=True)
    emitted=0; min_sep=float(ps.get('candidate_min_separation_px',80))*min(w/REF_W,h/REF_H); centers=[]
    for label,score in ranked:
        xa=stats[label,cv2.CC_STAT_LEFT]+x0; ya=stats[label,cv2.CC_STAT_TOP]+y0
        xb=xa+stats[label,cv2.CC_STAT_WIDTH]; yb=ya+stats[label,cv2.CC_STAT_HEIGHT]
        # Moderate padding captures leaf tips but never leaves the annotated ROI.
        pad=int(float(ps.get('instance_pad_px',18))*min(w/REF_W,h/REF_H))
        xa=max(x0,xa-pad); ya=max(y0,ya-pad); xb=min(x1,xb+pad); yb=min(y1,yb+pad)
        cx=(xa+xb)/2; cy=(ya+yb)/2
        if any(math.hypot(cx-qx,cy-qy)<min_sep for qx,qy in centers): continue
        centers.append((cx,cy))
        yield (xa,ya,xb,yb,(0,0,0,0),label)
        emitted+=1
        if emitted>=int(ps.get('max_instances',4)): break


def palm_analysis_instances(src, green, ps, templates, w, h):
    out=[]
    roi_poly = poly_mask(ps.get('polygons',[]),w,h) if ps.get('polygons') else green.copy()
    roi_poly *= green
    for bbox0 in find_trunk_candidates(src,green,ps,w,h):
        xa,ya,xb,yb,_,label=bbox0
        local_roi=roi_poly[ya:yb,xa:xb]
        best=None
        for temp in (templates or [{'id':'none','segments':[]}]):
            # Build alpha from dark pixels, then close leaf gaps. The green annotation is the hard spatial gate.
            crop=src[ya:yb,xa:xb]
            gray=cv2.cvtColor(crop.astype(np.uint8),cv2.COLOR_BGR2GRAY)
            hsv=cv2.cvtColor(crop.astype(np.uint8),cv2.COLOR_BGR2HSV)
            dark=((gray<=float(ps.get('dark_value_max',100))) & (local_roi>0.5) &
                  ((hsv[:,:,1]>=int(ps.get('dark_sat_min',5))) | (gray<=72))).astype(np.uint8)
            dark=cv2.morphologyEx(dark,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
            dark=cv2.dilate(dark,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),1)
            if temp.get('segments'):
                sk=render_sketch(temp,crop.shape[1],crop.shape[0],int(ps.get('sketch_thickness',4)))
                corridor=cv2.dilate((sk>0.03).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(ps.get('sketch_corridor_px',48))|1,)*2))
                # Sketch is a soft prior: keep palm dark pixels, but don't delete leaves that fall outside it.
                score=float((dark*corridor).sum())/(float(dark.sum())+1e-6)
            else: score=float(dark.sum())
            if best is None or score>best[0]: best=(score,temp,dark)
        if best is None: continue
        _,temp,alpha_local=best
        alpha=cv2.GaussianBlur(alpha_local.astype(np.float32),(0,0),float(ps.get('motion_feather',0.8))) * local_roi
        full=np.zeros(green.shape,np.float32); full[ya:yb,xa:xb]=np.clip(alpha,0,1)
        la=full[ya:yb,xa:xb]; ys,xs=np.where(la>0.18)
        if len(xs)<120: continue
        low=ys>np.percentile(ys,82); pivot_y=float(np.percentile(ys,90)); pivot_x=float(np.median(xs[low])) if np.any(low) else float(np.median(xs))
        info={'found':True,'template':str(temp.get('id','none')),'bbox':[int(xa),int(ya),int(xb),int(yb)],'pixel_area':int(np.count_nonzero(la>0.18)),'pivot':[float(pivot_x+xa),float(pivot_y+ya)]}
        out.append({'alpha':full,'bbox':(xa,ya,xb,yb),'cfg':ps,'pivot':(pivot_x+xa,pivot_y+ya),'info':info})
    return out

def apply_palm(fr, src, palm, t, w, h):
    alpha = palm['alpha']; xa, ya, xb, yb = palm['bbox']; ps = palm['cfg']
    a = alpha[ya:yb, xa:xb]
    crop = src[ya:yb, xa:xb]
    hh, ww = a.shape
    scale = min(w / REF_W, h / REF_H)
    amp = float(ps.get('sway_px', 10)) * scale
    f = float(ps.get('frequency_hz', 0.075)); phase = float(ps.get('phase', 0.0)); direction = float(ps.get('direction', 1.0))
    s = (0.82 * math.sin(2 * math.pi * f * t + phase) +
         0.18 * math.sin(4 * math.pi * f * t + phase + 0.55))
    py_global = float(palm['pivot'][1]); py = py_global - ya
    # 0 at pivot, 1 at the crown. Clamp so the bottom/stem barely moves.
    rows = np.arange(hh, dtype=np.float32)[:, None]
    top_dist = max(1.0, py - 3.0)
    rel = np.clip((py - rows) / top_dist, 0, 1) ** float(ps.get('vertical_power', 1.55))
    rel *= (a.max(axis=1, keepdims=True) > 0.05).astype(np.float32)
    dx = direction * amp * s * rel
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    # Add a tiny depth-like rotation around the trunk so leaves visibly follow the wind.
    rot_deg = float(ps.get('rotation_deg', 1.8)) * s
    cx = float(palm['pivot'][0] - xa); cy = float(palm['pivot'][1] - ya)
    ang = math.radians(-rot_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    rx = ca * (xx - cx) - sa * (yy - cy) + cx
    ry = sa * (xx - cx) + ca * (yy - cy) + cy
    mapx = (rx - dx).astype(np.float32); mapy = ry.astype(np.float32)
    wi = cv2.remap(crop, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
    wa = cv2.remap(a, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    pa = np.clip(wa * float(ps.get('motion_alpha', 1.0)), 0, 1)[..., None]
    fr[ya:yb, xa:xb] = wi * pa + fr[ya:yb, xa:xb] * (1 - pa)
    return fr


def strict_water(white, spec, w, h):
    m = white.copy()
    if spec.get('polygons'):
        m *= poly_mask(spec['polygons'], w, h)
    for poly in spec.get('exclusion_polygons', []):
        p = scale_points(poly, w, h)
        if len(p) >= 3:
            cv2.fillPoly(m, [p], 0)
    # require sufficiently wide water band; no automatic growth into adjacent objects
    return np.clip(cv2.GaussianBlur((m > float(spec.get('sample_mask_threshold', 0.4))).astype(np.float32),
                                   (0, 0), float(spec.get('feather', 0.8))), 0, 1)


def water_glints(mask, spec, w, h, seed):
    rng = np.random.default_rng(seed)
    bm = (mask > 0.65).astype(np.uint8)
    ys, xs = np.where(bm > 0)
    if len(xs) == 0: return []
    n = int(spec.get('highlight_count', 220))
    xa, xb, ya, yb = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    out = []; tries = 0
    while len(out) < n and tries < n * 35:
        tries += 1
        x = int(rng.integers(xa, xb + 1)); y = int(rng.integers(ya, yb + 1))
        if bm[y, x] == 0: continue
        out.append({
            'x': x, 'y': y,
            'w': float(rng.uniform(spec.get('min_width', 2), spec.get('max_width', 7))),
            'h': float(rng.uniform(spec.get('min_height', 0.5), spec.get('max_height', 1.5))),
            'phase': float(rng.uniform(0, math.tau)), 'amp': float(rng.uniform(.65, 1.0)),
            'dx': float(rng.uniform(-2.0, 2.0)), 'dy': float(rng.uniform(-.35, .35))
        })
    return out


def add_glints(fr, mask, glints, spec, t):
    layer = np.zeros(mask.shape, np.float32)
    for g in glints:
        ph = g['phase'] + t * float(spec.get('phase_speed', 1.6))
        a = g['amp'] * (0.35 + 0.65 * (0.5 + 0.5 * math.sin(ph)))
        x = int(round(g['x'] + g['dx'] * math.sin(t * .7 + g['phase'])))
        y = int(round(g['y'] + g['dy'] * math.sin(t * 1.2 + g['phase'])))
        cv2.ellipse(layer, (x, y), (max(1, int(g['w'])), max(1, int(g['h']))), 0, 0, 360,
                     float(a), -1, cv2.LINE_AA)
    layer *= mask
    core = cv2.GaussianBlur(layer, (0, 0), float(spec.get('glow_sigma', .7)))
    inten = np.clip(layer * 0.9 + core * 0.65, 0, 1.6) * float(spec.get('strength', 0.9))
    inten *= float(spec.get('alpha', 0.5))
    warm = np.array(spec.get('color_bgr', [92, 178, 255]), np.float32)
    white = np.array([255, 255, 255], np.float32)
    hot = np.clip(inten / max(float(spec.get('core_boost', 1.5)), 1e-3), 0, 1)[..., None]
    col = warm[None, None, :] * (1 - .65 * hot) + white[None, None, :] * .65 * hot
    return np.clip(fr + col * inten[..., None], 0, 255)


def refined_logo_mask(src, cfg, w, h):
    """Refine the manual eagle geometry with GrabCut and strong logo color/edge cues.
    This improves coverage of the baked logo without expanding the hole to the whole center ROI.
    """
    geom = complete_logo_geom(cfg, w, h)
    ys, xs = np.where(geom > 0)
    if len(xs) < 100:
        return geom.astype(np.uint8)
    pad = int(round(float(cfg.get('animation', {}).get('logo_refine_padding_px', 14)) * min(w/REF_W, h/REF_H)))
    x0, x1 = max(0, int(xs.min())-pad), min(w, int(xs.max())+pad+1)
    y0, y1 = max(0, int(ys.min())-pad), min(h, int(ys.max())+pad+1)
    crop = np.clip(src[y0:y1, x0:x1], 0, 255).astype(np.uint8)
    gm = (geom[y0:y1, x0:x1] > 0).astype(np.uint8)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:,:,1], hsv[:,:,2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    detail = gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32),(0,0),2.2)
    # Strong cues: neon border/interior + very dark logo stroke.
    cue = (((sat > 95) & (val > 105)) | ((sat < 60) & (val > 180)) | (val < 90) | (np.abs(detail) > 18)) & (gm > 0)
    gc = np.full(gm.shape, cv2.GC_PR_BGD, np.uint8)
    gc[gm > 0] = cv2.GC_PR_FGD
    gc[cue] = cv2.GC_FGD
    outer = cv2.dilate(gm, np.ones((9,9), np.uint8), iterations=1)
    gc[outer == 0] = cv2.GC_BGD
    bgd = np.zeros((1,65), np.float64); fgd = np.zeros((1,65), np.float64)
    try:
        cv2.grabCut(crop, gc, None, bgd, fgd, int(cfg.get('animation',{}).get('logo_refine_iters', 6)), cv2.GC_INIT_WITH_MASK)
        keep = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    except cv2.error:
        keep = gm
    # Never allow the refined mask to escape the annotated geometry by more than a thin antialias ring.
    allowed = cv2.dilate(gm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    keep &= allowed
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    full = np.zeros((h,w), np.uint8); full[y0:y1,x0:x1] = keep * 255
    return full


def logo_remove_mask(src, cfg, w, h):
    base = refined_logo_mask(src, cfg, w, h)
    k = int(round(float(cfg.get('animation', {}).get('logo_remove_dilation_px', 1.15)) * min(w/REF_W, h/REF_H)))
    if k > 0:
        base = cv2.dilate(base, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*k+1,2*k+1)))
    return base

def mirrored_sharp_fill(src, mask, cfg):
    """Crisp single-frame fallback: NS inpaint + mirrored source detail.
    It never reuses any generated frame. It is intentionally sharp; external clean plate remains preferred.
    """
    src8 = np.clip(src, 0, 255).astype(np.uint8)
    m = (mask > 0).astype(np.uint8)
    if not m.any(): return src8.astype(np.float32)
    anim = cfg.get('animation', {})
    ns = cv2.inpaint(src8, m * 255, int(anim.get('logo_inpaint_radius_px', 2)), cv2.INPAINT_NS).astype(np.float32)
    h, w = m.shape; axis = float(anim.get('background_symmetry_axis_x', REF_W * 0.5)) * w / REF_W
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mx = 2.0 * axis - xx
    mirror = cv2.remap(src8.astype(np.float32), mx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    mirror_known = cv2.remap(1.0 - m.astype(np.float32), mx, yy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Use mirrored texture only where it is actually outside the logo hole.
    blend = float(anim.get('mirror_fill_strength', .62))
    use = (m > 0) & (mirror_known > .5)
    out = ns.copy()
    out[use] = ns[use] * (1 - blend) + mirror[use] * blend
    # Preserve the inpainted low-frequency structure while injecting local detail from known areas.
    detail = src8.astype(np.float32) - cv2.GaussianBlur(src8.astype(np.float32), (0, 0), 1.15)
    out[m > 0] += detail[m > 0] * float(anim.get('hole_detail_reinject', 0.12))
    return np.clip(out, 0, 255).astype(np.float32)


def temporal_clean_plate(src, cfg, mask, neighbor_dir):
    paths = sorted([p for p in Path(neighbor_dir).glob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}])
    paths = paths[:int(cfg.get('animation', {}).get('neighbor_max_frames', 9))]
    if not paths: return None
    known = (mask == 0)
    frames = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None: continue
        im = cv2.resize(im, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        d = np.mean(np.abs(im.astype(np.float32) - src.astype(np.float32)), axis=2)
        med = float(np.median(d[known])) if np.any(known) else 0
        if med <= float(cfg.get('animation', {}).get('neighbor_known_median_maxdiff', 38)):
            frames.append(im.astype(np.float32))
    if not frames: return None
    stack = np.stack(frames)
    med = np.median(stack, axis=0)
    out = src.astype(np.float32).copy()
    out[mask > 0] = med[mask > 0]
    return out, {'mode': 'real_neighbor_temporal_median', 'frames_used': len(frames)}


def clean_plate(src, cfg, w, h, neighbor_dir=None, background=None):
    mask = logo_remove_mask(src, cfg, w, h)
    if background:
        im = cv2.imread(background, cv2.IMREAD_COLOR)
        if im is None: raise FileNotFoundError(background)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
        return im, mask, {'mode': 'external_clean_background'}
    if neighbor_dir:
        tc = temporal_clean_plate(src, cfg, mask, neighbor_dir)
        if tc is not None:
            out, meta = tc; return out, mask, meta
    out = mirrored_sharp_fill(src, mask, cfg)
    return out, mask, {'mode': 'ns_plus_mirror_sharp_fallback'}


def render(args, cfg, templates):
    src0 = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if src0 is None: raise FileNotFoundError(args.image)
    src = cv2.resize(src0, (args.width, args.height), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    w, h = args.width, args.height
    anim = cfg.get('animation', {}); env = cfg.get('environment', {})
    bg, logo_m, plate = clean_plate(src, cfg, w, h, args.neighbor_frame_dir, args.background)

    seg = cv2.imread(args.environment_mask or env.get('segmentation_mask', ''), cv2.IMREAD_COLOR) if (args.environment_mask or env.get('segmentation_mask')) else None
    palms, waters = [], []
    if seg is not None:
        seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
        green, white = direct_annotation_masks(seg)
        for ps in env.get('palms', []):
            palms.extend(palm_analysis_instances(src, green, ps, templates, w, h))
        for i, ws in enumerate(env.get('water', [])):
            wm = strict_water(white, ws, w, h)
            waters.append({'mask': wm, 'cfg': ws,
                           'glints': water_glints(wm, ws, w, h, int(env.get('seed', 4242)) + i * 1009)})
        # Remove all palm silhouettes from the plate before locally animating them.
        pm = np.zeros((h, w), np.uint8)
        for p in palms:
            pm = np.maximum(pm, (p['alpha'] > 0.12).astype(np.uint8) * 255)
        if np.any(pm):
            bg = cv2.inpaint(np.clip(bg, 0, 255).astype(np.uint8), pm, 1.2, cv2.INPAINT_TELEA).astype(np.float32)

    comps = cfg.get('components', [])
    outline_paths = outline_contours(cfg, w, h)
    manual_end = max([trace_done(c) for c in comps] or [0.0])
    auto_start = max(0.0, manual_end - float(anim.get('auto_outline_tail_sec', 0.40)))
    auto_end = manual_end
    rs = manual_end + float(anim.get('logo_fill_delay_sec', 0.18))
    rd = max(.01, float(anim.get('logo_fill_duration_sec', .72)))
    re = rs + rd
    rf = max(re + float(anim.get('ray_hold_after_fill_sec', 0.9)), auto_end + .05)
    rfd = max(.01, float(anim.get('ray_fade_duration_sec', 1.2)))

    yy, xx = np.mgrid[0:h, 0:w]
    xn = xx / max(w - 1, 1); mix = np.clip(.5 + (xn - .5) * 1.5, 0, 1)
    purple = np.array(cfg.get('colors', {}).get('purple_bgr', [235, 35, 255]), np.float32)
    cyan = np.array(cfg.get('colors', {}).get('cyan_bgr', [255, 225, 35]), np.float32)
    neon = purple[None, None, :] * (1 - mix[..., None]) + cyan[None, None, :] * mix[..., None]

    ff = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}',
          '-r', str(args.fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', str(anim.get('ffmpeg_preset', 'medium')),
          '-crf', str(anim.get('crf', 18)), '-pix_fmt', 'yuv420p', '-movflags', '+faststart', args.output]
    proc = subprocess.Popen(ff, stdin=subprocess.PIPE)
    total = int(round(args.duration * args.fps))
    for fi in range(total):
        t = fi / args.fps
        fr = bg.copy()
        for p in palms: fr = apply_palm(fr, src, p, t, w, h)
        for q in waters: fr = add_glints(fr, q['mask'], q['glints'], q['cfg'], t)

        # Ray stage: manual traces + guaranteed final completion from true geometric outline.
        ray_fade = 1.0 if t < rf else max(0.0, 1.0 - ease((t - rf) / rfd, anim.get('ray_fade_easing', 'ease_in_out')))
        for c in comps:
            tm = trace_mask(c, w, h, t)
            if t >= auto_start:
                tm = np.maximum(tm, auto_outline_mask_from_paths(outline_paths, w, h, t, auto_start, max(.001, auto_end - auto_start), cfg))
            tr = tm * ray_fade
            if tr.max() > 0:
                sm = cv2.GaussianBlur(tr, (0, 0), float(c.get('glow_small', 2.7)))
                bb = cv2.GaussianBlur(tr, (0, 0), float(c.get('glow_big', 9.0)))
                fr += neon * (sm[..., None] * float(c.get('glow_strength', 2.7)) +
                              bb[..., None] * float(c.get('big_glow_strength', .7)))

        # Logo reveal starts only after the full outline stage is complete.
        fillu = ease((t - rs) / rd, anim.get('logo_fill_easing', 'ease_out')) if t >= rs else 0.0
        if fillu > 0:
            for c in comps:
                a = fill_mask(c, w, h) * fillu * float(c.get('fill_alpha_max', 100)) / 100.0
                a = np.clip(a, 0, 1)[..., None]
                logo_src = src
                fr = fr * (1 - a) + logo_src * a
        proc.stdin.write(np.clip(fr, 0, 255).astype(np.uint8).tobytes())
    proc.stdin.close(); code = proc.wait()
    if code != 0: raise RuntimeError(f'ffmpeg exit {code}')
    return {
        'output': args.output, 'frames': total, 'duration_sec': total / args.fps,
        'clean_plate': plate,
        'timeline': {'manual_contour_end': manual_end, 'auto_outline_start': auto_start,
                     'full_contour_end': auto_end, 'reveal_start': rs, 'reveal_end': re,
                     'ray_fade_start': rf, 'ray_fade_end': rf + rfd},
        'palms_detected': len(palms), 'palms': [p['info'] for p in palms],
        'water_regions': len(waters), 'water_alpha_values': [float(q['cfg'].get('alpha', .5)) for q in waters]
    }


def save_debug(args, cfg, templates):
    out = Path(args.debug_dir); out.mkdir(parents=True, exist_ok=True)
    src = cv2.resize(cv2.imread(args.image), (args.width, args.height), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    bg, logo_m, plate = clean_plate(src, cfg, args.width, args.height, None, args.background)
    cv2.imwrite(str(out / 'clean_plate.png'), np.clip(bg, 0, 255).astype(np.uint8))
    cv2.imwrite(str(out / 'logo_remove_mask.png'), np.clip(logo_m * 255, 0, 255).astype(np.uint8))
    paths = outline_contours(cfg, args.width, args.height)
    om = np.zeros((args.height,args.width),np.float32)
    for _, pts in paths:
        if len(pts)>=2: cv2.polylines(om,[np.round(pts).astype(np.int32)],False,1.0,5,cv2.LINE_AA)
    cv2.imwrite(str(out / 'neon_completion_outline.png'), np.clip(om*255,0,255).astype(np.uint8))
    seg_path = args.environment_mask or cfg.get('environment', {}).get('segmentation_mask')
    if seg_path:
        seg = cv2.resize(cv2.imread(seg_path, cv2.IMREAD_COLOR), (args.width, args.height), interpolation=cv2.INTER_NEAREST)
        green, white = direct_annotation_masks(seg)
        cv2.imwrite(str(out / 'green_annotation.png'), (green * 255).astype(np.uint8))
        cv2.imwrite(str(out / 'water_annotation.png'), (white * 255).astype(np.uint8))
        palms = []
        for i, ps in enumerate(cfg.get('environment', {}).get('palms', [])):
            pp = palm_analysis_instances(src, green, ps, templates, args.width, args.height)
            palms.extend(pp)
            canvas = np.zeros_like(seg)
            for j, p in enumerate(pp):
                xa, ya, xb, yb = p['bbox']
                canvas[ya:yb, xa:xb, 1] = np.clip(p['alpha'][ya:yb, xa:xb] * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(str(out / f'palm_group_{i}.png'), canvas)
        allp = np.zeros((args.height, args.width), np.float32)
        for p in palms: allp = np.maximum(allp, p['alpha'])
        cv2.imwrite(str(out / 'palm_all.png'), np.clip(allp * 255, 0, 255).astype(np.uint8))
    with open(out / 'debug_report.json', 'w', encoding='utf-8') as f:
        json.dump({'clean_plate': plate, 'palms': len(palms) if seg_path else 0}, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument('--config', default='eagle_scene_v17.json')
    ap.add_argument('--output', default='eagle_v16.mp4')
    ap.add_argument('--duration', type=float, default=12)
    ap.add_argument('--fps', type=int, default=24)
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--height', type=int, default=540)
    ap.add_argument('--environment-mask', default=None)
    ap.add_argument('--background', default=None)
    ap.add_argument('--neighbor-frame-dir', default=None)
    ap.add_argument('--palm-templates', default='palm_sketches_v15.json')
    ap.add_argument('--debug-dir', default=None)
    a = ap.parse_args()
    cfg = load_json(a.config)
    templates = palm_templates_from_json(a.palm_templates)
    info = render(a, cfg, templates)
    if a.debug_dir: save_debug(a, cfg, templates)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
