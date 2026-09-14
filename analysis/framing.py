"""Conservative offline framing from anonymous detections and reviewed scene gates.

The caller supplies scene calibration and editorial close-up intervals. This is
not an identity tracker or a guarantee that an undetected performer is visible.
All detection bounds and coverage results remain available for review.
"""
from dataclasses import dataclass, field
import math
import numpy as np


@dataclass
class FramingConfig:
    floor: float = .865
    maximum_head_y: float = .60
    stage_anchor: tuple = (.38, .32, .77, .865)
    lookahead_seconds: float = 6
    smoothing_seconds: float = 20
    center_speed: float = .008
    width_speed: float = .012
    maximum_zoom: float = 1.8
    dramatic_zoom: float = 2.1
    dramatic_ranges: list = field(default_factory=list)
    wide_ranges: list = field(default_factory=list)
    corner_pin: list | None = None


def homography(corners):
    src = [(0, 0), (1, 0), (0, 1), (1, 1)]
    a, b = [], []
    for (x, y), (u, v) in zip(src, corners):
        a.extend([[x,y,1,0,0,0,-u*x,-u*y], [0,0,0,x,y,1,-v*x,-v*y]])
        b.extend([u,v])
    return np.append(np.linalg.solve(a,b),1).reshape(3,3)



def transform_box(box, matrix):
    x1,y1,x2,y2 = box
    points = np.array([[x1,y1,1],[x2,y1,1],[x1,y2,1],[x2,y2,1]]) @ matrix.T
    points = points[:,:2] / points[:,2:]
    return np.r_[points.min(axis=0), points.max(axis=0)]


def union(boxes):
    boxes = np.asarray(boxes)
    return np.r_[boxes[:,:2].min(axis=0), boxes[:,2:].max(axis=0)]


def in_ranges(t, ranges):
    return any(a <= t <= b for a,b in ranges)


def required_box(row, config, dramatic=False):
    people = [d['box'] for d in row['people'] if
              d['box'][1] < config.maximum_head_y and d['box'][3]-d['box'][1] >= .06]
    faces = [d['box'] for d in row['faces'] if d['box'][1] < config.maximum_head_y]
    if not people and not faces:
        return np.array([0,0,1,1]), True
    boxes = []
    for x1,y1,x2,y2 in people:
        w,h = x2-x1,y2-y1
        # In a reviewed close-up, retain head, arms, and upper body of EVERY actor.
        bottom = max(y1+.66*h, y1+.24) if dramatic else max(y2, config.floor)
        boxes.append([x1-max(.035,.20*w), y1-max(.045,.15*h),
                      x2+max(.035,.20*w), bottom+.025])
    for x1,y1,x2,y2 in faces:
        if any(px1 <= (x1+x2)/2 <= px2 and py1 <= (y1+y2)/2 <= py2
               for px1,py1,px2,py2 in people):
            continue
        w,h = x2-x1,y2-y1
        # Supplement body misses, especially the performer above the backdrop.
        bottom = y2+4*h if dramatic else max(y2+7*h, config.floor)
        boxes.append([x1-1.5*w-.015,y1-max(.04,.9*h),x2+1.5*w+.015,bottom+.02])
    anchor = list(config.stage_anchor)
    if dramatic:
        anchor[3] = min(anchor[3], .70)
    boxes.append(anchor)
    return np.clip(union(boxes), 0, 1), False


def plan(rows, config, *, dramatic=False, keyframe_seconds=2):
    settings=[config.lookahead_seconds, config.smoothing_seconds, config.center_speed,
              config.width_speed, keyframe_seconds]
    if not all(math.isfinite(x) and x>0 for x in settings):
        raise ValueError('Timing and speed settings must be finite and positive')
    if not (1 <= config.maximum_zoom <= config.dramatic_zoom):
        raise ValueError('Zoom limits must satisfy 1 <= maximum <= dramatic')
    if len(rows)<2:
        raise ValueError('At least two detection samples are required')
    times=np.array([r['time'] for r in rows],float)
    if not np.all(np.isfinite(times)) or not np.all(np.diff(times)>0):
        raise ValueError('Detection timestamps must increase')
    dt=float(np.median(np.diff(times)))
    if not np.allclose(np.diff(times), dt, atol=1e-5, rtol=1e-5):
        raise ValueError('Detection samples must have a uniform interval')
    matrix,corners,fit_scale=np.eye(3), None, 1.0
    raw=[]; missing=[]; close=[]
    for row in rows:
        tight=dramatic and in_ranges(row['time'],config.dramatic_ranges)
        box,unknown=required_box(row,config,tight)
        if in_ranges(row['time'],config.wide_ranges): box=np.array([0,0,1,1])
        raw.append(transform_box(box,matrix)); missing.append(unknown); close.append(tight)
    raw=np.array(raw)
    # Bridge short detector dropouts with the union of adjacent known bounds.
    # Long gaps remain wide. Original unknown flags are retained in the report.
    good=np.flatnonzero(~np.array(missing))
    for i in np.flatnonzero(missing):
        before=good[good<i];after=good[good>i]
        if len(before) and len(after):
            a,b=before[-1],after[0]
            if times[b]-times[a] <= 8 and not in_ranges(times[i],config.wide_ranges):
                raw[i]=union([raw[a],raw[b]])
    radius=max(1,round(config.lookahead_seconds/dt))
    bounds=np.array([union(raw[max(0,i-radius):i+radius+1]) for i in range(len(raw))])
    centers=(bounds[:,:2]+bounds[:,2:])/2
    smooth=max(1,round(config.smoothing_seconds/dt))
    kernel=np.convolve(np.ones(smooth),np.ones(smooth));kernel/=kernel.sum()
    centers=np.stack([np.convolve(np.pad(centers[:,j],(len(kernel)//2,)*2,mode='edge'),kernel,mode='valid') for j in range(2)],axis=1)
    # A backwards/forwards velocity projection prevents abrupt pans; containment
    # is restored by widening, never by dropping a performer constraint.
    speed=config.center_speed*fit_scale*dt
    for order in (range(len(times)-2,-1,-1),range(1,len(times))):
        for i in order:
            other=i+1 if order.step<0 else i-1
            centers[i]=np.clip(centers[i],centers[other]-speed,centers[other]+speed)
    widths=2*np.maximum(np.max(centers-bounds[:,:2],axis=1),np.max(bounds[:,2:]-centers,axis=1))
    minimum=np.where(close,fit_scale/config.dramatic_zoom,fit_scale/config.maximum_zoom)
    widths=np.maximum(widths,minimum)+.012*fit_scale
    speed=config.width_speed*fit_scale*dt
    for i in range(1,len(widths)):widths[i]=max(widths[i],widths[i-1]-speed)
    for i in range(len(widths)-2,-1,-1):widths[i]=max(widths[i],widths[i+1]-speed)
    widths=np.minimum(widths,1.0)
    lower=np.maximum(widths[:,None]/2,bounds[:,2:]-widths[:,None]/2)
    upper=np.minimum(1-widths[:,None]/2,bounds[:,:2]+widths[:,None]/2)
    if np.any(lower>upper+1e-8): raise ValueError('Infeasible source bounds')
    centers=np.clip(centers,lower,upper)
    step=max(1,round(keyframe_seconds/dt))
    indexes=sorted(set(range(0,len(times),step))|{len(times)-1})
    zoom=1/widths
    position=.5-zoom[:,None]*(centers-.5)
    keys=np.c_[times,zoom,position][:][indexes]
    interp=np.stack([np.interp(times,keys[:,0],keys[:,j]) for j in (1,2,3)],axis=1)
    lo=.5+interp[:,0,None]*(raw[:,:2]-.5)+(interp[:,1:]-.5)
    hi=.5+interp[:,0,None]*(raw[:,2:]-.5)+(interp[:,1:]-.5)
    violations=np.flatnonzero((lo.min(axis=1)<-1e-8)|(hi.max(axis=1)>1+1e-8))
    if len(violations):
        raise ValueError(f'Interpolated camera misses {len(violations)} required sample bounds')
    result = {'version':1,'mode':'dramatic' if dramatic else 'full_body',
            'matrix':matrix.tolist(),'fitted_corner_pin':corners,'perspective_fit_scale':fit_scale,
            'keys':keys.tolist(),'key_columns':['source_seconds','scale_multiplier','position_x','position_y'],
            'statistics':{'samples':len(rows),'keys':len(keys),'missing_detection_samples':sum(missing),
                          'coverage_violations':len(violations),'effective_zoom_min':float((zoom*fit_scale).min()),
                          'effective_zoom_max':float((zoom*fit_scale).max())},
            'review':[{'time':float(t),'required':box.tolist(),'unknown':bool(m)} for t,box,m in zip(times,raw,missing)]}
    if config.corner_pin is not None:
        add_adaptive_perspective(result, config.corner_pin, times, raw, config.lookahead_seconds)
    return result


def mapped_points(points, matrix):
    h=np.c_[np.array(points),np.ones(len(points))]@matrix.T
    return h[:,:2]/h[:,2:]


def covers_frame(corners):
    q=np.asarray(corners)[[0,1,3,2]]
    area=np.sum(q[:,0]*np.roll(q[:,1],-1)-q[:,1]*np.roll(q[:,0],-1))
    if not np.all(np.isfinite(q)) or area<=1e-10: return False
    targets=np.array([[0,0],[1,0],[1,1],[0,1]])
    for a,b in zip(q,np.roll(q,-1,axis=0)):
        delta=b-a; p=targets-a
        if np.any(delta[0]*p[:,1]-delta[1]*p[:,0]<-1e-8): return False
    return True


def add_adaptive_perspective(result, target, times, bounds, lookahead):
    unit=np.array([[0,0],[1,0],[0,1],[1,1]],float)
    target=np.asarray(target,float); keys=np.array(result['keys'])
    def transform(key, alpha):
        _,z,x,y=key
        motion=np.array([[z,0,x-.5*z],[0,z,y-.5*z],[0,0,1.]])
        return motion@homography(unit+alpha*(target-unit))
    def safe(key,alpha,box):
        matrix=transform(key,alpha)
        if not covers_frame(mapped_points(unit,matrix)): return False
        b=transform_box(box,matrix)
        return b[:2].min()>=-1e-8 and b[2:].max()<=1+1e-8
    alphas=[]
    for key in keys:
        window=bounds[np.abs(times-key[0])<=lookahead]
        box=union(window) if len(window) else bounds[np.argmin(abs(times-key[0]))]
        # Keyframes already contain their own lookahead. A union over this wider
        # sample window may be infeasible even at alpha=0; final coverage checks
        # still include every actual sample below.
        if not safe(key,0,box): box=bounds[np.argmin(abs(times-key[0]))]
        lo,hi=0.,1.
        for _ in range(14):
            mid=(lo+hi)/2
            if safe(key,mid,box):lo=mid
            else:hi=mid
        alphas.append(lo)
    alphas=np.array(alphas)
    # Slow changes in perspective; only lower the safe strength ceiling.
    for i in range(1,len(alphas)):alphas[i]=min(alphas[i],alphas[i-1]+.025*(keys[i,0]-keys[i-1,0]))
    for i in range(len(alphas)-2,-1,-1):alphas[i]=min(alphas[i],alphas[i+1]+.025*(keys[i+1,0]-keys[i,0]))
    for attempt in range(30):
        vertices=np.array([mapped_points(unit,transform(k,a)) for k,a in zip(keys,alphas)])
        interp=np.stack([np.interp(times,keys[:,0],vertices[:,p,c]) for p in range(4) for c in range(2)],axis=1).reshape(-1,4,2)
        bad=[]
        for i,(quad,box) in enumerate(zip(interp,bounds)):
            b=transform_box(box,homography(quad))
            if not covers_frame(quad) or b[:2].min() < -2e-6 or b[2:].max() > 1+2e-6:bad.append(i)
        if not bad:break
        # Reduce the neighboring keys involved in a failed interpolation. Never
        # relax coverage or reveal empty image borders to keep more perspective.
        for i in bad:
            j=max(0,min(len(keys)-2,np.searchsorted(keys[:,0],times[i])-1))
            alphas[j:j+2]*=.5
    if bad:raise ValueError(f'Adaptive perspective fails {len(bad)} coverage checks')
    result['corner_pin_keys']=[{'time':float(k[0]),'corners':q.tolist(),'strength':float(a)} for k,q,a in zip(keys,vertices,alphas)]
    result['statistics'].update(perspective_strength_min=float(alphas.min()),perspective_strength_max=float(alphas.max()),
                                perspective_coverage_violations=len(bad),black_border_samples=0)


def main():
    import argparse
    import json
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', type=Path, help='Completed analysis directory')
    parser.add_argument('config', type=Path, help='Scene-specific FramingConfig JSON')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dramatic', action='store_true')
    args=parser.parse_args()
    manifest=json.loads((args.analysis/'manifest.json').read_text())
    if manifest.get('status')!='complete':
        raise ValueError('Framing requires a completed analysis')
    rows=[json.loads(line) for line in (args.analysis/'detections.jsonl').read_text().splitlines()]
    if len(rows)!=manifest.get('frames'):
        raise ValueError('Analysis frame count does not match its completion manifest')
    config=FramingConfig(**json.loads(args.config.read_text()))
    result=plan(rows,config,dramatic=args.dramatic)
    with args.output.open('x') as output:
        json.dump(result,output,separators=(',',':'))
    print(json.dumps(result['statistics'],indent=2))


if __name__=='__main__':
    main()
