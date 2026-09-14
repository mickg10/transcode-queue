import io
import json

import numpy as np
import pytest

from analysis.core import DecodeConfig, frames, hardware_command
from analysis import analyze as runner
from analysis.framing import FramingConfig, plan, required_box, covers_frame, homography, transform_box


def detection(t, boxes=(), faces=()):
    return {'time':t,'people':[{'box':list(b),'confidence':.9} for b in boxes],
            'faces':[{'box':list(b),'confidence':.9} for b in faces]}


def test_hardware_command_keeps_decode_and_resize_before_python():
    command=hardware_command(DecodeConfig('a movie.mp4', start=17, duration=3))
    assert command[command.index('-hwaccel')+1]=='cuda'
    assert command[command.index('-c:v')+1]=='hevc_cuvid'
    assert command.index('-ss')<command.index('-i')
    assert 'scale_cuda=' in command[command.index('-vf')+1]
    assert 'hwdownload' in command[command.index('-vf')+1]
    assert command[-1]=='pipe:1'


class FakeProcess:
    def __init__(self, data, code=0):
        self.stdout=io.BytesIO(data)
        self.code=code
    def wait(self, timeout=None):return self.code
    def poll(self):return self.code


@pytest.mark.parametrize('data,code,message',[(b'abc',0,'incomplete frame'),(b'',1,'no software fallback')])
def test_decoder_failure_does_not_retry_software(monkeypatch,data,code,message):
    calls=[]
    def start(*args,**kwargs):
        calls.append(args[0]);return FakeProcess(data,code)
    monkeypatch.setattr('analysis.core.subprocess.Popen',start)
    with pytest.raises(RuntimeError,match=message):
        list(frames(DecodeConfig('test.mp4',width=2,height=2)))
    assert len(calls)==1


def test_decoder_reassembles_partial_pipe_reads(monkeypatch):
    class Partial(io.BytesIO):
        def read(self,size=-1):return super().read(min(size,5))
    process=FakeProcess(b'');process.stdout=Partial(bytes(range(24)))
    monkeypatch.setattr('analysis.core.subprocess.Popen',lambda *a,**kw:process)
    output=list(frames(DecodeConfig('test.mp4',width=2,height=2,start=10,fps=2)))
    assert [t for t,_ in output]==[10,10.5]
    assert output[1][1].reshape(-1).tolist()==list(range(12,24))


def test_failed_analysis_records_error_and_closes_reader(tmp_path,monkeypatch):
    closed=[]
    class Detector:
        def __init__(self,*a,**kw):pass
        def detect(self,frame):return {'people':[],'faces':[]}
    def reader(config):
        try:
            yield 0,np.zeros((2,2,3),np.uint8)
            raise RuntimeError('decode interrupted')
        finally:closed.append(True)
    monkeypatch.setattr(runner,'Performers',Detector)
    monkeypatch.setattr(runner,'frames',reader)
    monkeypatch.setattr(runner.cv2,'imwrite',lambda *a:True)
    output=tmp_path/'result'
    with pytest.raises(RuntimeError,match='decode interrupted'):
        runner.analyze(DecodeConfig('test.mp4'),tmp_path,output)
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['status']=='failed' and manifest['frames']==1
    assert closed==[True]
    previous=(output/'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):runner.analyze(DecodeConfig('test.mp4'),tmp_path,output)
    assert (output/'manifest.json').read_bytes()==previous


def test_failed_review_image_is_not_reported_complete(tmp_path,monkeypatch):
    class Detector:
        def __init__(self,*a,**kw):pass
        def detect(self,frame):return {'people':[],'faces':[]}
    monkeypatch.setattr(runner,'Performers',Detector)
    monkeypatch.setattr(runner,'frames',lambda c:(x for x in [(0,np.zeros((2,2,3),np.uint8))]))
    monkeypatch.setattr(runner.cv2,'imwrite',lambda *a:False)
    with pytest.raises(OSError):runner.analyze(DecodeConfig('test.mp4'),tmp_path,tmp_path/'result')
    assert json.loads((tmp_path/'result'/'manifest.json').read_text())['status']=='failed'


def test_elevated_face_supplements_ground_body():
    row=detection(0,[(.55,.4,.7,.8)],[(.48,.16,.51,.2)])
    bounds,unknown=required_box(row,FramingConfig())
    assert not unknown and bounds[1]<.16 and bounds[3]>.865


def test_interpolated_crop_contains_both_performers_and_entry():
    rows=[detection(t,[(.4,.3,.53,.82),(.66+.002*t,.36,.76+.002*t,.84)]) for t in np.arange(0,60,.5)]
    result=plan(rows,FramingConfig())
    keys=np.array(result['keys'])
    for row in result['review']:
        t=row['time'];z,x,y=[np.interp(t,keys[:,0],keys[:,j]) for j in [1,2,3]]
        b=np.array(row['required']).reshape(2,2)
        shown=z*(b-.5)+[x,y]
        assert shown.min()>=-1e-8 and shown.max()<=1+1e-8
        # The crop must also remain within available source pixels.
        assert x-.5*z<=1e-8 and y-.5*z<=1e-8
        assert x+.5*z>=1-1e-8 and y+.5*z>=1-1e-8


def test_long_detection_loss_returns_to_wide():
    rows=[detection(t,[(.45,.3,.6,.8)] if t<10 or t>40 else []) for t in np.arange(0,60,.5)]
    result=plan(rows,FramingConfig())
    key=min(result['keys'],key=lambda k:abs(k[0]-26))
    assert key[1:]==pytest.approx([1,.5,.5])
    assert result['statistics']['missing_detection_samples']>0


def test_dramatic_requires_editorial_ranges():
    rows=[detection(t,[(.45,.32,.6,.82)]) for t in np.arange(0,80,.5)]
    cfg=FramingConfig()
    full=plan(rows,cfg);disabled=plan(rows,cfg,dramatic=True)
    assert disabled['keys']==full['keys']
    cfg.dramatic_ranges=[[0,80]]
    close=plan(rows,cfg,dramatic=True)
    assert close['statistics']['effective_zoom_max']>full['statistics']['effective_zoom_max']


def test_perspective_keeps_pixels_and_required_performers():
    rows=[detection(t,[(.45,.3,.6,.82)]) for t in np.arange(0,30,.5)]
    cfg=FramingConfig(corner_pin=[[-.02,-.04],[1.12,-.08],[-.065,1.01],[1.18,1.21]])
    result=plan(rows,cfg)
    keys=result['corner_pin_keys']; times=np.array([k['time'] for k in keys]);vertices=np.array([k['corners'] for k in keys])
    for row in result['review']:
        quad=np.array([[np.interp(row['time'],times,vertices[:,i,j]) for j in [0,1]] for i in range(4)])
        assert covers_frame(quad)
        bounds=transform_box(row['required'],homography(quad))
        assert bounds[:2].min()>=-2e-6 and bounds[2:].max()<=1+2e-6
    assert not covers_frame(np.zeros((4,2)))


def test_irregular_or_nonfinite_timestamps_rejected():
    for times in [[0,1,3],[0,float('nan')]]:
        with pytest.raises(ValueError):plan([detection(t) for t in times],FramingConfig())
