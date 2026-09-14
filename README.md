# Transcode Queue

A small, unauthenticated web application for a trusted LAN. Browse mounted media
one directory at a time, upload videos, manage presets, and run a persistent
transcoding queue. Docker Compose deployment targets a QNAP with NVIDIA Container
Runtime. The web service and worker are one process; SQLite state lives in /data.

## QNAP deployment

~~~sh
cp .env.example .env
# Check the host paths in .env.
docker compose -f compose.yaml -f compose.qnap.yaml up -d --build
~~~

Open http://QNAP_ADDRESS:8765. The first start is paused. Queue jobs, then click
**Start queue**. Pause prevents the next job from starting; Cancel interrupts the
selected job. A restart recovers interrupted jobs, preserving completed files.
Keep /data on local storage, not an NFS share.

The base mount exposes media/photos. The QNAP override explicitly mounts the
staging photo subtree as the staging_m3 directory because that directory is
commonly a symlink outside the physical photos tree. Unconfigured symlinks that
escape the exposed roots are not browsable.

The QNAP runtime is named nvidia-runtime. On a standard NVIDIA Docker host, change
that to the installed runtime name or use the equivalent GPU reservation. The
application image includes FFmpeg built with NV Codec SDK 13.0 headers, compatible
with Linux NVIDIA drivers 570 and later. This supports the Blackwell HEVC 4:2:2
decode path without requiring the newer SDK 13.1 driver.

For a CPU-only development run:

~~~sh
mkdir -p test-media
docker compose up -d --build
~~~

## C50 proxy preset

- Output: foo/bar.mp4 → foo/proxy/bar.mp4.
- HEVC Main 10, square pixels, up to 3456 × 2304, preserving source aspect ratio.
- NVIDIA hardware decode, decoder resize, CUDA chroma conversion, and NVENC encode.
- NVENC p4, CQ 21, 20 Mbps target / 30 Mbps maximum, 12-frame GOP, no B frames.
- Preserve frame rate, source timecode, range/matrix/primaries/transfer metadata,
  and every audio track. No LUT is baked in.
- Audio is copied, including the C50's four mono 24-bit PCM tracks. The included
  FFmpeg supports PCM in both MP4 and MOV.

Automatic mode tries NVIDIA decode + encode, then CPU decode + NVIDIA encode,
then CPU decode + encode. It only falls back on an initialization failure before
frames are produced. A mid-encode error is reported rather than silently rerun.
Select **NVIDIA decode + encode** to require hardware processing.

An existing readable video output is skipped when its video duration matches
within one source frame. This duration-based rule is deliberately inexpensive;
it is not a content hash or a check that two different presets look identical.
The preset can also reuse existing Proxies/name_Proxy.mov files without copying.
The actual reused path is visible on the job.

Queue entries take a snapshot of their preset. Editing a preset changes new jobs.
An output can only have one queued/running job at a time. Newest-first ordering
uses Canon HYYMMDD_HHMMSS recording names, with modification time as a fallback.
Directory scans exclude hidden folders, proxy folders, renders, and temporary files.

Encodes use a hidden, per-job temporary output. Duration, frame count (when present),
dimensions, timecode, color metadata, and audio properties are checked before
publication. An existing mismatched output is retained as .previous-JOB_ID.
Source media is never overwritten. Uploads stream directly to the chosen media
directory and are published atomically without replacing existing files.

## API

Interactive API documentation is at /docs.

~~~sh
curl -X POST http://localhost:8765/api/jobs \
  -H 'Content-Type: application/json' \
  -d '{"sources":["foo/bar.mp4"],"preset_id":"c50-proxy","output":"foo/proxy/bar.mp4"}'

curl -X POST http://localhost:8765/api/queue-tree \
  -H 'Content-Type: application/json' \
  -d '{"path":"staging_m3/CANON_C50_2TB","recursive":true,"preset_id":"c50-proxy"}'

curl -X POST http://localhost:8765/api/control \
  -H 'Content-Type: application/json' -d '{"paused":false}'
~~~

The optional WAIT_FOR_MANIFEST variable can hold the queue until a pre-existing
nas_proxy_job.py batch has finished. It reads that batch's manifest only.

## Development and tests

~~~sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
DATA_DIR=/tmp/transcode-queue-test MEDIA_ROOT=/tmp .venv/bin/python -m pytest -q
~~~

Tests exercise path containment, duration comparisons, queue ordering and duplicate
insertion, preset snapshots, API controls, uploads, and safe output command creation.
GPU correctness additionally requires a real sample on the target NVIDIA host.

## Offline camera analysis

analysis/core.py supplies sampled video frames to Python using **hardware-decoding
FFmpeg**. It does not use OpenCV VideoCapture or silently fall back to CPU decoding.
It can start FFmpeg locally or through SSH on a GPU host. Only reduced-resolution
frames cross the pipe. Tracking outputs are offline artifacts for editorial review;
they do not modify a Premiere project by themselves.

The optional analysis dependencies are separate from the web worker:

~~~sh
pip install -r requirements-analysis.txt
python -m analysis.analyze decode.json --models models --output analysis-output --gpu
python -m analysis.framing analysis-output framing.json --output full-body.json
python -m analysis.framing analysis-output framing.json --output dramatic.json --dramatic
~~~

The analysis output directory must be new. A failed decoder, inference call, or
review-image write records a failed manifest and returns an error. Existing
analysis results are never replaced by a retry. Framing requires a completed
manifest with the matching number of detection records.

Example decode.json, with paths interpreted on the FFmpeg host:

~~~json
{"source":"/media/photos/example.mp4","width":1152,"height":768,"fps":2,"start":0,"duration":600,"codec":"hevc"}
~~~

FFmpeg must support the source format in its NVIDIA hardware decoder. The GPU
inference option also requires a compatible CUDA PyTorch and ONNX Runtime GPU
installation. A tested analysis environment used PyTorch 2.9.1+cu129,
onnxruntime-gpu 1.26.0, and OpenCV 4.11. Person inference uses CUDA; the lightweight
YuNet face detector uses CPU. These optional inference packages and model weights
are not included in the queue's web image. Omitting --gpu selects CPU inference,
while video decoding remains hardware-only.

Obtain object_detection_yolox_2022nov.onnx and face_detection_yunet_2023mar.onnx
from [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models).
Place them in the models directory and retain the upstream model licenses.
No face identities or recognition embeddings are generated.

Example framing.json (calibrate these normalized coordinates against the scene):

~~~json
{"floor":0.865,"maximum_head_y":0.60,"stage_anchor":[0.38,0.32,0.77,0.865],"dramatic_ranges":[[120,180]],"wide_ranges":[[0,20]]}
~~~

The planner combines body and face bounds, anticipates entrances, bridges short
detection gaps, and widens for longer uncertainty. Dramatic ranges are explicit
editorial choices; the detector does not infer dramatic meaning. Optional
corner_pin coordinates are ordered upper-left, upper-right, lower-left,
lower-right. Perspective strength relaxes where needed to retain performers and
avoid empty borders. If corner_pin_keys are present, they contain the combined
crop and perspective: apply them with native Motion at its identity settings.
Otherwise keys describe native Motion scale and normalized position.

The JSON includes every required sample boundary and interpolation checks.
Coverage proves containment of those constraints, not detection of every person
or coverage of unsampled motion. Review the accompanying frames and the final
native timeline before treating a generated camera move as finished.

## Licensing

Application code is MIT licensed. The image's FFmpeg build enables GPL components
(including x264 and x265); their licenses apply separately. The exact FFmpeg source
revision and NV codec headers used to build the image are included under
/opt/ffmpeg/share/source. Upstream sources:
[FFmpeg](https://github.com/FFmpeg/FFmpeg),
[NV codec headers](https://github.com/FFmpeg/nv-codec-headers),
[x264](https://www.videolan.org/developers/x264.html),
[x265](https://www.videolan.org/developers/x265.html).
