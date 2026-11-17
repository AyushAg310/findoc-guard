# Phase 3 - Video input and tracking: complete code explanation

Version **0.4.0** (patch 0.4.1) · Course outcomes **CO-4** (motion and tracking, the core of this phase),
**CO-3** (feature matching, used to re-find a lost document), **CO-5** groundwork (a pinhole
camera model and the keyframes that Phase 4 reconstructs in 3D).

Phase 3 lets the user film a cheque instead of photographing it:

```
load -> video -> track -> quality -> detect -> rectify -> ... (Phases 1-2, unchanged)
          |        |
   read frames   KLT optical flow follows the document through every frame;
   (streamed,    pick keyframes (sharp, spread viewpoints) for 3D in Phase 4
   downscaled)   and the single best frame for the 2D pipeline
```

Why video at all? A photo is one chance: if it is blurred, glared or taken at a bad angle, the
user retakes. A 3-4 second sweep gives ~100 chances, and the system picks the best one. Video
is also the input that Phase 4 needs: depth comes from **parallax**, i.e. seeing the same point
from different viewpoints.

---

## 1. Big picture

| Step | Technique | File |
| --- | --- | --- |
| Get test data | Render hand-held videos of a cheque with a pinhole camera: sweep, hand shake, wrist jerks, physically based motion blur, optional thumb or camera pan-away; exact per-frame ground truth | `findoc/synth/camera.py`, `findoc/synth/video.py` |
| Read video | Stream frames, keep at most 180 evenly spaced, downscale to 960 px; later re-read only the chosen frames at full resolution | `findoc/io.py` |
| Follow the document | Shi-Tomasi corners + pyramidal Lucas-Kanade optical flow, forward-backward check, reference-anchored RANSAC homography | `findoc/track/klt.py` |
| Keep it accurate | Drift correction (snap to the page detector, or verify against a SIFT "anchor" view) at every re-detection | `klt.py` |
| Survive loss | Mark the document lost when too few points survive; re-localise by SIFT matching against the anchor | `klt.py` |
| Score frames | Sharpness measured on the *rectified* document; coverage; frontalness | `klt.py`, `keyframes.py` |
| Choose frames | Keyframes: sharp + enough viewpoint change (projective shape change); best frame: sharpness x frontalness x coverage | `findoc/track/keyframes.py` |
| Hand over | The best frame becomes `doc.original` with tracked corners, so Phases 1-2 run unchanged | `findoc/track/stages.py` |
| Show it | Keyframe contact sheet, annotated tracking video, Streamlit video mode | `findoc/track/viz.py`, CLI, demo |
| Measure | Tracking error vs. ground truth and vs. a per-frame detector; keyframe blur; best vs. first vs. random frame | `experiments/phase3_tracking_eval.py` |

---

## 2. How a run flows

`findoc synth --kind video --n 1 -o data/synthetic/video` writes `video_0000.mp4`, its ground
truth `video_0000_truth.json` and `labels.json`. Then
`findoc process data/synthetic/video/video_0000.mp4 -s outputs`:

1. `load`: for a video, nothing yet (photos are decoded here as before).
2. `video` (`video_load_stage`): `video_info` reads frame count, fps and size;
   `read_video` streams the file and keeps every `stride`-th frame (stride chosen so at most
   `video_max_frames` = 180 are kept), downscaled so the longest side is at most
   `video_track_side` = 960 px. Frames go to `doc.frames` (excluded from JSON), their original
   frame numbers to `doc.frame_numbers`, and `doc.video` records fps, stride and scale.
3. `track` (`track_stage`):
   - `track_document` finds the document with the Phase 1 page detector in one of the first
     frames, then follows it frame by frame (section 3.4), producing per-frame corners,
     number of tracked points, motion, sharpness, coverage and "fully in view".
   - `select_keyframes` picks the keyframes and the best frame (section 3.5).
   - Results are stored in original-video pixels: `doc.track` (one `TrackedFrame` per frame),
     `doc.keyframes`, `doc.tracking` (summary numbers), `doc.best_frame`.
   - `read_frames` re-reads only the best frame and keyframes **at full resolution** (second
     pass). The best frame becomes `doc.original`, its tracked corners `doc.corners`, and
     `doc.detection_method = "tracked"`. Keyframe images go to `doc.keyframe_images` for
     Phase 4.
   - If no frame is sharp with the whole document in view, the status becomes
     `rejected_quality` with a "hold steadier" message.
4. `quality`, `detect`, `rectify`, `enhance`, `align`, `fields`, `rules`: exactly Phases 1-2.
   `detect_stage` sees `detection_method == "tracked"` and keeps the tracked corners
   (it only computes coverage).
5. The CLI prints a `Tracking:` line and the keyframe list; `--save-dir` also writes
   `<name>_tracking.mp4` (outline per frame: blue = best, green = keyframe, yellow = other),
   `<name>_keyframes.jpg` (contact sheet) and `<name>_best.jpg`.

For a photo, the `video` and `track` stages return immediately, so nothing changes.

---

## 3. File-by-file walkthrough

### 3.1 `findoc/synth/camera.py`: a pinhole camera

The whole phase is tested on synthetic videos, so we need to render what a phone camera would
see. A **pinhole camera** maps a 3D point X (world) to a pixel:

```
x_cam = R X + t          # world -> camera coordinates (rotation R, translation t)
[u v 1]^T ~ K x_cam      # perspective projection, K = [[f 0 cx] [0 f cy] [0 0 1]]
```

- `Intrinsics(f, cx, cy)` holds K. `Intrinsics.for_frame(w, h, hfov_deg)` computes the focal
  length from the horizontal field of view: `f = (w/2) / tan(hfov/2)` (phones: ~60-70 degrees).
- `Pose(R, t)` with `center = -R^T t` (where the camera is in the world).
- `look_at(eye, target, up, roll_deg)` builds R so the camera at `eye` looks at `target`
  (camera axes: z forward, x right, y down; optional roll about the viewing axis).
- `project(K, pose, points)` projects (N, 3) points.
- `plane_homography(K, pose)`: the key formula. The document lies on the plane Z = 0, so for a
  point (X, Y, 0): `R [X Y 0]^T + t = X r1 + Y r2 + t`, hence

  ```
  [u v 1]^T ~ K [r1 r2 t] [X Y 1]^T     =>     H = K [r1 r2 t]
  ```

  A flat document seen by a pinhole camera is **exactly** a homography of its texture. This is
  why rendering is a single `cv2.warpPerspective` per sub-frame, and why tracking can model the
  document's motion between frames with a homography.

### 3.2 `findoc/synth/video.py`: synthetic hand-held videos

- `desk_texture`: a procedural desk (wood-grain waves, blotches, noise) so that the background
  has texture, as real desks do. A plain background would make tracking unrealistically easy.
- `_canvas`: one large image with the document pasted in the middle of the desk, plus the
  canvas-to-world transform (world origin = document centre, 1 world unit = 1 template pixel).
- `camera_path`: camera centres for every frame: a left-to-right **sweep** (limited so the
  viewing angle stays within `tilt_deg` = 22 degrees), a slow forward/back drift, the camera
  slightly farther away at the ends of the sweep, plus low-frequency **hand shake**
  (`_smooth_noise`: random noise smoothed with a Hanning window, so it wobbles rather than
  jitters).
- `render_video`: for each frame,
  - the pose is interpolated along the path (`pose_at(u)`, u = fractional frame index);
  - **hand jerks** (`jerks` = 2): a Gaussian pulse (sigma 1.2 frames) added to where the camera
    *looks*. A flick of the wrist is a rotation, which sweeps the whole image sideways; this is
    what produces the heavy motion blur seen in real phone videos. (A first attempt moved the
    camera position instead; because the camera keeps looking at the document, translation
    barely moved the image and produced almost no blur.)
  - **motion blur**: the shutter is open for `shutter` = 0.5 of the frame interval. We render
    `subframes` = 5 warps at times spread across the open shutter and average them. Blur is
    therefore *caused* by camera motion, exactly as in a real camera, not a filter added
    afterwards.
  - small roll noise (about 2 degrees), image noise.
  - ground truth (`VideoTruth`): document corners at mid-shutter, `in_frame` (all four corners
    inside the image), `blur_px` (how far the document centre moves while the shutter is open)
    and the camera centre.
  - `exit_frame=True` pans the camera away mid-clip so the document leaves the view and comes
    back; `thumb=True` paints a thumb holding the cheque by its bottom-left corner (`add_thumb`),
    hiding that corner. Both are used by the evaluation.
- `cheque_video(seed, fraud, ...)` renders a cheque (Phase 2 renderer) and films it;
  `generate_videos` writes a dataset: `video_XXXX.mp4`, `video_XXXX_truth.json`, `labels.json`
  (same cheque labels as photos). `write_video` uses the `mp4v` codec (plays in VLC; browsers
  may not play it).

### 3.3 `findoc/io.py`: reading video efficiently

- `video_info`: frame count, fps, width, height (raises `ValueError` if the file can't be
  opened).
- `read_video(path, max_frames, max_side)`: **first pass**. `cap.grab()` advances without
  decoding to an image; `cap.retrieve()` decodes only the frames we keep. With 180 frames at
  960x540 this is about 280 MB at most, instead of several GB for a full-resolution 30 s clip.
- `read_frames(path, numbers)`: **second pass**, full resolution, for only the frames that
  were chosen. It reads sequentially rather than seeking, because seeking (`CAP_PROP_POS_FRAMES`)
  is unreliable for many phone codecs (it jumps to the nearest keyframe).

### 3.4 `findoc/track/klt.py`: following the document

**Optical flow.** Brightness constancy says a point keeps its intensity as it moves:
`I(x + u, y + v, t + 1) = I(x, y, t)`. A first-order Taylor expansion gives one equation per
pixel, `I_x u + I_y v + I_t = 0`, with two unknowns (u, v). **Lucas-Kanade** assumes all pixels
in a small window (here 21x21) move together and solves the over-determined system by least
squares:

```
[ sum Ix^2   sum IxIy ] [u]     [ sum IxIt ]
[ sum IxIy   sum Iy^2 ] [v] = - [ sum IyIt ]
```

The 2x2 matrix is the **structure tensor**. It is invertible (well conditioned) only when the
window has gradients in two directions, i.e. at corners. On a straight edge one eigenvalue is
~0 and motion along the edge is unknown: the **aperture problem**.

- `FEATURE_PARAMS` / `detect_features`: **Shi-Tomasi** corners (`cv2.goodFeaturesToTrack`)
  keep points whose *smaller* eigenvalue of the structure tensor is large, which are exactly
  the points where Lucas-Kanade is well conditioned. Up to 600 points, at least 7 px apart, only
  inside the document (`quad_mask`, shrunk 3% so the page edge and desk don't count).
- `LK_PARAMS`: `maxLevel=3` builds an image **pyramid** (each level half the size). Flow is
  solved at the coarsest level first, then refined. A 40 px motion is only 5 px at level 3,
  inside the small-motion assumption.
- **Forward-backward check** (`FB_MAX_ERROR` = 1 px): track t-1 -> t, then track the result
  back t -> t-1. A good point returns to where it started. Points that don't (occluded,
  blurred, on a repeated pattern) are dropped.
- **Reference-anchored homography**: every point remembers its position in the *reference*
  frame (where it was detected), `p_ref`. The document is planar, so reference and current
  positions are related by one homography, fitted with RANSAC (3 px). The document corners in
  the reference frame are mapped through it. Chaining frame-to-frame homographies instead
  would multiply small errors at every frame; anchoring keeps the error from growing
  until the next re-detection. A non-convex result is rejected.
- **Re-detection with drift correction**: when fewer than `MIN_TRACKS` = 150 points survive
  (they fall off as blur and viewpoint change), new Shi-Tomasi corners are detected in the
  current outline and this frame becomes the new reference. A reset freezes whatever error the
  current estimate has, so errors would accumulate across resets (drift). So at each reset:
  1. `snap_to_detection`: run the page detector; if its quad is clear of the border and agrees
     with the track within `SNAP_MAX` = 1% of the diagonal, adopt its corners;
  2. otherwise `Anchor.locate`: verify against the stored appearance (below) and adopt that.

  Measured on clean clips: without correction 0.43% corner error, with it 0.13%.
- **Loss and re-localisation**: with fewer than `MIN_FOR_HOMOGRAPHY` = 20 points the document
  is marked lost (`corners = None`). `Anchor` stores SIFT features of the document from the
  first frame it was found. In each lost frame, `Anchor.locate` matches them against the whole
  frame (Lowe ratio 0.75 + RANSAC, at least 25 inliers: the Phase 2 recipe, CO-3) and maps the
  anchor's corners into the frame. Matching appearance finds the *whole* document even when
  part of it is still outside the view. The page detector is the last resort, accepted only
  when its quad is clear of the image border: a document cut off by the border gives a smaller,
  wrong quad (an early version did this and ended up 72 px off after re-acquisition; using the
  anchor brought it to about 8 px).
- `region_sharpness`: Laplacian variance of the document **after rectifying it** to a
  rectangle (at most 600 px wide, never upsampled). A first version measured the bounding box
  in the frame, which includes a viewpoint-dependent amount of textured desk; its sharpness
  barely correlated with true blur. On the rectified document, rank correlation with true blur
  is about -0.94 at 1280x720 (-0.65 at 640x360, where the document is small).
- `FrameTrack` / `TrackResult`: the per-frame record and the summary counters
  (`lost`, `redetections`, `reacquisitions`, `snaps`). `draw_track` overlays one frame.

### 3.5 `findoc/track/keyframes.py`: choosing frames

Two consumers want different frames. The 2D pipeline wants **one best frame**; 3D
reconstruction (Phase 4) wants **several sharp frames from different viewpoints**.

- `viewpoint_change(a, b)`: how differently two outlines are *shaped*. Moving the camera
  sideways or closer moves and scales the outline, which says nothing about the viewing angle.
  What changes with the angle is the projective shape (a rectangle becomes a trapezoid). So we
  fit the best **similarity** (translation, rotation, uniform scale:
  `cv2.estimateAffinePartial2D`) from one outline to the other and measure the remaining corner
  distance, divided by the document diagonal. It is 0 for pure pan, zoom or roll.
- `frontalness`: shorter / longer of opposite sides, multiplied for both pairs; 1.0 head-on.
- `usable_frames`: tracked, fully in view, and sharp: at least `min_sharpness` *and* at least
  0.75 x the clip's median sharpness. The relative test matters: absolute sharpness depends on
  the document and the camera, but a frame at under 75% of the clip's typical sharpness is
  smeared by motion.
- `best_frame`: maximises (sharpness / max) x frontalness x (coverage / max).
- `select_keyframes`: first keyframe = sharpest usable frame at the start; then scan forward
  and, once the viewpoint has changed by `min_view_change` since the last keyframe, take the
  sharpest usable frame in the next `window` = 6 frames. If fewer than 8 keyframes come out,
  halve the threshold and repeat (at most 4 times). Never more than 25.

### 3.6 `findoc/track/stages.py`: the pipeline stages

`video_load_stage` and `track_stage` (section 2). Two details:

- Tracking runs on downscaled frames, so corners are multiplied by `1/scale` before they are
  stored (`_round`): everything in `doc.track` and `doc.keyframes` is in original-video
  pixels, so it can be compared with ground truth and used with the full-resolution frames.
- If the second pass fails to read the best frame (unusual codecs), the tracking-size frame is
  used instead, with corners at that scale.
- If no frame is usable, `rejection_reasons` says *why*, most fundamental cause first: too
  little detail to track (fewer than 20 points per frame on average), the document never fully
  inside the frame, no outline found at all, or (if none of those) blur. (v0.4.1: the first real
  test video, a blank notebook page larger than the frame, was rejected correctly but with a
  generic "hold steadier" message.)
- "Fully in view" (`_coverage_and_inside` in `klt.py`) requires every corner to be at least a
  few pixels inside the image. The page detector clips outlines that run off the image to the
  border, so a corner exactly on the border means the page continues beyond it.

### 3.7 `findoc/types.py`, `config.py`, `pipeline.py`, `preprocess/stages.py`: what changed

- `Document` gains `frames` and `keyframe_images` (excluded from JSON), `frame_numbers`,
  `video`, `track` (list of `TrackedFrame`), `tracking`, `keyframes` (list of `Keyframe`) and
  `best_frame`. New models `TrackedFrame` and `Keyframe`.
- `Settings`: `video_max_frames` (180), `video_track_side` (960), `keyframe_min_view_change`
  (0.008), `keyframe_max` (25). All can be set with `FINDOC_...` environment variables.
- `default_pipeline`: `load -> video -> track -> quality -> ...`. `load_stage` no longer
  prints "Video input registered (processing arrives in Phase 3)".
- `detect_stage`: keeps tracked corners for video and just records coverage.

### 3.8 `findoc/track/viz.py`, CLI and demo

- `contact_sheet`: thumbnails in a grid with labels. `keyframe_sheet`: the keyframes at full
  resolution with their outline, frame number, sharpness and viewpoint change.
- `write_tracking_video`: every tracked frame with its outline and metrics.
- CLI: `findoc synth --kind video --n N --seconds S`; `process` prints `Tracking:` and
  keyframes, and saves the tracking video, contact sheet and best frame with `-s`.
- Streamlit: upload a video or choose **Cheque video**; section "0. Video tracking (KLT) and
  keyframes" shows the tracking summary, a sharpness / tracked-points chart, the contact sheet,
  and a download for the tracking video. The 2D sections below show the best frame.

### 3.9 `experiments/phase3_tracking_eval.py`: results

`python experiments/phase3_tracking_eval.py --n 18` (about 5 minutes on a laptop CPU):
18 synthetic 4 s clips at 1280x720 / 30 fps, six per capture condition: **clean**,
**thumb** (thumb hides the bottom-left corner) and **exit** (camera pans away; the document
leaves the view and returns). Corner error = mean distance of the four corners from ground
truth, as % of the frame diagonal (0.1% = 1.5 px).

| Condition | KLT error | Page detector every frame | KLT in-view frames tracked | Best-frame template id | Best-frame field IoU |
| --- | --- | --- | --- | --- | --- |
| clean | 0.130% | 0.137% | 100% | 100% | 0.990 |
| thumb | 1.438% | 1.276% | 100% | 100% | 0.990 |
| exit | 0.305% | 0.165% | 99.7% | 100% | 0.991 |

Time per frame: KLT 12.4 ms (on 960 px frames), page detector 5.2 ms (1280 px). Whole pipeline
about 2.9 s per 4 s clip.

| Keyframes | Selected | Same number, evenly spaced |
| --- | --- | --- |
| Mean motion blur | 1.01 px | 5.73 px |
| Worst motion blur | 1.87 px | 25.90 px |
| Frames blurred more than 3 px | 0% | 20.5% |

About 9.6 keyframes per clip, covering 23.7 of the 26.1 degrees of viewing angle in the clip.

| Frame sent to the 2D pipeline | Blur | Template identified | Field-box IoU |
| --- | --- | --- | --- |
| Selected best frame | 0.43 px | 100% | 0.990 |
| First frame | 1.19 px | 22.2% | 0.988 |
| Random frame | 8.85 px | 50.0% | 0.990 |

**What the numbers say (be ready to explain this honestly):**

1. **On these synthetic clips, tracking is not more accurate than detection.** The desk
   contrasts well with the cheque, so the Phase 1 detector finds the page in every frame, to
   0.14%. KLT matches it on clean clips (0.13%) and is about 2x slower per frame. It took two
   fixes to get there (drift correction: 0.43% -> 0.13%; anchor re-localisation: 6.7% -> 0.3%
   worst exit clip).
2. **Why track, then?** (a) Keyframe selection needs a *consistent* outline in every frame to
   measure viewpoint change between frames, including frames where detection fails: on real
   captures (busy desks, white paper on a white table) edge detection fails far more often
   than on synthetic data. (b) Phase 4's Structure-from-Motion needs **point tracks**: the
   same physical point followed across frames. KLT produces exactly that; per-frame detection
   produces only four corners. (c) Tracking knows where the document is when its edges are
   hidden or out of view (99.7% of in-view frames tracked through pan-away and back).
3. **Thumb occlusion fools both** (about 1.3-1.4%): the thumb sits on the corner, and the
   tracker is initialised from, and snaps to, the detector. This does **not** hurt the 2D
   result (IoU 0.990), because Phase 2's SIFT registration to the template corrects the
   outline. Phase 4 can use the same trick (template-registered corners) for keyframes.
4. **Selection matters a lot.** Keyframes avoid blur completely (0% vs 20.5%). The first frame
   was rejected by the coverage gate in 78% of clips (the camera starts farther away and at an
   angle), and a random frame often lands in a blur burst.

With `--real-ocr` the script also reads the fields with EasyOCR on each of the three frames
(run on the RTX 3050 laptop, `experiments/results/phase3_real_ocr.json`; a frame whose template
was not identified scores 0):

| Frame | Field accuracy (EasyOCR) |
| --- | --- |
| Selected best frame | 94.4% |
| First frame | 19.0% |
| Random frame | 44.4% |

This is the clearest result of the phase: choosing the frame roughly doubles to quintuples what
OCR gets right. It is below the 100% of Phase 2 photos; the evaluation records accuracy per
clip, not per field, so finding out which fields fail on video frames is a follow-up.

### 3.10 Tests (141 in total, 95% coverage)

`tests/test_track.py` uses small, fast clips (640x360, 1.5 s, 3 sub-frames; shared by a
module-scoped fixture):

- camera: focal length from FOV; `plane_homography` agrees with `project` to 1e-6.
- synthetic video: lengths, ground truth, jerks give blur bursts, `exit_frame` really leaves
  the view, dataset files, fraud flag passes through.
- tracking: corners within 1% of the diagonal, start from given corners, recovery after a cut
  (blank frames inserted -> lost -> re-acquired), mask and drawing.
- keyframes: `viewpoint_change` ~0 for a similarity and >0.02 for a keystone, frontalness,
  keyframes sorted and less blurred than the worst frame, empty input, rejection reasons
  (blank page, page larger than the frame, blur), outlines on the border not counted as in view.
- video I/O: stride, scale, second-pass reading, bad file.
- pipeline: end to end on a video, nothing image-like in the JSON; CLI `synth --kind video`
  and `process -s` on a 1280x720 clip (template found).

Changed tests: the stage order now includes `video` and `track`; the old test "video input is
registered but not processed" became "an unreadable video fails cleanly".

---

## 4. Concepts used in this phase

| Concept | In one paragraph |
| --- | --- |
| Pinhole camera, K, [R t] | A 3D point is rotated and translated into camera coordinates and divided by depth; K converts to pixels with the focal length and principal point. |
| Plane-induced homography | Points on a plane Z = 0 project through H = K [r1 r2 t], a 3x3 homography. A flat document's image under any camera pose is a homography of its texture. |
| Motion blur | The sensor integrates light while the shutter is open; if the image moves, the result is the average along the motion path. We simulate it by averaging sub-frames. |
| Optical flow, brightness constancy | Each point keeps its brightness as it moves; linearised, that is one equation `Ix u + Iy v + It = 0` per pixel. |
| Lucas-Kanade | Assume constant motion in a window and solve the 2x2 least-squares system (the structure tensor). |
| Aperture problem | Along a straight edge only the motion perpendicular to it is observable; corners fix both directions. |
| Shi-Tomasi | Keep points whose smaller structure-tensor eigenvalue is large: good to track. |
| Image pyramid | Coarse-to-fine solving makes large motions small at the coarse levels. |
| Forward-backward error | Track there and back; disagreement flags unreliable tracks without any ground truth. |
| RANSAC homography | Fit to random 4-point samples, keep the model with the most inliers; robust to outliers. |
| Drift | Errors accumulate when each estimate is built on the previous one; fixed by anchoring to a reference and correcting against an independent measurement. |
| Re-localisation | Recognise a stored view (SIFT + RANSAC) to recover after tracking is lost. |
| Laplacian variance | High-frequency energy; drops with blur. Comparable only for the same content at the same scale, hence rectifying first. |
| Similarity vs. projective change | A similarity cannot turn a rectangle into a trapezoid; what is left after removing the best similarity measures viewpoint change. |
| Parallax and baselines | Depth needs the same point seen from different positions; keyframes are spread out so that Phase 4 has parallax. |
| Two-pass video reading | Track on small frames, then decode only the chosen frames at full resolution: bounded memory, full detail where it matters. |

---

## 5. Try it yourself

1. Generate and process a video, then open the outputs:
   ```
   findoc synth --kind video --n 1 --seed 3 -o data/synthetic/video
   findoc process data/synthetic/video/video_0003.mp4 --no-ocr -s outputs
   ```
   Open `outputs/video_0003_tracking.mp4` in VLC. Where are the blurred frames? Are they
   keyframes?
2. Film a cheque-sized piece of paper with your phone for 3-4 s, sweeping sideways and
   tilting a little. Copy it to the laptop and run `findoc process` on it. How many frames
   were lost? Which frame was chosen as best?
3. In `klt.py`, set `FB_MAX_ERROR = 100` (effectively off) and rerun
   `pytest tests/test_track.py`. Then run the evaluation with `--n 6`. What happens to the
   corner error?
4. Set `SNAP_MAX = 0` (no drift correction) and compare the clean-condition error with the
   table above.
5. In a Python shell, compute `viewpoint_change` for a rectangle and the same rectangle
   rotated by 30 degrees and scaled by 2, then for a trapezoid. Explain the two results.
6. Run `streamlit run demo/streamlit_app.py`, choose **Cheque video**, and compare the
   sharpness chart with the keyframes chosen.
7. Run `python experiments/phase3_tracking_eval.py --n 6 --real-ocr` on the GPU laptop. Is
   OCR field accuracy different on the best frame and on a random frame?

---

## 6. Viva and interview questions

**Q: What does Phase 3 add, in one sentence?**
A: Video input: the document is tracked through a short phone video with KLT optical flow, the
sharpest well-placed frame is sent to the existing 2D pipeline, and a set of sharp keyframes
from different viewpoints is kept for 3D reconstruction.

**Q: Explain the Lucas-Kanade method.**
A: Brightness constancy gives `Ix u + Iy v + It = 0` per pixel. Assuming constant motion in a
21x21 window gives many equations for (u, v), solved by least squares through the 2x2
structure tensor. It needs texture in two directions (corners), and the pyramidal version
handles motions larger than the window.

**Q: Why Shi-Tomasi corners, not every pixel?**
A: Lucas-Kanade is only well conditioned where the structure tensor has two large
eigenvalues. Shi-Tomasi selects exactly those points. Elsewhere the aperture problem makes
motion ambiguous.

**Q: What is the forward-backward check?**
A: Track a point to the next frame and back again. If it does not return to within 1 px, the
track is unreliable (occlusion, blur, repeated texture) and is dropped. It needs no ground
truth.

**Q: Why fit a homography to the tracked points?**
A: The document is planar, so the image motion of all its points is one homography. Fitting it
with RANSAC averages hundreds of points and rejects outliers; the corners follow from it even
when they are off-screen or hidden by a thumb.

**Q: What is drift and how do you limit it?**
A: Small errors pile up when each estimate is built on the previous one. Points are anchored to
a reference frame rather than chained frame to frame, and at every reference reset the outline
is corrected against the page detector (if they agree within 1% of the diagonal) or against a
stored SIFT view of the document. This brought the error on clean clips from 0.43% to 0.13% of
the diagonal.

**Q: What happens if the document leaves the view?**
A: With fewer than 20 reliable points it is marked lost. Each later frame is matched against
the stored SIFT anchor view (ratio test + RANSAC, 25 inliers minimum) to re-localise it. 99.7%
of in-view frames were tracked in pan-away clips, with about one re-localisation per clip.

**Q: Your per-frame detector was as accurate as the tracker. Why track at all?**
A: On synthetic clips with a contrasting desk, edge detection works in every frame, so yes.
Tracking matters because (1) real backgrounds break edge detection much more often, (2)
keyframe selection needs consistent outlines in every frame, and (3) Phase 4's
Structure-from-Motion needs point correspondences across frames, which the detector does not
provide. I report this honestly rather than claim tracking is more accurate.

**Q: How do you measure blur without ground truth?**
A: Laplacian variance of the document after warping it to a fixed rectangle, so every frame
shows the same content at the same scale. On the raw bounding box the score depended on how
much desk was visible. Rectified, its rank correlation with true blur is about -0.94.

**Q: How do you choose keyframes?**
A: Usable frames are tracked, fully in view and at least 75% as sharp as the clip's median. A
new keyframe is added once the viewpoint has changed enough since the last one, and it is the
sharpest usable frame in a 6-frame window. Result: 0% of keyframes blurred versus 20.5% for
evenly spaced frames, while still covering 24 of 26 degrees of viewing angle.

**Q: How do you measure viewpoint change from 2D outlines?**
A: Remove the best similarity transform (translation, rotation, scale) between the two outlines
and measure what is left, relative to the diagonal. Panning, zooming or rolling the camera
leaves it at zero; changing the viewing angle turns the rectangle into a trapezoid, which a
similarity cannot undo.

**Q: How do you pick the best frame?**
A: Highest product of relative sharpness, frontalness and relative coverage. Sent to the 2D
pipeline, it was identified 100% of the time, against 22% for the first frame (too far away and
rejected by the coverage gate) and 50% for a random frame (often blurred). With real OCR, field
accuracy was 94.4% on the best frame, 19% on the first and 44% on a random frame.

**Q: Why render synthetic videos?**
A: Exact ground truth for every frame (corners, blur, visibility, camera pose) and full control
over the conditions (jerks, thumb, pan-away) at no labelling cost. The rendering is physically
based: plane-induced homography for geometry and shutter integration for blur.

**Q: How is the video read without running out of memory?**
A: Two passes. First, frames are streamed, subsampled to at most 180 and downscaled to 960 px
for tracking. Second, only the best frame and keyframes are decoded again at full resolution.

**Q: What are the limitations?**
A: Synthetic evaluation only so far (real captures in Phase 8); a thumb over a corner biases
the initial outline (the 2D result is corrected by template registration); the tracker is about
2x slower than detection per frame; and `mp4v` output does not play in every browser.
