# Phase 5 - 3D reconstruction: complete code explanation

Version **0.6.0** · Course outcome **CO-5** (multi-view geometry: structure from motion,
triangulation, bundle adjustment, camera calibration), building on **CO-4** (the video and
keyframes of Phase 3) and **CO-1/CO-3** (features and homographies).

## Why this phase exists

Up to Phase 4, the flat cheque image is made with **one homography**: the four tracked corners
are mapped to a rectangle. A homography is exact only for a **flat** page. Real cheques are
rarely flat: they come out of a wallet curled, folded in half, or wavy. Under a homography the
middle of a folded cheque is squeezed or stretched, and every later stage (template alignment,
field crops, tamper detection in Phase 6) sees a distorted page.

A video gives many views of the same page from slightly different positions. From those views
we can recover the page's **3D shape** and then unroll it flat, the way you would smooth a
crumpled receipt with your hand:

```
keyframes -> is the page flat? --yes--> keep the 2D homography (exact for a flat page)
               | no
               v
           structure from motion (camera poses + 3D points)
               -> page points -> smooth surface -> mesh seen from the best frame
               -> ARAP: unroll the mesh with as little stretching as possible
               -> remap the best frame onto the unrolled mesh -> flat page + correspondence map
```

The result on synthetic hand-held videos (section 3.9): with 3D, the error left on a folded
cheque falls from **0.179% to 0.072%** of its width (2.5x smaller), and on the other bends by
about a third. Flat pages are recognised as flat and keep the 2D result.

---

## 1. Big picture

| Step | Technique | File |
| --- | --- | --- |
| Bent cheques with exact ground truth | Developable surfaces (generalised cylinders) from a curvature profile; triangle rasterisation | `findoc/synth/surface.py`, `findoc/synth/video.py` |
| Camera poses and 3D points | Incremental SfM: SIFT, essential matrix, homography decomposition, PnP, triangulation, bundle adjustment, focal-length search | `findoc/recon3d/sfm.py` |
| Is the page flat? | Median 90th-percentile homography residual between the best frame and each keyframe | `findoc/recon3d/pipeline.py` (`plane_residual`) |
| Page surface | Robust thin-plate-spline height field | `findoc/recon3d/flatten.py` (`fit_surface`) |
| Mesh | Image grid inside the page outline, rays cast onto the surface | `flatten.py` (`grid_in_quad`, `cast_rays`) |
| Unrolling | ARAP parameterisation with cotangent weights, run to convergence | `flatten.py` (`arap_flatten`) |
| Flat image | Per-pixel sampling map, `cv2.remap` | `flatten.py` (`unwarp`) |
| Pipeline stage | `recon3d` after `detect`; `rectify` skips when 3D succeeded | `findoc/recon3d/stages.py`, `findoc/pipeline.py`, `findoc/preprocess/stages.py` |
| Calibration | Zhang's method with a printed checkerboard; `findoc calibrate` | `findoc/recon3d/calibrate.py`, `findoc/cli.py` |
| Outputs | Point cloud and mesh (PLY), preview image, flat map | `findoc/recon3d/export.py` |
| Measure | 3D vs. 2D on 15 synthetic videos, focal estimated and known | `experiments/phase5_recon_eval.py` |

No new heavy dependencies: everything is NumPy, OpenCV and SciPy. (The roadmap had planned
COLMAP and Open3D; writing SfM by hand keeps the code explainable and the install light.)

---

## 2. How a run flows

`findoc process cheque.mp4` runs the Phase 3 stages (load video, track, pick the best frame and
keyframes) and the Phase 1 quality gate and detection. Then the new **`recon3d` stage**:

1. Takes the best frame plus up to 9 keyframes spread over the clip, and the page corners in
   the best frame.
2. Extracts SIFT features in all of them and runs the **planarity test**. If a single
   homography explains the page in every view to within 1.3 px (90th percentile, scaled for
   resolution), the page is flat: the stage records `reason = "page is flat..."` and stops. The
   2D path then runs exactly as in v0.5.
3. Otherwise runs **SfM**. Without a calibrated focal length it tries several focal lengths
   and keeps the one with the lowest reprojection error.
4. Keeps the 3D points that project inside the page outline, fits a smooth surface, builds a
   56 x 28 mesh over the page as seen in the best frame, unrolls it with ARAP and resamples
   the best frame. It sets `doc.flat`, `doc.flattening = "3d"` and `doc.flat_map`.
5. Safety checks fall back to 2D with a reason: fewer than 40 page points, the camera looking
   along the page (a collapsed reconstruction), or an unrolled mesh that stretches more than
   6% (paper does not stretch, so the 3D shape must be wrong).

`rectify` then skips (the page is already flat) and the cheque stages run on the 3D-flattened
page. The console shows a `3D:` line with views, points, focal length, stretch and timing;
`--save` also writes `<name>_points.ply`, `<name>_mesh.ply`, `<name>_recon3d.jpg` and
`<name>_flatmap.npz`.

Settings: `FINDOC_RECON3D=off` disables the stage; `FINDOC_FOCAL_PX=<pixels>` gives the
calibrated focal length (`findoc calibrate board.mp4 --save` writes it to `.env`).

---

## 3. File-by-file walkthrough

### 3.1 `findoc/synth/surface.py`: bent paper with exact ground truth

To measure flattening we need videos of bent cheques **and** the true flat cheque. Paper bends
without stretching, so a bent sheet is a **developable surface**. The module models the common
cases as generalised cylinders: the sheet bends only along one direction `a`, following a
profile curve described by its curvature k(u) along the arc length u.

- `Bend`: the kind (`flat`, `curl` = one end lifted, `curl2` = both ends, `fold` = a crease with
  a rounding radius, `wave`) and its parameters. `curvature(u)` returns k(u).
- `DocSurface.points(st)`: integrates the curvature numerically (angle phi = integral of k,
  then x and z = integrals of cos phi and -sin phi) and maps paper coordinates (s, t) to 3D.
  Because the profile is parametrised by arc length, the map is an exact isometry: the true
  flattened page is simply the original texture.
- `lift()`: moves the surface so it rests on the desk (Z = 0) with its lowest point.
- `mesh(nx, ny)`: a triangle mesh with texture coordinates.
- `render_surface`: draws the textured mesh through the camera. For every triangle it computes
  the affine map from image to texture, rasterises triangle IDs, and builds one `cv2.remap`
  lookup for the whole frame. `_triangle_shading` darkens triangles that face away from the
  light, so a fold shows shading like a real one.
- `random_bend(kind, rng)`: a random bend of each kind in the ranges used by the evaluation (a
  15-40 degree fold, curls with radius 0.6-1.5 page widths, gentle waves).

### 3.2 `findoc/synth/video.py`: curved cheques in the hand-held videos

`render_video(..., bend=...)` renders the page with `render_surface` instead of a flat
homography. The ground truth gains `meta["poses"]` (camera R, t per frame), `meta["K"]`,
`meta["bend"]` and `meta["lift"]`, which the evaluation uses. `to_world` converts paper
coordinates to world points. `generate_videos(..., bend="fold")` and
`findoc synth --kind video --bend fold` write bent-cheque videos (any other name is rejected).

### 3.3 `findoc/recon3d/sfm.py`: structure from motion

SfM recovers the camera pose of every keyframe and the 3D positions of points seen in several
of them, using only correspondences and the pinhole model x ~ K (R X + t).

- **`extract`**: SIFT (up to 4000 features) on frames downscaled to at most 1600 px; positions
  are kept in full-resolution pixels.
- **`match_pair`**: ratio-test matches, then the **essential matrix** with RANSAC
  (`cv2.findEssentialMat`, 1.5 px). A cheque on a desk is almost a *plane*. For a plane the
  fundamental matrix (8-point) is degenerate, while the calibrated 5-point essential-matrix
  method still works, which is why a focal length must be assumed. OpenCV can throw on
  degenerate input, so the call is wrapped in `try/except`.
- **`build_tracks`**: union-find chains pairwise matches into one track per physical point.
  Tracks that would contain two keypoints of the same image are dropped.
- **`triangulate`**: linear (DLT) triangulation from any number of views: each view gives two
  rows of A X = 0, solved by SVD.
- **`_init_candidates`**: the starting pair needs parallax. Pose candidates come from the
  essential matrix (`recoverPose`) **and** from decomposing the pair's homography
  (`cv2.decomposeHomographyMat`). For a near-planar scene the essential matrix is
  ill-conditioned and the homography decomposition is often the correct one. Candidates with
  less than 80% of points in front of both cameras are discarded; the best two are kept.
- **`_grazing`**: rejects reconstructions in which the cameras look along the page (over 70
  degrees from its normal). That is a collapsed solution, not a real one.
- **`_grow`**: registers the remaining views one at a time by **PnP + RANSAC** against the points
  already in 3D, triangulates newly visible tracks (only if the triangulation angle is at least
  1 degree), and bundle-adjusts.
- **`bundle_adjust`**: SciPy `least_squares` over all poses (axis-angle + translation), all
  points and, optionally, the shared focal length; Huber loss; a **sparse Jacobian pattern** so
  thousands of points solve in seconds (each residual depends on one camera and one point
  only). Observations above 4 px afterwards are removed.
- **`reconstruct`**: if `focal_guess` is given (calibrated), it runs once and keeps the focal
  fixed. Otherwise it scans `FOCAL_SCAN` = 0.6-1.2 x the longer image side, then refines around
  the best at +-0.06, re-runs the winner with the same random seed (`cv2.setRNGSeed(0)`, so
  RANSAC is reproducible), and finally lets bundle adjustment refine the focal. In the
  evaluation the estimated focal was within 15% of the true one (1037-1229 px against 1065 px).

### 3.4 `findoc/recon3d/pipeline.py`: from keyframes to a flat page

- **`plane_residual`**: for each keyframe, matches features with the reference frame inside the
  page outline, filters them with a fundamental matrix, fits a homography with RANSAC and takes
  the 90th percentile of the residuals; the median over views is the score. A flat page scores
  about 0.5-1 px (feature noise); a bent page scores several pixels.
- **`_inside` / `_near`**: which projected points fall inside the outline (page) or in a band
  around it (desk).
- **`flatten_document`**: the steps of section 2, with the safety checks. It returns a
  `Flattening` with the flat image, the sampling maps, the reconstruction and statistics. The
  statistics end up in `doc.recon3d` and in the JSON output.

### 3.5 `findoc/recon3d/flatten.py`: surface, mesh, unrolling, resampling

- **`fit_surface`**: PCA of the page points gives a plane and a local frame (a, b along the page,
  h = height). Height is modelled as a **thin-plate spline** h(a, b) with smoothing (SciPy
  `RBFInterpolator`): the smoothest surface that passes near the points. It is refitted three
  times, each time dropping points more than 3 robust standard deviations (MAD) off. Those are
  mismatches, and desk points that happen to project inside the outline when the page curls up.
  The smoothing is 1e-3; lighter smoothing followed SfM noise into bumps. `planarity` is the RMS
  height / diagonal. If at least 60 desk points are found, a smoother fit to the desk measures
  any doming of the whole reconstruction and `relief` subtracts it. On the synthetic desks there
  are rarely that many, so `relief` is usually just the height.
- **`grid_in_quad`**: a regular grid of pixels inside the page outline in the best frame, with
  its triangles.
- **`cast_rays`**: intersects each pixel's viewing ray with the surface. The crossing of
  g(lambda) = height(X(lambda)) - h(a, b) is **bracketed** by sampling 41 values around the ray's
  intersection with the plane, then refined by 30 bisection steps, which always converges. (A
  fixed-point iteration diverged on steep folds.) Rays that miss give NaN.
- **`arap_flatten`**: "as rigid as possible" parameterisation (Liu et al., 2008). Each triangle
  is laid flat in its own 2D frame (`_local_frames`). Then two steps alternate:
  - local step: for every triangle, the rotation that best maps its flat shape onto its current
    2D shape (2x2 SVD, the orthogonal Procrustes problem; reflections are fixed);
  - global step: given the rotations, one sparse linear solve of the **cotangent Laplacian**
    gives all vertex positions. The matrix never changes, so it is factorised once.

  It iterates **until the vertices stop moving** (largest move below 1e-5 of the mesh size, at
  most 200 iterations). `stretch` = RMS relative error of edge lengths. Degenerate input
  (non-finite vertices, zero-area triangles) is handled; a broken mesh reports stretch = inf.
- **`unwarp`**: rasterises the flat mesh at output resolution and, for every output pixel,
  interpolates its triangle's *image* positions barycentrically. That gives `map_x` and `map_y`:
  where in the photo each flat-page pixel comes from. One `cv2.remap` builds the flat page. The
  same maps are the **correspondence map** Phase 6 will use to draw tamper evidence back onto
  the original photo.

### 3.6 `findoc/recon3d/stages.py`, `pipeline.py`, `preprocess/stages.py`, `types.py`, `config.py`

- `recon3d_stage` (video only): chooses the views (best frame first, as reference), passes the
  calibrated focal length if set, stores the result in `doc.recon3d` (including the frame
  numbers used) and, on success, `doc.flat`, `doc.flattening = "3d"`, `doc.flat_map`,
  `doc.recon`. It adds a readable message either way.
- `findoc/pipeline.py`: registers `recon3d` between `detect` and `rectify`.
- `rectify_stage`: returns early when the page was already flattened from 3D.
- `Document`: new fields `flattening`, `recon3d`, `flat_map` and `recon`. The last two are
  excluded from the JSON, so the JSON contract only gains `flattening` and `recon3d`.
- `Settings`: `recon3d = "auto" | "off"` and `focal_px`.

### 3.7 `findoc/recon3d/calibrate.py` and `findoc calibrate`

Zhang's method: each view of a checkerboard gives a homography H = K [r1 r2 t]; because r1 and
r2 are orthonormal, every view gives two equations on K. `cv2.calibrateCamera` solves for K and
lens distortion and refines everything by reprojection error.

- `find_corners`: `findChessboardCorners` + `cornerSubPix`.
- `calibrate`: needs at least 3 views; returns focal length, field of view, RMS error.
- `load_images`: a folder of photos or every n-th frame of a video.
- `save_focal`: writes `FINDOC_FOCAL_PX` into `.env` (replacing an old value).

A phone's focal length in pixels is fixed for a given video resolution, so it is measured once.
It makes 3D about three times faster (no focal search) and removes one source of error.

### 3.8 `findoc/recon3d/export.py`

`write_ply_points` (coloured point cloud) and `write_ply_mesh` (the unrolled-from mesh in 3D)
write ASCII PLY files that MeshLab, Blender or CloudCompare open. `preview` draws the page
points on the best frame, coloured by height. `save_recon` writes all of them plus the flat map.

### 3.9 Results (`experiments/phase5_recon_eval.py`)

Fifteen synthetic hand-held videos (4 s, 1280x720, three per bend, with a deliberate camera
jerk), each processed twice: `FINDOC_RECON3D=off` (2D) and `auto` (3D). **Unwarping error**: SIFT
matches between the flattened page and the true flat cheque, fitted with the best homography,
then the median remaining displacement (% of cheque width). A homography is allowed because any
scale or perspective left over is harmless; what matters is the *non-linear* distortion a
homography cannot fix.

Focal length estimated from the video (`experiments/results/phase5.json`):

| Bend | 3D used | Error 2D (median / p90) | Error 3D (median / p90) | Time 2D / 3D |
| --- | --- | --- | --- | --- |
| flat | 0% (correctly) | 0.244% / 0.768% | same | 2.1 / 2.9 s |
| wave | 100% | 0.397% / 1.370% | **0.270%** / 1.184% | 1.7 / 50 s |
| curl | 33% | 0.170% / 0.871% | **0.166%** / 0.832% | 2.1 / 18 s |
| curl2 | 100% | 0.376% / 1.241% | **0.258%** / 1.355% | 1.8 / 56 s |
| fold | 100% | 0.179% / 0.459% | **0.072%** / 0.327% | 1.9 / 52 s |

Focal length known, as after `findoc calibrate` (`experiments/results/phase5_calibrated.json`):

| Bend | Error 2D | Error 3D | Time 3D |
| --- | --- | --- | --- |
| wave | 0.397% | **0.277%** | 16 s |
| curl | 0.170% | **0.163%** | 9 s |
| curl2 | 0.376% | **0.276%** | 22 s |
| fold | 0.179% | **0.094%** | 19 s |

How to read this:

- **Fold gains most** (2-2.5x): a crease is exactly what a homography cannot model.
- **Two of the three curl clips were judged flat**: their curl is gentle, so 2D was already
  good and the planarity test correctly kept it. Template recognition was 100% in every case.
- **3D points are accurate to 0.2-0.5% of the page width** (column `acc3d` in the JSON).
- **Cost**: 3D adds 15-55 s per 4-second clip on a laptop CPU; most of it is the focal search,
  which calibration removes. The 90th percentile is slightly worse than 2D on curl2: the worst
  pixels are near the page border, where the surface is extrapolated.

#### How we got here: three lessons worth telling in a viva

1. **Check your metric before your method.** The first evaluation said the SfM points were off
   by 0.5-1% of the page width, and we spent effort making them better (dense KLT tracking
   through every frame, with blur and camera-jerk detection). The tracking worked (tracks within
   0.7 px of ground truth), but the 3D did not improve. The reason: the metric aligned the
   reconstruction to the truth with a plain least-squares similarity, and a few desk points
   that project inside the outline (matched to the page surface far above them) dragged the
   fit. With a **trimmed** fit the points were already 0.15-0.25% accurate. The dense-tracking
   code was removed again: complexity without benefit does not ship.
2. **Test each stage with perfect input.** Feeding the *exact* true surface into the flattening
   still gave errors worse than 2D on curl2 (0.6%), while unwarping with the exact texture
   coordinates gave 0.03-0.09%. So surface and resampling were fine and **ARAP** was the
   culprit: it was stopped after a fixed 25 iterations. Started from a projection onto the page
   plane, a strongly curled page needs around 100. Iterating until the vertices stop moving fixed
   it (curl2 from exact 3D: 0.60% -> 0.13%), and 3D went from losing to 2D to winning on every
   bend. `test_arap_converges_on_strong_curl` keeps it fixed.
3. **Know when 2D is exact.** For a flat page the homography is the correct model, and any
   3D reconstruction can only add noise. The planarity test runs first and is cheap.

### 3.10 Tests (`tests/test_recon3d.py`)

22 tests (some parametrised over bends), all on synthetic data with known answers:

- Surfaces are isometric: edge lengths on the 3D mesh equal those on the paper.
- `triangulate` recovers a point exactly from noise-free views; `build_tracks` merges chains
  and drops inconsistent ones.
- SfM on a rendered curled-page clip registers the views with reprojection error under 1 px
  and recovers the focal length from the images alone (within 20%).
- `fit_surface` tells a bent sheet from a flat one, and `cast_rays` hits the surface.
- ARAP recovers the true layout of a curled page; the convergence test above; broken input
  gives stretch = inf.
- `unwarp` reproduces a texture through a known mapping; `plane_residual` separates flat from
  bent pages.
- `flatten_document` on a curled-cheque video returns a 3D result and on a flat one keeps 2D.
- The stage is skipped for photos and with `FINDOC_RECON3D=off`.
- PLY files are valid; calibration recovers a known focal length from rendered checkerboards.

---

## 4. Concepts used in this phase

- **Pinhole camera, intrinsics K, pose (R, t)**: x ~ K (R X + t). The focal length f in pixels
  = image width / (2 tan(horizontal FOV / 2)).
- **Epipolar geometry**: two views of a point satisfy x2^T E x1 = 0 (calibrated coordinates),
  with E = [t]x R. Five correspondences determine E; RANSAC rejects mismatches.
- **Planar degeneracy**: if all points lie on a plane, the fundamental matrix is not unique,
  and the essential matrix gets fragile as the scene approaches a plane. A plane is described
  by a homography H = K (R + t n^T / d) K^-1, which `decomposeHomographyMat` splits into up
  to four (R, t, n) solutions.
- **Triangulation (DLT)**: each view gives two linear equations in the homogeneous point.
- **PnP**: camera pose from known 3D points and their 2D images (at least 4; RANSAC for outliers).
- **Bundle adjustment**: non-linear least squares over all cameras and points. The Jacobian is
  sparse (block structure), which is what makes it tractable.
- **Scale ambiguity**: images alone never fix the size of the world; SfM is defined up to a
  similarity transform. Flattening does not need scale: the output is resampled to the page's
  aspect ratio.
- **Developable surfaces and isometry**: paper can bend but not stretch, so lengths on the page
  are preserved. That is why "unroll with as little stretch as possible" (ARAP) is the right
  objective, and why a large `stretch` means the 3D is wrong.
- **Thin-plate spline**: the interpolant that minimises bending energy; with smoothing it becomes
  a regularised fit. **MAD** (median absolute deviation) gives a robust spread for outlier
  rejection.
- **Bisection**: once a sign change is bracketed, halving the interval always converges.
- **Cotangent Laplacian**: the standard discrete Laplacian on a triangle mesh; edge weights are
  the cotangents of the opposite angles.
- **Camera calibration (Zhang)**: homographies of a known plane constrain K.
- **Umeyama similarity and trimmed fitting**: the closed-form best rotation + scale +
  translation between two point sets. Fitting on the inliers of the previous fit makes it
  robust to gross outliers.

---

## 5. Try it yourself

1. Run `python experiments/phase5_recon_eval.py --seeds 1` and then with `--calibrated`.
   Compare the times. Where does the extra time go without calibration? (Hint:
   `doc.recon3d["sfm_s"]` and `Reconstruction.focal_search`.)
2. In `arap_flatten`, set `iters=25` and re-run the evaluation for `curl2` only. How much worse
   does it get? Then run `pytest -k strong_curl`.
3. Change `PLANE_P90_PX` in `pipeline.py` from 1.3 to 3.0. Which curl clips switch to 2D, and does
   the error change? What does that tell you about the threshold?
4. Process a synthetic fold video with `--save` and open `<name>_points.ply` and `_mesh.ply` in
   MeshLab (or https://3dviewer.net). Can you see the crease?
5. Print a checkerboard, film it with your phone at the resolution you use for cheques, and run
   `findoc calibrate board.mp4`. Compare the focal length with the phone's stated field of view.
6. In `fit_surface`, set `smoothing=1e-4`. Look at `acc3d` and the unwarping error. Explain the
   trade-off between following the data and following noise.

---

## 6. Viva and interview questions

**Q: Why not always use 3D?**
A flat page is exactly a homography, and the planarity test detects it cheaply. 3D adds
seconds and noise for no gain. It also needs camera motion: a perfectly still video cannot be
reconstructed.

**Q: Why the essential matrix and not the fundamental matrix?**
The scene is nearly planar. For a plane the fundamental matrix is degenerate (a family of
solutions fits). The 5-point essential-matrix method, with an assumed focal length, is not
degenerate for planes; bundle adjustment then refines the focal.

**Q: How do you get the focal length without calibration?**
Run SfM for several candidate focal lengths and keep the one with the lowest reprojection
error, then refine it in bundle adjustment. For a near-planar scene this is ill-conditioned
(focal length and depth trade off), so it is slower and less reliable than calibrating once.
In our tests the estimate was within 15%.

**Q: What does bundle adjustment minimise, and why is it fast enough?**
The sum of squared (Huber-robustified) reprojection errors over all observations, with respect
to all camera poses, all points and the focal. Each residual depends on one camera and one
point, so the Jacobian is very sparse; telling SciPy the sparsity pattern makes the solve scale
to thousands of points.

**Q: What is ARAP and why is it the right tool here?**
"As rigid as possible": find 2D positions for the mesh vertices so every triangle is as close
as possible to a rotated copy of its 3D shape. It alternates per-triangle best rotations (SVD)
with one sparse linear solve. Paper is developable, so a perfect unrolling with zero stretch
exists, and ARAP finds it up to reconstruction noise.

**Q: Your first ARAP had a bug that tests didn't catch. What was it and how did you find it?**
A fixed 25 iterations. The unit test used a mesh built in paper coordinates, which converges
quickly; the pipeline's mesh is a regular grid in the *image*, skewed in 3D, which converges
slowly. We found it by testing each stage with perfect input: exact 3D still flattened badly,
exact texture coordinates did not. Now it iterates to convergence, and a test uses an
image-grid mesh.

**Q: How do you know the 3D is wrong at run time, without ground truth?**
Several checks: enough page points; the reference camera must look at the page, not along it;
and the unrolled mesh must not stretch (paper cannot). Any failure falls back to 2D with a
recorded reason.

**Q: What is the correspondence map and what is it for?**
For each pixel of the flat page, the position in the original frame it came from (`map_x`,
`map_y`). Phase 6 finds tampering on the flat page and uses the map to draw the evidence on the
original photo, which is what a reviewer trusts.

**Q: Why remove the dense-tracking code you wrote?**
It made the tracks better but not the 3D, once the evaluation metric was fixed. Code that does
not improve a measured result is a maintenance cost; the finding is documented here instead.

**Q: What would you improve next?**
Fit the surface as a developable model directly (a ruled surface) instead of a generic smooth
height field; use the page border in the image as a constraint where 3D points are sparse;
and speed up the focal search, e.g. by estimating the focal from vanishing points of the
page edges.
