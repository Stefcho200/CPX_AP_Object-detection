# CPX Vision – object recognizer (YOLOv8n-seg, Ultralytics)

Rebuilt from github.com/Stefcho200/CPX_AP_Object-detection with the original
`core/` + `gui/` layout restored.

## Start

Double-click `run.bat` (or `.venv\Scripts\python main.py`).

Main menu:
- **Detect Products**: live camera, locks onto the most confident object and
  draws its mask + label. Uses the newest `models/cpx_model_vN.pt`.
- **Training**: record data, clean it, train.
- **Settings**: pick the camera.

## Detect something else (the only steps you need)

Every folder in `data/` is one class. The model learns whatever folders exist.

1. **Edit `project.json`**
   - `app_title`: e.g. `"Tool Recognizer"`
   - `twin_groups`: set to `[]` unless some of your objects look identical
     and can only be told apart by printed text
   - `training.epochs`: 50–100 is enough for a demo
2. **Remove the old classes**: move/delete the CPX folders in `data/` (it
   starts empty on this PC).
3. **Record each object**: Training → Train Model on Product → *Train model
   on new product* → type the class name (e.g. `screwdriver`) → hold the
   object in front of the camera, rotate it, press Record.
   - The **"AI-assisted mode"** checkbox labels frames with the current
     model, so it only works once a model already knows the object. For a
     NEW object **untick it** and use the threshold/contour sliders (put the
     object on a plain background that contrasts with it), or use
     *Record & manually label data* (click on the object and SAM draws the
     mask; it downloads `mobile_sam.pt` into `models/` the first time).
   - After the first model is trained, AI-assisted mode makes recording
     more data for the same objects much faster.
   - Aim for 150–300 varied frames per object. Then Training Data Cleaning →
     *Prune* to remove duplicate frames.
4. **Train**: Training → *Run training*. The result is saved as the next
   `models/cpx_model_vN.pt`; Detect picks it up automatically.
5. **Demo**: Detect Products.

## Files

| Path | What |
|---|---|
| `project.json` | title, training settings, twin groups – **edit this** |
| `config.json` | camera choice (written by Settings) |
| `record_settings.json` | per-class threshold/contour settings for manual labelling |
| `data/<class>/session_*/images, labels` | recorded frames + YOLO polygon labels |
| `models/cpx_model_vN.pt` | trained models (highest N is used) |
| `models/yolov8n-seg.pt` | COCO-pretrained starting weights |
| `runs/cpx_detector/` | training output: results.csv, confusion_matrix.png, curves |
| `core/` | logic (dataset, training, inference, camera, OCR, SAM) |
| `gui/` | Tkinter windows |

> Restart the app after editing `project.json`; it is read at startup.

## Changes vs. the GitHub version

- Folder structure restored (GitHub had all files flattened).
- `project.json` + `core/project_config.py`: title, training settings and twin
  groups are configurable instead of hard-coded.
- Training: run folder is now absolute (`runs/cpx_detector`) – previously
  Ultralytics nested it under `runs/segment/...` and `best.pt` was not copied
  to `models/`. Epochs/patience/batch come from `project.json` (was a fixed
  1000 epochs). Falls back to COCO-pretrained `yolov8n-seg.pt` instead of
  random init. mAP in the progress window is now read after validation.
- `yolo_dataset/` is built next to the project, not in the current directory.
- Camera set to auto-detect (the original pointed at index 1 / an office IP cam).

Bugs fixed during the full debug pass (all modes tested end-to-end with a
simulated camera):
- Record + manual-record windows crashed with an IP camera (`cap.grab()` on
  the IP frame reader); all windows now share `camera_utils.read_frame()`.
- Closed camera windows kept their frame loop running in the background.
- New products inherited a 20 px mask margin (masks saved too large) – now 0.
- Record window: AI mode was forced on for products the model has never
  seen, and the saved AI/manual choice was ignored. Now: saved choice, else
  AI only if the model knows the product; asks before enabling AI otherwise.
- Manual record: auto-labeling ran on the UI thread (window froze); it now
  runs in the background, and for an unknown product you can skip straight
  to click-labeling with SAM.
- SAM helper process kept running after commit/discard; clicks outside the
  image are clamped.
- Two sessions/batches started within the same second overwrote each other.
- Dataset builder: twin groups listed in `project.json` became (empty)
  classes even with no data; empty product folders became phantom classes;
  a class with 1 frame got no training image; non-ASCII (Cyrillic) class
  names crashed `data.yaml` writing.
- "View training data" offered delete; it is read-only now.
- Viewer (760x680) and Record window (940x900) did not fit a 1080p screen at
  125% scaling – bottom buttons were cut off. Both now size to the screen.
- Training: could be started twice in parallel; the progress callback broke
  on Ultralytics 8.4 (`loss_items` is a dict now).
- Invalid product names (`/ : * ?` ...) are rejected.
- EasyOCR is only loaded when twin groups are configured.
