"""End-to-end GUI test for CPX_Vision, run against a COPY of the project.
Drives every window/flow programmatically with a fake camera, auto-answers
dialogs, and reports every exception raised inside Tk callbacks.

usage: python gui_test.py <project_copy_dir>
"""
import os, sys, time, json, glob, traceback, threading

PROJ = sys.argv[1]
os.chdir(PROJ)
sys.path.insert(0, PROJ)

# The SAM worker is a multiprocessing (spawn) child. Without __file__ on
# __main__, the child does not re-execute this whole test script.
import __main__
try:
    del __main__.__file__
except AttributeError:
    pass

import numpy as np, cv2

RESULTS = []   # (name, ok, detail)
TK_ERRORS = []
DIALOGS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


# ---------------------------------------------------------------- fake camera
class FakeCam:
    """Light background with a dark rotating rectangle that moves around."""
    def __init__(self, ip_style=False):
        self.t = 0
        self.open = True
        if ip_style:          # LatestFrameReader has only read()/release()
            self.grab = None
            del self.grab

    def _frame(self):
        self.t += 1
        img = np.full((480, 640, 3), 205, np.uint8)
        cx = 320 + int(120 * np.sin(self.t / 15))
        cy = 240 + int(60 * np.cos(self.t / 20))
        box = cv2.boxPoints(((cx, cy), (180, 110), (self.t * 3) % 180)).astype(np.int32)
        cv2.fillPoly(img, [box], (45, 50, 60))
        noise = np.random.default_rng(self.t).integers(-4, 5, img.shape)
        return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)

    def grab(self):
        return self.open

    def retrieve(self):
        return (True, self._frame()) if self.open else (False, None)

    def read(self):
        return self.retrieve()

    def set(self, *a):
        return True

    def isOpened(self):
        return self.open

    def release(self):
        self.open = False


class FakeIPCam:
    def __init__(self):
        self._c = FakeCam()
    def read(self):
        return self._c.read()
    def release(self):
        self._c.release()


CAM_FACTORY = [FakeCam]

import tkinter as tk
from tkinter import messagebox, simpledialog

ANSWERS = {"askyesno": True, "askstring": "widget"}


def _dlg(kind):
    def f(*a, **k):
        DIALOGS.append((kind, a[0] if a else "", (a[1] if len(a) > 1 else "")[:120]))
        if kind == "askyesno":
            ans = ANSWERS["askyesno"]
            return ans(a) if callable(ans) else ans
        return None
    return f


for k in ("showinfo", "showwarning", "showerror", "askyesno"):
    setattr(messagebox, k, _dlg(k))
simpledialog.askstring = lambda *a, **k: (DIALOGS.append(("askstring", a[0], "")), ANSWERS["askstring"])[1]

import core.camera_utils as camu
fake_open = lambda: CAM_FACTORY[0]()
camu.open_camera = fake_open
camu.list_available_cameras = lambda max_index=4: [
    {"type": "local", "index": 0, "usable": True, "preview": FakeCam()._frame()},
    {"type": "local", "index": 1, "usable": False, "preview": None}]

import main as app_main
import gui.detect_window as dw, gui.record_window as rw, gui.manual_label_window as mlw
import gui.settings_window as sw, gui.training_menu as tm
for mod in (dw, rw, mlw):
    mod.open_camera = fake_open
sw.list_available_cameras = camu.list_available_cameras

from core import dataset_utils as du, cleanup_utils as cu, manual_capture_utils as mcu
from core import i18n

root = app_main.MainMenu()
root.report_callback_exception = lambda et, ev, tb: TK_ERRORS.append(
    "".join(traceback.format_exception(et, ev, tb)))


def pump(seconds=0.5):
    # Run the REAL mainloop for `seconds` (like the app does), so worker
    # threads can call widget.after() exactly as they do in production.
    root.after(int(seconds * 1000), root.quit)
    root.mainloop()


def errors_since(n):
    return TK_ERRORS[n:]


def step(name, fn):
    n = len(TK_ERRORS)
    try:
        fn()
        errs = errors_since(n)
        check(name, not errs, errs[0].strip().splitlines()[-1] if errs else "")
        for e in errs:
            print(e)
    except Exception as e:
        traceback.print_exc()
        check(name, False, f"{type(e).__name__}: {e}")


def windows(cls):
    return [w for w in root.winfo_children() if isinstance(w, cls)] + \
           [w for t in root.winfo_children() for w in t.winfo_children() if isinstance(w, cls)]


# ================================================================ main menu
def t_main():
    pump(0.3)
    root.open_detect(); pump(1.5)
    root.open_training(); pump(0.6)
    root.open_settings(); pump(1.0)
    root.open_detect(); pump(0.2)  # second click focuses, no duplicate
    check("main: one detect window only", len(windows(dw.DetectWindow)) == 1)
step("main menu opens Detect / Training / Settings", t_main)


def t_lang_theme():
    tb = [w for w in root.winfo_children() if w.__class__.__name__ == "TopBar"][0]
    tb._select_language("bg"); pump(0.5)
    check("i18n: title in Bulgarian on detect window", windows(dw.DetectWindow)[0].title() == i18n.t("detect_window_title"))
    tb._select_language("en"); pump(0.3)
    tb._toggle_theme(); pump(0.3); tb._toggle_theme(); pump(0.3)
step("language switch EN->BG->EN and theme toggle", t_lang_theme)


def t_settings():
    s = windows(sw.SettingsWindow)[0]
    s.select_local_camera(0); pump(0.2)
    check("settings: camera 0 saved", json.load(open("config.json"))["camera_source"]["value"] == 0)
    s.use_auto(); pump(0.2)
    check("settings: auto saved", json.load(open("config.json"))["camera_source"]["value"] is None)
    s.on_close(); pump(0.3)
step("settings: select camera / auto", t_settings)


# ================================================================ detect window
def t_detect_state_machine():
    d = windows(dw.DetectWindow)[0]
    check("detect: model loaded", d.detector.loaded, d.detector.model_path)
    frame = np.zeros((480, 640, 3), np.uint8)
    poly = np.array([[100, 100], [300, 100], [300, 300], [100, 300]], np.float32)
    det = [("CPX-AP-I-PN-M12", 0.9, (100, 100, 300, 300), poly)]
    d._reset_lock()
    r = [d._process_detections(det, frame) for _ in range(3)]
    check("detect: locks after 3 confident frames", r[0] is None and r[1] is None and r[2] is not None and d.state == "LOCKED")
    moved = [("CPX-AP-I-PN-M12", 0.3, (110, 105, 310, 305), poly)]
    check("detect: keeps tracking at low conf", d._process_detections(moved, frame) is not None)
    d.last_seen_time -= 1.0
    d._process_detections([], frame)
    check("detect: unlocks after timeout", d.state == "SEARCHING")
    pump(1.0)  # live loop with the real model on fake frames
    d.on_close(); pump(0.5)
step("detect: lock / track / lose state machine + live loop", t_detect_state_machine)


# ================================================================ record window (new product, manual mode)
menu = None
def t_record_new():
    global menu
    menu = windows(tm.TrainingMenu)[0]
    menu.train_new(); pump(1.0)
    r = windows(rw.RecordWindow)[-1]
    pump(0.5)
    check("record: record button + session list visible on this screen",
          r.record_btn.winfo_ismapped() and r.session_listbox.winfo_ismapped(),
          f"window h={r.winfo_height()} screen={r.winfo_screenheight()}")
    check("record: new product starts in MANUAL mode", r.use_ai_mode.get() is False)
    check("record: new product margin defaults to 0", r.margin_var.get() == 0, r.margin_var.get())
    r.invert_var.set(True)                       # dark object on light background
    frame = FakeCam()._frame()
    _, poly, bbox = r.compute_mask_and_bbox_contour(frame)
    check("record: manual contour finds the object", poly is not None and len(poly) >= 4, None if poly is None else len(poly))
    r.toggle_record(); pump(2.0); r.toggle_record(); pump(0.3)
    sess = du.list_sessions("widget")
    imgs = glob.glob(os.path.join(du.DATA_DIR, "widget", sess[-1], "images", "*.jpg")) if sess else []
    lbls = glob.glob(os.path.join(du.DATA_DIR, "widget", sess[-1], "labels", "*.txt")) if sess else []
    check("record: frames + labels saved", len(imgs) > 10 and len(imgs) == len(lbls), f"{len(imgs)} imgs / {len(lbls)} labels")
    if lbls:
        parts = open(lbls[0]).read().split()
        vals = list(map(float, parts[1:]))
        check("record: label is valid YOLO polygon", parts[0] == "0" and len(vals) % 2 == 0 and all(0 <= v <= 1 for v in vals))
    # color tools
    r.open_color_picker(); pump(0.3)
    r.exclude_colors.append([0, 0, 200]); r._update_color_count_label()
    r.open_color_manager(); pump(0.3)
    r.clear_colors(); r.reset_to_default(); pump(0.2)
    # switching to AI mode for an unknown product asks first
    ANSWERS["askyesno"] = False
    r.use_ai_mode.set(True); r._on_mode_toggle(); pump(0.2)
    check("record: AI mode for unknown product is refused when user says No", r.use_ai_mode.get() is False)
    ANSWERS["askyesno"] = True
    r.on_close(); pump(0.5)
    check("record: camera loop stopped after close", r._closed)
step("record (new product, manual contour mode)", t_record_new)


def t_record_ip():
    CAM_FACTORY[0] = FakeIPCam
    r = rw.RecordWindow(root, "widget"); pump(1.0)
    r.open_color_picker(); pump(0.3)
    r.on_close(); pump(0.3)
    CAM_FACTORY[0] = FakeCam
step("record with IP-style camera (no grab/retrieve)", t_record_ip)


def t_record_known_ai():
    r = rw.RecordWindow(root, "CPX-AP-I-PN-M12"); pump(1.0)
    check("record: known product -> AI mode allowed", r.model_knows_product)
    r.on_close(); pump(0.3)
step("record window for a product the model knows", t_record_known_ai)


# ================================================================ manual record + review (SAM)
def t_manual():
    ANSWERS["askyesno"] = False   # "model doesn't know product" -> No = skip auto-label
    m = mlw.ManualRecordWindow(root, "gadget"); pump(0.5)
    m.toggle_recording(); pump(1.0); m.toggle_recording(); pump(0.2)
    check("manual: raw frames captured", m.frame_count > 5, m.frame_count)
    m.run_autolabel_and_review(); pump(1.0)
    ANSWERS["askyesno"] = True
    rev = windows(mlw.ManualReviewWindow)
    check("manual: review window opened", len(rev) == 1)
    if not rev:
        return
    rev = rev[0]
    rev.jump(1); rev.jump(-1); rev.on_scrub("2"); rev.reset_zoom(); pump(0.2)
    # SAM click on the object center (frame 2)
    img = rev.current_frame_img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray < 100)
    fx, fy = float(xs.mean()), float(ys.mean())
    rev._add_point_and_refine(fx, fy, 1)
    t0 = time.time()
    while getattr(rev, "_sam_busy", False) and time.time() - t0 < 240:
        pump(0.5)
    poly = rev._current_polygon()
    check("manual: SAM click produced a mask", poly is not None and len(poly) >= 3,
          "SAM: " + str(getattr(rev.sam_helper, "error", None)))
    if poly is not None:
        mask = np.zeros(gray.shape, np.uint8); cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
        obj = (gray < 100).astype(np.uint8)
        iou = (mask & obj).sum() / max(1, (mask | obj).sum())
        check("manual: SAM mask matches object (IoU>0.7)", iou > 0.7, f"IoU={iou:.2f}")
    rev.clear_points(); pump(0.2)
    rev.remove_current_frame(); pump(0.2)
    rev._add_point_and_refine(fx, fy, 1)
    while getattr(rev, "_sam_busy", False):
        pump(0.5)
    sam_proc = rev.sam_helper._process if rev.sam_helper else None
    rev.commit_batch(); pump(0.5)
    check("manual: SAM process stopped after commit", sam_proc is None or not sam_proc.is_alive())
    sess = du.list_sessions("gadget")
    check("manual: committed session exists", len(sess) == 1 and
          len(os.listdir(os.path.join(du.DATA_DIR, "gadget", sess[0], "labels"))) >= 1)
step("manual record -> skip auto-label -> SAM review -> commit", t_manual)


def t_resume_batch():
    b = mcu.new_temp_batch("gadget")
    b2 = mcu.new_temp_batch("gadget")
    check("manual: same-second batches get distinct folders", b != b2)
    cv2.imwrite(os.path.join(b, "images", "00000.jpg"), FakeCam()._frame())
    menu.resume_batch_flow(); pump(0.3)
    rv = mlw.ManualReviewWindow(root, b, "gadget"); pump(0.3)
    rv.discard_batch(); pump(0.2)
    check("manual: discard removes batch", not os.path.exists(b))
    mcu.discard_batch(b2)
step("resume / discard pending batch", t_resume_batch)


def t_auto_label_known():
    # Known product: auto-label path runs in a worker thread
    m = mlw.ManualRecordWindow(root, "CPX-AP-I-PN-M12"); pump(0.3)
    m.toggle_recording(); pump(0.5); m.toggle_recording()
    m.run_autolabel_and_review()
    t0 = time.time()
    while not windows(mlw.ManualReviewWindow) and time.time() - t0 < 60:
        pump(0.2)
    rv = windows(mlw.ManualReviewWindow)
    check("manual: threaded auto-label finished and opened review", len(rv) == 1)
    for w in rv:
        w.discard_batch()
    pump(0.3)
step("manual record with auto-label (known product, threaded)", t_auto_label_known)


# ================================================================ cleaning tools
def t_distribution():
    menu.distribution_flow(); pump(0.8)
    d = [w for w in root.winfo_children() if w.__class__.__name__ == "DistributionWindow"]
    check("distribution: window drawn", d and d[0].entries_sorted, [e["label"] for e in d[0].entries_sorted] if d else None)
    if d:
        d[0].record_more_data(); pump(0.3); d[0].destroy()
step("distribution chart", t_distribution)


def t_prune():
    sess = du.list_sessions("widget")[-1]
    path = os.path.join(du.DATA_DIR, "widget", sess)
    before = len(os.listdir(os.path.join(path, "images")))
    # duplicate a frame to guarantee near-identical neighbours
    src = sorted(os.listdir(os.path.join(path, "images")))[0]
    import shutil
    for i in range(3):
        shutil.copy(os.path.join(path, "images", src), os.path.join(path, "images", f"{src[:-4]}_dup{i}.jpg"))
        shutil.copy(os.path.join(path, "labels", src[:-4] + ".txt"), os.path.join(path, "labels", f"{src[:-4]}_dup{i}.txt"))
    menu.prune_session_flow(); pump(0.2)
    win = tk.Toplevel(root)
    menu._run_prune(win, "widget", sess, mode="dedupe", similarity_threshold=3.0); pump(0.2)
    after = len(os.listdir(os.path.join(path, "images")))
    check("prune: dedupe removed the duplicates", after <= before, f"{before}+3 -> {after}")
    check("prune: images/labels still paired", after == len(os.listdir(os.path.join(path, "labels"))))
    menu._pick_prune_mode_all(du.list_products()); pump(0.2)
step("prune (dedupe)", t_prune)


def t_viewers():
    n = len(TK_ERRORS)
    menu._open_viewer("all", None, None, delete_enabled=False); pump(0.5)
    v = [w for w in root.winfo_children() if w.__class__.__name__ == "CleanupViewer"][-1]
    check("view-only: no flag/delete controls", v.flag_btn.master.winfo_manager() == "")
    check("view-only: window fits on screen",
          v.winfo_height() <= v.winfo_screenheight() - 40, f"{v.winfo_height()} vs screen {v.winfo_screenheight()}")
    v.toggle_play(); pump(0.6); v.toggle_play(); v.jump_next_session(); v.jump_prev_session(); v.on_close()
    menu._open_viewer("product", "widget", None, delete_enabled=True); pump(0.5)
    v = [w for w in root.winfo_children() if w.__class__.__name__ == "CleanupViewer"][-1]
    check("review&remove: flag controls visible", v.flag_btn.winfo_ismapped())
    check("review&remove: window fits on screen",
          v.winfo_height() <= v.winfo_screenheight() - 40, f"{v.winfo_height()} vs screen {v.winfo_screenheight()}")
    victim = v.frames[0]["image_path"]
    v.toggle_current_flag(); v.delete_flagged(); pump(0.3)
    check("review&remove: flagged frame deleted", not os.path.exists(victim))
    v.on_close(); pump(0.2)
step("training-data viewer (view-only and review&remove)", t_viewers)


def t_relabel():
    menu._confirm_and_relabel("product", "CPX-AP-I-PN-M12", None)  # no data -> info dialog
    menu._confirm_and_relabel("product", "widget", None)
    t0 = time.time()
    while not cu.has_pending_staging() and time.time() - t0 < 60:
        pump(0.3)
    pump(0.5)
    flags = cu.load_flags()
    check("relabel: ran over widget frames (unknown object -> flagged, nothing staged)",
          len(flags) > 0 or cu.has_pending_staging(), f"flagged={len(flags)} staged={len(cu.load_staging_manifest())}")
    cu.save_flags(set())
    menu._open_staged_review_entry(); pump(0.3)
    for v in [w for w in root.winfo_children() if w.__class__.__name__ == "CleanupViewer"]:
        if v.review_mode:
            v.discard_staging()
    cu.discard_staged_relabels()
step("relabel with newest model -> staged review", t_relabel)


# ================================================================ training
def t_training():
    pj = json.load(open("project.json"))
    pj["training"].update({"epochs": 3, "patience": 3, "batch": 4})
    pj["twin_groups"] = []
    json.dump(pj, open("project.json", "w"), indent=2)
    models_before = set(os.listdir("models"))
    menu.run_training(); pump(0.5)
    menu.run_training(); pump(0.3)   # second click must be refused
    check("training: second start refused", any(d[0] == "showwarning" for d in DIALOGS[-3:]))
    t0 = time.time()
    while menu._training_thread.is_alive() and time.time() - t0 < 900:
        pump(1.0)
    new = set(os.listdir("models")) - models_before
    check("training: new model version saved", len(new) == 1, new)
    check("training: results.csv + confusion matrix written",
          os.path.exists("runs/cpx_detector/results.csv") and
          os.path.exists("runs/cpx_detector/confusion_matrix.png"))
    yaml_txt = open("yolo_dataset/data.yaml", encoding="utf-8").read()
    check("training: only classes with data", '"widget"' in yaml_txt and "nc: 2" in yaml_txt, yaml_txt.splitlines()[-2:])
    from core.infer import CPXDetector
    d = CPXDetector()
    check("training: Detect now uses new model with new classes",
          sorted(d.class_names()) == ["gadget", "widget"], d.class_names())
step("run training (3 epochs) -> new model -> detector picks it up", t_training)


# ================================================================ delete flows
def t_delete():
    sess = du.list_sessions("widget")[0]
    menu._confirm_delete_session("widget", sess); pump(0.2)
    check("delete: session removed", sess not in du.list_sessions("widget"))
    menu._confirm_delete_product("gadget"); pump(0.2)
    check("delete: product removed", "gadget" not in du.list_products())
step("delete session / product", t_delete)


def t_bad_name():
    ANSWERS["askstring"] = "bad/name"
    menu.train_new(); pump(0.3)
    check("new product: invalid name rejected", not os.path.exists(os.path.join(du.DATA_DIR, "bad")) and
          any(d[0] == "showerror" for d in DIALOGS[-2:]))
    ANSWERS["askstring"] = "Кутия"   # Cyrillic name must work
    menu.train_new(); pump(0.8)
    for r in windows(rw.RecordWindow):
        r.on_close()
    check("new product: Cyrillic name accepted", "Кутия" in du.list_products())
step("new product name validation", t_bad_name)


pump(0.5)
root.destroy()
print("\n================ SUMMARY ================")
fails = [r for r in RESULTS if not r[1]]
print(f"{len(RESULTS) - len(fails)}/{len(RESULTS)} checks passed")
for name, ok, detail in fails:
    print("FAIL:", name, detail)
print("dialogs shown:", len(DIALOGS))
for d in DIALOGS:
    if d[0] in ("showerror", "showwarning"):
        print("  ", d)
