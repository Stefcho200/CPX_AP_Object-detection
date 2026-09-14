import os
import threading
import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
from core import dataset_utils as du
from core import cleanup_utils as cu
from core.infer import CPXDetector
from gui.record_window import RecordWindow
from gui.cleanup_viewer import CleanupViewer
from gui.manual_label_window import ManualRecordWindow, ManualReviewWindow
from gui.distribution_window import DistributionWindow
from core.i18n import t, tf, on_language_change
import theme
from gui.animation import DropdownAnimationMixin

class TrainingMenu(tk.Toplevel, DropdownAnimationMixin):
    def __init__(self, master, anchor_widget=None):
        super().__init__(master)

        self.withdraw()

        self.title(t("training_window_title"))
        theme.style_window(self)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self._training_thread = None

        self.title_label = tk.Label(self, text=t("training_title"), bg=theme.BG_DARK, fg=theme.ACCENT_ORANGE,
                 font=theme.FONT_TITLE)
        self.title_label.pack(pady=15)

        self._top_buttons = []
        top_level_options = [
            ("train_model_on_product", self.open_train_submenu),
            ("training_data_cleaning", self.open_cleaning_submenu),
            ("run_training", self.run_training),
        ]

        for key, cmd in top_level_options:
            b = tk.Button(self, text=t(key), command=cmd)
            theme.style_button(b)
            b.pack(pady=6, padx=10, fill="x")
            self._top_buttons.append((key, b))


        self._unsub_lang = on_language_change(self._on_language_changed)

        self.animate_dropdown_open(width=420, anchor_widget=anchor_widget)

    def _on_language_changed(self, lang_code):
        self.title(t("training_window_title"))
        self.title_label.configure(text=t("training_title"))
        for key, btn in self._top_buttons:
            btn.configure(text=t(key))

    # ------------------------------------------------------------------
    # Submenu: Train Model on Product
    # ------------------------------------------------------------------

    def open_train_submenu(self):
        win = tk.Toplevel(self)
        win.title(t("train_model_on_product"))
        theme.style_window(win)
        win.minsize(520, 0)

        tk.Label(win, text=t("train_submenu_title"), bg=theme.BG_DARK,
                 fg=theme.ACCENT_ORANGE, font=theme.FONT_LABEL).pack(pady=10)

        options = [
            ("train_new_product", lambda: (win.destroy(), self.train_new())),
            ("upgrade_existing", lambda: (win.destroy(), self.upgrade_existing())),
            ("record_manual_label", lambda: (win.destroy(), self.manual_label_flow())),
            ("resume_batch", lambda: (win.destroy(), self.resume_batch_flow())),
            ("relabel_session", lambda: (win.destroy(), self.retag_session())),
            ("delete_data", lambda: (win.destroy(), self.open_delete_submenu())),
        ]

        for key, cmd in options:
            b = tk.Button(win, text=t(key), command=cmd)
            theme.style_button(b)
            b.pack(pady=6, padx=10, fill="x")


        close_btn = tk.Button(win, text=t("close"), command=win.destroy)
        theme.style_button(close_btn)
        close_btn.pack(pady=10)

    def resume_batch_flow(self):
        from core import manual_capture_utils as mcu
        from gui.manual_label_window import ManualReviewWindow

        pending = mcu.list_pending_batches()
        if not pending:
            messagebox.showinfo(t("nothing_pending_title"), t("nothing_pending_batches"))
            return

        win = tk.Toplevel(self)
        win.title(t("resume_batch_title"))
        theme.style_window(win)

        tk.Label(win, text=t("resume_batch_desc"),
                 bg=theme.BG_DARK, fg=theme.CREAM, font=theme.FONT_LABEL).pack(pady=10)

        for product, batch_path, count in pending:
            label = f"{product}  —  {os.path.basename(batch_path)}  ({count})"
            b = tk.Button(
                win, text=label,
                command=lambda p=product, bp=batch_path: (
                    win.destroy(), ManualReviewWindow(self.master, bp, p))
            )
            theme.style_button(b)
            b.pack(fill="x", padx=10, pady=3)

        close_btn = tk.Button(win, text=t("close"), command=win.destroy)
        theme.style_button(close_btn)
        close_btn.pack(pady=10)

    def open_delete_submenu(self):
        win = tk.Toplevel(self)
        win.title(t("delete_data"))
        theme.style_window(win)
        win.minsize(520, 0)

        tk.Label(win, text=t("delete_submenu_title"), bg=theme.BG_DARK,
                 fg=theme.ACCENT_ORANGE, font=theme.FONT_LABEL).pack(pady=10)

        options = [
            ("delete_session", lambda: (win.destroy(), self.delete_session_flow())),
            ("delete_product", lambda: (win.destroy(), self.delete_product_flow())),
        ]

        for key, cmd in options:
            b = tk.Button(win, text=t(key), command=cmd)
            theme.style_button(b)
            b.pack(pady=6, padx=10, fill="x")


        close_btn = tk.Button(win, text=t("close"), command=win.destroy)
        theme.style_button(close_btn)
        close_btn.pack(pady=10)

    # ------------------------------------------------------------------
    # Submenu: Training Data Cleaning
    # ------------------------------------------------------------------

    def open_cleaning_submenu(self):
        win = tk.Toplevel(self)
        win.title(t("training_data_cleaning"))
        theme.style_window(win)
        win.minsize(520, 0)

        tk.Label(win, text=t("cleaning_submenu_title"), bg=theme.BG_DARK,
                 fg=theme.ACCENT_ORANGE, font=theme.FONT_LABEL).pack(pady=10)

        options = [
            ("distribution", lambda: (win.destroy(), self.distribution_flow())),
            ("prune_frames", lambda: (win.destroy(), self.prune_session_flow())),
            ("relabel_newest_model", lambda: (win.destroy(), self.relabel_flow())),
            ("review_remove", lambda: (win.destroy(), self.review_and_remove_flow())),
            ("view_training_data", lambda: (win.destroy(), self.view_data_flow())),
        ]

        for key, cmd in options:
            b = tk.Button(win, text=t(key), command=cmd)
            theme.style_button(b)
            b.pack(pady=6, padx=10, fill="x")


        if cu.has_pending_staging():
            tk.Frame(win, bg=theme.CREAM, height=1).pack(fill="x", padx=10, pady=6)
            review_btn = tk.Button(
                win, text=t("review_staged"),
                command=lambda: (win.destroy(), self._open_staged_review_entry())
            )
            theme.style_button(review_btn)
            review_btn.pack(pady=6, padx=10)

        close_btn = tk.Button(win, text=t("close"), command=win.destroy)
        theme.style_button(close_btn)
        close_btn.pack(pady=10)

    # ------------------------------------------------------------------
    # Recording flows
    # ------------------------------------------------------------------

    def train_new(self):
        name = simpledialog.askstring(t("new_product_prompt_title"), t("new_product_prompt_text"))
        if not name:
            return
        du.create_product(name)
        RecordWindow(self.master, name)

    def distribution_flow(self):
        products = du.list_products()
        if not products:
            messagebox.showinfo(t("no_data_title"), t("no_data_products"))
            return
        DistributionWindow(self.master)

    def upgrade_existing(self):
        products = du.list_products()
        if not products:
            messagebox.showinfo(t("no_data_title"), t("no_data_products"))
            return
        self._pick_from_list(t("select_product_upgrade"), products,
                              lambda p: RecordWindow(self.master, p))

    def manual_label_flow(self):
        products = du.list_products()
        if not products:
            messagebox.showinfo(t("no_data_title"), t("no_data_products"))
            return
        self._pick_from_list(t("select_product_record"), products,
                              lambda p: ManualRecordWindow(self.master, p))

    # ------------------------------------------------------------------
    # Re-tagging ("Relabel a recorded session")
    # ------------------------------------------------------------------

    def retag_session(self):
        products = du.list_products()
        if not products:
            messagebox.showinfo(t("no_data_title"), t("no_data_products"))
            return

        def pick_product(prod):
            sessions = du.list_sessions(prod)
            if not sessions:
                messagebox.showinfo(t("empty_title"), t("no_data_sessions"))
                return
            self._pick_from_list(t("select_session"), sessions,
                                  lambda s: self._pick_new_tag(prod, s, products))

        self._pick_from_list(t("select_product_current"), products, pick_product)

    def _pick_new_tag(self, prod, session, products):
        session_path = os.path.join(du.DATA_DIR, prod, session)

        def apply_new_tag(new_p):
            du.retag_session(session_path, new_p)
            messagebox.showinfo(t("retag_done_title"), tf("retag_done_text", product=new_p))

        self._pick_from_list(tf("retag_to_prompt", session=session), products, apply_new_tag)

    # ------------------------------------------------------------------
    # Pruning redundant frames
    # ------------------------------------------------------------------

    def prune_session_flow(self):
        products = du.list_products()
        if not products:
            messagebox.showinfo(t("no_data_title"), t("no_data_products"))
            return

        win = tk.Toplevel(self)
        win.title(t("select_scope_prune"))
        theme.style_window(win)
        tk.Label(win, text=t("select_scope_prune"), bg=theme.BG_DARK, fg=theme.CREAM,
                 font=theme.FONT_LABEL).pack(pady=5)

        all_btn = tk.Button(
            win, text=t("all_products_sessions"),
            command=lambda: (win.destroy(), self._pick_prune_mode_all(products))
        )
        theme.style_button(all_btn)
        all_btn.pack(fill="x", padx=10, pady=(10, 4))

        tk.Frame(win, bg=theme.CREAM, height=1).pack(fill="x", padx=10, pady=6)

        for prod in products:
            b = tk.Button(
                win, text=prod,
                command=lambda p=prod: (win.destroy(), self._pick_product_sessions(p))
            )
            theme.style_button(b)
            b.pack(fill="x", padx=10, pady=3)

    def _pick_product_sessions(self, prod):
        sessions = du.list_sessions(prod)
        if not sessions:
            messagebox.showinfo(t("empty_title"), t("no_data_sessions"))
            return
        self._pick_from_list(t("select_session_prune"), sessions,
                              lambda s: self._pick_prune_mode(prod, s))

    def _pick_prune_mode(self, prod, session):
        win = tk.Toplevel(self)
        win.title(tf("pruning_title", session=session))
        theme.style_window(win)

        tk.Label(win, text=tf("pruning_title", session=session), bg=theme.BG_DARK,
                 fg=theme.CREAM, font=theme.FONT_LABEL).pack(pady=10)

        tk.Label(win, text=t("similarity_dedupe_desc"),
                 bg=theme.BG_DARK, fg=theme.ACCENT_YELLOW, font=("Consolas", 9),
                 justify="left").pack(pady=(0, 5))

        thresh_var = tk.DoubleVar(value=3.0)
        thresh_frame = tk.Frame(win, bg=theme.BG_DARK)
        thresh_frame.pack()
        tk.Label(thresh_frame, text=t("similarity_threshold_label"), bg=theme.BG_DARK,
                 fg=theme.CREAM, font=theme.FONT_LABEL).pack(side="left", padx=5)
        tk.Entry(thresh_frame, textvariable=thresh_var, width=6).pack(side="left")

        dedupe_btn = tk.Button(
            win, text=t("run_similarity_dedupe"),
            command=lambda: self._run_prune(
                win, prod, session, mode="dedupe",
                similarity_threshold=thresh_var.get()))
        theme.style_button(dedupe_btn)
        dedupe_btn.pack(pady=8, fill="x", padx=20)

        tk.Label(win, text=t("decimate_desc"),
                 bg=theme.BG_DARK, fg=theme.ACCENT_YELLOW, font=("Consolas", 9),
                 justify="left").pack(pady=(5, 5))

        n_var = tk.IntVar(value=3)
        n_frame = tk.Frame(win, bg=theme.BG_DARK)
        n_frame.pack()
        tk.Label(n_frame, text=t("keep_every_nth"), bg=theme.BG_DARK,
                 fg=theme.CREAM, font=theme.FONT_LABEL).pack(side="left", padx=5)
        tk.Entry(n_frame, textvariable=n_var, width=5).pack(side="left")

        decimate_btn = tk.Button(
            win, text=t("run_decimation"),
            command=lambda: self._run_prune(
                win, prod, session, mode="decimate", decimate_n=n_var.get()))
        theme.style_button(decimate_btn)
        decimate_btn.pack(pady=10, fill="x", padx=20)

    def _run_prune(self, win, prod, session, mode, decimate_n=3, similarity_threshold=3.0):
        session_path = os.path.join(du.DATA_DIR, prod, session)
        confirm = messagebox.askyesno(
            t("confirm_pruning_title"),
            tf("confirm_pruning_text", session=session, mode=mode)
        )
        if not confirm:
            return

        kept, removed = du.prune_session(
            session_path, mode=mode, decimate_n=decimate_n,
            similarity_threshold=similarity_threshold
        )
        win.destroy()
        messagebox.showinfo(
            t("pruning_complete_title"),
            tf("pruning_complete_text", session=session, kept=kept, removed=removed)
        )

    def _pick_prune_mode_all(self, products):
        win = tk.Toplevel(self)
        win.title(t("prune_all_title"))
        theme.style_window(win)

        total_sessions = sum(len(du.list_sessions(p)) for p in products)

        tk.Label(win, text=tf("prune_all_summary", n=len(products), total=total_sessions),
                 bg=theme.BG_DARK, fg=theme.ACCENT_ORANGE, font=theme.FONT_LABEL).pack(pady=10)

        tk.Label(win, text=t("similarity_dedupe_desc"),
                 bg=theme.BG_DARK, fg=theme.ACCENT_YELLOW, font=("Consolas", 9),
                 justify="left").pack(pady=(0, 5))

        thresh_var = tk.DoubleVar(value=3.0)
        thresh_frame = tk.Frame(win, bg=theme.BG_DARK)
        thresh_frame.pack()
        tk.Label(thresh_frame, text=t("similarity_threshold_label"), bg=theme.BG_DARK,
                 fg=theme.CREAM, font=theme.FONT_LABEL).pack(side="left", padx=5)
        tk.Entry(thresh_frame, textvariable=thresh_var, width=6).pack(side="left")

        dedupe_btn = tk.Button(
            win, text=t("run_similarity_dedupe_all"),
            command=lambda: self._run_prune_all(
                win, products, mode="dedupe",
                similarity_threshold=thresh_var.get()))
        theme.style_button(dedupe_btn)
        dedupe_btn.pack(pady=8, fill="x", padx=20)

        tk.Label(win, text=t("decimate_desc"),
                 bg=theme.BG_DARK, fg=theme.ACCENT_YELLOW, font=("Consolas", 9),
                 justify="left").pack(pady=(5, 5))

        n_var = tk.IntVar(value=3)
        n_frame = tk.Frame(win, bg=theme.BG_DARK)
        n_frame.pack()
        tk.Label(n_frame, text=t("keep_every_nth"), bg=theme.BG_DARK,
                 fg=theme.CREAM, font=theme.FONT_LABEL).pack(side="left", padx=5)
        tk.Entry(n_frame, textvariable=n_var, width=5).pack(side="left")

        decimate_btn = tk.Button(
            win, text=t("run_decimation_all"),
            command=lambda: self._run_prune_all(
                win, products, mode="decimate", decimate_n=n_var.get()))
        theme.style_button(decimate_btn)
        decimate_btn.pack(pady=10, fill="x", padx=20)

    def _run_prune_all(self, win, products, mode, decimate_n=3, similarity_threshold=3.0):
        confirm = messagebox.askyesno(
            t("confirm_pruning_all_title"),
            tf("confirm_pruning_all_text", n=len(products), mode=mode)
        )
        if not confirm:
            return

        win.destroy()

        total_kept = 0
        total_removed = 0
        per_product_results = []

        for prod in products:
            kept, removed = du.prune_product(
                prod, mode=mode, decimate_n=decimate_n,
                similarity_threshold=similarity_threshold
            )
            total_kept += kept
            total_removed += removed
            per_product_results.append(f"  {prod}: kept {kept}, removed {removed}")

        summary = "\n".join(per_product_results)
        messagebox.showinfo(
            t("pruning_complete_title"),
            tf("pruning_complete_all_text", kept=total_kept, removed=total_removed, summary=summary)
        )

    # ------------------------------------------------------------------
    # Re-labeling with newest model (staged, not immediate)
    # ------------------------------------------------------------------

    def relabel_flow(self):
        products = du.list_products()
        if not products:
            messagebox.showinfo(t("no_data_title"), t("no_data_products"))
            return

        win = tk.Toplevel(self)
        win.title(t("relabel_window_title"))
        theme.style_window(win)

        tk.Label(win, text=t("select_scope_relabel"),
                 bg=theme.BG_DARK, fg=theme.CREAM, font=theme.FONT_LABEL).pack(pady=5)

        all_btn = tk.Button(
            win, text=t("all_products_sessions"),
            command=lambda: (win.destroy(), self._confirm_and_relabel("all", None, None))
        )
        theme.style_button(all_btn)
        all_btn.pack(fill="x", padx=10, pady=(10, 4))

        tk.Frame(win, bg=theme.CREAM, height=1).pack(fill="x", padx=10, pady=6)

        for prod in products:
            b = tk.Button(
                win, text=prod,
                command=lambda p=prod: (win.destroy(), self._pick_relabel_session(p))
            )
            theme.style_button(b)
            b.pack(fill="x", padx=10, pady=3)

    def _pick_relabel_session(self, product):
        sessions = du.list_sessions(product)
        if not sessions:
            messagebox.showinfo(t("empty_title"), t("no_data_sessions"))
            return

        win = tk.Toplevel(self)
        win.title(tf("relabel_session_title", product=product))
        theme.style_window(win)

        all_btn = tk.Button(
            win, text=tf("all_sessions_of", product=product),
            command=lambda: (win.destroy(), self._confirm_and_relabel("product", product, None))
        )
        theme.style_button(all_btn)
        all_btn.pack(fill="x", padx=10, pady=(10, 4))

        tk.Frame(win, bg=theme.CREAM, height=1).pack(fill="x", padx=10, pady=6)

        for sess in sessions:
            b = tk.Button(
                win, text=sess,
                command=lambda s=sess: (win.destroy(), self._confirm_and_relabel("session", product, s))
            )
            theme.style_button(b)
            b.pack(fill="x", padx=10, pady=3)

    def _confirm_and_relabel(self, scope, product, session):
        frames = cu.list_frames(scope=scope, product=product, session=session)
        if not frames:
            messagebox.showinfo(t("empty_title"), t("no_data_frames"))
            return

        confirm = messagebox.askyesno(
            t("confirm_relabel_title"),
            tf("confirm_relabel_text", n=len(frames))
        )
        if not confirm:
            return

        detector = CPXDetector(conf_threshold=0.5)
        if not detector.loaded:
            messagebox.showerror(t("no_model_title"), tf("no_model_text", error=detector.load_error))
            return

        win = tk.Toplevel(self)
        win.title(t("relabel_in_progress_title"))
        theme.style_window(win)
        win.geometry("420x190")
        win.resizable(False, False)

        status_label = tk.Label(win, text=t("starting"), bg=theme.BG_DARK,
                                 fg=theme.CREAM, font=theme.FONT_LABEL)
        status_label.pack(pady=15)

        progress = ttk.Progressbar(win, orient="horizontal", length=380, mode="determinate")
        progress.pack(pady=5)

        review_btn = tk.Button(win, text=t("review_staged_now"),
                                command=lambda: (win.destroy(), self._open_staged_review()),
                                state="disabled")
        theme.style_button(review_btn)
        review_btn.pack(pady=(10, 5))

        close_btn = tk.Button(win, text=t("close"), command=win.destroy, state="disabled")
        theme.style_button(close_btn)
        close_btn.pack(pady=5)

        flag_set = cu.load_flags()

        def on_progress(done, total):
            def update_ui():
                progress["value"] = (done / total) * 100
                status_label.configure(text=tf("processing_frames", done=done, total=total))
            win.after(0, update_ui)

        def worker():
            relabeled, flagged, manifest = cu.relabel_frames(
                frames, detector, conf_threshold=0.5,
                progress_callback=on_progress, flag_set=flag_set
            )

            def finish():
                status_label.configure(
                    text=tf("relabel_done_text", relabeled=relabeled, flagged=flagged))
                close_btn.configure(state="normal")
                review_btn.configure(state="normal")

            win.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def _open_staged_review(self):
        manifest = cu.load_staging_manifest()
        if not manifest:
            messagebox.showinfo(t("nothing_staged_title"), t("nothing_staged_text"))
            return
        CleanupViewer(self.master, manifest, title=t("review_staged"), review_mode=True)

    def _open_staged_review_entry(self):
        if not cu.has_pending_staging():
            messagebox.showinfo(t("nothing_pending_title"), t("nothing_staged_text"))
            return
        self._open_staged_review()

    # ------------------------------------------------------------------
    # Viewing / manual review of training data
    # ------------------------------------------------------------------

    def review_and_remove_flow(self):
        self._pick_viewer_scope(review_mode=False, delete_enabled=True)

    def view_data_flow(self):
        self._pick_viewer_scope(review_mode=False, delete_enabled=False)

    def _pick_viewer_scope(self, review_mode, delete_enabled):
        products = du.list_products()
        if not products:
            messagebox.showinfo(t("no_data_title"), t("no_data_products"))
            return

        win = tk.Toplevel(self)
        win.title(t("select_scope_view"))
        theme.style_window(win)

        tk.Label(win, text=t("select_scope_view"),
                 bg=theme.BG_DARK, fg=theme.CREAM, font=theme.FONT_LABEL).pack(pady=5)

        all_btn = tk.Button(
            win, text=t("all_products_sessions"),
            command=lambda: (win.destroy(), self._open_viewer("all", None, None, delete_enabled))
        )
        theme.style_button(all_btn)
        all_btn.pack(fill="x", padx=10, pady=(10, 4))

        tk.Frame(win, bg=theme.CREAM, height=1).pack(fill="x", padx=10, pady=6)

        for prod in products:
            b = tk.Button(
                win, text=prod,
                command=lambda p=prod: (win.destroy(), self._pick_view_session(p, delete_enabled))
            )
            theme.style_button(b)
            b.pack(fill="x", padx=10, pady=3)

    def _pick_view_session(self, product, delete_enabled):
        sessions = du.list_sessions(product)
        if not sessions:
            messagebox.showinfo(t("empty_title"), t("no_data_sessions"))
            return

        win = tk.Toplevel(self)
        win.title(tf("view_title", product=product))
        theme.style_window(win)

        all_btn = tk.Button(
            win, text=tf("all_sessions_of", product=product),
            command=lambda: (win.destroy(), self._open_viewer("product", product, None, delete_enabled))
        )
        theme.style_button(all_btn)
        all_btn.pack(fill="x", padx=10, pady=(10, 4))

        tk.Frame(win, bg=theme.CREAM, height=1).pack(fill="x", padx=10, pady=6)

        for sess in sessions:
            b = tk.Button(
                win, text=sess,
                command=lambda s=sess: (win.destroy(), self._open_viewer("session", product, s, delete_enabled))
            )
            theme.style_button(b)
            b.pack(fill="x", padx=10, pady=3)

    def _open_viewer(self, scope, product, session, delete_enabled):
        frames = cu.list_frames(scope=scope, product=product, session=session)
        scope_label = t("all_products_label") if scope == "all" else (
            product if scope == "product" else f"{product} / {session}")
        CleanupViewer(self.master, frames,
                      title=tf("viewer_all_title", scope=scope_label),
                      review_mode=False)

    # ------------------------------------------------------------------
    # Deletion flows
    # ------------------------------------------------------------------

    def delete_session_flow(self):
        products = du.list_products()
        if not products:
            messagebox.showinfo(t("no_data_title"), t("no_data_products"))
            return

        def pick_product(prod):
            sessions = du.list_sessions(prod)
            if not sessions:
                messagebox.showinfo(t("empty_title"), t("no_data_sessions"))
                return
            self._pick_from_list(t("delete_session"), sessions,
                                  lambda s: self._confirm_delete_session(prod, s))

        self._pick_from_list(t("select_product_current"), products, pick_product)

    def _confirm_delete_session(self, prod, session):
        confirm = messagebox.askyesno(
            t("confirm_deletion_title"),
            tf("confirm_delete_session_text", session=session)
        )
        if confirm:
            session_path = os.path.join(du.DATA_DIR, prod, session)
            du.delete_session(session_path)
            messagebox.showinfo(t("deleted_title"), tf("deleted_session_text", session=session))

    def delete_product_flow(self):
        products = du.list_products()
        if not products:
            messagebox.showinfo(t("no_data_title"), t("no_data_products"))
            return
        self._pick_from_list(t("select_product_delete"), products,
                              self._confirm_delete_product)

    def _confirm_delete_product(self, product):
        confirm = messagebox.askyesno(
            t("confirm_deletion_title"),
            tf("confirm_delete_product_text", product=product)
        )
        if confirm:
            du.delete_product(product)
            messagebox.showinfo(t("deleted_title"), tf("deleted_product_text", product=product))

    # ------------------------------------------------------------------
    # Shared picker dialog
    # ------------------------------------------------------------------

    def _pick_from_list(self, title, items, on_select):
        win = tk.Toplevel(self)
        win.title(title)
        theme.style_window(win)
        tk.Label(win, text=title, bg=theme.BG_DARK, fg=theme.CREAM,
                 font=theme.FONT_LABEL).pack(pady=5)

        for item in items:
            b = tk.Button(win, text=item,
                          command=lambda i=item: (win.destroy(), on_select(i)))
            theme.style_button(b)
            b.pack(fill="x", padx=10, pady=3)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def run_training(self):
        from core.train_model import train

        win = tk.Toplevel(self)
        win.title(t("training_in_progress_title"))
        theme.style_window(win)
        win.geometry("460x200")
        win.resizable(False, False)

        tk.Label(win, text=t("training_yolo"), bg=theme.BG_DARK,
                 fg=theme.ACCENT_ORANGE, font=theme.FONT_LABEL).pack(pady=10)

        style = ttk.Style()
        style.theme_use('default')
        style.configure(
            "Retro.Horizontal.TProgressbar",
            troughcolor=theme.BG_PANEL,
            background=theme.ACCENT_ORANGE,
            thickness=20
        )

        progress = ttk.Progressbar(
            win, style="Retro.Horizontal.TProgressbar",
            orient="horizontal", length=390, mode="determinate"
        )
        progress.pack(pady=10)

        status_label = tk.Label(win, text=t("starting"), bg=theme.BG_DARK,
                                 fg=theme.CREAM, font=theme.FONT_LABEL, justify="left")
        status_label.pack(pady=5)

        close_btn = tk.Button(win, text=t("close"), command=win.destroy, state="disabled")
        theme.style_button(close_btn)
        close_btn.pack(pady=5)

        def on_training_win_close():
            if self._training_thread is not None and self._training_thread.is_alive():
                messagebox.showwarning(
                    t("training_still_running_title"),
                    t("training_still_running_text")
                )
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_training_win_close)

        epochs_total = 1000

        def on_progress(epoch, total, box_loss, cls_loss, map50, error=None):
            def update_ui():
                if error:
                    status_label.configure(text=error, fg="red")
                    close_btn.configure(state="normal")
                    return

                progress["value"] = (epoch / total) * 100
                map_text = f"{map50*100:.1f}%" if map50 is not None else "n/a"
                status_label.configure(
                    text=tf("epoch_status", epoch=epoch, total=total,
                            box_loss=box_loss, cls_loss=cls_loss, map50=map_text)
                )

                if epoch == total:
                    status_label.configure(fg=theme.ACCENT_YELLOW)
                    close_btn.configure(state="normal")

            win.after(0, update_ui)

        def training_thread():
            train(epochs=epochs_total, progress_callback=on_progress)
            win.after(0, lambda: close_btn.configure(state="normal"))

        self._training_thread = threading.Thread(target=training_thread, daemon=True)
        self._training_thread.start()

    def on_close(self):
        self.destroy()
