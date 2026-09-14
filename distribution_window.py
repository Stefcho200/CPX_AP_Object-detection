import tkinter as tk
from core import dataset_utils as du
from core import twin_groups as tg
from gui.record_window import RecordWindow
from core.i18n import t, tf, on_language_change
import theme


class DistributionWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title(t("training_data_title"))
        theme.style_window(self)
        self.geometry("720x560")
        self.minsize(720, 560)

        self.title_label = tk.Label(self, text=t("distribution_title"), bg=theme.BG_DARK,
                 fg=theme.ACCENT_ORANGE, font=theme.FONT_LABEL)
        self.title_label.pack(pady=10)

        self.legend_label = tk.Label(
            self, text=t("distribution_legend"),
            bg=theme.BG_DARK, fg=theme.CREAM, font=("Consolas", 9), wraplength=680)
        self.legend_label.pack(pady=(0, 5))

        self.canvas = tk.Canvas(self, bg=theme.BG_PANEL, highlightthickness=0)
        self.canvas.pack(padx=10, pady=10, fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", self._on_resize)

        self.info_label = tk.Label(self, text="", bg=theme.BG_DARK, fg=theme.CREAM,
                                    font=("Consolas", 9), justify="left")
        self.info_label.pack(pady=(0, 10))

        btn_row = tk.Frame(self, bg=theme.BG_DARK)
        btn_row.pack(pady=5)
        self.record_btn = tk.Button(btn_row, text=t("record_more_data"), command=self.record_more_data)
        theme.style_button(self.record_btn)
        self.record_btn.pack(side="left", padx=5)
        self.refresh_btn = tk.Button(btn_row, text=t("refresh"), command=self.draw_chart)
        theme.style_button(self.refresh_btn)
        self.refresh_btn.pack(side="left", padx=5)

        self.close_btn = tk.Button(self, text=t("close"), command=self.destroy)
        theme.style_button(self.close_btn)
        self.close_btn.pack(pady=10)

        self._unsub_lang = on_language_change(self._on_language_changed)

        self.entries_sorted = []
        self.expanded_keys = set()
        self.click_regions = []
        self.draw_chart()

    def _on_language_changed(self, lang_code):
        self.title(t("training_data_title"))
        self.title_label.configure(text=t("distribution_title"))
        self.legend_label.configure(text=t("distribution_legend"))
        self.record_btn.configure(text=t("record_more_data"))
        self.refresh_btn.configure(text=t("refresh"))
        self.close_btn.configure(text=t("close"))
        self.draw_chart()

    def _on_resize(self, event):
        if event.width > 50 and event.height > 50:
            self.draw_chart()

    def draw_chart(self):
        self.canvas.delete("all")
        self.click_regions = []

        counts = du.count_images_per_product()
        if not counts:
            self.canvas.create_text(340, 190, text=t("no_data_products"),
                                     fill=theme.CREAM, font=theme.FONT_LABEL)
            self.info_label.configure(text="")
            self.entries_sorted = []
            return

        entries = tg.build_display_entries(counts)
        entries.sort(key=lambda e: e["count"])
        self.entries_sorted = entries

        rows = []
        for entry in entries:
            rows.append(("main", entry))
            if entry["expandable"] and entry["key"] in self.expanded_keys:
                for member_name, member_count, member_color in entry["members"]:
                    rows.append(("member", {
                        "label": member_name.replace("CPX-AP-I-", ""),
                        "count": member_count,
                        "color": member_color,
                        "parent_key": entry["key"],
                    }))

        max_count = max(r[1]["count"] for r in rows) or 1

        margin_left = 190
        margin_top = 15
        margin_bottom = 20
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 50 or canvas_h < 50:
            return
        chart_w = canvas_w - margin_left - 60
        chart_h = canvas_h - margin_top - margin_bottom
        row_h = chart_h / max(1, len(rows))
        row_h = min(row_h, 34)

        y_cursor = margin_top
        for kind, data in rows:
            y0 = y_cursor
            y1 = y0 + row_h * 0.72
            bar_w = (data["count"] / max_count) * chart_w

            indent = 22 if kind == "member" else 0
            x0 = margin_left + indent
            x1 = x0 + bar_w

            self.canvas.create_rectangle(x0, y0, x1, y1, fill=data["color"], outline="")

            label_text = data["label"]
            if kind == "main" and data.get("expandable"):
                marker = "▾" if data["key"] in self.expanded_keys else "▸"
                label_text = f"{marker} {label_text}"

            font = ("Consolas", 9, "bold") if kind == "main" else ("Consolas", 8)
            self.canvas.create_text(margin_left - 10 + (0 if kind == "main" else indent),
                                     (y0 + y1) / 2, anchor="e",
                                     text=label_text, fill=theme.CREAM, font=font)
            self.canvas.create_text(x1 + 8, (y0 + y1) / 2, anchor="w",
                                     text=str(data["count"]), fill=theme.CREAM, font=("Consolas", 8))

            if kind == "main" and data.get("expandable"):
                self.click_regions.append((y0, y1, data["key"]))

            y_cursor += row_h

        weakest = entries[:3]
        weakest_text = ", ".join(f"{e['label']} ({e['count']})" for e in weakest)
        self.info_label.configure(
            text=f"{t('weakest_label')}: {weakest_text}")

    def on_canvas_click(self, event):
        for y0, y1, key in self.click_regions:
            if y0 <= event.y <= y1:
                if key in self.expanded_keys:
                    self.expanded_keys.discard(key)
                else:
                    self.expanded_keys.add(key)
                self.draw_chart()
                return

    def record_more_data(self):
        if not self.entries_sorted:
            return
        win = tk.Toplevel(self)
        win.title(t("record_more_data"))
        theme.style_window(win)
        tk.Label(win, text=t("record_more_data"),
                 bg=theme.BG_DARK, fg=theme.CREAM, font=theme.FONT_LABEL).pack(pady=10)

        for entry in self.entries_sorted:
            if entry["expandable"]:
                group_label = tk.Label(win, text=f"— {entry['label']} —",
                                        bg=theme.BG_DARK, fg=entry["color"], font=("Consolas", 9, "bold"))
                group_label.pack(pady=(8, 2))
                for member_name, member_count, _ in sorted(entry["members"], key=lambda m: m[1]):
                    b = tk.Button(win, text=f"{member_name}   ({member_count})",
                                  command=lambda p=member_name: (win.destroy(), RecordWindow(self.master, p)))
                    theme.style_button(b)
                    b.pack(fill="x", padx=20, pady=2)
            else:
                b = tk.Button(win, text=f"{entry['label']}   ({entry['count']})",
                              command=lambda p=entry["label"]: (win.destroy(), RecordWindow(self.master, p)))
                theme.style_button(b)
                b.pack(fill="x", padx=10, pady=3)

        close_btn = tk.Button(win, text=t("close"), command=win.destroy)
        theme.style_button(close_btn)
        close_btn.pack(pady=10)
