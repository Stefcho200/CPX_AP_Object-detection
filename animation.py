import tkinter as tk


class DropdownAnimationMixin:
    """
    Mixin for tk.Toplevel subclasses that gives them a 'dropping down'
    open animation instead of just appearing instantly.

    Usage:
        class MyMenu(tk.Toplevel, DropdownAnimationMixin):
            def __init__(self, master, anchor_widget=None):
                super().__init__(master)
                ... build widgets ...
                self.animate_dropdown_open(width=420, anchor_widget=anchor_widget)

    Height is auto-measured from the actual packed content (so it never
    clips/squishes a button again, even as menus grow), unless an
    explicit height is passed in.

    The window is withdrawn (hidden) while its position is computed and
    set, then revealed already in place — this avoids the OS window
    manager flashing it at a default location first (which could look
    like it was spawning from the middle of the triggering button).

    After the animation finishes, the window becomes freely movable and
    resizable again, just like a normal window.
    """

    def animate_dropdown_open(self, width, height=None, anchor_widget=None,
                               x_offset=0, y_offset=6, duration=220, steps=16):
        # Hide immediately so the WM's default placement is never shown.
        self.withdraw()
        self.update_idletasks()

        # Auto-measure the natural required height of the window's content
        # if not explicitly given, so buttons never get squished/clipped,
        # even if more options are added later.
        if height is None:
            height = self.winfo_reqheight()

        if anchor_widget is not None:
            x = anchor_widget.winfo_rootx() + x_offset
            y = anchor_widget.winfo_rooty() + anchor_widget.winfo_height() + y_offset
        else:
            master = self.master
            mx = master.winfo_rootx()
            my = master.winfo_rooty()
            mw = master.winfo_width()
            x = mx + max(0, (mw - width) // 2)
            y = my + 40

        # Keep window fully on-screen horizontally/vertically
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        x = min(max(0, x), max(0, screen_w - width))
        y = min(max(0, y), max(0, screen_h - height))

        self._dd_final_x = x
        self._dd_final_y = y
        self._dd_final_w = width
        self._dd_final_h = height

        # While collapsed/animating, disable resize handles so the
        # animation itself can't be fought by manual resizing.
        self.resizable(False, False)

        # Set final geometry BEFORE revealing, but starting at 1px tall,
        # so the window only ever appears already correctly positioned.
        self.geometry(f"{width}x1+{x}+{y}")
        self.deiconify()

        self._dd_step(0, steps, duration)

    def _dd_step(self, i, steps, duration):
        progress = (i + 1) / steps
        eased = 1 - (1 - progress) ** 2  # ease-out: fast start, gentle settle
        cur_h = max(1, int(self._dd_final_h * eased))
        self.geometry(f"{self._dd_final_w}x{cur_h}+{self._dd_final_x}+{self._dd_final_y}")

        if i + 1 < steps:
            self.after(max(1, duration // steps), self._dd_step, i + 1, steps, duration)
        else:
            # Snap to exact final geometry to avoid any rounding drift,
            # then hand full move/resize control back to the user.
            self.geometry(f"{self._dd_final_w}x{self._dd_final_h}+"
                           f"{self._dd_final_x}+{self._dd_final_y}")
            self.resizable(True, True)
