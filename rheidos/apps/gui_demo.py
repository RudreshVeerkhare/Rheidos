"""
Panda3D DirectGUI Playground (standalone)
----------------------------------------

Install:
  pip install panda3d

Run:
  python gui_demo.py

Notes:
- Uses base.pixel2d (pixel coordinates; origin top-left; Z goes downward negative).
- Demonstrates DirectGUI widgets + callbacks + simple layout + resize handling.
"""

from direct.showbase.ShowBase import ShowBase
from direct.gui.DirectGui import (
    DirectFrame,
    DirectLabel,
    DirectButton,
    DirectCheckButton,
    DirectRadioButton,
    DirectOptionMenu,
    DirectEntry,
    DirectSlider,
    DirectWaitBar,
    DirectScrolledFrame,
    DirectDialog,
)
from panda3d.core import TextNode, loadPrcFileData, ClockObject


# Window config BEFORE ShowBase initializes
loadPrcFileData("", "window-title Panda3D GUI Playground")
loadPrcFileData("", "win-size 1200 720")
loadPrcFileData("", "show-frame-rate-meter 1")


class GuiPlayground(ShowBase):
    def __init__(self):
        super().__init__()

        self.clock = ClockObject.getGlobalClock()

        # Make a simple 3D background so it's obvious GUI is separate
        self.disableMouse()
        self.camera.setPos(0, -12, 3)
        self.camera.lookAt(0, 0, 1.5)
        self._make_simple_scene()

        # GUI parent: pixel2d uses pixel-ish coordinates
        self.ui = self.pixel2d

        # Layout constants
        self.sidebar_w = 360
        self.pad = 16
        self.row_h = 42

        # State
        self.main_visible = True
        self.items = [f"Item {i}" for i in range(1, 31)]

        # Radio button shared variable (MUST be a list; Panda mutates it by reference)
        self.radio_var = ["A"]  # default selection

        # Build UI
        self._build_ui()

        # Resize handling
        self.accept("window-event", self._on_window_event)

        # Animate progress bar
        self.progress_t = 0.0
        self.taskMgr.add(self._tick, "tick-ui")

    # -----------------------
    # Scene decoration
    # -----------------------
    def _make_simple_scene(self):
        try:
            p = self.loader.loadModel("models/panda")
            p.reparentTo(self.render)
            p.setScale(0.005)
            p.setPos(0, 0, 0)
        except Exception:
            pass

        try:
            env = self.loader.loadModel("models/environment")
            env.reparentTo(self.render)
            env.setScale(0.1)
            env.setPos(-3, 8, -1)
        except Exception:
            pass

    # -----------------------
    # Window utilities
    # -----------------------
    def _win_size(self):
        props = self.win.getProperties()
        return props.getXSize(), props.getYSize()

    def _on_window_event(self, win):
        if not win:
            return
        self._reflow()

    # -----------------------
    # UI construction
    # -----------------------
    def _build_ui(self):
        w, h = self._win_size()

        # Sidebar container
        self.sidebar = DirectFrame(
            parent=self.ui,
            frameSize=(0, self.sidebar_w, -h, 0),
            frameColor=(0.12, 0.12, 0.13, 0.96),
        )

        # Main container
        self.main = DirectFrame(
            parent=self.ui,
            frameSize=(self.sidebar_w, w, -h, 0),
            frameColor=(0.08, 0.08, 0.09, 0.92),
        )

        # Sidebar content (simple vertical stack)
        x = self.pad
        y = -self.pad

        self.title = DirectLabel(
            parent=self.sidebar,
            text="DirectGUI Playground",
            text_scale=18,
            text_align=TextNode.ALeft,
            text_fg=(1, 1, 1, 1),
            frameColor=(0, 0, 0, 0),
            pos=(x, 0, y),
        )
        y -= 36

        self.subtitle = DirectLabel(
            parent=self.sidebar,
            text="Poke widgets. Break UI. Learn.",
            text_scale=12,
            text_align=TextNode.ALeft,
            text_fg=(0.75, 0.75, 0.78, 1),
            frameColor=(0, 0, 0, 0),
            pos=(x, 0, y),
        )
        y -= 26

        # Helper: full-width sidebar button row
        def row_button(label, command):
            nonlocal y
            btn = DirectButton(
                parent=self.sidebar,
                text=label,
                text_align=TextNode.ALeft,
                text_scale=14,
                text_pos=(12, -12),
                frameSize=(0, self.sidebar_w - 2 * self.pad, -34, 0),
                frameColor=(0.18, 0.18, 0.20, 1),
                relief=1,
                pos=(x, 0, y),
                command=command,
            )
            y -= self.row_h
            return btn

        self.btn_dialog = row_button("Open Dialog", self._open_dialog)
        self.btn_toggle_main = row_button("Toggle Main Panel", self._toggle_main)
        self.btn_add_item = row_button("Add List Item", self._add_item)
        self.btn_remove_item = row_button("Remove Last Item", self._remove_item)
        self.btn_quit = row_button("Quit", self.userExit)

        y -= 10

        # Checkbox: toggle status label
        self.check_label = DirectLabel(
            parent=self.sidebar,
            text="Show status label",
            text_scale=13,
            text_align=TextNode.ALeft,
            text_fg=(0.9, 0.9, 0.9, 1),
            frameColor=(0, 0, 0, 0),
            pos=(x + 48, 0, y - 6),
        )
        self.check_show_status = DirectCheckButton(
            parent=self.sidebar,
            pos=(x + 10, 0, y - 10),
            command=self._on_toggle_status,
            indicatorValue=True,
            boxPlacement="left",
        )
        y -= self.row_h

        # Radio buttons: Mode A/B/C
        self.radio_header = DirectLabel(
            parent=self.sidebar,
            text="Radio group",
            text_scale=13,
            text_align=TextNode.ALeft,
            text_fg=(0.9, 0.9, 0.9, 1),
            frameColor=(0, 0, 0, 0),
            pos=(x, 0, y),
        )
        y -= 24

        self.radio_buttons = []
        for name in ["A", "B", "C"]:
            rb = DirectRadioButton(
                parent=self.sidebar,
                text=f"Mode {name}",
                text_scale=12,
                text_align=TextNode.ALeft,
                text_pos=(28, -6),
                variable=self.radio_var,
                value=[name],
                pos=(x, 0, y),
                command=self._on_radio,
                extraArgs=[name],
            )
            self.radio_buttons.append(rb)
            y -= 26

        # Make them exclusive
        for rb in self.radio_buttons:
            rb.setOthers(self.radio_buttons)

        # Set default selection (check/uncheck are the supported methods)
        default = self.radio_var[0]
        for rb, name in zip(self.radio_buttons, ["A", "B", "C"]):
            if name == default:
                rb.check()      # selects and updates others :contentReference[oaicite:2]{index=2}
            else:
                rb.uncheck()    # keeps indicator consistent :contentReference[oaicite:3]{index=3}

        y -= 8

        # Dropdown / option menu
        self.dropdown_label = DirectLabel(
            parent=self.sidebar,
            text="Dropdown",
            text_scale=13,
            text_align=TextNode.ALeft,
            text_fg=(0.9, 0.9, 0.9, 1),
            frameColor=(0, 0, 0, 0),
            pos=(x, 0, y),
        )
        y -= 26

        self.dropdown = DirectOptionMenu(
            parent=self.sidebar,
            items=["Low", "Medium", "High", "Maximum Chaos"],
            initialitem=1,
            text_scale=12,
            text_align=TextNode.ALeft,
            text_pos=(10, -8),
            frameSize=(0, self.sidebar_w - 2 * self.pad, -30, 0),
            pos=(x, 0, y),
            command=self._on_dropdown,
        )
        y -= self.row_h

        # Entry box (press Enter)
        self.entry_label = DirectLabel(
            parent=self.sidebar,
            text="Entry (press Enter)",
            text_scale=13,
            text_align=TextNode.ALeft,
            text_fg=(0.9, 0.9, 0.9, 1),
            frameColor=(0, 0, 0, 0),
            pos=(x, 0, y),
        )
        y -= 26

        self.entry = DirectEntry(
            parent=self.sidebar,
            initialText="Type something…",
            text_scale=12,
            width=28,
            frameColor=(0.18, 0.18, 0.20, 1),
            pos=(x, 0, y),
            command=self._on_entry_submit,
        )
        y -= self.row_h

        # Slider
        self.slider_label = DirectLabel(
            parent=self.sidebar,
            text="Slider: 50",
            text_scale=13,
            text_align=TextNode.ALeft,
            text_fg=(0.9, 0.9, 0.9, 1),
            frameColor=(0, 0, 0, 0),
            pos=(x, 0, y),
        )
        y -= 26

        self.slider = DirectSlider(
            parent=self.sidebar,
            range=(0, 100),
            value=50,
            pageSize=1,
            pos=(x + (self.sidebar_w - 2 * self.pad) / 2, 0, y),
            command=self._on_slider,
        )
        # Shrink it horizontally so it fits the sidebar
        self.slider.setScale((0.58, 1.0, 1.0))
        y -= self.row_h

        # -----------------------
        # Main panel content
        # -----------------------
        self.main_title = DirectLabel(
            parent=self.main,
            text="Main Panel",
            text_scale=18,
            text_align=TextNode.ALeft,
            text_fg=(1, 1, 1, 1),
            frameColor=(0, 0, 0, 0),
            pos=(self.sidebar_w + self.pad, 0, -self.pad),
        )

        self.status = DirectLabel(
            parent=self.main,
            text="Status: ready",
            text_scale=13,
            text_align=TextNode.ALeft,
            text_fg=(0.8, 0.8, 0.82, 1),
            frameColor=(0, 0, 0, 0),
            pos=(self.sidebar_w + self.pad, 0, -(self.pad + 36)),
        )

        # Progress bar
        self.progress = DirectWaitBar(
            parent=self.main,
            text="Progress",
            text_scale=12,
            value=0,
            range=100,
            frameSize=(0, 420, -24, 0),
            frameColor=(0.15, 0.15, 0.16, 1),
            barColor=(0.35, 0.70, 0.95, 1),
            pos=(self.sidebar_w + self.pad, 0, -(self.pad + 70)),
        )

        # Scrolled list area
        self.list_header = DirectLabel(
            parent=self.main,
            text="Scrolled List (click items)",
            text_scale=13,
            text_align=TextNode.ALeft,
            text_fg=(0.9, 0.9, 0.9, 1),
            frameColor=(0, 0, 0, 0),
            pos=(self.sidebar_w + self.pad, 0, -(self.pad + 112)),
        )

        self.scroller = DirectScrolledFrame(
            parent=self.main,
            frameSize=(0, 520, -420, 0),   # will reflow
            canvasSize=(0, 500, -800, 0),  # will update
            frameColor=(0.12, 0.12, 0.13, 1),
            scrollBarWidth=14,
            pos=(self.sidebar_w + self.pad, 0, -(self.pad + 140)),
        )
        self.canvas = self.scroller.getCanvas()

        self._rebuild_list()
        self._reflow()

    def _reflow(self):
        w, h = self._win_size()

        self.sidebar["frameSize"] = (0, self.sidebar_w, -h, 0)
        self.main["frameSize"] = (self.sidebar_w, w, -h, 0)

        self.main_title.setPos(self.sidebar_w + self.pad, 0, -self.pad)
        self.status.setPos(self.sidebar_w + self.pad, 0, -(self.pad + 36))
        self.progress.setPos(self.sidebar_w + self.pad, 0, -(self.pad + 70))
        self.list_header.setPos(self.sidebar_w + self.pad, 0, -(self.pad + 112))

        # Scroller fits to remaining space
        sc_x = self.sidebar_w + self.pad
        sc_top = -(self.pad + 140)
        sc_w = max(520, w - self.sidebar_w - 2 * self.pad)
        sc_h = max(220, h - 220)

        self.scroller.setPos(sc_x, 0, sc_top)
        self.scroller["frameSize"] = (0, sc_w, -sc_h, 0)

        self._update_canvas_size()

    # -----------------------
    # List content
    # -----------------------
    def _rebuild_list(self):
        for child in self.canvas.getChildren():
            child.removeNode()

        self.item_widgets = []
        x = 10
        y = -10
        row = 34

        for idx, name in enumerate(self.items):
            btn = DirectButton(
                parent=self.canvas,
                text=name,
                text_align=TextNode.ALeft,
                text_scale=12,
                text_pos=(10, -10),
                frameSize=(0, 460, -28, 0),
                frameColor=(0.16, 0.16, 0.18, 1),
                relief=1,
                pos=(x, 0, y),
                command=self._on_list_click,
                extraArgs=[idx, name],
            )
            self.item_widgets.append(btn)
            y -= row

        self._update_canvas_size()

    def _update_canvas_size(self):
        num = len(self.items)
        row = 34
        top_pad = 10
        bottom_pad = 10
        content_h = top_pad + bottom_pad + num * row
        self.scroller["canvasSize"] = (0, 500, -content_h, 0)

    def _add_item(self):
        self.items.append(f"Item {len(self.items) + 1}")
        self.status["text"] = f"Status: added item ({len(self.items)} total)"
        self._rebuild_list()

    def _remove_item(self):
        if self.items:
            removed = self.items.pop()
            self.status["text"] = f"Status: removed {removed} ({len(self.items)} left)"
            self._rebuild_list()
        else:
            self.status["text"] = "Status: list already empty"

    # -----------------------
    # Widget callbacks
    # -----------------------
    def _open_dialog(self):
        DirectDialog(
            dialogName="demo-dialog",
            text="Hello from DirectDialog.\n\nThis is a popup.\nClick buttons below.",
            buttonTextList=["Neat", "Close"],
            command=self._on_dialog_choice,
        )

    def _on_dialog_choice(self, choice_index):
        self.status["text"] = f"Status: dialog choice = {choice_index}"

    def _toggle_main(self):
        self.main_visible = not self.main_visible
        if self.main_visible:
            self.main.show()
            self.status["text"] = "Status: main panel shown"
        else:
            self.main.hide()

    def _on_toggle_status(self, checked):
        show = bool(checked)
        if show:
            self.status.show()
        else:
            self.status.hide()

    def _on_radio(self, name):
        # name is passed via extraArgs
        self.status["text"] = f"Status: radio mode = {name} (radio_var={self.radio_var[0]})"

    def _on_dropdown(self, choice):
        self.status["text"] = f"Status: dropdown = {choice}"

    def _on_entry_submit(self, text):
        self.status["text"] = f"Status: entry = {text}"

    def _on_slider(self, value=None):
        # Panda may pass the value; if not, read from widget
        val = int(float(value)) if value is not None else int(self.slider["value"])
        self.slider_label["text"] = f"Slider: {val}"
        self.status["text"] = f"Status: slider = {val}"

    def _on_list_click(self, idx, name):
        self.status["text"] = f"Status: clicked {name} (index {idx})"

    # -----------------------
    # Animation tick
    # -----------------------
    def _tick(self, task):
        dt = self.clock.getDt()
        self.progress_t += dt
        v = int((self.progress_t * 20) % 101)
        self.progress["value"] = v
        return task.cont


if __name__ == "__main__":
    GuiPlayground().run()
