"""
Homography de-risk harness
Testing harness to ensure that homography is possible at a reasonable rate.

Two modes:
  python bench.py collect  <image_dir> <label_dir>   # click rink landmarks per frame
  python bench.py evaluate <label_dir>               # leave-one-out reprojection error (feet)

Coordinate system: NHL rink, CENTER-ICE ORIGIN, units = FEET.
  x = length  in [-100, 100]   (goal lines at +/-89, blue lines at +/-25)
  y = width   in [-42.5, 42.5] (centerline at 0)
This matches the convention used by NHL/analytics tracking data.
"""
import os, sys, json, glob
import numpy as np
import cv2

# --- Rink landmarks (verified against NHL spec). Naming convention: ---
#   feature_END_SIDE   END: E = +x end (x>0), W = -x end (x<0)
#                      SIDE: N = +y (one side), S = -y (other side)
LANDMARKS = {
    "center": (0.0, 0.0),
    # blue lines meet the straight side boards (E line at +25, W at -25)
    "blue_E_N": ( 25.0,  42.5), "blue_E_S": ( 25.0, -42.5),
    "blue_W_N": (-25.0,  42.5), "blue_W_S": (-25.0, -42.5),
    # neutral-zone faceoff dots: 5 ft from blue line, 22 off center
    "dotNZ_E_N": ( 20.0,  22.0), "dotNZ_E_S": ( 20.0, -22.0),
    "dotNZ_W_N": (-20.0,  22.0), "dotNZ_W_S": (-20.0, -22.0),
    # end-zone faceoff dots: 20 ft from goal line, 22 off center
    "dotEZ_E_N": ( 69.0,  22.0), "dotEZ_E_S": ( 69.0, -22.0),
    "dotEZ_W_N": (-69.0,  22.0), "dotEZ_W_S": (-69.0, -22.0),
    # goal post BASES only (on goal line, 3 off center).
    # WARNING: never click the crossbar or net top -- those sit 48" off the ice
    # and are NOT on the z=0 plane; using them corrupts the homography.
    "post_E_N": ( 89.0,  3.0), "post_E_S": ( 89.0, -3.0),
    "post_W_N": (-89.0,  3.0), "post_W_S": (-89.0, -3.0),
    # trapezoid corners ON the goal line (11 off center)
    "trapGL_E_N": ( 89.0,  11.0), "trapGL_E_S": ( 89.0, -11.0),
    "trapGL_W_N": (-89.0,  11.0), "trapGL_W_S": (-89.0, -11.0),
    # trapezoid corners AT the end boards (14 off center)
    "trapEB_E_N": ( 100.0,  14.0), "trapEB_E_S": ( 100.0, -14.0),
    "trapEB_W_N": (-100.0,  14.0), "trapEB_W_S": (-100.0, -14.0),
    # goal line meets the (curved) corner boards. At x=+/-89 the boards are on the
    # 28 ft corner arc centered at (+/-72, +/-14.5): y = 14.5 + sqrt(28^2 - 17^2) = 36.75.
    "goalBrd_E_N": ( 89.0,  36.75), "goalBrd_E_S": ( 89.0, -36.75),
    "goalBrd_W_N": (-89.0,  36.75), "goalBrd_W_S": (-89.0, -36.75),
}

def loo_feet_errors(img_pts, rink_pts):
    """Leave-one-out: fit image->rink homography on all-but-one clicked point,
    predict the held-out point, return Euclidean error in FEET for each."""
    n = len(img_pts); errs = []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        H, _ = cv2.findHomography(img_pts[m], rink_pts[m], 0)  # 0 = least-squares DLT
        if H is None:
            continue
        proj = cv2.perspectiveTransform(img_pts[i].reshape(1, 1, 2), H)[0, 0]
        errs.append(float(np.hypot(*(proj - rink_pts[i]))))
    return errs

class _FrameState:
    """Pure (GUI-free) labeling state so undo/skip/place logic is testable."""
    def __init__(self, names, clicks):
        self.names, self.clicks, self.i = names, clicks, 0
        while self.i < len(names) and names[self.i] in clicks:  # resume: skip done
            self.i += 1

    def current(self):
        return self.names[self.i] if self.i < len(self.names) else None

    def place(self, x, y):
        name = self.current()
        if name is None:
            return None
        self.clicks[name] = [float(x), float(y)]
        self.i += 1
        return name

    def skip(self):
        name = self.current()
        if name is None:
            return None
        self.clicks.pop(name, None)
        self.i += 1
        return name

    def undo(self):
        if self.i <= 0:
            return None
        self.i -= 1
        name = self.names[self.i]
        self.clicks.pop(name, None)
        return name  # caller removes this marker

def collect(image_dir, label_dir):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    os.makedirs(label_dir, exist_ok=True)
    names = list(LANDMARKS.keys())
    paths = [p for p in sorted(glob.glob(os.path.join(image_dir, "*")))
             if p.lower().endswith((".png", ".jpg", ".jpeg"))]
    print("Controls:  LEFT-click = place current anchor   RIGHT-click = skip (not visible)\n"
          "           Undo button / 'u' = step back one    Save & Exit button / 's' = stop session\n"
          "           close the window = commit this frame and go to the next\n")

    session = {"stop": False}
    for path in paths:
        if session["stop"]:
            break
        base = os.path.basename(path)
        out = os.path.join(label_dir, base + ".json")
        clicks = {}
        if os.path.exists(out):                      # resume a partially-labeled frame
            try:
                clicks = {k: v for k, v in json.load(open(out)).items() if k in LANDMARKS}
            except Exception:
                clicks = {}
        st = _FrameState(names, clicks)
        artists, exited = {}, {"save": False}

        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(15, 8))
        plt.subplots_adjust(bottom=0.13)
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])

        def draw(name):
            x, y = clicks[name]
            m, = ax.plot([x], [y], marker="+", ms=13, mew=2, color="lime")
            t = ax.annotate(name, (x, y), color="lime", fontsize=8,
                            xytext=(6, 6), textcoords="offset points")
            artists[name] = (m, t)

        def erase(name):
            for a in artists.pop(name, ()):
                a.remove()

        def title():
            name = st.current()
            n = len(clicks)
            if name:
                ax.set_title(f"[{base}]  ({st.i + 1}/{len(names)})   place: {name}   "
                             f"(right-click=skip)   |  anchors: {n}")
            else:
                ax.set_title(f"[{base}]  done — close window for next frame   |  anchors: {n}")
            fig.canvas.draw_idle()

        for nm in list(clicks):                      # redraw resumed points
            draw(nm)

        def on_click(event):
            if event.inaxes != ax:                   # ignore clicks on buttons
                return
            if event.button == 1 and event.xdata is not None:
                name = st.place(event.xdata, event.ydata)
                if name:
                    erase(name); draw(name)
            elif event.button == 3:
                st.skip()
            else:
                return
            title()

        def on_undo(_=None):
            name = st.undo()
            if name:
                erase(name); title()

        def on_save(_=None):
            exited["save"] = True; session["stop"] = True; plt.close(fig)

        def on_key(event):
            if event.key == "u": on_undo()
            elif event.key == "s": on_save()

        ax_u = plt.axes([0.32, 0.03, 0.16, 0.06]); b_u = Button(ax_u, "Undo (u)")
        ax_s = plt.axes([0.52, 0.03, 0.16, 0.06]); b_s = Button(ax_s, "Save & Exit (s)")
        b_u.on_clicked(on_undo); b_s.on_clicked(on_save)
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)
        title()
        plt.show(block=True)                         # returns on window close or Save & Exit

        json.dump(clicks, open(out, "w"), indent=2)
        tag = "  [SAVE & EXIT]" if exited["save"] else ""
        print(f"{base:30s} -> {len(clicks)} anchors saved{tag}")
    print("\nStopped — resume anytime; done frames are skipped." if session["stop"]
          else "\nAll frames labeled.")

def evaluate(label_dir):
    medians, anchor_counts = [], []
    print(f"{'frame':32s} {'anchors':>7s} {'median_ft':>10s} {'max_ft':>8s}")
    for path in sorted(glob.glob(os.path.join(label_dir, "*.json"))):
        clicks = json.load(open(path))
        names = [n for n in clicks if n in LANDMARKS]
        anchor_counts.append(len(names))
        if len(names) < 5:
            print(f"{os.path.basename(path):32s} {len(names):7d}  SKIP (<5, no LOO)")
            continue
        img_pts  = np.array([clicks[n]      for n in names], np.float64)
        rink_pts = np.array([LANDMARKS[n]   for n in names], np.float64)
        errs = loo_feet_errors(img_pts, rink_pts)
        if not errs:
            continue
        med, mx = float(np.median(errs)), float(np.max(errs))
        medians.append(med)
        print(f"{os.path.basename(path):32s} {len(names):7d} {med:10.2f} {mx:8.2f}")
    if medians:
        a = np.array(medians)
        print("\n=== AGGREGATE ===")
        print(f"frames scored : {len(a)}")
        print(f"anchors/frame : median {int(np.median(anchor_counts))}, "
              f"min {min(anchor_counts)}, frac<4 = {np.mean(np.array(anchor_counts)<4):.0%}")
        print(f"median-of-medians : {np.median(a):.2f} ft")
        print(f"p90 of medians    : {np.percentile(a, 90):.2f} ft")

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "collect":
        collect(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 3 and sys.argv[1] == "evaluate":
        evaluate(sys.argv[2])
    else:
        print(__doc__)