from pathlib import Path
import json

import cv2
import numpy as np


# ============================================================
# Project / dataset settings
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"

LOAD_EXISTING_ANNOTATIONS = False
SKIP_ALREADY_ANNOTATED = True

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Window / UI settings
# ============================================================

WINDOW_NAME = "Plate GT Annotator"
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900

# Main image zoom
MAIN_ZOOM = 1.0
MIN_MAIN_ZOOM = 1.0
MAX_MAIN_ZOOM = 20.0
MAIN_ZOOM_STEP = 1.25

# Small magnifier
MAGNIFIER_SIZE = 260
MAGNIFIER_ZOOM = 8.0
MIN_MAGNIFIER_ZOOM = 2.0
MAX_MAGNIFIER_ZOOM = 20.0
MAGNIFIER_ZOOM_STEP = 1.25

CORNER_HIT_RADIUS = 14
EDGE_HIT_DISTANCE = 10
POINT_RADIUS = 5

# Fine keyboard adjustment
FINE_STEP = 1
COARSE_STEP = 5


# ============================================================
# Runtime state
# ============================================================

original_image = None

plates = []
current_quad = []

# Base scale = fit whole image inside the window.
base_fit_scale = 1.0

# Current transform from image -> display:
# display = image * display_scale + display_offset
# display_scale = base_fit_scale * MAIN_ZOOM
display_scale = 1.0
display_offset_x = 0.0
display_offset_y = 0.0

mouse_image_x = 0
mouse_image_y = 0

# Corner / edge editing
drag_mode = None
drag_corner_idx = None
drag_edge_idx = None
drag_start_image = None
drag_original_quad = None

# Middle-button panning
pan_start_display = None
pan_original_offset = None

# The currently selected corner for keyboard fine adjustment.
active_corner_idx = None


# ============================================================
# Coordinate transforms / view handling
# ============================================================


def update_display_scale():
    global display_scale
    display_scale = base_fit_scale * MAIN_ZOOM


def clamp_view_offsets():
    """Keep the image inside the viewport while allowing full panning."""
    global display_offset_x
    global display_offset_y

    image_h, image_w = original_image.shape[:2]

    scaled_w = image_w * display_scale
    scaled_h = image_h * display_scale

    # Horizontal
    if scaled_w <= WINDOW_WIDTH:
        display_offset_x = (WINDOW_WIDTH - scaled_w) / 2.0
    else:
        min_x = WINDOW_WIDTH - scaled_w
        max_x = 0.0
        display_offset_x = float(
            np.clip(display_offset_x, min_x, max_x)
        )

    # Vertical
    if scaled_h <= WINDOW_HEIGHT:
        display_offset_y = (WINDOW_HEIGHT - scaled_h) / 2.0
    else:
        min_y = WINDOW_HEIGHT - scaled_h
        max_y = 0.0
        display_offset_y = float(
            np.clip(display_offset_y, min_y, max_y)
        )


def reset_main_view():
    global base_fit_scale
    global MAIN_ZOOM
    global display_offset_x
    global display_offset_y

    image_h, image_w = original_image.shape[:2]

    scale_x = WINDOW_WIDTH / image_w
    scale_y = WINDOW_HEIGHT / image_h
    base_fit_scale = min(scale_x, scale_y)

    MAIN_ZOOM = 1.0
    update_display_scale()

    scaled_w = image_w * display_scale
    scaled_h = image_h * display_scale

    display_offset_x = (WINDOW_WIDTH - scaled_w) / 2.0
    display_offset_y = (WINDOW_HEIGHT - scaled_h) / 2.0


def image_to_display(x, y):
    dx = int(round(x * display_scale + display_offset_x))
    dy = int(round(y * display_scale + display_offset_y))
    return dx, dy


def display_to_image_float(x, y):
    ix = (x - display_offset_x) / display_scale
    iy = (y - display_offset_y) / display_scale
    return ix, iy


def display_to_image(x, y):
    ix, iy = display_to_image_float(x, y)

    ix = int(round(ix))
    iy = int(round(iy))

    ix = int(np.clip(ix, 0, original_image.shape[1] - 1))
    iy = int(np.clip(iy, 0, original_image.shape[0] - 1))

    return ix, iy


def zoom_main_at(display_x, display_y, zoom_in):
    """Zoom main image while keeping the pixel under the cursor fixed."""
    global MAIN_ZOOM
    global display_offset_x
    global display_offset_y

    image_x, image_y = display_to_image_float(
        display_x,
        display_y,
    )

    if zoom_in:
        new_zoom = MAIN_ZOOM * MAIN_ZOOM_STEP
    else:
        new_zoom = MAIN_ZOOM / MAIN_ZOOM_STEP

    new_zoom = float(
        np.clip(
            new_zoom,
            MIN_MAIN_ZOOM,
            MAX_MAIN_ZOOM,
        )
    )

    if np.isclose(new_zoom, MAIN_ZOOM):
        return

    MAIN_ZOOM = new_zoom
    update_display_scale()

    # Preserve the image coordinate underneath the cursor.
    display_offset_x = display_x - image_x * display_scale
    display_offset_y = display_y - image_y * display_scale

    clamp_view_offsets()


def build_display_base():
    """Render only the currently visible image area into the fixed window."""
    display = np.zeros(
        (WINDOW_HEIGHT, WINDOW_WIDTH, 3),
        dtype=np.uint8,
    )

    image_h, image_w = original_image.shape[:2]

    # Visible source rectangle in original-image coordinates.
    src_x1 = max(
        0,
        int(np.floor((0 - display_offset_x) / display_scale)),
    )
    src_y1 = max(
        0,
        int(np.floor((0 - display_offset_y) / display_scale)),
    )
    src_x2 = min(
        image_w,
        int(np.ceil((WINDOW_WIDTH - display_offset_x) / display_scale)),
    )
    src_y2 = min(
        image_h,
        int(np.ceil((WINDOW_HEIGHT - display_offset_y) / display_scale)),
    )

    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return display

    crop = original_image[src_y1:src_y2, src_x1:src_x2]

    dst_x1 = int(round(src_x1 * display_scale + display_offset_x))
    dst_y1 = int(round(src_y1 * display_scale + display_offset_y))

    target_w = max(1, int(round((src_x2 - src_x1) * display_scale)))
    target_h = max(1, int(round((src_y2 - src_y1) * display_scale)))

    interpolation = (
        cv2.INTER_AREA
        if display_scale < 1.0
        else cv2.INTER_LINEAR
    )

    resized = cv2.resize(
        crop,
        (target_w, target_h),
        interpolation=interpolation,
    )

    # Clip destination rectangle to the window.
    dst_x2 = dst_x1 + target_w
    dst_y2 = dst_y1 + target_h

    clip_x1 = max(0, dst_x1)
    clip_y1 = max(0, dst_y1)
    clip_x2 = min(WINDOW_WIDTH, dst_x2)
    clip_y2 = min(WINDOW_HEIGHT, dst_y2)

    if clip_x2 <= clip_x1 or clip_y2 <= clip_y1:
        return display

    resize_x1 = clip_x1 - dst_x1
    resize_y1 = clip_y1 - dst_y1
    resize_x2 = resize_x1 + (clip_x2 - clip_x1)
    resize_y2 = resize_y1 + (clip_y2 - clip_y1)

    display[
        clip_y1:clip_y2,
        clip_x1:clip_x2,
    ] = resized[
        resize_y1:resize_y2,
        resize_x1:resize_x2,
    ]

    return display


# ============================================================
# Drawing helpers
# ============================================================


def draw_corner(image, x, y, color, label=None, selected=False):
    radius = POINT_RADIUS + (2 if selected else 0)

    cv2.circle(
        image,
        (x, y),
        radius,
        color,
        -1,
        cv2.LINE_AA,
    )

    cv2.circle(
        image,
        (x, y),
        radius + 2,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if selected:
        cv2.circle(
            image,
            (x, y),
            radius + 5,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if label is not None:
        cv2.putText(
            image,
            str(label),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_quad(image, quad, color, show_labels=True, show_active=False):
    if not quad:
        return

    display_points = [
        image_to_display(x, y)
        for x, y in quad
    ]

    for i in range(len(display_points)):
        p1 = display_points[i]

        if len(display_points) == 4:
            p2 = display_points[(i + 1) % 4]
        elif i < len(display_points) - 1:
            p2 = display_points[i + 1]
        else:
            p2 = None

        if p2 is not None:
            cv2.line(
                image,
                p1,
                p2,
                color,
                2,
                cv2.LINE_AA,
            )

    for i, (x, y) in enumerate(display_points):
        selected = (
            show_active
            and active_corner_idx is not None
            and i == active_corner_idx
        )

        draw_corner(
            image,
            x,
            y,
            color,
            label=(i + 1 if show_labels else None),
            selected=selected,
        )


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    vx = x2 - x1
    vy = y2 - y1

    wx = px - x1
    wy = py - y1

    length_sq = vx * vx + vy * vy

    if length_sq == 0:
        return np.hypot(px - x1, py - y1)

    t = (wx * vx + wy * vy) / length_sq
    t = np.clip(t, 0.0, 1.0)

    proj_x = x1 + t * vx
    proj_y = y1 + t * vy

    return np.hypot(px - proj_x, py - proj_y)


def find_corner_hit(display_x, display_y):
    if len(current_quad) != 4:
        return None

    for i, (ix, iy) in enumerate(current_quad):
        dx, dy = image_to_display(ix, iy)

        if np.hypot(display_x - dx, display_y - dy) <= CORNER_HIT_RADIUS:
            return i

    return None


def find_edge_hit(display_x, display_y):
    if len(current_quad) != 4:
        return None

    pts = [
        image_to_display(x, y)
        for x, y in current_quad
    ]

    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]

        distance = point_to_segment_distance(
            display_x,
            display_y,
            x1,
            y1,
            x2,
            y2,
        )

        if distance <= EDGE_HIT_DISTANCE:
            return i

    return None


def clamp_point(x, y):
    x = int(np.clip(x, 0, original_image.shape[1] - 1))
    y = int(np.clip(y, 0, original_image.shape[0] - 1))
    return [x, y]


# ============================================================
# Magnifier
# ============================================================


def image_to_magnifier(
    image_x,
    image_y,
    src_x1,
    src_y1,
    source_size,
    mag_x1,
    mag_y1,
):
    local_x = image_x - src_x1
    local_y = image_y - src_y1

    scale = MAGNIFIER_SIZE / source_size

    mx = int(round(mag_x1 + local_x * scale))
    my = int(round(mag_y1 + local_y * scale))

    return mx, my


def draw_magnifier(display):
    source_size = int(round(MAGNIFIER_SIZE / MAGNIFIER_ZOOM))
    source_size = max(source_size, 5)

    half = source_size // 2

    cx = mouse_image_x
    cy = mouse_image_y

    x1 = cx - half
    y1 = cy - half
    x2 = x1 + source_size
    y2 = y1 + source_size

    crop = np.zeros(
        (source_size, source_size, 3),
        dtype=np.uint8,
    )

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(original_image.shape[1], x2)
    src_y2 = min(original_image.shape[0], y2)

    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        crop[
            dst_y1:dst_y2,
            dst_x1:dst_x2,
        ] = original_image[
            src_y1:src_y2,
            src_x1:src_x2,
        ]

    magnified = cv2.resize(
        crop,
        (MAGNIFIER_SIZE, MAGNIFIER_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )

    margin = 15

    mx2 = display.shape[1] - margin
    mx1 = mx2 - MAGNIFIER_SIZE

    my1 = 60
    my2 = my1 + MAGNIFIER_SIZE

    display[my1:my2, mx1:mx2] = magnified

    # Quad overlay in the magnifier.
    if len(current_quad) == 4:
        mag_points = []

        for px, py in current_quad:
            qx, qy = image_to_magnifier(
                px,
                py,
                x1,
                y1,
                source_size,
                mx1,
                my1,
            )
            mag_points.append((qx, qy))

        for i in range(4):
            p1 = mag_points[i]
            p2 = mag_points[(i + 1) % 4]

            cv2.line(
                display,
                p1,
                p2,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        for i, (qx, qy) in enumerate(mag_points):
            selected = (
                active_corner_idx is not None
                and i == active_corner_idx
            )

            cv2.circle(
                display,
                (qx, qy),
                7 if selected else 6,
                (0, 0, 255),
                -1,
                cv2.LINE_AA,
            )

            cv2.circle(
                display,
                (qx, qy),
                10 if selected else 8,
                (0, 255, 255) if selected else (255, 255, 255),
                2 if selected else 1,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                str(i + 1),
                (qx + 8, qy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.rectangle(
        display,
        (mx1, my1),
        (mx2, my2),
        (255, 255, 255),
        2,
    )

    center = MAGNIFIER_SIZE // 2

    cv2.line(
        display,
        (mx1 + center - 18, my1 + center),
        (mx1 + center + 18, my1 + center),
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.line(
        display,
        (mx1 + center, my1 + center - 18),
        (mx1 + center, my1 + center + 18),
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        display,
        f"Mag {MAGNIFIER_ZOOM:.1f}x | ({cx}, {cy})",
        (mx1, my2 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


# ============================================================
# Quad geometry
# ============================================================


def order_quad_points(points):
    pts = np.array(points, dtype=np.float32)

    sums = pts.sum(axis=1)
    diffs = pts[:, 0] - pts[:, 1]

    tl = pts[np.argmin(sums)]
    br = pts[np.argmax(sums)]
    tr = pts[np.argmax(diffs)]
    bl = pts[np.argmin(diffs)]

    return [
        tl.astype(int).tolist(),
        tr.astype(int).tolist(),
        br.astype(int).tolist(),
        bl.astype(int).tolist(),
    ]


def move_active_corner(dx, dy):
    if len(current_quad) != 4 or active_corner_idx is None:
        return

    x, y = current_quad[active_corner_idx]
    current_quad[active_corner_idx] = clamp_point(
        x + dx,
        y + dy,
    )


# ============================================================
# Rendering
# ============================================================


def render():
    display = build_display_base()

    for plate in plates:
        draw_quad(
            display,
            plate["corners"],
            (0, 255, 0),
            show_labels=False,
            show_active=False,
        )

    if current_quad:
        draw_quad(
            display,
            current_quad,
            (0, 0, 255),
            show_labels=True,
            show_active=True,
        )

    # Top information bars.
    cv2.rectangle(
        display,
        (0, 0),
        (display.shape[1], 72),
        (0, 0, 0),
        -1,
    )

    if len(current_quad) < 4:
        status = (
            f"Create quad TL->TR->BR->BL "
            f"({len(current_quad)}/4)"
        )
    else:
        selected_text = (
            f"corner {active_corner_idx + 1}"
            if active_corner_idx is not None
            else "no corner selected"
        )
        status = (
            f"Edit quad | {selected_text} | ENTER confirm"
        )

    line1 = (
        f"{status} | Main zoom {MAIN_ZOOM:.2f}x | "
        "Wheel main zoom | Shift+Wheel magnifier | Middle-drag pan"
    )

    line2 = (
        "Click/drag corner or edge | WASD/Arrows 1px | "
        "Shift+WASD 5px | V reset view | R reset quad | "
        "U undo plate | S save | Q quit"
    )

    cv2.putText(
        display,
        line1,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        display,
        line2,
        (12, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    draw_magnifier(display)

    return display


# ============================================================
# Mouse interaction
# ============================================================


def get_wheel_delta(flags):
    try:
        return cv2.getMouseWheelDelta(flags)
    except AttributeError:
        # Fallback for older OpenCV builds.
        return 1 if flags > 0 else -1


def mouse_callback(event, x, y, flags, param):
    global mouse_image_x
    global mouse_image_y

    global MAGNIFIER_ZOOM

    global drag_mode
    global drag_corner_idx
    global drag_edge_idx
    global drag_start_image
    global drag_original_quad

    global pan_start_display
    global pan_original_offset

    global display_offset_x
    global display_offset_y

    global active_corner_idx

    mouse_image_x, mouse_image_y = display_to_image(x, y)

    # --------------------------------------------------------
    # Wheel zoom
    # --------------------------------------------------------
    if event == cv2.EVENT_MOUSEWHEEL:
        delta = get_wheel_delta(flags)
        zoom_in = delta > 0

        shift_pressed = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)

        if shift_pressed:
            if zoom_in:
                MAGNIFIER_ZOOM *= MAGNIFIER_ZOOM_STEP
            else:
                MAGNIFIER_ZOOM /= MAGNIFIER_ZOOM_STEP

            MAGNIFIER_ZOOM = float(
                np.clip(
                    MAGNIFIER_ZOOM,
                    MIN_MAGNIFIER_ZOOM,
                    MAX_MAGNIFIER_ZOOM,
                )
            )
        else:
            zoom_main_at(x, y, zoom_in)

            # Mouse image coordinate can change after zoom/clamping.
            mouse_image_x, mouse_image_y = display_to_image(x, y)

        return

    # --------------------------------------------------------
    # Middle-button pan
    # --------------------------------------------------------
    if event == cv2.EVENT_MBUTTONDOWN:
        pan_start_display = np.array(
            [x, y],
            dtype=np.float32,
        )
        pan_original_offset = np.array(
            [display_offset_x, display_offset_y],
            dtype=np.float32,
        )
        return

    if event == cv2.EVENT_MOUSEMOVE and pan_start_display is not None:
        current_display = np.array(
            [x, y],
            dtype=np.float32,
        )

        delta = current_display - pan_start_display

        display_offset_x = float(pan_original_offset[0] + delta[0])
        display_offset_y = float(pan_original_offset[1] + delta[1])

        clamp_view_offsets()

        mouse_image_x, mouse_image_y = display_to_image(x, y)
        return

    if event == cv2.EVENT_MBUTTONUP:
        pan_start_display = None
        pan_original_offset = None
        return

    # Right click resets only the current unconfirmed quad.
    if event == cv2.EVENT_RBUTTONDOWN:
        current_quad.clear()
        active_corner_idx = None
        return

    # --------------------------------------------------------
    # Left-button creation / editing
    # --------------------------------------------------------
    if event == cv2.EVENT_LBUTTONDOWN:
        # Initial quad creation.
        if len(current_quad) < 4:
            current_quad.append(
                [mouse_image_x, mouse_image_y]
            )

            if len(current_quad) == 4:
                ordered = order_quad_points(current_quad)
                current_quad.clear()
                current_quad.extend(ordered)
                active_corner_idx = 0

            return

        # Existing quad editing.
        corner_idx = find_corner_hit(x, y)

        if corner_idx is not None:
            drag_mode = "corner"
            drag_corner_idx = corner_idx
            active_corner_idx = corner_idx

            drag_original_quad = [
                point.copy()
                for point in current_quad
            ]
            return

        edge_idx = find_edge_hit(x, y)

        if edge_idx is not None:
            drag_mode = "edge"
            drag_edge_idx = edge_idx

            drag_start_image = np.array(
                [mouse_image_x, mouse_image_y],
                dtype=np.float32,
            )

            drag_original_quad = [
                point.copy()
                for point in current_quad
            ]
            return

    if event == cv2.EVENT_MOUSEMOVE:
        if drag_mode == "corner":
            current_quad[drag_corner_idx] = clamp_point(
                mouse_image_x,
                mouse_image_y,
            )

        elif drag_mode == "edge":
            current = np.array(
                [mouse_image_x, mouse_image_y],
                dtype=np.float32,
            )

            delta = current - drag_start_image

            i1 = drag_edge_idx
            i2 = (drag_edge_idx + 1) % 4

            for idx in (i1, i2):
                original = np.array(
                    drag_original_quad[idx],
                    dtype=np.float32,
                )

                moved = original + delta

                current_quad[idx] = clamp_point(
                    moved[0],
                    moved[1],
                )

    if event == cv2.EVENT_LBUTTONUP:
        # Do not reorder after every drag. Reordering can change which
        # logical corner is selected while the user is fine-adjusting.
        # The initial 4-click creation already canonicalizes the order.
        drag_mode = None
        drag_corner_idx = None
        drag_edge_idx = None
        drag_start_image = None
        drag_original_quad = None


# ============================================================
# Annotation I/O
# ============================================================


def load_existing_annotation(annotation_path):
    if not annotation_path.exists():
        return []

    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("plates", [])


def save_annotation(image_path, annotation_path):
    data = {
        "image": image_path.name,
        "width": original_image.shape[1],
        "height": original_image.shape[0],
        "plates": plates,
    }

    with annotation_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(
        f"Saved {annotation_path.name} "
        f"({len(plates)} plates)"
    )


# ============================================================
# Keyboard handling
# ============================================================


def handle_fine_adjustment(key):
    """Return True when key was consumed as a corner movement."""
    if len(current_quad) != 4 or active_corner_idx is None:
        return False

    # WASD: lowercase = 1 px, uppercase = 5 px.
    movement = {
        ord("w"): (0, -FINE_STEP),
        ord("a"): (-FINE_STEP, 0),
        ord("s"): (0, FINE_STEP),
        ord("d"): (FINE_STEP, 0),
        ord("W"): (0, -COARSE_STEP),
        ord("A"): (-COARSE_STEP, 0),
        ord("S"): (0, COARSE_STEP),
        ord("D"): (COARSE_STEP, 0),
    }

    if key in movement:
        dx, dy = movement[key]
        move_active_corner(dx, dy)
        return True

    # Common Windows waitKeyEx arrow codes.
    arrow_movement = {
        2490368: (0, -FINE_STEP),   # Up
        2621440: (0, FINE_STEP),    # Down
        2424832: (-FINE_STEP, 0),   # Left
        2555904: (FINE_STEP, 0),    # Right
        # Some OpenCV builds expose the lower 16-bit codes instead.
        65362: (0, -FINE_STEP),
        65364: (0, FINE_STEP),
        65361: (-FINE_STEP, 0),
        65363: (FINE_STEP, 0),
    }

    if key in arrow_movement:
        dx, dy = arrow_movement[key]
        move_active_corner(dx, dy)
        return True

    return False


# ============================================================
# Per-image annotation loop
# ============================================================


def annotate_image(image_path):
    global original_image
    global plates
    global current_quad
    global MAGNIFIER_ZOOM
    global active_corner_idx

    annotation_path = (
        ANNOTATIONS_DIR
        / f"{image_path.stem}.json"
    )

    original_image = cv2.imread(str(image_path))

    if original_image is None:
        print(f"Could not read {image_path}")
        return

    if LOAD_EXISTING_ANNOTATIONS:
        plates = load_existing_annotation(annotation_path)
    else:
        plates = []

    current_quad = []
    active_corner_idx = None
    MAGNIFIER_ZOOM = 8.0

    reset_main_view()

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback,
    )

    while True:
        display = render()

        cv2.imshow(
            WINDOW_NAME,
            display,
        )

        # waitKeyEx is used so arrow keys are available.
        key = cv2.waitKeyEx(20)

        if key == -1:
            continue

        # Fine adjustment takes priority only while a corner is active.
        # To avoid conflict with S=save, lowercase s is movement only
        # while a corner is selected. Save is handled with Ctrl+S or
        # after confirming the quad (when there is no current quad).
        if handle_fine_adjustment(key):
            continue

        if key in (10, 13):
            if len(current_quad) != 4:
                print("Need 4 corners first.")
                continue

            plates.append(
                {
                    "corners": [
                        point.copy()
                        for point in current_quad
                    ]
                }
            )

            print(f"Confirmed plate {len(plates)}")

            current_quad = []
            active_corner_idx = None

        elif key in (ord("r"), ord("R")):
            current_quad = []
            active_corner_idx = None

        elif key in (ord("u"), ord("U")):
            if plates:
                plates.pop()
                print("Removed last confirmed plate.")

        elif key in (ord("v"), ord("V")):
            reset_main_view()

        elif key in (ord("s"), ord("S")):
            if current_quad:
                print(
                    "Current quad is not confirmed. "
                    "Press ENTER or R."
                )
                continue

            save_annotation(
                image_path,
                annotation_path,
            )
            break

        elif key in (ord("q"), ord("Q")):
            print("Closed image without saving.")
            break

    cv2.destroyWindow(WINDOW_NAME)


# ============================================================
# Main dataset loop
# ============================================================


def main():
    image_paths = sorted(
        path
        for path in RAW_DIR.iterdir()
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(
            f"No images found in {RAW_DIR}"
        )

    print(f"Found {len(image_paths)} images.")

    for idx, image_path in enumerate(
        image_paths,
        start=1,
    ):
        annotation_path = (
            ANNOTATIONS_DIR
            / f"{image_path.stem}.json"
        )

        if SKIP_ALREADY_ANNOTATED and annotation_path.exists():
            print(
                f"[{idx}/{len(image_paths)}] "
                f"Skipping already annotated: {image_path.name}"
            )
            continue

        print()
        print("=" * 70)
        print(
            f"[{idx}/{len(image_paths)}] "
            f"{image_path.name}"
        )
        print("=" * 70)

        annotate_image(image_path)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
