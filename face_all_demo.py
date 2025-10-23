# -*- coding: utf-8 -*-
"""
face_all_demo.py
----------------
Demo tổng hợp: Eye (EAR + blink) + Mouth (MAR + open) + Head (roll, pitch 3D, off-road/nodding).

Phím tắt:
  q / ESC  - Thoát
  f        - Bật/tắt lật gương (mirror)
  e        - Bật/tắt vẽ polyline mắt
  m        - Bật/tắt vẽ polyline miệng
  v        - Bật/tắt vẽ viz roll/pitch
  n        - Bật/tắt vẽ face mesh
  s        - Bật/tắt smoothing
  o        - Bật/tắt overlay thông số
  g        - Bật/tắt lưới tham chiếu (grid)
  r        - RESET counters/filters/baseline

Yêu cầu:
  pip install opencv-python mediapipe

Đặt file này cùng thư mục với:
  geom_types.py, geom_utils.py,
  eye_utils.py, mouth_utils.py, head_utils.py,
  ear_calc.py, mar_calc.py, head_calc.py,
  moving_average.py
"""

import argparse
import sys
import time
import math

try:
    import cv2
except Exception:
    print("Cần OpenCV: pip install opencv-python", file=sys.stderr); raise

try:
    import mediapipe as mp
except Exception:
    print("Cần MediaPipe: pip install mediapipe", file=sys.stderr); raise

# ====== Modules của bạn ======
from geom_types import Point
from geom_utils import pt_from_landmark, pts_from_landmarks

from eye_utils import LEFT_EYE_IDX, RIGHT_EYE_IDX, draw_eye_polyline
from mouth_utils import MOUTH_IDX, draw_mouth_polyline
from head_utils import (
    FOREHEAD_IDX, CHIN_IDX, LEFT_EYE_OUTER, RIGHT_EYE_OUTER,
    draw_roll_viz, draw_pitch_viz
)

from ear_calc import eye_aspect_ratio
from mar_calc import mouth_aspect_ratio
from head_calc import head_roll_deg, head_pitch_rel_eyes_deg, head_pitch_3d_deg

from moving_average import MovingAverage, MovingAverageAngle, ExponentialMovingAverage


# ====== Cấu hình ngưỡng mặc định ======
CFG = dict(
    # Blink
    BLINK_EAR_TH=0.20,
    BLINK_MIN_FRAMES=3,

    # Mouth open
    MOUTH_MAR_TH=0.65,
    MOUTH_MIN_FRAMES=5,

    # Head nod (drowsiness) theo pitch 3D magnitude
    NOD_PITCH3D_HIGH=25.0,
    NOD_PITCH3D_RECOVER=15.0,
    NOD_MIN_DOWN_FRAMES=5,
    NOD_ALERT_WINDOW_SEC=30,
    NOD_ALERT_COUNT=2,

    # Off-road / quay đầu
    IPD_SHRINK_TH=0.70,        # ipd/baseline < 0.70 → quay ngang đáng kể
    OFFCENTER_X_FRAC=0.30,     # |cx_face - cx_frame| / w > 0.30 → lệch tâm mạnh
    OFFROAD_MIN_SEC=2.0,       # duy trì > 2s → cảnh báo

    # Smoothing
    SMOOTH_EAR_WIN=3,
    SMOOTH_MAR_WIN=3,
    SMOOTH_ROLL_WIN=5,         # dùng MovingAverageAngle
    SMOOTH_PITCH_REL_WIN=5,
    SMOOTH_PITCH3D_WIN=5,

    # EMA baseline IPD
    IPD_BASELINE_ALPHA=0.02,
)


# ====== Helpers UI ======
COL_OK   = (60, 220, 60)    # xanh lá
COL_WARN = (0, 165, 255)    # cam
COL_BAD  = (0, 0, 255)      # đỏ
COL_DIM  = (200, 200, 200)  # xám nhạt
COL_TXT  = (230, 230, 230)

def draw_panel(img, x, y, lines, scale=0.6, default_color=COL_TXT, bg=(0, 0, 0), alpha=0.35):
    """
    Ô thông số bán trong suốt với màu từng dòng.
    lines: list[str] hoặc list[tuple[str, (b,g,r)]]
    """
    pad = 8
    line_h = int(18 * scale + 8)

    # Lấy text để tính kích thước
    def _str_of(line):
        return line[0] if isinstance(line, (tuple, list)) else line

    w = int(max(cv2.getTextSize(_str_of(s), cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for s in lines) + pad * 2)
    h = int(line_h * len(lines) + pad * 2)
    x2, y2 = x + w, y + h

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), bg, thickness=-1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)

    for i, line in enumerate(lines):
        if isinstance(line, (tuple, list)) and len(line) >= 2:
            s, color = line[0], line[1]
        else:
            s, color = line, default_color
        yy = y + pad + (i + 1) * line_h - 6
        cv2.putText(img, s, (x + pad, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_banner(img, text, color_bg=(0, 0, 255)):
    h, w = img.shape[:2]
    bar_h = max(40, h // 14)
    cv2.rectangle(img, (0, 0), (w, bar_h), color_bg, -1)
    cv2.putText(img, text, (10, int(bar_h*0.72)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)


def draw_badge(img, text, x, y, fg=(255,255,255), bg=(40,160,40), pad=6, scale=0.55, thickness=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    w = tw + pad*2; h = th + pad*2
    cv2.rectangle(img, (x, y), (x+w, y+h), bg, -1)
    cv2.putText(img, text, (x+pad, y+h-pad-1), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv2.LINE_AA)


def now_sec(): return time.time()


class TimeWindowCounter:
    """Đếm sự kiện trong cửa sổ thời gian trượt."""
    def __init__(self):
        from collections import deque
        self.ts = deque()
    def add(self, t: float):
        self.ts.append(t)
    def count_in(self, win_sec: float, t_now: float) -> int:
        while self.ts and (t_now - self.ts[0]) > win_sec:
            self.ts.popleft()
        return len(self.ts)
    def reset(self):
        self.ts.clear()


# ====== Màu theo ngưỡng ======
def color_for_ear(ear_avg, th):
    if ear_avg is None: return COL_DIM
    if ear_avg < th: return COL_BAD
    if ear_avg < th * 1.1: return COL_WARN
    return COL_OK

def color_for_mar(mar, th):
    if mar is None: return COL_DIM
    if mar > th: return COL_BAD
    if mar > th * 0.9: return COL_WARN
    return COL_OK

def color_for_pitch3d(p3d, hi, rec):
    if p3d is None: return COL_DIM
    if p3d > hi: return COL_BAD
    if p3d > rec: return COL_WARN
    return COL_OK

def color_for_ipd_ratio(ratio, th):
    if ratio is None: return COL_DIM
    if ratio < th: return COL_BAD
    if ratio < th + 0.05: return COL_WARN
    return COL_OK


# ====== Chương trình chính ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="Chỉ số camera")
    ap.add_argument("--no-mirror", action="store_true", help="Không lật gương")
    ap.add_argument("--min-detect", type=float, default=0.5)
    ap.add_argument("--min-track", type=float, default=0.5)
    ap.add_argument("--blink-th", type=float, default=CFG["BLINK_EAR_TH"])
    ap.add_argument("--blink-frames", type=int, default=CFG["BLINK_MIN_FRAMES"])
    ap.add_argument("--mar-th", type=float, default=CFG["MOUTH_MAR_TH"])
    ap.add_argument("--open-frames", type=int, default=CFG["MOUTH_MIN_FRAMES"])
    ap.add_argument("--win", default="Face ALL Demo", help="Tên cửa sổ")
    ap.add_argument("--viz", action="store_true", help="Vẽ trực quan roll/pitch")
    ap.add_argument("--verbose", action="store_true", help="In trạng thái tức thời ra console")
    args = ap.parse_args()

    mirror = not args.no_mirror
    show_eye_poly = True
    show_mouth_poly = True
    show_viz = bool(args.viz)
    show_mesh = False
    show_overlay = True
    show_grid = True
    smoothing = True

    # Bộ lọc làm mượt
    filt_ear_l = MovingAverage(CFG["SMOOTH_EAR_WIN"])
    filt_ear_r = MovingAverage(CFG["SMOOTH_EAR_WIN"])
    filt_mar   = MovingAverage(CFG["SMOOTH_MAR_WIN"])

    filt_roll      = MovingAverageAngle(window=CFG["SMOOTH_ROLL_WIN"], mode="(-180,180]")
    filt_pitch_rel = MovingAverage(window=CFG["SMOOTH_PITCH_REL_WIN"])
    filt_pitch3d   = MovingAverage(window=CFG["SMOOTH_PITCH3D_WIN"])

    # Baseline IPD: EMA
    ipd_baseline = ExponentialMovingAverage(alpha=CFG["IPD_BASELINE_ALPHA"])

    # Blink state
    blink_count = 0
    under_thresh_frames = 0
    closing_flag = False

    # Mouth open state
    open_count = 0
    over_thresh_frames = 0
    opening_flag = False

    # Nodding state
    nod_down_frames = 0
    nod_events = TimeWindowCounter()
    nod_state_down = False

    # Off-road state
    offroad_start_t = None
    offroad_active = False
    offroad_reason = ""

    # ==== HÀM RESET (phím r) ====
    def reset_all():
        nonlocal blink_count, under_thresh_frames, closing_flag
        nonlocal open_count, over_thresh_frames, opening_flag
        nonlocal nod_down_frames, nod_state_down
        nonlocal offroad_start_t, offroad_active, offroad_reason

        # Counters & flags
        blink_count = 0
        under_thresh_frames = 0
        closing_flag = False

        open_count = 0
        over_thresh_frames = 0
        opening_flag = False

        nod_down_frames = 0
        nod_state_down = False
        nod_events.reset()

        offroad_start_t = None
        offroad_active = False
        offroad_reason = ""

        # Filters & baselines
        filt_ear_l.reset(); filt_ear_r.reset(); filt_mar.reset()
        filt_roll.reset();  filt_pitch_rel.reset();  filt_pitch3d.reset()
        ipd_baseline.reset()
        print("[RESET] Đã reset counters, filters, baseline.")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"Không mở được camera {args.cam}", file=sys.stderr); sys.exit(1)

    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=args.min_detect,
        min_tracking_confidence=args.min_track,
    )

    prev_t = time.time(); fps = 0.0
    cv2.namedWindow(args.win, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if mirror: frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mesh.process(rgb)
            t_now = now_sec()

            face_found = bool(results.multi_face_landmarks)

            # Defaults / reset per frame
            ear_l_v = ear_r_v = ear_avg_v = None
            mar_v = None
            roll_v = pitch_rel_v = pitch3d_v = None
            ipd_ratio = None
            nod_now = False
            offroad_now = False
            offroad_now_reason = ""

            if face_found:
                fl = results.multi_face_landmarks[0].landmark

                # ====== Eye: EAR & blink ======
                le_pts = pts_from_landmarks(fl, LEFT_EYE_IDX, w, h)
                re_pts = pts_from_landmarks(fl, RIGHT_EYE_IDX, w, h)

                ear_l = eye_aspect_ratio(le_pts)
                ear_r = eye_aspect_ratio(re_pts)
                ear_l_v = filt_ear_l.push(ear_l) if smoothing else ear_l
                ear_r_v = filt_ear_r.push(ear_r) if smoothing else ear_r
                ear_avg_v = (ear_l_v + ear_r_v) / 2.0

                if show_eye_poly:
                    draw_eye_polyline(frame, le_pts, True, (0,255,0), 1)
                    draw_eye_polyline(frame, re_pts, True, (0,255,0), 1)

                # Blink logic
                if ear_avg_v < args.blink_th:
                    under_thresh_frames += 1; closing_flag = True
                else:
                    if under_thresh_frames >= args.blink_frames:
                        blink_count += 1
                    under_thresh_frames = 0; closing_flag = False

                # ====== Mouth: MAR & open ======
                mouth_pts = pts_from_landmarks(fl, MOUTH_IDX, w, h)
                mar = mouth_aspect_ratio(mouth_pts)
                mar_v = filt_mar.push(mar) if smoothing else mar

                if show_mouth_poly:
                    draw_mouth_polyline(frame, mouth_pts, True, (0,0,255), 1)

                if mar_v > args.mar_th:
                    over_thresh_frames += 1; opening_flag = True
                else:
                    if over_thresh_frames >= args.open_frames:
                        open_count += 1
                    over_thresh_frames = 0; opening_flag = False

                # ====== Head: roll / pitch ======
                fore_px: Point = pt_from_landmark(fl, FOREHEAD_IDX, w, h)
                chin_px: Point = pt_from_landmark(fl, CHIN_IDX, w, h)
                le_outer: Point = pt_from_landmark(fl, LEFT_EYE_OUTER, w, h)
                re_outer: Point = pt_from_landmark(fl, RIGHT_EYE_OUTER, w, h)

                roll = head_roll_deg(le_outer, re_outer)
                pitch_rel = head_pitch_rel_eyes_deg(fore_px, chin_px, le_outer, re_outer)
                pitch3d = head_pitch_3d_deg(fl[FOREHEAD_IDX], fl[CHIN_IDX])

                if smoothing:
                    roll_v = filt_roll.push(roll)
                    pitch_rel_v = filt_pitch_rel.push(pitch_rel)
                    pitch3d_v = filt_pitch3d.push(pitch3d)
                else:
                    roll_v, pitch_rel_v, pitch3d_v = roll, pitch_rel, pitch3d

                # Viz
                if show_viz:
                    try:
                        draw_roll_viz(frame, le_outer, re_outer, roll_v if roll_v is not None else roll)
                        # Vẽ pitch dùng giá trị pitch3d_v (độ lớn) để tránh nhầm pitch 2D
                        draw_pitch_viz(frame, fore_px, chin_px, pitch3d_v if pitch3d_v is not None else pitch3d)
                    except Exception:
                        pass

                # ====== Nodding (drowsiness) ======
                if pitch3d_v is not None:
                    nod_now = pitch3d_v > CFG["NOD_PITCH3D_HIGH"]
                    if pitch3d_v > CFG["NOD_PITCH3D_HIGH"]:
                        nod_down_frames += 1; nod_state_down = True
                    else:
                        if nod_state_down and nod_down_frames >= CFG["NOD_MIN_DOWN_FRAMES"] and pitch3d_v < CFG["NOD_PITCH3D_RECOVER"]:
                            nod_events.add(t_now)
                        nod_down_frames = 0; nod_state_down = False

                # ====== Off-road (quay đầu / lệch tâm) ======
                dx = re_outer[0] - le_outer[0]
                dy = re_outer[1] - le_outer[1]
                ipd = math.hypot(dx, dy)
                base_ipd = ipd_baseline.push(ipd)
                ipd_ratio = (ipd / base_ipd) if (base_ipd and base_ipd > 1e-6) else 1.0
                ipd_shrink_flag = ipd_ratio < CFG["IPD_SHRINK_TH"]

                cx_face = 0.5 * (le_outer[0] + re_outer[0])
                offcenter = abs(cx_face - (w * 0.5)) / max(1.0, w)
                offcenter_flag = offcenter > CFG["OFFCENTER_X_FRAC"]

                if ipd_shrink_flag or offcenter_flag:
                    offroad_now = True
                    offroad_now_reason = "IPD-shrink" if ipd_shrink_flag else "Off-center"

                if ipd_shrink_flag or offcenter_flag:
                    if offroad_start_t is None:
                        offroad_start_t = t_now
                    dur = t_now - offroad_start_t
                    if dur >= CFG["OFFROAD_MIN_SEC"]:
                        offroad_active = True
                        offroad_reason = "IPD-shrink" if ipd_shrink_flag else "Off-center"
                else:
                    offroad_start_t = None
                    offroad_active = False
                    offroad_reason = ""

                # Vẽ mesh nếu cần
                if show_mesh:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=results.multi_face_landmarks[0],
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                    )

            else:
                # ====== RESET khi mất mặt: chỉ reset trạng thái tạm (không chạm counters tổng) ======
                under_thresh_frames = 0; closing_flag = False
                over_thresh_frames = 0; opening_flag = False
                nod_down_frames = 0; nod_state_down = False
                offroad_start_t = None; offroad_active = False; offroad_reason = ""
                offroad_now = False; offroad_now_reason = ""

            # ====== Grid tham chiếu ======
            if show_grid:
                cx, cy = w // 2, h // 2
                cv2.line(frame, (0, cy), (w, cy), (80, 80, 80), 1)
                cv2.line(frame, (cx, 0), (cx, h), (80, 80, 80), 1)

            # ====== Badges tức thời ======
            badge_y = 10
            if 'w' in locals():
                # Off-road ngay tức thì
                if offroad_now:
                    dur_now = (t_now - offroad_start_t) if offroad_start_t else 0.0
                    txt = f"OFF-ROAD ({offroad_now_reason}) {dur_now:.1f}s"
                    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    draw_badge(frame, txt, w - 10 - (tw + 12), badge_y, (255,255,255), (30,120,200))
                    badge_y += th + 16
                # Nodding tức thì
                if nod_now:
                    txt = "NODDING"
                    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    draw_badge(frame, txt, w - 10 - (tw + 12), badge_y, (255,255,255), (200,120,30))
                    badge_y += th + 16

            # Verbose console (tuỳ chọn)
            if args.verbose and (nod_now or offroad_now):
                p3d_txt = f"{pitch3d_v:.1f}" if pitch3d_v is not None else "nan"
                ipd_txt = f"{ipd_ratio:.2f}" if ipd_ratio is not None else "nan"
                print(f"[{time.strftime('%H:%M:%S')}] nod_now={nod_now}, offroad_now={offroad_now}({offroad_now_reason}), "
                      f"pitch3d={p3d_txt}, ipd_ratio={ipd_txt}")

            # FPS
            now_t = time.time(); dt = now_t - prev_t; prev_t = now_t
            fps = (0.9 * fps + 0.1 * (1.0 / dt)) if dt > 0 and fps > 0 else ((1.0 / dt) if dt > 0 else fps)

            # ====== Banner cảnh báo ======
            alerts = []
            if nod_events.count_in(CFG["NOD_ALERT_WINDOW_SEC"], t_now) >= CFG["NOD_ALERT_COUNT"]:
                alerts.append("⚠️ DROWSINESS: GỤC ĐẦU NHIỀU")
            if offroad_active:
                alerts.append(f"⚠️ EYES OFF-ROAD: {offroad_reason}".strip())
            if alerts:
                draw_banner(frame, " | ".join(alerts), color_bg=(0, 0, 255))

            # ====== Overlay thông số (có màu) ======
            if show_overlay:
                ear_color = color_for_ear(ear_avg_v, args.blink_th)
                mar_color = color_for_mar(mar_v, args.mar_th)
                p3d_color = color_for_pitch3d(pitch3d_v, CFG["NOD_PITCH3D_HIGH"], CFG["NOD_PITCH3D_RECOVER"])
                ipd_color = color_for_ipd_ratio(ipd_ratio, CFG["IPD_SHRINK_TH"])

                lines = [
                    (f"FPS: {fps:5.1f} | Face: {'YES' if face_found else 'NO'}", COL_TXT),
                    # Eyes
                    (f"EAR L: {ear_l_v:.3f}" if ear_l_v is not None else "EAR L: --", ear_color),
                    (f"EAR R: {ear_r_v:.3f}" if ear_r_v is not None else "EAR R: --", ear_color),
                    (f"EAR Avg: {ear_avg_v:.3f}" if ear_avg_v is not None else "EAR Avg: --", ear_color),
                    (f"Blink Th: {args.blink_th:.2f} | MinFrames: {args.blink_frames} | Blink Count: {blink_count} {'(closing...)' if closing_flag else ''}", ear_color),
                    # Mouth
                    (f"MAR: {mar_v:.3f}" if mar_v is not None else "MAR: --", mar_color),
                    (f"Open Th: {args.mar_th:.2f} | MinFrames: {args.open_frames} | Open Count: {open_count} {'(opening...)' if opening_flag else ''}", mar_color),
                    # Head
                    (f"Roll (deg): {roll_v:+6.1f}" if roll_v is not None else "Roll (deg): --", COL_TXT),
                    (f"Pitch rel-eyes (deg): {pitch_rel_v:6.1f}" if pitch_rel_v is not None else "Pitch rel-eyes (deg): --", COL_TXT),
                    (f"Pitch 3D (deg): {pitch3d_v:6.1f}" if pitch3d_v is not None else "Pitch 3D (deg): --", p3d_color),
                    # Off-road
                    (f"IPD ratio: {ipd_ratio:.2f}" if ipd_ratio is not None else "IPD ratio: --", ipd_color),
                    (f"Off-road: {'YES' if (offroad_active or offroad_now) else 'NO'}", COL_BAD if (offroad_active or offroad_now) else COL_OK),
                    # Help
                    ("Keys: q Esc-quit | f-mirror | e-eye | m-mouth | v-viz | n-mesh | s-smooth | o-overlay | g-grid | r-reset", COL_DIM),
                ]
                draw_panel(frame, 10, 10, lines, 0.6, COL_TXT, (0,0,0), 0.35)

            cv2.imshow(args.win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')): break
            elif key == ord('f'): mirror = not mirror
            elif key == ord('e'): show_eye_poly = not show_eye_poly
            elif key == ord('m'): show_mouth_poly = not show_mouth_poly
            elif key == ord('v'): show_viz = not show_viz
            elif key == ord('n'): show_mesh = not show_mesh
            elif key == ord('s'): smoothing = not smoothing
            elif key == ord('o'): show_overlay = not show_overlay
            elif key == ord('g'): show_grid = not show_grid
            elif key == ord('r'): reset_all()

    finally:
        try: mesh.close()
        except Exception: pass
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
