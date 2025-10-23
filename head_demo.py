# -*- coding: utf-8 -*-
"""
head_demo.py
------------
Demo theo dõi "góc đầu" + cảnh báo mất chú ý (phiên bản bỏ pitch 2D trán→cằm theo trục ảnh).

Giữ & dùng:
- head_roll_deg: roll 2D từ đường nối hai khóe mắt
- head_pitch_rel_eyes_deg: pitch 2D **tương đối theo đường mắt** (roll-invariant)
- head_pitch_3d_deg: pitch 3D (độ lớn) từ (x,y,z) MediaPipe
- Smoothing từ moving_average.py (MA/EMA/MA-angle)

Cảnh báo:
- GỤC ĐẦU (nodding/microsleep): dựa trên pitch_3d (độ lớn) vượt ngưỡng rồi hồi lại
- QUAY ĐẦU / NHÌN LỆCH: dựa trên IPD-shrink so với baseline EMA + lệch tâm theo trục X

Phím tắt:
  q / ESC  - Thoát
  f        - Bật/tắt lật gương (mirror)
  v        - Bật/tắt vẽ trực quan (viz roll/pitch)
  n        - Bật/tắt vẽ mesh (tesselation)
  s        - Bật/tắt smoothing
  o        - Bật/tắt overlay thông số
  g        - Bật/tắt grid tham chiếu

Yêu cầu:
  pip install opencv-python mediapipe
Đặt file này cùng thư mục với:
  geom_types.py, geom_utils.py, head_utils.py, head_calc.py, moving_average.py
"""

import argparse
import sys
import time
import math

try:
    import cv2
except Exception:
    print("Cần OpenCV: pip install opencv-python", file=sys.stderr)
    raise

try:
    import mediapipe as mp
except Exception:
    print("Cần MediaPipe: pip install mediapipe", file=sys.stderr)
    raise

from geom_types import Point
from head_utils import (
    FOREHEAD_IDX, CHIN_IDX, LEFT_EYE_OUTER, RIGHT_EYE_OUTER,
    pt_from_landmark, draw_roll_viz, draw_pitch_viz
)
from head_calc import (
    head_roll_deg, head_pitch_rel_eyes_deg, head_pitch_3d_deg
)
from moving_average import MovingAverage, MovingAverageAngle, ExponentialMovingAverage


def draw_panel(img, x, y, lines, scale=0.6, color=(240, 240, 240), bg=(0, 0, 0), alpha=0.35):
    pad = 8
    line_h = int(18 * scale + 8)
    w = int(max([cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for s in lines]) + pad * 2)
    h = int(line_h * len(lines) + pad * 2)
    x2, y2 = x + w, y + h
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), bg, thickness=-1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    for i, s in enumerate(lines):
        yy = y + pad + (i + 1) * line_h - 6
        cv2.putText(img, s, (x + pad, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_banner(img, text, color_bg=(0, 0, 255)):
    h, w = img.shape[:2]
    bar_h = max(40, h // 14)
    cv2.rectangle(img, (0, 0), (w, bar_h), color_bg, -1)
    cv2.putText(img, text, (10, int(bar_h*0.72)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)


def draw_badge(img, text, x, y, fg=(255,255,255), bg=(40,160,40), pad=6, scale=0.55, thickness=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    w = tw + pad*2
    h = th + pad*2
    cv2.rectangle(img, (x, y), (x+w, y+h), bg, -1)
    cv2.putText(img, text, (x+pad, y+h-pad-1), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv2.LINE_AA)


def now_sec():
    return time.time()


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


def main():
    ap = argparse.ArgumentParser()
    # Camera & pipeline
    ap.add_argument("--cam", type=int, default=0, help="Chỉ số camera (mặc định 0)")
    ap.add_argument("--no-mirror", action="store_true", help="Không lật gương (mặc định có lật)")
    ap.add_argument("--min-detect", type=float, default=0.5, help="min_detection_confidence")
    ap.add_argument("--min-track", type=float, default=0.5, help="min_tracking_confidence")
    ap.add_argument("--win", default="Head Demo (3D+rel-eyes)", help="Tên cửa sổ")
    ap.add_argument("--viz", action="store_true", help="Bật vẽ viz roll/pitch")
    ap.add_argument("--verbose", action="store_true", help="In log trạng thái ra console")

    # Thresholds & timings (cho phép override CFG)
    ap.add_argument("--nod-high", type=float, default=25.0, help="pitch3D > nod_high coi là cúi sâu")
    ap.add_argument("--nod-recover", type=float, default=15.0, help="pitch3D < nod_recover coi là ngóc lên")
    ap.add_argument("--nod-min-frames", type=int, default=5, help="Tối thiểu khung giữ trạng thái cúi")
    ap.add_argument("--nod-window", type=float, default=30.0, help="Cửa sổ đếm gục (s)")
    ap.add_argument("--nod-count", type=int, default=2, help="Số lần gục trong cửa sổ để cảnh báo")

    ap.add_argument("--ipd-shrink-th", type=float, default=0.70, help="ipd/baseline < th → quay ngang đáng kể")
    ap.add_argument("--offcenter-x-frac", type=float, default=0.30, help="|cx_face - cx_frame|/w > frac → lệch tâm")
    ap.add_argument("--offroad-min-sec", type=float, default=2.0, help="Thời gian duy trì off-road để cảnh báo")

    # Smoothing windows & EMA alpha
    ap.add_argument("--smooth-roll-win", type=int, default=5, help="MA-angle cửa sổ cho roll")
    ap.add_argument("--smooth-pitch-rel-win", type=int, default=5, help="MA cửa sổ cho pitch_rel_eyes")
    ap.add_argument("--smooth-pitch3d-win", type=int, default=5, help="MA cửa sổ cho pitch3d")
    ap.add_argument("--ipd-alpha", type=float, default=0.02, help="EMA alpha cho baseline IPD")

    args = ap.parse_args()

    mirror = not args.no_mirror
    show_viz = bool(args.viz)
    show_mesh = False
    smoothing = True
    show_overlay = True
    show_grid = True

    # Smoothing filters
    filt_roll = MovingAverageAngle(window=max(1, args.smooth_roll_win), mode="(-180,180]")
    filt_pitch_rel = MovingAverage(window=max(1, args.smooth_pitch_rel_win))
    filt_pitch3d = MovingAverage(window=max(1, args.smooth_pitch3d_win))

    # Baseline IPD: EMA
    ipd_baseline = ExponentialMovingAverage(alpha=max(1e-6, min(1.0, args.ipd_alpha)))

    # Nodding state
    nod_down_frames = 0
    nod_events = TimeWindowCounter()
    nod_state_down = False

    # Off-road
    offroad_start_t = None
    offroad_active = False
    offroad_reason = ""

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"Không mở được camera {args.cam}", file=sys.stderr)
        sys.exit(1)

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

    prev_t = time.time()
    fps = 0.0

    cv2.namedWindow(args.win, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mesh.process(rgb)
            t_now = now_sec()

            # Defaults
            face_found = results.multi_face_landmarks and len(results.multi_face_landmarks) > 0
            roll_v = pitch_rel_v = pitch3d_v = None
            ipd_ratio = None
            offroad_alert_now = False

            # Trạng thái tức thời (để badge)
            nod_now = False
            offroad_now = False
            offroad_now_reason = ""

            if face_found:
                fl = results.multi_face_landmarks[0].landmark

                # Pixel points
                fore_px: Point = pt_from_landmark(fl, FOREHEAD_IDX, w, h)
                chin_px: Point = pt_from_landmark(fl, CHIN_IDX, w, h)
                le_outer: Point = pt_from_landmark(fl, LEFT_EYE_OUTER, w, h)
                re_outer: Point = pt_from_landmark(fl, RIGHT_EYE_OUTER, w, h)

                # --- Góc ---
                roll = head_roll_deg(le_outer, re_outer)                                   # (-180..180]
                pitch_rel = head_pitch_rel_eyes_deg(fore_px, chin_px, le_outer, re_outer)  # [0..180]
                pitch3d = head_pitch_3d_deg(fl[FOREHEAD_IDX], fl[CHIN_IDX])                # [0..~90]

                if smoothing:
                    roll_v = filt_roll.push(roll)
                    pitch_rel_v = filt_pitch_rel.push(pitch_rel)
                    pitch3d_v = filt_pitch3d.push(pitch3d)
                else:
                    roll_v, pitch_rel_v, pitch3d_v = roll, pitch_rel, pitch3d

                # --- GỤC ĐẦU (nodding) qua pitch_3d magnitude ---
                if pitch3d_v is not None:
                    nod_now = pitch3d_v > args.nod_high  # hiển thị tức thời

                    # logic đếm 1 lần gục
                    if pitch3d_v > args.nod_high:
                        nod_down_frames += 1
                        nod_state_down = True
                    else:
                        if nod_state_down and nod_down_frames >= args.nod_min_frames and pitch3d_v < args.nod_recover:
                            nod_events.add(t_now)  # ghi nhận 1 lần gục
                        nod_down_frames = 0
                        nod_state_down = False

                # --- QUAY ĐẦU / NHÌN LỆCH ---
                dx = re_outer[0] - le_outer[0]
                dy = re_outer[1] - le_outer[1]
                ipd = math.hypot(dx, dy)

                base_ipd = ipd_baseline.push(ipd)
                ipd_ratio = ipd / base_ipd if base_ipd else 1.0
                ipd_shrink_flag = ipd_ratio < args.ipd_shrink_th

                cx_face = 0.5 * (le_outer[0] + re_outer[0])
                offcenter = abs(cx_face - (w * 0.5)) / max(1.0, w)
                offcenter_flag = offcenter > args.offcenter_x_frac

                # Trạng thái tức thời (không chờ đủ thời lượng)
                if ipd_shrink_flag or offcenter_flag:
                    offroad_now = True
                    offroad_now_reason = "IPD-shrink" if ipd_shrink_flag else "Off-center"

                # Cảnh báo theo thời lượng
                if ipd_shrink_flag or offcenter_flag:
                    if offroad_start_t is None:
                        offroad_start_t = t_now
                    dur = t_now - offroad_start_t
                    if dur >= args.offroad_min_sec:
                        offroad_active = True
                        offroad_alert_now = True
                        offroad_reason = "IPD-shrink" if ipd_shrink_flag else "Off-center"
                else:
                    offroad_start_t = None
                    offroad_active = False
                    offroad_reason = ""

                # --- Vẽ trực quan (chỉ vẽ nếu bật) ---
                if show_viz:
                    try:
                        # Vẽ hình 2D; text hiển thị giá trị pitch3d_v (magnitude)
                        draw_roll_viz(frame, le_outer, re_outer, roll_v if roll_v is not None else roll)
                        draw_pitch_viz(frame, fore_px, chin_px, pitch3d_v if pitch3d_v is not None else pitch3d)
                    except Exception:
                        pass

                if show_mesh:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=results.multi_face_landmarks[0],
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                    )

            # Grid tham chiếu
            if show_grid:
                cx, cy = w // 2, h // 2
                cv2.line(frame, (0, cy), (w, cy), (80, 80, 80), 1)
                cv2.line(frame, (cx, 0), (cx, h), (80, 80, 80), 1)

            # ====== BADGES trạng thái tức thời ======
            badge_y = 10
            # OFF-ROAD ngay lập tức (kèm thời lượng đang duy trì)
            if offroad_now:
                dur_now = (t_now - offroad_start_t) if offroad_start_t else 0.0
                txt = f"OFF-ROAD ({offroad_now_reason}) {dur_now:.1f}s"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                draw_badge(frame, txt, w - 10 - (tw + 12), badge_y, fg=(255,255,255), bg=(30,120,200))
                badge_y += th + 16
            # NODDING ngay lập tức
            if nod_now:
                txt = "NODDING"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                draw_badge(frame, txt, w - 10 - (tw + 12), badge_y, fg=(255,255,255), bg=(200,120,30))
                badge_y += th + 16

            # Verbose console (fix f-string cho điều kiện)
            if args.verbose and (nod_now or offroad_now):
                pitch3d_print = f"{pitch3d_v:.1f}" if pitch3d_v is not None else "nan"
                ipd_ratio_print = f"{ipd_ratio:.2f}" if ipd_ratio is not None else "nan"
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"nod_now={nod_now}, offroad_now={offroad_now}({offroad_now_reason}), "
                    f"pitch3d={pitch3d_print}, ipd_ratio={ipd_ratio_print}"
                )

            # FPS
            now_t = time.time()
            dt = now_t - prev_t
            prev_t = now_t
            fps = (0.9 * fps + 0.1 * (1.0 / dt)) if dt > 0 and fps > 0 else ((1.0 / dt) if dt > 0 else fps)

            # BANNER cảnh báo
            alerts = []
            if nod_events.count_in(args.nod_window, t_now) >= args.nod_count:
                alerts.append("⚠️ DROWSINESS: GỤC ĐẦU NHIỀU")
            if offroad_active or offroad_alert_now:
                alerts.append(f"⚠️ EYES OFF-ROAD: {offroad_reason}".strip())
            if alerts:
                draw_banner(frame, " | ".join(alerts), color_bg=(0, 0, 255))

            # Overlay thông số
            if show_overlay:
                lines = [
                    f"FPS: {fps:5.1f} | Face: {'YES' if face_found else 'NO'}",
                    f"Roll (deg): {roll_v:+6.1f}" if roll_v is not None else "Roll (deg): --",
                    f"Pitch rel-eyes (deg): {pitch_rel_v:6.1f}" if pitch_rel_v is not None else "Pitch rel-eyes (deg): --",
                    f"Pitch 3D (deg): {pitch3d_v:6.1f}" if pitch3d_v is not None else "Pitch 3D (deg): --",
                    f"Nod count (last {int(args.nod_window)}s): {nod_events.count_in(args.nod_window, t_now)}",
                    f"IPD ratio: {ipd_ratio:.2f}" if ipd_ratio is not None else "IPD ratio: --",
                    f"Off-road: {'YES' if (offroad_active or offroad_alert_now) else 'NO'}",
                    "Keys: q Esc-quit | v-viz | n-mesh | f-mirror | s-smooth | o-overlay | g-grid",
                ]
                draw_panel(frame, 10, 10, lines, scale=0.6, color=(230,230,230), bg=(0,0,0), alpha=0.35)

            cv2.imshow(args.win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')): break
            elif key == ord('v'): show_viz = not show_viz
            elif key == ord('n'): show_mesh = not show_mesh
            elif key == ord('f'): mirror = not mirror
            elif key == ord('s'): smoothing = not smoothing
            elif key == ord('o'): show_overlay = not show_overlay
            elif key == ord('g'): show_grid = not show_grid

    finally:
        mesh.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
