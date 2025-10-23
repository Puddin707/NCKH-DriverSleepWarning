# -*- coding: utf-8 -*-
"""
eye_demo.py
py eye_demo.py
-----------
Demo theo dõi mắt bằng MediaPipe Face Mesh + EAR:
- Vẽ 6 điểm viền mỗi mắt (eye_utils)
- Tính EAR (Eye Aspect Ratio) mắt trái/phải + trung bình (ear_calc)
- Phát hiện nháy mắt đơn giản bằng ngưỡng EAR & số khung liên tiếp

Phím tắt:
  q / ESC  - Thoát
  f        - Bật/tắt lật gương (mirror)
  v        - Bật/tắt vẽ polyline mắt
  s        - Bật/tắt smoothing (MovingAverage)
  o        - Bật/tắt overlay thông số

Yêu cầu:
  pip install opencv-python mediapipe
Đặt file này cùng thư mục với:
  geom_types.py, eye_utils.py, ear_calc.py, moving_average.py
"""

import argparse
import sys
import time

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

# Modules do bạn cung cấp
from geom_types import Point  # gợi ý kiểu, không ảnh hưởng runtime
from eye_utils import (
    LEFT_EYE_IDX, RIGHT_EYE_IDX,
    draw_eye_polyline,
    pts_from_landmarks as eye_pts_from,
)
from ear_calc import eye_aspect_ratio
from moving_average import MovingAverage  # dùng bộ lọc chung


def draw_panel(img, x, y, lines, scale=0.6, color=(240, 240, 240), bg=(0, 0, 0), alpha=0.35):
    """Ô thông số bán trong suốt."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="Chỉ số camera (mặc định 0)")
    ap.add_argument("--no-mirror", action="store_true", help="Không lật gương (mặc định có lật)")
    ap.add_argument("--min-detect", type=float, default=0.5, help="min_detection_confidence")
    ap.add_argument("--min-track", type=float, default=0.5, help="min_tracking_confidence")
    ap.add_argument("--blink-th", type=float, default=0.20, help="Ngưỡng EAR coi là nhắm (mặc định 0.20)")
    ap.add_argument("--blink-frames", type=int, default=3, help="Số khung EAR<th để tính 1 lần nháy (mặc định 3)")
    ap.add_argument("--smooth-win", type=int, default=3, help="Cửa sổ MovingAverage (mặc định 3)")
    ap.add_argument("--win", default="Eye Demo", help="Tên cửa sổ")
    args = ap.parse_args()

    mirror = not args.no_mirror
    show_poly = True
    show_overlay = True
    smoothing = True

    filt_l = MovingAverage(max(1, args.smooth_win))
    filt_r = MovingAverage(max(1, args.smooth_win))

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"Không mở được camera {args.cam}", file=sys.stderr)
        sys.exit(1)

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

    # Biến đếm nháy
    blink_count = 0
    under_thresh_frames = 0
    closing_flag = False  # để hiển thị '(closing...)' ngắn hạn

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
            res = mesh.process(rgb)

            ear_l_v = ear_r_v = ear_avg_v = None
            face_found = res.multi_face_landmarks and len(res.multi_face_landmarks) > 0

            if face_found:
                fl = res.multi_face_landmarks[0].landmark

                # 6 điểm mỗi mắt
                le_pts = eye_pts_from(fl, LEFT_EYE_IDX, w, h)
                re_pts = eye_pts_from(fl, RIGHT_EYE_IDX, w, h)

                # EAR thô
                ear_l = eye_aspect_ratio(le_pts)
                ear_r = eye_aspect_ratio(re_pts)

                # Smoothing
                if smoothing:
                    ear_l_v = filt_l.push(ear_l)
                    ear_r_v = filt_r.push(ear_r)
                else:
                    ear_l_v = ear_l
                    ear_r_v = ear_r

                ear_avg_v = (ear_l_v + ear_r_v) / 2.0 if (ear_l_v is not None and ear_r_v is not None) else None

                # Vẽ polyline mắt
                if show_poly:
                    draw_eye_polyline(frame, le_pts, True, (0, 255, 0), 1)
                    draw_eye_polyline(frame, re_pts, True, (0, 255, 0), 1)

                # Phát hiện nháy mắt đơn giản
                if ear_avg_v is not None:
                    if ear_avg_v < args.blink_th:
                        under_thresh_frames += 1
                        closing_flag = True
                    else:
                        if under_thresh_frames >= args.blink_frames:
                            blink_count += 1
                        under_thresh_frames = 0
                        closing_flag = False

            # FPS
            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

            # Overlay
            if show_overlay:
                lines = [
                    f"FPS: {fps:5.1f} | Face: {'YES' if face_found else 'NO'}",
                    f"EAR L: {ear_l_v:.3f}" if ear_l_v is not None else "EAR L: --",
                    f"EAR R: {ear_r_v:.3f}" if ear_r_v is not None else "EAR R: --",
                    f"EAR Avg: {ear_avg_v:.3f}" if ear_avg_v is not None else "EAR Avg: --",
                    f"Blink Th: {args.blink_th:.2f} | MinFrames: {args.blink_frames} | SmoothWin: {max(1, args.smooth_win)}",
                    f"Blink Count: {blink_count} {'(closing...)' if closing_flag else ''}",
                    "Keys: q Esc-quit | f-mirror | v-polyline | s-smooth | o-overlay",
                ]
                draw_panel(frame, 10, 10, lines, scale=0.6, color=(230, 230, 230), bg=(0, 0, 0), alpha=0.35)

            cv2.imshow(args.win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('f'):
                mirror = not mirror
            elif key == ord('v'):
                show_poly = not show_poly
            elif key == ord('s'):
                smoothing = not smoothing
            elif key == ord('o'):
                show_overlay = not show_overlay

    finally:
        mesh.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
