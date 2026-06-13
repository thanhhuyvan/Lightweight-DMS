"""
ear_calculator.py — Module trích xuất landmark vùng mắt và tính toán EAR.

Chức năng:
    - Nhận danh sách 468 điểm mốc (MediaPipe Face Mesh).
    - Lọc ra nhóm chỉ mục mắt trái và mắt phải.
    - Tính EAR cho từng mắt theo công thức Euclid.
    - Trả về EAR_avg (float) cho mỗi khung hình.

Dependency-free output:
    Hàm `compute_ear_avg()` chỉ cần danh sách landmarks + kích thước ảnh,
    trả về 1 giá trị float duy nhất. Không phụ thuộc vào pipeline.

Test độc lập:
    python -m src.ear_calculator          (từ PROJECT_ROOT)
    hoặc: python src/ear_calculator.py    (chạy trực tiếp)
"""

import math
from typing import List, Optional, Tuple

# ──────────────────────────────────────────────
# Chỉ mục landmark vùng mắt (MediaPipe 468-point mesh)
# ──────────────────────────────────────────────
#   Thứ tự mỗi mắt: [p1, p2, p3, p4, p5, p6]
#
#       p2    p3
#   p1 ──────────── p4
#       p6    p5
#
#   EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
# ──────────────────────────────────────────────
RIGHT_EYE_IDXS: List[int] = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDXS: List[int] = [362, 385, 387, 263, 373, 380]


def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Khoảng cách Euclid giữa 2 điểm 2D."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_single_ear(
    landmarks,
    eye_idxs: List[int],
    img_w: int,
    img_h: int,
) -> float:
    """
    Tính Eye Aspect Ratio cho MỘT mắt.

    Parameters
    ----------
    landmarks : list
        Danh sách 468+ đối tượng landmark (có thuộc tính .x, .y chuẩn hoá 0-1).
    eye_idxs : list[int]
        6 chỉ mục landmark theo thứ tự [p1, p2, p3, p4, p5, p6].
    img_w, img_h : int
        Kích thước ảnh gốc (pixel) để chuyển toạ độ chuẩn hoá → pixel.

    Returns
    -------
    float
        Giá trị EAR (thường ~0.20–0.35 khi mở, <0.18 khi nhắm).
    """
    # Chuyển toạ độ chuẩn hoá → pixel
    pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_idxs]

    # Khoảng cách dọc (vertical)
    v1 = _euclidean(pts[1], pts[5])  # ||p2 - p6||
    v2 = _euclidean(pts[2], pts[4])  # ||p3 - p5||

    # Khoảng cách ngang (horizontal)
    h = _euclidean(pts[0], pts[3])   # ||p1 - p4||

    # Công thức EAR (thêm epsilon tránh chia cho 0)
    ear = (v1 + v2) / (2.0 * h + 1e-6)
    return ear


def compute_ear_avg(landmarks, img_w: int, img_h: int) -> Optional[float]:
    """
    Tính EAR trung bình cả hai mắt.

    Parameters
    ----------
    landmarks : list or None
        Danh sách 468+ đối tượng landmark (MediaPipe NormalizedLandmark).
        Nếu None hoặc rỗng, trả về None.
    img_w, img_h : int
        Kích thước ảnh gốc (pixel).

    Returns
    -------
    float or None
        EAR_avg = (EAR_right + EAR_left) / 2, hoặc None nếu landmarks không hợp lệ.
    """
    # Kiểm tra null-safety
    if landmarks is None or len(landmarks) < 400:
        return None

    try:
        ear_right = compute_single_ear(landmarks, RIGHT_EYE_IDXS, img_w, img_h)
        ear_left = compute_single_ear(landmarks, LEFT_EYE_IDXS, img_w, img_h)
        return (ear_right + ear_left) / 2.0
    except (IndexError, AttributeError):
        return None


# ══════════════════════════════════════════════
# Test độc lập bằng camera
# ══════════════════════════════════════════════
def run_camera_test():
    """Mở webcam, detect khuôn mặt và in EAR realtime. Nhấn 'q' để thoát."""
    import os
    import tempfile
    import urllib.request

    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    # ── Tự động tải model nếu chưa có ──
    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    )
    # Lưu vào thư mục temp (ASCII path — tránh lỗi unicode)
    model_path = os.path.join(tempfile.gettempdir(), "face_landmarker.task")
    if not os.path.exists(model_path):
        print("Đang tải model face_landmarker.task...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("Tải xong!")

    # ── Khởi tạo Face Landmarker (Tasks API) ──
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # ── Mở camera ──
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Không thể mở camera. Kiểm tra kết nối webcam.")
        landmarker.close()
        return

    print("=" * 50)
    print("  EAR Live Test — Nhấn 'q' để thoát")
    print("  Mở mắt bình thường → EAR ~ 0.25–0.35")
    print("  Nhắm mắt           → EAR < 0.18")
    print("=" * 50)

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            h, w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            results = landmarker.detect(mp_image)

            # Tạo bản vẽ trên frame BGR (cho OpenCV imshow)
            display_frame = frame_bgr.copy()

            if results.face_landmarks:
                face_lms = results.face_landmarks[0]

                # ── Tính EAR_avg bằng hàm module ──
                ear_avg = compute_ear_avg(face_lms, w, h)

                if ear_avg is not None:
                    # Xác định trạng thái
                    status = "CLOSED" if ear_avg < 0.20 else "OPEN"
                    color_bgr = (0, 0, 255) if status == "CLOSED" else (0, 200, 0)

                    # In ra console
                    print(f"EAR_avg = {ear_avg:.4f}  [{status}]")

                    # Vẽ text EAR lên frame
                    cv2.putText(
                        display_frame,
                        f"EAR: {ear_avg:.3f} [{status}]",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color_bgr,
                        2,
                    )

                    # Vẽ landmark mắt phải & trái
                    for idx in RIGHT_EYE_IDXS + LEFT_EYE_IDXS:
                        lm = face_lms[idx]
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(display_frame, (cx, cy), 3, (0, 100, 255), -1)
                else:
                    cv2.putText(
                        display_frame,
                        "Error computing EAR",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
            else:
                cv2.putText(
                    display_frame,
                    "No face detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            # Hiển thị bằng OpenCV
            cv2.imshow("EAR Live Test", display_frame)

            # Thoát nếu nhấn 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopped by user.")
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("Camera test ended.")


if __name__ == "__main__":
    run_camera_test()
