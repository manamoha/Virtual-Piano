import cv2
import numpy as np
import os

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # ابعاد جدید با نسبت 10:7
    maxWidth, maxHeight = 1000, 700
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def save_patch(image, x, y, w, h, label, key_name, index):
    patch = image[y:y+h, x:x+w]
    folder = f'dataset/{label}/{key_name}'
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(f'{folder}/{key_name}_{index}.png', patch)

# تنظیمات
ip_address = "http://192.168.1.100:8080/video"  # آدرس IP گوشی
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open phone camera")
    exit()

# تعریف ۸ ستون (نوت‌ها) با ابعاد جدید
key_regions = [
    (0, 0, 125, 700),   # C4
    (125, 0, 125, 700), # D4
    (250, 0, 125, 700), # E4
    (375, 0, 125, 700), # F4
    (500, 0, 125, 700), # G4
    (625, 0, 125, 700), # A4
    (750, 0, 125, 700), # B4
    (875, 0, 125, 700)  # C5
]
key_names = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']
index = {'finger': 0, 'no_finger': 0}
fixed_corners = None

print("Press 'f' to save with finger, 'n' for no finger, 's' to fix corners, 'q' to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        # print("Error: Could not read frame")
        break
    
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    cv2.imwrite('debug_frame.png', frame)
    cv2.imwrite('debug_edges.png', edges)
    # print("Saved debug images: debug_frame.png, debug_edges.png")
    cv2.imshow("Edges", edges)
    
    warped = None
    screenCnt = None
    if fixed_corners is None:
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for c in contours:
            # print(f"Contour area: {cv2.contourArea(c)}")
            if cv2.contourArea(c) < 500:
                # print("Contour too small")
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # print(f"Number of corners: {len(approx)}")
            if len(approx) == 4:
                screenCnt = approx
                break
    
    if fixed_corners is not None:
        screenCnt = fixed_corners
    
    if screenCnt is not None:
        detected = frame.copy()
        cv2.drawContours(detected, [screenCnt], -1, (0, 255, 0), 2)
        cv2.imshow("Detected", detected)
        warped = four_point_transform(frame, screenCnt.reshape(4, 2))
        
        # نمایش ۸ نوت
        for i, (x, y, w, h) in enumerate(key_regions):
            cv2.rectangle(warped, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(warped, key_names[i], (x+10, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.imshow("Warped", warped)
    else:
        # print("Failed to detect paper")
        cv2.imshow("Detected", frame)
        cv2.imshow("Warped", np.zeros((700, 1000, 3), dtype=np.uint8))
    
    key = cv2.waitKey(1) & 0xFF
    # print(f"Key pressed: {chr(key) if key != 255 else 'None'}")
    if key == ord('q'):
        break
    elif key == ord('s'):
        if screenCnt is not None:
            fixed_corners = screenCnt
            print("Corners fixed!")
        else:
            print("Cannot fix corners: No paper detected")
    elif key == ord('f') and warped is not None:
        x, y, w, h = key_regions[0]  # فقط C4
        save_patch(warped, x, y, w, h, 'finger', 'C4', index['finger'])
        print(f"Saved finger image: C4({index['finger']}).png")
        index['finger'] += 1
    elif key == ord('n') and warped is not None:
        x, y, w, h = key_regions[0]  # فقط C4
        save_patch(warped, x, y, w, h, 'no_finger', 'C4', index['no_finger'])
        print(f"Saved no_finger image: C4({index['no_finger']}).png")
        index['no_finger'] += 1

cap.release()
cv2.destroyAllWindows()