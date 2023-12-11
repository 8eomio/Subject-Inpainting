import numpy as np
import cv2

# Global variables
drawing = False  # True if mouse is pressed
ix, iy = -1, -1

# Function to draw with a brush
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing

    brush_size = 50  # Adjust the size of the brush as needed

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(mask, (x, y), brush_size, 1, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(mask, (x, y), brush_size, 1, -1)

# Read the image
image = cv2.imread('./barn/source/burntbarn_5065_GT.png')
h, w = image.shape[:2]
mask = np.zeros((h, w), dtype=np.uint8)

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while True:
    img_with_mask = cv2.bitwise_and(image, image, mask=mask)
    combined_display = np.hstack([image, img_with_mask])
    cv2.imshow('image', combined_display)

    k = cv2.waitKey(1) & 0xFF

    # 's' 키를 눌렀을 때 마스크를 저장
    if k == ord('s'):
        scaled_mask = (mask * 255).astype(np.uint8)
        cv2.imwrite('./examples/mask/mask.png', scaled_mask)
        print(f"mask saved")
    # 'ESC' 키를 눌렀을 때 루프 종료
    elif k == 27:
        break

cv2.destroyAllWindows()

