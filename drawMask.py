import cv2
import numpy as np
from pathlib import Path

# Initialize global variables for drawing
drawing = False
ix, iy = -1, -1
mask_polygons = []
current_polygon = []

# Mouse callback function for drawing
def draw_mask(event, x, y, flags, param):
    global ix, iy, drawing, current_polygon

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        current_polygon.append((x, y))

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(param[0], (ix, iy), (x, y), (255, 255, 255), 2)
            ix, iy = x, y
            current_polygon.append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(param[0], (ix, iy), current_polygon[0], (255, 255, 255), 2)  # Close the polygon
        mask_polygons.append(current_polygon)
        current_polygon = []


def create_mask(frame, polygons):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for polygon in polygons:
        cv2.fillPoly(mask, np.array([polygon], dtype=np.int32), 255)
    return mask


def list_video_files(directory, extension='mp4'):
    """List video files in the specified directory with the given extension."""
    directory_path = Path(directory)
    return list(directory_path.glob(f'*.{extension}'))


def main(directory, correlative):
    video_files = list_video_files(directory)

    for video_path in video_files:

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print("Error: Could not open video.")
            continue

        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read the first frame.")
            continue

        video_name = video_path.stem
        partial_name = video_name.rsplit('_', 1)[0]
        camera_name = partial_name.rsplit('_', 1)[0]
        mask_path = Path('mask') / f"{camera_name}_{correlative}.jpg"

        # Let the user draw the mask on the first frame
        clone = first_frame.copy()  # Assuming first_frame is defined
        cv2.namedWindow('Draw Mask')
        cv2.setMouseCallback('Draw Mask', draw_mask, [clone])  # Assuming draw_mask is defined

        print(
            "Draw the mask on the first frame by clicking and "
            "dragging the mouse. Press 'd' when done."
        )

        while True:
            cv2.imshow('Draw Mask', clone)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('d'):  # Done drawing
                break
        # Save the mask to a jpg file inside mask folder
        mask = create_mask(first_frame, mask_polygons)  # Assuming mask_polygons and create_mask are defined

        cv2.imwrite(str(mask_path), mask)
        cv2.destroyAllWindows()
        break

if __name__ == "__main__":

    camera_name = "ctr_526_527_pebbles_sag"
    correlative_number = 10

    videos_directory = Path('D:/analitica/storeVideos/videos/GMIN')
    directory = str(videos_directory / camera_name / "20240405" / "mask_10")

    main(directory, correlative=correlative_number)
    print(f"Directorio {directory} procesado.")