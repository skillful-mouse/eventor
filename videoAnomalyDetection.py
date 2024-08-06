import cv2
import numpy as np
#import os
from pathlib import Path
import glob

# Importing the QT_LOGGING_RULES to avoid the warning of the OpenCV
# os.environ["QT_LOGGING_RULES"] = "qt.*=false;*.debug=false;*.warning=false;*.critical=false"

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

def process_video(video_path, min_anomaly_duration=2.0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    video_name = video_path.stem

    # Find the last underscore's index and then slice the string to remove the last segment
    partial_name = video_name.rsplit('_', 1)[0]
    # Find the new last underscore's index in the modified string to remove the date part
    camera_name = partial_name.rsplit('_', 1)[0]

    # Construct the mask path using pathlib
    mask_path = Path('mask') / f"{camera_name}.jpg"

    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        
        # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # mask_image = [np.array([[p[0], p[1]] for p in contours[0].squeeze()])]
        #mask = create_mask(first_frame, mask_image)  # Assuming create_mask is defined elsewhere
    else:
        # Let the user draw the mask on the first frame
        clone = first_frame.copy()  # Assuming first_frame is defined
        cv2.namedWindow('Draw Mask')
        cv2.setMouseCallback('Draw Mask', draw_mask, [clone])  # Assuming draw_mask is defined

        while True:
            cv2.imshow('Draw Mask', clone)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('d'):  # Done drawing
                break
        # Save the mask to a jpg file inside mask folder
        mask = create_mask(first_frame, mask_polygons)  # Assuming mask_polygons and create_mask are defined

        cv2.imwrite(str(mask_path), mask)
        cv2.destroyAllWindows()

    # Motion detection and anomaly duration tracking
    fps = cap.get(cv2.CAP_PROP_FPS)
    min_frames = fps * min_anomaly_duration
    prev_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_frame_gray = cv2.GaussianBlur(prev_frame_gray, (21, 21), 0)
    motion_frames = 0

    # Create a folder with the name of the video
    video_folder = Path.cwd() / video_path.stem  # This creates a Path object for the new directory
    video_folder.mkdir(exist_ok=True)


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply the mask
        frame_masked = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Frame differencing, thresholding, and contour detection
        frame_diff = cv2.absdiff(prev_frame_gray, gray)
        thresh = cv2.threshold(frame_diff, 5, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False

        if motion_frames >= min_frames:
            print("Anomaly detected")
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 10 == 0:
                cv2.imwrite(f"{video_name}/{video_name}_frame{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.jpg", frame)

        for contour in contours:
            if cv2.contourArea(contour) < 500:  # Adjust as needed
                continue
            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if motion_detected:
            motion_frames += 1
        else:               
            motion_frames = 0

        
        #cv2.imshow("Frame", frame)
        prev_frame_gray = gray

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def list_video_files(directory, extension='mp4'):
    """List video files in the specified directory with the given extension."""
    directory_path = Path(directory)
    return list(directory_path.glob(f'*.{extension}'))

def log_processed_video(video_log_file, video_name):
    """Log the processed video to a file, ensuring no more than 1000 lines."""
    with open(video_log_file, 'a+') as file:
        file.seek(0)
        lines = file.readlines()
        if len(lines) >= 1000:
            lines = lines[1:]
        lines.append(video_name + '\n')
        file.seek(0)
        file.truncate()
        file.writelines(lines)

def main(directory):
    video_files = list_video_files(directory)
    video_log_file =  'processed_videos.txt'

    for video_path in video_files:
        video_name = video_path.name
        print(f'Processing {video_name}...')
        process_video(video_path, min_anomaly_duration=0.2) # Assuming process_video needs a string path
        log_processed_video(str(video_log_file), video_name) # Assuming log_processed_video needs a string path
        print(f'Finished processing {video_name}')


# Example usage:
if __name__ == "__main__":
    directory = r'D:\analitica\storeVideos\videos\GMIN\correa_461_sag\20240324'
    main(directory)
