import time
import cv2
import numpy as np
#import os
from pathlib import Path

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

def process_video(video_path, mask_correlative, min_anomaly_duration=1):
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
    mask_path = Path('mask') / f"{camera_name}_{mask_correlative}.jpg"

    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        
    else:
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

    # Motion detection and anomaly duration tracking
    fps = cap.get(cv2.CAP_PROP_FPS)
    min_frames = fps * min_anomaly_duration
    prev_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_frame_gray = cv2.GaussianBlur(prev_frame_gray, (21, 21), 0)
    motion_frames = 0
    n_frame = 0

    # Create a folder with the name of the video
    video_folder = Path.cwd() / partial_name  # This creates a Path object for the new directory
    video_folder.mkdir(exist_ok=True)


    while True:
        ret = cap.grab()
        if not ret:
            break

        # Skip some operations if the frame number is not a multiple of 5
        n_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if n_frame % 5 != 0:
            continue

        ret, frame = cap.retrieve()
        if not ret:
            break

        # Apply the masks
        frame_masked = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Frame differencing, thresholding, and contour detection
        frame_diff = cv2.absdiff(prev_frame_gray, gray)
        thresh = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)[1]                                 
        
        ## SEGUNDO PARAMETRO ES EL UMBRAL (ENTRE MAS PEQUEÑO MEJOR SE DETECTAN CAMBIOS EN ESCENAS LEJANAS)
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False

        sum_contour = 0

        for contour in contours:
            if cv2.contourArea(contour) < 500:  # Adjust as needed
                continue
            sum_contour += cv2.contourArea(contour)
            (x, y, w, h) = cv2.boundingRect(contour)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if sum_contour > 1000:
            motion_detected = True

        if motion_detected:
            motion_frames += 5
        else:               
            motion_frames = 0

        if motion_frames >= min_frames:
            print(f"Anomaly detected in frame {n_frame}")
            if n_frame % 10 == 0:
                print(f"Save frame {n_frame}")
                cv2.imwrite(f"{partial_name}/{video_name}_frame{n_frame}.jpg", frame)

        prev_frame_gray = gray
        

        # cv2.imshow("Frame", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

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

def main(directory, mask_correlative):
    video_files = list_video_files(directory)
    video_log_file =  'processed_videos.txt'

    for video_path in video_files:
        video_name = video_path.name
        t_start = time.time()
        print(f'Processing {video_name}...')
        
        process_video(video_path, mask_correlative, min_anomaly_duration=3) # Assuming process_video needs a string path
        log_processed_video(str(video_log_file), video_name) # Assuming log_processed_video needs a string path
        
        t_end = time.time()
        print(f'Finished processing {video_name}. Time taken: {t_end - t_start:.2f}s\n')


# Example usage:
if __name__ == "__main__":

    # FIXME: Modificar ruta del directorio
    videos_directory = Path('D:/analitica/storeVideos/videos/GMIN')

    # Usar como directory el nombre string de la camara
    # Por defecto correr el "dia de ayer"
    # Crear el analizador de argumentos
    # parser = argparse.ArgumentParser(description='Procesa el nombre de una cámara.')
    # parser.add_argument('nombre_camara', type=str, help='El nombre de la cámara')
    # args = parser.parse_args()
    # print(f"El nombre de la cámara proporcionado es: {args.nombre_camara}")
    
    # Calcular la fecha de ayer y formatear (yyyymmdd)
    # ayer = datetime.now() - timedelta(days=1)
    # fecha_ayer_formato = ayer.strftime('%Y%m%d')

    # directory = str(videos_directory / args.nombre_camara / fecha_ayer_formato)
    # for fecha in ["20240319", "20240320", "20240321", "20240322", "20240323", "20240324"]:
    #     directory = str(videos_directory / "ctr_516_sag" / fecha)
    #     main(directory)
    #     print(f"Directorio {directory} procesado.")

    directory_camera = str(videos_directory / "alimentacion_carguio_bolas_sag1")

    main(str(Path(directory_camera) / "20240401"), 0)
    main(str(Path(directory_camera) / "20240402"), 0)
    main(str(Path(directory_camera) / "20240403" / "mask_0"), 0)
    main(str(Path(directory_camera) / "20240403" / "mask_1"), 1)
    main(str(Path(directory_camera) / "20240403" / "mask_2"), 2)
    main(str(Path(directory_camera) / "20240404"), 2)
    main(str(Path(directory_camera) / "20240405"), 2)

# ctr_526_527_pebbles_sag
    
    directory_camera = str(videos_directory / "correa_461_sag")
    main(str(Path(directory_camera) / "20240401"), 0)
    main(str(Path(directory_camera) / "20240402"), 0)
    main(str(Path(directory_camera) / "20240403"), 0)
    main(str(Path(directory_camera) / "20240404"), 0)
    main(str(Path(directory_camera) / "20240405"), 0)


    directory_camera = str(videos_directory / "ctr_101_chsec")
    main(str(Path(directory_camera) / "20240401"), 0)
    main(str(Path(directory_camera) / "20240402"), 0)
    main(str(Path(directory_camera) / "20240403"), 0)
    main(str(Path(directory_camera) / "20240404"), 0)
    main(str(Path(directory_camera) / "20240405" / "mask_0"), 0)
    main(str(Path(directory_camera) / "20240405" / "mask_1"), 1)

    directory_camera = str(videos_directory / "ctr_526_527_pebbles_sag")
    main(str(Path(directory_camera) / "20240401" / "mask_0"), 0)
    main(str(Path(directory_camera) / "20240401" / "mask_1"), 1)
    main(str(Path(directory_camera) / "20240401" / "mask_2"), 2)
    main(str(Path(directory_camera) / "20240401" / "mask_3"), 3)

    main(str(Path(directory_camera) / "20240402" / "mask_3"), 3)
    main(str(Path(directory_camera) / "20240402" / "mask_4"), 4)
    main(str(Path(directory_camera) / "20240402" / "mask_5"), 5)
    main(str(Path(directory_camera) / "20240402" / "mask_6"), 6)

    main(str(Path(directory_camera) / "20240403" / "mask_6"), 6)
    main(str(Path(directory_camera) / "20240403" / "mask_7"), 7)

    main(str(Path(directory_camera) / "20240404" / "mask_7"), 7)

    main(str(Path(directory_camera) / "20240405" / "mask_7"), 7)
    main(str(Path(directory_camera) / "20240405" / "mask_8"), 8)
    main(str(Path(directory_camera) / "20240405" / "mask_9"), 9)
    main(str(Path(directory_camera) / "20240405" / "mask_10"), 10)



    # Delete all mask files in the mask folder
    # mask_folder = Path('mask')
    # for mask_file in mask_folder.glob('*.jpg'):
    #     mask_file.unlink()