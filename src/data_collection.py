import cv2
import os
import time

def collect_data():
    data_dir = 'data/raw'
    for gesture in ['rock', 'paper', 'scissors']:
        path = os.path.join(data_dir, gesture)
        os.makedirs(path, exist_ok=True)
        print(f"Directory created at: {path}")
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("\n--- Data Collection Started ---")
    print("Press 'r' for ROCK, 'p' for PAPER, 's' for SCISSORS.")
    print("Press 'q' to quit.")

    last_capture_time = 0
    capture_interval = 1

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)

        cv2.imshow('Gesture collector', frame)

        current_time = time.time()

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nQuitting data collection.")
            break

        if current_time - last_capture_time > capture_interval:
            if key == ord('r'):
                last_capture_time = current_time
                count = len(os.listdir(os.path.join(data_dir, 'rock'))) + 1
                file_name = f"rock_{count}.jpg"
                file_path = os.path.join(data_dir, 'rock', file_name)
                cv2.imwrite(file_path, frame)
                print(f"Captured: {file_path}")
            elif key == ord('p'):
                last_capture_time = current_time
                count = len(os.listdir(os.path.join(data_dir, 'paper'))) + 1
                file_name = f"paper_{count}.jpg"
                file_path = os.path.join(data_dir, 'paper', file_name)
                cv2.imwrite(file_path, frame)
                print(f"Captured: {file_path}")
            elif key == ord('s'):
                last_capture_time = current_time
                count = len(os.listdir(os.path.join(data_dir, 'scissors'))) + 1
                file_name = f"scissors_{count}.jpg"
                file_path = os.path.join(data_dir, 'scissors', file_name)
                cv2.imwrite(file_path, frame)
                print(f"Captured: {file_path}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    collect_data()