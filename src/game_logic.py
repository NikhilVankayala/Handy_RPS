import cv2
import tensorflow as tf
import numpy as np
import os
import joblib
import random
import time

def play_game():
    """
    Runs the Rock, Paper, Scissors game using a webcam and a pre-trained model.

    The game detects the user's hand gesture in real-time, makes a random choice
    for the computer, and determines the winner of each round.
    """

    # --- Step 1: Load the trained model and class labels ---
    print("Loading model and class labels...")
    model_path = 'models/rps_model.keras'
    labels_path = 'models/class_labels.pkl'

    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        print("Error: Model or class labels not found.")
        print("Please ensure you have run train_model.py successfully.")
        return

    model = tf.keras.models.load_model(model_path)
    class_labels = joblib.load(labels_path)
    label_map = {v: k for k, v in class_labels.items()}
    print("Model and labels loaded successfully.")

    # --- Step 2: Initialize game variables ---
    user_score = 0
    computer_score = 0
    round_count = 0
    last_round_time = time.time()
    round_duration = 3 # seconds
    outcome_message = ""
    user_choice_display = ""
    computer_choice_display = ""

    # --- Step 3: Initialize the webcam ---
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n--- Game Started ---")
    print("Show your hand gesture in the camera.")
    print("The game will make a choice every 3 seconds.")
    print("Press 'q' to quit.")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        # Flip the frame for a more natural mirror effect
        frame = cv2.flip(frame, 1)
        
        # --- Step 4: Preprocess the frame for model prediction ---
        # Resize the frame to match the model's input size
        img = cv2.resize(frame, (128, 128))
        # Normalize the pixel values
        img = img / 255.0
        # Add a batch dimension to the image
        img = np.expand_dims(img, axis=0)

        # --- Step 5: Make a prediction every few seconds ---
        current_time = time.time()
        if current_time - last_round_time >= round_duration:
            round_count += 1
            last_round_time = current_time

            # Get the model's prediction
            prediction = model.predict(img, verbose=0)
            predicted_class_index = np.argmax(prediction)
            user_choice = label_map[predicted_class_index]
            user_choice_display = f"Your choice: {user_choice}"

            # Computer makes a random choice
            computer_choice = random.choice(list(class_labels.keys()))
            computer_choice_display = f"Computer's choice: {computer_choice}"

            # --- Step 6: Determine the winner ---
            if user_choice == computer_choice:
                outcome_message = "It's a tie!"
            elif (user_choice == 'rock' and computer_choice == 'scissors') or \
                 (user_choice == 'paper' and computer_choice == 'rock') or \
                 (user_choice == 'scissors' and computer_choice == 'paper'):
                outcome_message = "You win!"
                user_score += 1
            else:
                outcome_message = "Computer wins!"
                computer_score += 1

        # --- Step 7: Display information on the frame ---
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Display scores
        cv2.putText(frame, f"Score - You: {user_score} | Computer: {computer_score}", 
                    (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display choices
        cv2.putText(frame, user_choice_display, (10, 60), font, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, computer_choice_display, (frame.shape[1] // 2, 60), font, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Display outcome
        cv2.putText(frame, outcome_message, (10, 90), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display the countdown for the next round
        time_left = round_duration - (current_time - last_round_time)
        countdown_message = f"Next round in: {max(0, int(time_left))}"
        cv2.putText(frame, countdown_message, (frame.shape[1] - 250, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the final output frame
        cv2.imshow('Handy RPS', frame)

        # Break the loop if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting game.")
            break

    # --- Step 8: Cleanup ---
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_game()
