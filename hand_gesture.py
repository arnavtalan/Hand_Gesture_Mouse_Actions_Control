import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Initialize hand tracking module
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Function to map values from one range to another
def map_values(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# Function to get the coordinates of the wrist
def get_wrist_landmark(image, results):
    image_height, image_width, _ = image.shape
    wrist_landmark = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            wrist_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            wrist_x = int(wrist_landmark.x * image_width)
            wrist_y = int(wrist_landmark.y * image_height)
            return wrist_x, wrist_y
    return None

# Function to move mouse cursor
def move_cursor(x, y, frame_width, frame_height):
    screen_width, screen_height = pyautogui.size()
    target_x = map_values(x, 0, frame_width, 0, screen_width)
    target_y = map_values(y, 0, frame_height, 0, screen_height)
    pyautogui.moveTo(target_x, target_y)

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def get_index_finger_landmark(image, results):
    image_height, image_width, _ = image.shape
    index_finger_landmark = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_x = int(index_finger_landmark.x * image_width)
            index_finger_y = int(index_finger_landmark.y * image_height)
            return index_finger_x, index_finger_y
    return None

def get_thumb_landmark(image, results):
    image_height, image_width, _ = image.shape
    thumb_landmark = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            thumb_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_x = int(thumb_landmark.x * image_width)
            thumb_y = int(thumb_landmark.y * image_height)
            return thumb_x, thumb_y
    return None

def get_middle_finger_landmark(image, results):
    image_height, image_width, _ = image.shape
    middle_finger_landmark = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            middle_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_finger_x = int(middle_finger_landmark.x * image_width)
            middle_finger_y = int(middle_finger_landmark.y * image_height)
            return middle_finger_x, middle_finger_y
    return None

# Main function
def main():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            # Flip the image horizontally for a later selfie-view display
            image = cv2.flip(image, 1)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Process the image and get hand landmarks
            results = hands.process(image_rgb)

            # If landmarks are detected
            if results.multi_hand_landmarks:
                # Draw hand landmarks on the image
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the coordinates of the wrist
                wrist = get_wrist_landmark(image, results)
                if wrist:
                    # Move mouse cursor
                    move_cursor(wrist[0], wrist[1], image_width, image_height)
                    
                index_finger = get_index_finger_landmark(image, results)
                thumb = get_thumb_landmark(image, results)
                if index_finger and thumb:
                    # Calculate distance between index finger tip and thumb tip
                    distance = calculate_distance(index_finger, thumb)
                    # Move mouse cursor if distance is below threshold
                    if distance < 23:  # Adjust threshold as needed
                        pyautogui.click()  # Perform selection action
                        
                        
                middle_finger = get_middle_finger_landmark(image, results)
                thumb = get_thumb_landmark(image, results)
                if middle_finger and thumb:
                    # Calculate distance between middle finger tip and thumb tip
                    distance = calculate_distance(middle_finger, thumb)
                    # Move mouse cursor if distance is below threshold
                    if distance < 30:  # Adjust threshold as needed
                        pyautogui.doubleClick()  # Perform selection action

            # Display the image
            cv2.imshow('Hand Tracking', image)

            # Exit when 'q' is pressed
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
