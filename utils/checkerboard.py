import cv2
import numpy as np

def generate_checkerboard(num_squares_width, num_squares_height, square_size_mm):
    # Create a black image
    board = np.zeros((num_squares_height * square_size_mm, num_squares_width * square_size_mm), dtype=np.uint8)

    # Create the checkerboard pattern
    for i in range(num_squares_height):
        for j in range(num_squares_width):
            if (i + j) % 2 == 1:
                board[i * square_size_mm:(i + 1) * square_size_mm, j * square_size_mm:(j + 1) * square_size_mm] = 255

    return board

if __name__ == '__main__':
    # Parameters for the checkerboard pattern
    num_squares_width = 6  # Number of squares in width
    num_squares_height = 8  # Number of squares in height
    square_size_mm = 100  # Size of each square in millimeters

    # Generate the checkerboard pattern
    checkerboard = generate_checkerboard(num_squares_width, num_squares_height, square_size_mm)

    # Display or save the checkerboard pattern
    cv2.imwrite('checkerboard_pattern.png', checkerboard)
    cv2.imshow('Checkerboard Pattern', checkerboard)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
