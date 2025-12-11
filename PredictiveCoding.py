import numpy as np
from PIL import Image


##############################################
#                MAIN MENU
##############################################

def menu():
    """Display main menu and return user choice."""
    print("\n============ Vector Quantization Menu ============")
    print("1. Compress Image")
    print("2. Decompress Image")
    print("3. Exit")
    print("==================================================\n")
    
    while True:
        try:
            choice = int(input("Enter your choice (1-3): "))
            if choice in [1, 2, 3]:
                return choice
            print("Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main_menu():
    pass


##############################################
#           USER INPUT FUNCTIONS
##############################################

def get_image_path():
    """Ask user for the image file path."""
    path = input("Enter the path to the grayscale image file: ").strip()
    
    if path == "":
        raise ValueError("Image path cannot be empty.")
    
    if not path.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise ValueError("Unsupported file format. Please provide a PNG or JPG image.")
    
    return path


def get_quantization_bits():
    """Ask user for number of bits (default = 2)."""
    while True:
        bits_input = input("Enter number of bits for quantization (default 2): ").strip()
        if bits_input == "":
            return 2
        try:
            bits = int(bits_input)
            if bits <= 0 or bits > 8:
                print("Please enter a positive integer between 1 and 8.")
                continue
            return bits
        except ValueError:
            print("Invalid input. Please enter a positive integer.")


##############################################
#          IMAGE LOADING / SAVING
##############################################

def load_image(image_path):
    """Load image as numpy array."""
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    return np.array(img)

def save_image(array, filename):
    """Save numpy array as image."""
    img = Image.fromarray(array.astype(np.uint8))
    img.save(filename)


##############################################
#        2D PREDICTOR (MANDATORY)
##############################################

def predictor_2d(image):
    """
    Adaptive 2D Predictor (as in the provided lecture slide):

    B   C   D
    A   X

    A = left        = (i, j-1)
    B = up-left     = (i-1, j-1)
    C = up          = (i-1, j)

    Predictor rule:
        if B <= min(A, C):  P = max(A, C)
        elif B >= max(A, C): P = min(A, C)
        else:                P = A + C - B
    """
    img = image.astype(np.int16) # int16 is used to avoid overflow
    rows, cols = img.shape
    predicted = np.zeros_like(img, dtype=np.int16)

    for i in range(rows):
        for j in range(cols):
            if i == 0 and j == 0:
                predicted[i, j] = img[i, j]
                continue
        
            A = img[i, j-1]
            B = img[i-1, j-1]
            C = img[i-1, j]

            if B <= min(A, C):
                P = max(A, C)
            elif B >= max(A, C):
                P = min(A, C)
            else:
                P = A + C - B

            predicted[i, j] = P
    return predicted

##############################################
#         ERROR COMPUTATION
##############################################

def compute_error(original, predicted):
    """Compute prediction error (original - predicted)."""
    if original.shape != predicted.shape:
        raise ValueError("Original and predicted images must have the same shape.")
    
    error = original.astype(np.int16) - predicted.astype(np.int16)
    return error


##############################################
#          UNIFORM QUANTIZER
##############################################

def compute_quantization_params(bits , min_val=-255, max_val=255):
    """
    Given number of bits:
      - compute number of levels
      - find step size
      - generate quantization codebook (binary)
      - generate dequantized values
    Returns: step, q_levels, binary_codes, dequant_values
    """

    q_levels = 2 ** bits
    step = (max_val - min_val) / q_levels
    binary_codes = [format(i, f'0{bits}b') for i in range(q_levels)]
    dequant_values = []
    for i in range(q_levels):
        dequant_values.append(min_val + (i + 0.5) * step)  # Centering around 0
        
    return step, q_levels, binary_codes, dequant_values
    


def quantize_error(error, step , min_val=-255, q_levels=None):
    """Quantize prediction error."""
    q_error = np.floor((error - min_val) / step).astype(np.int16)
    q_error = np.clip(q_error, 0, q_levels - 1)  # Ensure within range
    return q_error


def dequantize(q_error, step, min_val=-255):
    """Convert quantized values back to integers."""
    deq = ((q_error + 0.5) * step + min_val).astype(np.int16)      # midpoint reconstruction
    return deq


##############################################
#        IMAGE RECONSTRUCTION
##############################################

def reconstruct_image(predicted, dequantized):
    """Reconstruct image from predicted + dequantized error."""
    reconstructed = predicted.astype(np.int16) + dequantized.astype(np.int16)
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    return reconstructed


##############################################
#         COMPRESSION / DECOMPRESSION
##############################################

def compress_image():
    """
    - Get user inputs
    - Load image
    - Apply predictor
    - Compute error
    - Quantize error
    - Dequantize
    - Reconstruct
    - Display 6 images
    - Calculate compression ratio
    """
    pass


def decompress_image():
    """Reverse steps for decompression."""
    pass


##############################################
#          DISPLAY REQUIRED IMAGES
##############################################

def show_results(original, predicted, error, q_error, deq, final):
    """
    Display the six required images:
     1. Original
     2. Predicted
     3. Error
     4. Quantized error
     5. De-quantized error
     6. Decompressed image
    """
    pass


##############################################
#          COMPRESSION RATIO
##############################################

def compute_compression_ratio(original_bits, compressed_bits):
    """Return compression ratio."""
    if compressed_bits == 0:
        return float('inf')  # Avoid division by zero
    return original_bits / compressed_bits


##############################################
#               START
##############################################

if __name__ == "__main__":
    main_menu()
