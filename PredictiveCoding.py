import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

##############################################
# MAIN MENU
##############################################
def main_menu():
    print("\n============ Predictive Coding Menu ============")
    print("1. Compress Image (Grayscale or RGB)")
    print("2. Decompress Image")
    print("3. Exit")
    print("==================================================\n")

    while True:
        try:
            choice = int(input("Enter your choice (1-3): "))
            if choice == 1:
                compress_image()
            elif choice == 2:
                decompress_image()
            elif choice == 3:
                print("Exiting the program. Goodbye!")
                break
        except ValueError:
            print("Invalid input. Please enter a number.")

##############################################
# USER INPUT
##############################################
def get_image_path():
    path = input("Enter the image path: ").strip()
    if not path:
        raise ValueError("Image path cannot be empty.")
    return path

def get_quantization_bits():
    while True:
        bits_input = input("Enter number of bits for quantization (default 2): ").strip()
        if not bits_input:
            return 2
        try:
            bits = int(bits_input)
            if 1 <= bits <= 8:
                return bits
            print("Please enter a number between 1 and 8.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

##############################################
# LOAD / SAVE IMAGE
##############################################
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return np.array(img)

def save_image(array, filename):
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    Image.fromarray(array.astype(np.uint8)).save(filename)

##############################################
# 2D ADAPTIVE PREDICTOR
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
    if image.ndim == 2:
        return _predictor_2d_channel(image)
    pred = np.zeros_like(image, dtype=np.int16)
    for c in range(image.shape[2]):
        pred[:, :, c] = _predictor_2d_channel(image[:, :, c])
    return pred

def _predictor_2d_channel(channel):
    img = channel.astype(np.int16)
    h, w = img.shape
    pred = np.zeros_like(img, dtype=np.int16)
    for i in range(h):
        for j in range(w):
            if i == 0 and j == 0:
                pred[i,j] = img[i,j]
                continue
            A = img[i,j-1] if j > 0 else img[i,j]
            C = img[i-1,j] if i > 0 else img[i,j]
            B = img[i-1,j-1] if i > 0 and j > 0 else img[i,j]
            if B <= min(A, C):
                P = max(A, C)
            elif B >= max(A, C):
                P = min(A, C)
            else:
                P = A + C - B
            pred[i,j] = P
    return pred

##############################################
# ERROR & QUANTIZATION
##############################################
def compute_error(original, predicted):
    return original.astype(np.int16) - predicted.astype(np.int16)

def compute_quantization_params(bits):
    q_levels = 2 ** bits
    step = 510.0 / q_levels
    return step, q_levels

def quantize_error(error, step, q_levels):
    return np.clip(np.floor((error + 255) / step), 0, q_levels-1).astype(np.int16)

def dequantize(q_error, step):
    return ((q_error + 0.5) * step - 255).astype(np.int16)

##############################################
# RECONSTRUCTION
##############################################
def reconstruct_image(predicted, dequantized_error):
    return np.clip(predicted.astype(np.int16) + dequantized_error, 0, 255).astype(np.uint8)

##############################################
# DISPLAY - FIXED FOR NUMPY 2.0+
##############################################
def show_results(original, predicted, error, q_error, deq, reconstructed):
    def to_rgb(arr):
        return np.stack([arr]*3, axis=-1) if arr.ndim == 2 else arr

    def error_vis(e):
        e = e.astype(float)
        mn, mx = e.min(), e.max()
        if mx == mn:
            return np.zeros_like(e, dtype=np.uint8)
        return ((e - mn) / (mx - mn) * 255).astype(np.uint8)

    titles = ["Original", "Predicted", "Error", "Quantized Error", "Dequantized Error", "Reconstructed"]
    imgs = [original, predicted, error, q_error, deq, reconstructed]

    plt.figure(figsize=(15, 10))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        if i >= 2 and i <= 4:  # error images
            vis = error_vis(imgs[i])
            plt.imshow(to_rgb(vis))
        else:
            plt.imshow(to_rgb(imgs[i]))
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

##############################################
# COMPRESS & DECOMPRESS
##############################################
def compress_image():
    print("\n----- Compressing Image -----")
    path = get_image_path()
    bits = get_quantization_bits()

    original = load_image(path)
    is_color = original.ndim == 3
    print(f"Image shape: {original.shape} → {'RGB' if is_color else 'Grayscale'}")

    predicted = predictor_2d(original)
    error = compute_error(original, predicted)

    step, q_levels = compute_quantization_params(bits)

    # Quantize
    if is_color:
        q_error = np.zeros_like(error, dtype=np.int16)
        for c in range(3):
            q_error[:,:,c] = quantize_error(error[:,:,c], step, q_levels)
    else:
        q_error = quantize_error(error, step, q_levels)

    # Dequantize for display
    if is_color:
        deq = np.zeros_like(q_error, dtype=np.int16)
        for c in range(3):
            deq[:,:,c] = dequantize(q_error[:,:,c], step)
    else:
        deq = dequantize(q_error, step)

    reconstructed = reconstruct_image(predicted, deq)

    np.savez("compressed_output.npz",
             q_error=q_error, predicted=predicted, bits=bits,
             shape=original.shape, is_color=is_color)

    ratio = (original.size * 8) / (q_error.size * bits)
    print(f"Compressed file → compressed_output.npz")
    print(f"Compression Ratio = {ratio:.2f} : 1")

    show_results(original, predicted, error, q_error, deq, reconstructed)

def decompress_image():
    print("\n----- Decompressing Image -----")
    try:
        data = np.load("compressed_output.npz")
    except FileNotFoundError:
        print("No compressed file found!")
        return

    q_error = data["q_error"]
    predicted = data["predicted"]
    bits = int(data["bits"])
    is_color = data.get("is_color", False)

    step, _ = compute_quantization_params(bits)

    if is_color:
        deq = np.zeros_like(q_error, dtype=np.int16)
        for c in range(3):
            deq[:,:,c] = dequantize(q_error[:,:,c], step)
    else:
        deq = dequantize(q_error, step)

    final = reconstruct_image(predicted, deq)
    save_image(final, "decompressed_image.png")
    print("Decompressed image saved → decompressed_image.png")
    show_results(final, predicted, final-predicted.astype(np.int16), q_error, deq, final)

##############################################
# START
##############################################
if __name__ == "__main__":
    main_menu()