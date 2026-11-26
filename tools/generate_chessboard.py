"""
generate_chessboard.py

Generates a printable chessboard calibration pattern (10x7 squares =
9x6 internal corners, matching the default in calibrate_camera())
sized to fit exactly on an A4 page, and saves it as a high-DPI PNG
ready to print at 100% scale (no "fit to page" scaling, which would
distort the real square size).

Requirements:
    pip install pillow
"""

from PIL import Image
import math

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

SQUARES_X = 10        # number of squares across (9 internal corners)
SQUARES_Y = 7         # number of squares down (6 internal corners)
SQUARE_SIZE_MM = 20   # real-world size of each square's edge, in mm

DPI = 300             # print resolution — 300 is standard high-quality print
OUTPUT_FILE = "chessboard_A4.png"

A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297

# ----------------------------------------------------------------------

def mm_to_px(mm, dpi):
    return int(round(mm / 25.4 * dpi))

def main():
    board_width_mm = SQUARES_X * SQUARE_SIZE_MM
    board_height_mm = SQUARES_Y * SQUARE_SIZE_MM

    if board_width_mm > A4_WIDTH_MM or board_height_mm > A4_HEIGHT_MM:
        raise ValueError(
            f"Board size {board_width_mm}x{board_height_mm}mm does not fit "
            f"on A4 ({A4_WIDTH_MM}x{A4_HEIGHT_MM}mm). Reduce SQUARE_SIZE_MM "
            f"or SQUARES_X/Y."
        )

    page_w_px = mm_to_px(A4_WIDTH_MM, DPI)
    page_h_px = mm_to_px(A4_HEIGHT_MM, DPI)
    square_px = mm_to_px(SQUARE_SIZE_MM, DPI)

    board_w_px = SQUARES_X * square_px
    board_h_px = SQUARES_Y * square_px

    # Center the board on the page.
    offset_x = (page_w_px - board_w_px) // 2
    offset_y = (page_h_px - board_h_px) // 2

    img = Image.new("L", (page_w_px, page_h_px), color=255)  # white page

    for row in range(SQUARES_Y):
        for col in range(SQUARES_X):
            if (row + col) % 2 == 0:
                x0 = offset_x + col * square_px
                y0 = offset_y + row * square_px
                x1 = x0 + square_px
                y1 = y0 + square_px
                for y in range(y0, y1):
                    for x in range(x0, x1):
                        img.putpixel((x, y), 0)  # black square

    img.save(OUTPUT_FILE, dpi=(DPI, DPI))

    print(f"Saved {OUTPUT_FILE}")
    print(f"Board: {SQUARES_X}x{SQUARES_Y} squares "
          f"({SQUARES_X - 1}x{SQUARES_Y - 1} internal corners)")
    print(f"Square size: {SQUARE_SIZE_MM}mm (use this as square_size_m="
          f"{SQUARE_SIZE_MM / 1000} in calibrate_camera())")
    print(f"Board physical size: {board_w_px / DPI * 25.4:.1f}mm x "
          f"{board_h_px / DPI * 25.4:.1f}mm")
    print("IMPORTANT: print at 100% scale / 'actual size' — "
          "do NOT use 'fit to page' or 'shrink to fit', or the real "
          "square size will no longer match SQUARE_SIZE_MM.")

if __name__ == "__main__":
    main()
 