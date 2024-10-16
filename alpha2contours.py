import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# 1. 透過画像を読み込む処理
def load_image(image_path):
    image = Image.open(image_path).convert("RGBA")
    return np.array(image)

# 2. 透過部分とそうでない部分の境界を線分（座標配列）を抽出する処理
def extract_boundary(image):
    # アルファチャンネルを抽出
    alpha_channel = image[:, :, 3]

    # 透過部分と不透過部分の境界を検出するため、2値化
    thresholded = (alpha_channel > 0).astype(np.uint8) * 255  
    # 自作の輪郭検出関数を使用
    contours = find_contours_custom(thresholded)
    
    return contours

# 2点間の直線上のピクセルを求めるブレゼンハムのアルゴリズム
def bresenham_line(x0, y0, x1, y1):
    """2点間の直線上にあるピクセルの座標を返す"""
    pixels = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        pixels.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return pixels

def find_contours_custom(binary_image):
    """
    二値化された画像のエッジ（輪郭）をピクセル単位で追跡し、輪郭を検出する自作関数。
    
    Parameters:
    - binary_image: 2DのNumPy配列（0または255の値を持つ二値画像）
    
    Returns:
    - contours: すべての輪郭をリストとして返す（各輪郭は座標点のリスト）
    """
    contours = []
    height, width = binary_image.shape
    
    sideEdge = []
    prePixel = 0
    linePixels = bresenham_line(0, 0, width - 1, 0)
    prePixel = binary_image[linePixels[0][0], linePixels[0][1]]
    for x, y in linePixels:
        print(x, y, binary_image[x, y])
        if binary_image[x, y] != prePixel:
            prePixel = binary_image[x, y]
            sideEdge.append((x, y))
    
    
    linePixels = bresenham_line(0, 0, 0, height - 1)
    prePixel = binary_image[linePixels[0][0], linePixels[0][1]]
    for x, y in linePixels:
        print(x, y, binary_image[x, y])
        if binary_image[x, y] != prePixel:
            prePixel = binary_image[x, y]
            sideEdge.append((x, y))

    linePixels = bresenham_line(width - 1, 0, 0, height - 1)
    prePixel = binary_image[linePixels[0][0], linePixels[0][1]]
    for x, y in linePixels:
        print(x, y, binary_image[x, y])
        if binary_image[x, y] != prePixel:
            prePixel = binary_image[x, y]
            sideEdge.append((x, y))
    
    contours = sideEdge

    return contours

# 3. 抽出した線分のXYグラフと画像を同時に表示する処理
def plot_image_and_boundary(image, contours):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # 左側に元の画像を表示
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')  # 軸を非表示
    
    # カラーマップを定義
    cmap = plt.cm.viridis
    num_contours = len(contours)

    # 右側に境界線をプロット（インデックスに基づいて色を変える）
    for i, contour in enumerate(contours):
        contour = contour.squeeze()
        color = cmap(i / num_contours)  # 輪郭のインデックスに応じて色を取得
        
        # 境界線を描画
        ax[1].plot(contour[:, 0], -contour[:, 1], color=color)
        
        # 頂点座標を丸で示す
        ax[1].plot(contour[:, 0], -contour[:, 1], 'o', color=color, markersize=4)
    
    ax[1].set_title("Boundary Plot with Gradual Colors and Vertices")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")
    ax[1].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()

# メイン処理
def main(image_path):
    image = load_image(image_path)
    
    # 境界線を抽出
    contours = extract_boundary(image)
    
    # 画像と境界線のXYグラフを同時に表示
    plot_image_and_boundary(image, contours)

# 画像のパスを指定して実行
image_path = "image.png"
main(image_path)
