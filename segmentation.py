import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
import math
import copy

# ===== get starting points ====================================

def get_starting_points(img, nb_points):
    starting_points = []
    for i in range(nb_points):
        x = random.randint(0, img.shape[0])
        y = random.randint(0, img.shape[1])
        starting_points.append((x, y))

    return starting_points

# ===== segmentation ===========================================

def segmentation(img, nb_points):
    # select random starting points
    starting_points = get_starting_points(img, nb_points)
    print(starting_points)
    print(img.shape)

    tab = np.zeros(img.shape[:2])
    groupes = [[]]
    avg_groupes_color = np.zeros((nb_points+1, 3))
    test_groupes = np.zeros((img.shape[0] * img.shape[1], nb_points+1))

    for i, (x, y) in zip(range(1, nb_points+1), starting_points):
        # print(x, " / ", y)
        print(i)
        tab[x, y] = i
        groupes.append([(x, y)])
        avg_groupes_color[i] = img[x, y]

    for i in tqdm(range(10)):
        for j in range(1, nb_points+1):
            check = 0
            pts = copy.copy(groupes[j])
            # print("==========================")
            # print(len(pts))
            # print(enumerate(pts))
            # print()
            for ptx, pty in pts:
                # print(len(pts))
                for (diff_x, diff_y) in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
                    v_pt = (ptx + diff_x, pty + diff_y)
                    v_ptx = v_pt[0]
                    v_pty = v_pt[1]
                    check += 1

                    if 0 <= v_ptx < img.shape[0] and 0 <= v_pty < img.shape[1]:
                        # si le point n'appartient a aucun groupe
                        if tab[v_ptx, v_pty] == 0:
                            if test_groupes[v_ptx * v_ptx, j-1] == 0:
                                test_groupes[v_ptx * v_pty, j-1] = 1
                                v_img = img[v_pt]

                                dist = np.linalg.norm(v_img - avg_groupes_color[j])

                                if dist < 40:
                                    # print(" ", len(groupes[j]))
                                    tab[v_pt[0], v_pt[1]] = j
                                    groupes[j].append(v_pt)
                                    # print(avg_groupes_color[j])
                                    avg_groupes_color[j] = (avg_groupes_color[j] * len(groupes[j]) + v_img) / (len(groupes[j]) + 1)
                                    # print(avg_groupes_color[j])
            # print("check: ", check)

    for j in range(1, nb_points+1):
        pts = groupes[j]
        print("====================================")
        print(j)
        print(len(pts))
        for (ptx, pty) in pts:
            # print(ptx, " / ", pty)
            pass

    return tab

# ===== smoothing ==============================================

def smoothing(img):
    # smoothing
    kernel = np.ones((7, 7), np.float32)/49
    dst = cv.filter2D(img, -1, kernel)

    # denoising
    dst = cv.bilateralFilter(dst, 9, 750, 750)

    return dst

# ===== main ===================================================

def main():
    img = cv.imread('bird1.jpg')
    img = smoothing(img)

    print(img.shape)

    tab = segmentation(img, 5)
    plt.imshow(tab)
    plt.show()


if __name__ == "__main__":
    main()
