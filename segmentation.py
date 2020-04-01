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
    # tableau comptant le nombre de voisins deja dans le groupe
    neightbours_in_groupe = np.zeros((img.shape[0] * img.shape[1], nb_points+1))

    for i, (x, y) in zip(range(1, nb_points+1), starting_points):
        # print(x, " / ", y)
        print(i)
        tab[x, y] = i
        groupes.append([(x, y)])
        avg_groupes_color[i] = img[x, y]

    check = 0
    check2 = 0

    for i in tqdm(range(300)):
        check = 0
        check2 = 0
        for j in range(1, nb_points+1):
            # check = 0
            # check2 = 0
            pts = copy.copy(groupes[j])
            # print("==========================")
            # print(len(pts))
            # print()
            for ptx, pty in pts:
                # print("--------------------")
                # print(len(pts))
                # print(neightbours_in_groupe[ptx * img.shape[1] + pty, j])
                if neightbours_in_groupe[ptx * img.shape[1] + pty, j] < 4:
                    for (diff_x, diff_y) in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
                        v_ptx = ptx + diff_x
                        v_pty = pty + diff_y
                        # check += 1

                        if 0 <= v_ptx < img.shape[0] and 0 <= v_pty < img.shape[1]:
                            # si le point n'appartient a aucun groupe
                            if tab[v_ptx, v_pty] == 0:
                                if test_groupes[v_ptx * img.shape[1] + v_pty, j-1] == 0:
                                    test_groupes[v_ptx * v_pty, j-1] = 1
                                    v_img = img[v_ptx, v_pty]

                                    dist = np.linalg.norm(v_img - avg_groupes_color[j])

                                    if dist < 50:
                                        # print("add")
                                        # print(" ", len(groupes[j]))
                                        tab[v_ptx, v_pty] = j
                                        groupes[j].append((v_ptx, v_pty))
                                        # print(" ", len(groupes[j]))
                                        # print(avg_groupes_color[j])
                                        avg_groupes_color[j] = (avg_groupes_color[j] * len(groupes[j]) + v_img) / (len(groupes[j]) + 1)
                                        neightbours_in_groupe[ptx * img.shape[1] + pty, j] += 1
                                        # print(avg_groupes_color[j])
                            elif tab[v_ptx, v_pty] == j:
                                neightbours_in_groupe[ptx * img.shape[1] + pty, j] += 1
                else:
                    pass
                    # check2 += 1
            # print("check: ", check, " / ", check2)
        # print("check: ", check, " / ", check2)

    for j in range(1, nb_points+1):
        pts = groupes[j]
        print("====================================")
        print(j)
        print(len(pts))
        for (ptx, pty) in pts:
            # print(ptx, " / ", pty)
            pass

    image_segmente = np.zeros(img.shape[:])
    # print()

    for j in range(1, nb_points+1):
        # print(avg_groupes_color[j, 0])
        pts = groupes[j]
        for ptx, pty in pts:
            image_segmente[ptx, pty, 0] = avg_groupes_color[j, 0]
            image_segmente[ptx, pty, 1] = avg_groupes_color[j, 1]
            image_segmente[ptx, pty, 2] = avg_groupes_color[j, 2]
            # print(image_segmente[ptx, pty, 0])

    image_segmente = np.round(image_segmente)
    image_segmente = image_segmente.astype('int')
    # print(image_segmente)

    return image_segmente

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
    img = cv.imread('simple_human.jpg')
    img = smoothing(img)

    print(img.shape)

    tab = segmentation(img, 10)
    print(type(tab))

    # plt.imshow(tab)
    # plt.show()


if __name__ == "__main__":
    main()
