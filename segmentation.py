import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
import math
import copy
from scipy.signal import convolve2d
import bottleneck as bn
from scipy.ndimage import convolve1d

# ===== get starting points ====================================

def get_starting_points(img, nb_points):
    histr = cv.calcHist([img],[0],None,[256],[0,256])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    plt.imshow(gray, cmap='gray')
    plt.show()

    plt.plot(histr)
    plt.show()

    k = 9
    d = k // 2

    # z = bn.partition(-ravel, nb_points)[:nb_points]

    starting_points = []
    histr = convolve1d(histr, np.ones(k))
    histr[:10] = 0
    histr[250:] = 0

    for _ in range(nb_points):
        imax = np.argmax(histr)
        print(imax)

        a = max(imax - d, 0)
        b = min(imax + d, 255)
        print((a, b))
        histr[a:b] = 0
        print(histr[255])

        (cx, cy) = np.where(gray == imax)

        starting_points.append((cx[0], cy[0]))

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

def segmentation2(img, nb_points):
    # select random starting points
    starting_points = get_starting_points(img, nb_points)
    print(starting_points)
    print(img.shape)

    tab = np.zeros(img.shape[:2], dtype='int')
    groupes = [[]]
    avg_groupes_color = np.zeros((nb_points+1, 3))
    test_groupes = np.zeros((img.shape[0] * img.shape[1], nb_points+1))
    # tableau comptant le nombre de voisins deja dans le groupe
    neightbours_in_groupe = np.zeros((img.shape[0] * img.shape[1], nb_points+1))

    for i, (x, y) in zip(range(1, nb_points+1), starting_points):
        tab[x, y] = i
        groupes.append([(x, y)])
        avg_groupes_color[i] = img[x, y]

    for i in tqdm(range(1000)):
        # TODO: faire une convo pour voir les cells ayant des voisins dans un groupe
        kernel = np.array([[0, 1, 0],
                          [1, -100, 1],
                          [0, 1, 0]])
        no_groupes = convolve2d(tab, kernel, "same")
        no_groupes = np.where(no_groupes > 0)

        for ptx, pty in zip(no_groupes[0], no_groupes[1]):
            for (diff_x, diff_y) in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
                v_ptx = ptx + diff_x
                v_pty = pty + diff_y

                if 0 <= v_ptx < img.shape[0] and 0 <= v_pty < img.shape[1]:
                    g_v_pt = tab[v_ptx, v_pty]
                    # print(g_v_pt)

                    if g_v_pt > 0:
                        p_img = img[ptx, pty]

                        # print(avg_groupes_color[g_v_pt])
                        dist = np.linalg.norm(p_img - avg_groupes_color[g_v_pt])

                        if dist < 50:
                            tab[v_ptx, v_pty] = g_v_pt
                            groupes[g_v_pt].append((v_ptx, v_pty))
                            avg_groupes_color[g_v_pt] = np.round(avg_groupes_color[g_v_pt] * len(groupes[g_v_pt]) + p_img) / (len(groupes[g_v_pt]) + 1)
                            # neightbours_in_groupe[ptx * img.shape[1] + pty, g_v_pt] += 1

    # for i in tqdm(range(300)):
        # for j in range(1, nb_points+1):
            # pts = copy.copy(groupes[j])
            # for ptx, pty in pts:
                # if neightbours_in_groupe[ptx * img.shape[1] + pty, j] < 4:
                    # for (diff_x, diff_y) in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
                        # v_ptx = ptx + diff_x
                        # v_pty = pty + diff_y

                        # if 0 <= v_ptx < img.shape[0] and 0 <= v_pty < img.shape[1]:
                            # if tab[v_ptx, v_pty] == 0:
                                # if test_groupes[v_ptx * img.shape[1] + v_pty, j-1] == 0:
                                    # test_groupes[v_ptx * v_pty, j-1] = 1
                                    # v_img = img[v_ptx, v_pty]

                                    # dist = np.linalg.norm(v_img - avg_groupes_color[j])

                                    # if dist < 50:
                                        # tab[v_ptx, v_pty] = j
                                        # groupes[j].append((v_ptx, v_pty))
                                        # avg_groupes_color[j] = (avg_groupes_color[j] * len(groupes[j]) + v_img) / (len(groupes[j]) + 1)
                                        # neightbours_in_groupe[ptx * img.shape[1] + pty, j] += 1
                            # elif tab[v_ptx, v_pty] == j:
                                # neightbours_in_groupe[ptx * img.shape[1] + pty, j] += 1

    for j in range(1, nb_points+1):
        pts = groupes[j]
        print("====================================")
        print(j)
        print(len(pts))


    image_segmente = np.zeros(img.shape[:])

    for j in range(1, nb_points+1):
        pts = groupes[j]
        print(avg_groupes_color[ptx, pty])
        for ptx, pty in pts:
            image_segmente[ptx, pty, 0] = avg_groupes_color[j, 0]
            image_segmente[ptx, pty, 1] = avg_groupes_color[j, 1]
            image_segmente[ptx, pty, 2] = avg_groupes_color[j, 2]

    image_segmente = np.round(image_segmente)
    image_segmente = image_segmente.astype('int')

    return image_segmente


def segmentation3(img, nb_points):
    # select random starting points
    starting_points = get_starting_points(img, nb_points)
    print(starting_points)
    # print(img.shape)

    tab = np.zeros(img.shape[:2], dtype='int')
    groupes = [[]]
    nb_in_a_groupe = 0
    nb_pixel = img.shape[0] * img.shape[1]
    avg_groupes_color = np.zeros((nb_points+1, 3))
    stack = []
    nb_in_groupes = np.zeros(nb_points+1)

    for i, (x, y) in zip(range(1, nb_points+1), starting_points):
        tab[x, y] = i
        groupes.append([(x, y)])
        avg_groupes_color[i] = img[x, y]
        stack.append((x, y))

    max_step = 100000000
    step = 0

    while len(stack) > 0 and step < max_step:
        if step % 100 == 0:
            print(nb_in_a_groupe / nb_pixel * 100, end="\r")
        ptx, pty = stack.pop(0)
        # groupe du point
        g = tab[ptx, pty]

        for (diff_x, diff_y) in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            v_ptx = ptx + diff_x
            v_pty = pty + diff_y

            if 0 <= v_ptx < img.shape[0] and 0 <= v_pty < img.shape[1]:
                if tab[v_ptx, v_pty] == 0:
                    v_img = img[v_ptx, v_pty]
                    dist = np.linalg.norm(v_img - avg_groupes_color[g])

                    if dist < 70:
                        tab[v_ptx, v_pty] = g
                        # avg_groupes_color[g] = (avg_groupes_color[g] * nb_in_groupes[g] + v_img) / (nb_in_groupes[g] + 1)
                        nb_in_groupes[g] += 1
                        stack.append((v_ptx, v_pty))
                        nb_in_a_groupe += 1

        step += 1

    # for j in range(1, nb_points+1):
        # pts = groupes[j]
        # print("====================================")
        # print(j)
        # print(len(pts))

    # no_groupes = np.where(tab == 0)
    # for ptx, pty in zip(no_groupes[0], no_groupes[1]):
        # dist_array = np.zeros(nb_points+1)
        # dist_array[0] = 50

        # for g in range(1, nb_points+1):
            # v_img = img[ptx, pty]
            # dist_array[g] = np.linalg.norm(v_img - avg_groupes_color[g])

        # imin = np.argmin(dist_array)
        # if imin > 0:
            # tab[v_ptx, v_pty] = imin
            # groupes[imin].append((v_ptx, v_pty))


    image_segmente = np.zeros(img.shape[:])

    for j in tqdm(range(1, nb_points+1)):
        no_groupes = np.where(tab == j)
        for ptx, pty in zip(no_groupes[0], no_groupes[1]):
            image_segmente[ptx, pty, 0] = avg_groupes_color[j, 0]
            image_segmente[ptx, pty, 1] = avg_groupes_color[j, 1]
            image_segmente[ptx, pty, 2] = avg_groupes_color[j, 2]

    image_segmente = np.round(image_segmente)
    image_segmente = image_segmente.astype('int')

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

    tab = segmentation3(img, 60)

    plt.imshow(tab)
    plt.show()


if __name__ == "__main__":
    main()
