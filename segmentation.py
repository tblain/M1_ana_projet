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
from collections import deque

# ===== get germes ====================================

def get_germes_histo(img, nb_points):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    histr = cv.calcHist([gray],[0],None,[256],[0,256])

    plt.imshow(gray, cmap='gray')
    plt.show()

    plt.plot(histr)
    plt.show()

    k = 15
    d = k // 2

    # z = bn.partition(-ravel, nb_points)[:nb_points]

    starting_points = []
    histr = convolve1d(histr, np.ones(k))
    histr[:10] = 0
    histr[250:] = 0

    for i in range(nb_points):
        print("------")
        imax = np.argmax(histr)
        print(imax)

        a = max(imax - 1, 0)
        b = min(imax + 1, 255)
        print((a, b))
        # histr[a:b] = np.log(histr[a:b])
        histr[a:b] /= 2
        print("[a:b]: ", histr[a:b])

        (cx, cy) = np.where(gray == imax)

        starting_points.append((cx[0], cy[0]))

    return starting_points

def get_germes_regular(img, nb_points):
    """
    place des germes a intervalles regulieres
    """
    print("============================")
    print("Seed selection")

    germes = []
    dist_between_pts = img.shape[0] * img.shape[1] // nb_points

    for i in range(nb_points):
        # on ajoute une petite variation aleatoire
        diff_x = random.randint(-5, 5)
        diff_y = random.randint(-5, 5)

        n = dist_between_pts * i

        sx = n % img.shape[0] + diff_x
        sy = n // img.shape[0] + diff_y

        sx = max(min(sx, img.shape[0]-1), 0)
        sy = max(min(sy, img.shape[1]-1), 0)

        germes.append((sx, sy))

    return germes

def init_segmentation(img, nb_points):
    # select random starting points
    germes = get_germes_regular(img, nb_points)
    print("============================")
    print("INIT GROWING")
    # print(starting_points)
    # print(img.shape)

    # tableau contenant pour chaque point l'index du groupe auquel il appartient ou alors 0
    tab = np.zeros(img.shape[:2], dtype='int')

    groupes = [[]]
    checked = np.zeros([img.shape[0], img.shape[0], nb_points+1], dtype='int')

    # tableau qui contient pour chaque groupe la couleur moyenne
    avg_grp_col = np.zeros((nb_points+1, 3))
    # pile qui contiendra les cases ajoutes a un groupe et qui n'ont pas encore ete traitees
    stack = deque()


    nb_in_groupes = np.zeros(nb_points+1)

    # on parcourt les germes pour remplir les differents tableaux
    for i, (x, y) in zip(range(1, nb_points+1), germes):
        tab[x, y] = i
        tab[x, y] = i
        groupes.append([(x, y)])
        avg_grp_col[i] = img[x, y]
        stack.append((x, y))

    return tab, groupes, avg_grp_col, stack, nb_in_groupes, checked

# ===== segmentation ===========================================

def segmentation(img, nb_points):
    nb_pixel = img.shape[0] * img.shape[1]
    nb_in_a_groupe = 0

    # palier a partir duquel la difference entre 2 couleurs est trop elevee
    thresh = 30

    largeur = img.shape[1]
    hauteur = img.shape[0]

    tab, groupes, avg_grp_col, stack, nb_in_groupes, checked = init_segmentation(img, nb_points)

    print("============================")
    print("GROWING")
    print("Nb pixels: ", nb_pixel)

    max_step = 1000000
    step = 0

    while len(stack) > 0:# and step < max_step:
        # print(len(stack), end="\r")
        if step % 2000 == 0:
            print(nb_in_a_groupe / nb_pixel * 100, end="\r")

        ptx, pty = stack.popleft()

        # groupe du point
        g = tab[ptx, pty]

        # on parcours les voisins du point
        for (diff_x, diff_y) in [(0, 1), (1, 0), (-1, 0), (0, -1)]:#, (-1, -1), (1, 1), (-1, 1), (1, -1)]:
            v_ptx = ptx + diff_x
            v_pty = pty + diff_y

            # on verifie que le point est bien dans l'image
            if 0 <= v_ptx < hauteur and 0 <= v_pty < largeur:
                # groupe du voisin
                v_g = tab[v_ptx, v_pty]

                # si le point n'appartient a aucune case
                if v_g == 0:
                    # on verifie que le point n'a pas deja ete check
                    # if checked[v_ptx, v_pty, g] != 0: continue
                    # checked[v_ptx, v_pty, g] = 1

                    v_img = img[v_ptx, v_pty]

                    # si la couleur du point est similaire a celle du groupe
                    if distance(v_img, avg_grp_col[g]) < 1 * thresh:
                        tab[v_ptx, v_pty] = g

                        # on met a jour les infos du groupe
                        if step % 1 == 0:
                            avg_grp_col[g] = (avg_grp_col[g] * nb_in_groupes[g] + v_img) / (nb_in_groupes[g] + 1)
                        nb_in_groupes[g] += 1

                        nb_in_a_groupe += 1

                        stack.append((v_ptx, v_pty))
                elif v_g != g:  # on tente de merge les groupes de ces 2 cases adjacentes
                    if True: #step % 1 == 0:
                            if distance(avg_grp_col[v_g], avg_grp_col[g]) < thresh:
                                tab[tab == v_g] = g

                                # on met a jour la couleur du groupe
                                avg_grp_col[g] = (avg_grp_col[g] * nb_in_groupes[g] + avg_grp_col[v_g] * nb_in_groupes[v_g]) / (nb_in_groupes[g] + nb_in_groupes[v_g])


        step += 1

    # for j in range(1, nb_points+1):
        # pts = groupes[j]
        # print("====================================")
        # print(j)
        # print(len(pts))

    # no_groupes = np.where(tab == 05    # for ptx, pty in zip(no_groupes[0], no_groupes[1]):
        # dist_array = np.zeros(nb_points+1)
        # dist_array[0] = 50

        # for g in range(1, nb_points+1):
            # v_img = img[ptx, pty]
            # dist_array[g] = np.linalg.norm(v_img - avg_grp_col[g])

        # imin = np.argmin(dist_array)
        # if imin > 0:
            # tab[v_ptx, v_pty] = imin
            # groupes[imin].append((v_ptx, v_pty))



    return creation_image_segmente(img, tab, nb_points, avg_grp_col)

def creation_image_segmente(img, tab, nb_points, avg_grp_col):
    image_segmente = np.zeros(img.shape[:])

    # reconstruction d'une image avec la couleur de chaque point etant la couleur du groupe auquel il appartient
    for j in tqdm(range(1, nb_points+1)):
        groupe = np.where(tab == j)
        col = j * (200 / nb_points) + 15
        col0 = random.random() * 150 + 30
        col1 = random.random() * 150 + 30
        col2 = random.random() * 150 + 30
        for ptx, pty in zip(groupe[0], groupe[1]):
            image_segmente[ptx, pty, 0] = col0
            image_segmente[ptx, pty, 1] = col1
            image_segmente[ptx, pty, 2] = col2

    image_segmente = np.round(image_segmente)
    image_segmente = image_segmente.astype('int64')

    return image_segmente

# ===== smoothing ==============================================

def smoothing(img):
    print("========================")
    print("Smoothing")

    # smoothing
    t = 9 # taille du kernel
    kernel = np.ones((t, t), np.float32) / t**2
    dst = cv.filter2D(img, -1, kernel)

    # denoising
    dst = cv.bilateralFilter(dst, 20, 750, 750)

    return dst

# ===== Fonctions utilitaires ==================================

def distance(a, b):
    # return np.sum(np.abs(a-b))
    return np.linalg.norm(a - b)

# ===== main ===================================================

def main():
    img_bgr = cv.imread('simple_human.jpg')

    # on convertit l'image en rgb
    img = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

    # img = smoothing(img)

    # plt.imshow(img)
    # plt.show()

    tab = segmentation(img, 500)

    plt.imshow(tab)
    plt.show()


if __name__ == "__main__":
    main()
