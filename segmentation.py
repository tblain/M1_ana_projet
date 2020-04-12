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

# ===== segmentation ===========================================

def segmentation(img, nb_points):

    # select random starting points
    germes = get_germes_regular(img, nb_points)
    print("============================")
    print("GROWING")
    # print(starting_points)
    # print(img.shape)

    # tableau contenant pour chaque point l'index du groupe auquel il appartient ou alors 0
    tab = np.zeros(img.shape[:2], dtype='int')

    groupes = [[]]

    # tableau qui contient pour chaque groupe la couleur moyenne
    avg_grp_col = np.zeros((nb_points+1, 3))
    # pile qui contiendra les cases ajoutes a un groupe et qui n'ont pas encore ete traitees
    stack = []

    # palier a partir duquel la difference entre 2 couleurs est trop elevee
    thresh = 30

    nb_in_a_groupe = 0
    nb_pixel = img.shape[0] * img.shape[1]
    nb_in_groupes = np.zeros(nb_points+1)

    # on parcourt les germes pour remplir les differents tableaux
    for i, (x, y) in zip(range(1, nb_points+1), germes):
        tab[x, y] = i
        groupes.append([(x, y)])
        avg_grp_col[i] = img[x, y]
        stack.append((x, y))

    max_step = 1000000
    step = 0

    while len(stack) > 0 and step < max_step:
        if step % 200 == 0:
            print(nb_in_a_groupe / nb_pixel * 100, end="\r")

        ptx, pty = stack.pop(0)

        # groupe du point
        g = tab[ptx, pty]

        # on parcours les voisins du point
        for (diff_x, diff_y) in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            v_ptx = ptx + diff_x
            v_pty = pty + diff_y

            # on verifie que le point est bien dans l'image
            if 0 <= v_ptx < img.shape[0] and 0 <= v_pty < img.shape[1]:
                # groupe du voisin
                v_g = tab[v_ptx, v_pty]

                # si le point n'appartient a aucune case
                if v_g == 0:
                    v_img = img[v_ptx, v_pty]

                    # si la couleur du point est similaire a celle du groupe
                    if distance(v_img, avg_grp_col[g]) < thresh:
                        tab[v_ptx, v_pty] = g

                        # on met a jour les infos du groupe
                        avg_grp_col[g] = (avg_grp_col[g] * nb_in_groupes[g] + v_img) / (nb_in_groupes[g] + 1)
                        nb_in_groupes[g] += 1

                        nb_in_a_groupe += 1

                        stack.append((v_ptx, v_pty))
                elif v_g != g:  # on tente de merge les groupes de ces 2 cases adjacentes
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


    image_segmente = np.zeros(img.shape[:])

    # reconstruction d'une image avec la couleur de chaque point etant la couleur du groupe auquel il appartient
    for j in tqdm(range(1, nb_points+1)):
        groupe = np.where(tab == j)
        for ptx, pty in zip(groupe[0], groupe[1]):
            image_segmente[ptx, pty, 0] = avg_grp_col[j, 0]
            image_segmente[ptx, pty, 1] = avg_grp_col[j, 1]
            image_segmente[ptx, pty, 2] = avg_grp_col[j, 2]

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
    return np.linalg.norm(a - b)

# ===== main ===================================================

def main():
    img_bgr = cv.imread('simple_human.jpg')

    # on convertit l'image en rgb
    img = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

    img = smoothing(img)

    plt.imshow(img)
    plt.show()

    tab = segmentation(img, 1000)

    plt.imshow(tab)
    plt.show()


if __name__ == "__main__":
    main()
