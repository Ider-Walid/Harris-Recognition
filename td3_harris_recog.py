# -*-coding:UTF8 -*
#------------------------------------------
# TD3 descripteurs
#------------------------------------------
"""
@author: T. Chateau & C. Teulière
"""
import cv2
from matplotlib import pyplot as plt
import numpy as np

###############################################################
def main():
    #------------------------
    # Chargement de l'image
    #------------------------
    filenameref = "toys.jpg"
    filenametest = "toy_rot.jpg"
    imgrefcol = cv2.imread(filenameref)
    imgtestcol = cv2.imread(filenametest)
    #Conversion en niveaux de gris
    imgref=cv2.cvtColor( imgrefcol, cv2.COLOR_BGR2GRAY )
    imgtest=cv2.cvtColor( imgtestcol, cv2.COLOR_BGR2GRAY )

    #------------------------------------
    # Calcul des points et descripteurs :
    #------------------------------------
    # Calcul des points d'intérêt avec des fonctions OpenCV
    #harris_ref = cv2.goodFeaturesToTrack(imgref,maxCorners = 300, qualityLevel = 0.01, minDistance = 4, useHarrisDetector=True, k=0.04)
    #kp_ref_harris = [cv2.KeyPoint(x[0,0], x[0,1], 1) for x in harris_ref]
    
    alg = cv2.SIFT_create()    
    #kp_ref, ds_ref = alg.compute(imgref, kp_ref_harris) 
    kpr, dr=alg.detectAndCompute(imgref,None)
    kpt, dt=alg.detectAndCompute(imgtest,None)
    imgrefcol = cv2.drawKeypoints(imgrefcol, kpr, imrefRGB, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    imgtestcol = cv2.drawKeypoints(imgtestcol, kpt, imrefRGB, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #------------
    # Matching :
    #------------
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descripteurs.
    matches = bf.match(dr,dt)

    # Classement des appariements en fonction de la distance.
    matches = sorted(matches, key = lambda x:x.distance)

    #------------------------
    # Affichage du résultat : 
    #------------------------ 
    # Affichage des 10 premiers match.
    imgres = cv2.drawMatches(imgrefcol,kpr,imgtestcol,kpt,matches[:10], None, flags=2) 
    # image
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.imshow(imgres)
    plt.show()
 
if __name__ == "__main__":
    main()

