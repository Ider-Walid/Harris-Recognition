# -*-coding:UTF8 -*
#------------------------------------------
# TD3 détection de points caractéristiques
#------------------------------------------
"""
@author: T. Chateau & C. Teulière
"""
import cv2
from matplotlib import pyplot as plt
import numpy as np


#------------------------------------------
# non maxima suppression
def nms(S):
    W1 = np.array([\
        (1, 0, 0),\
        (0, -1, 0),\
        (0, 0, 0)]\
        ,dtype = float)
    C1 = cv2.filter2D(S, cv2.CV_32F, W1)<0 
    W2 = np.array([\
        (0, 1, 0),\
        (0, -1, 0),\
        (0, 0, 0)]\
        ,dtype = float)
    C2 = cv2.filter2D(S, cv2.CV_32F, W2)<0 
    W3 = np.array([\
        (0, 0, 1),\
        (0, -1, 0),\
        (0, 0, 0)]\
        ,dtype = float)
    C3 = cv2.filter2D(S, cv2.CV_32F, W3)<0 
    W4 = np.array([\
        (0, 0, 0),\
        (1, -1, 0),\
        (0, 0, 0)]\
        ,dtype = float)
    C4 = cv2.filter2D(S, cv2.CV_32F, W4)<0 
    W5 = np.array([\
        (0, 0, 0),\
        (0, -1, 1),\
        (0, 0, 0)]\
        ,dtype = float)
    C5 = cv2.filter2D(S, cv2.CV_32F, W5)<0 
    W6 = np.array([\
        (0, 0, 0),\
        (0, -1, 0),\
        (1, 0, 0)]\
        ,dtype = float)
    C6 = cv2.filter2D(S, cv2.CV_32F, W6)<0 
    W7 = np.array([\
        (0, 0, 0),\
        (0, -1, 0),\
        (0, 1, 0)]\
        ,dtype = float)
    C7 = cv2.filter2D(S, cv2.CV_32F, W7)<0 
    W8 = np.array([\
        (0, 0, 0),\
        (0, -1, 0),\
        (0, 0, 1)]\
        ,dtype = float)
    C8 = cv2.filter2D(S, cv2.CV_32F, W8)<0 
    Sf = C1 & C2 & C3 & C4 & C5 & C6 & C7 & C8
    return Sf

###############################################################
# Fonction principale :
def main():
    #------------------------
    # Chargement de l'image
    #------------------------
    if len(sys.argv) > 1:
        filename = "images/"+sys.argv[1]+".png" #Récupère le nom du fichier s'il est donné en argument, exemple: python3 TD2_filtrage.py "nomimage"
    else:
        filename = "images/toy1.jpg" #Image par défaut

    im = cv2.imread(filename)
    #Conversion en niveaux de gris
    img=cv2.cvtColor( im, cv2.COLOR_BGR2GRAY )

    #------------------------
    # Calcul des gradients :
    #------------------------
    # TO BE COMPLETED
    # Définition des noyaux
    #Wgx = 
    #Wgy =
    # Gradients selon x et y :
    #Ix =
    #Iy =

    #---------------------------------
    # Calcul de la matrice de Harris :
    #---------------------------------
    #TO BE COMPLETED
    #Ixx =
    #Iyy =
    #Ixy =

    #Filtrage :
    #Wf = np.ones((5,5))/25

    #----------------------------
    # Calcul du score de Harris :
    #----------------------------
    # 
    # Calcul de la trace et du det
    # TO BE COMPLETED
    # Calcul du score de Harris 
    # TO BE COMPLETED
    # S = 

    #--------------------------
    # Seuillage et nms :
    #--------------------------
    # Seuillage par rapport à une fraction du score max
    #r = 0.01
    #mask = S>r*np.max(S.flatten())
    #S2 = 
    # ou seuillage selon un seuil fixe
    #th = 300000
    #mask = S>th
    #S2 = 

    print("score max",np.max(S.flatten()))
    #non maxima suppression
    S3 = nms(S2)
    # récupération des coordonnées
    ptsl, ptsc  = np.nonzero(S3)

    #------------------------
    # Affichage du résultat : 
    #------------------------ 
    # image
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    #plt.subplot(321) 
    plt.imshow(img, 'gray',vmin=0, vmax=255)
    plt.title('original image')
    plt.plot(ptsc,ptsl,'+r')
    #plt.subplot(222) 
    #plt.imshow(S)
    #plt.title('Harris score')
    #plt.subplot(223) 
    #plt.imshow(S2)
    #plt.title('seuillage')
    #plt.subplot(224) 
    #plt.imshow(S3)
    #plt.title('Après nms')
    plt.show()
 
if __name__ == "__main__":
    main()
