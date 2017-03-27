# facerecog
A Python implementation of Face Recognition using Sparse Representation from the original paper:

John Wright et al, "Robust Face Recognition via Sparse Representation", PAMI 2009.

## What did we do ?

In this work we basically did :
* Implement in Python one version the John Wright's paper : 
	* Relaxed version of the classification algorithm using sparse representation (algorithm 1 with tolerance p.214 of the paper)
	* We applied the algorithm in the following situations : basic images, noisy images (with gaussian noise), and occlusion
* Implement PCA + SVM algorithm in order to compare the results in the same situations.
* Try CNN on the same database


## Datasets used

* Extended Yale B Database
 * 16128 images of 28 human subjects under 9 poses and 64 illumination conditions.
 * http://vision.ucsd.edu/~leekc/ExtYaleDatabase/download.html
* AR Face Database
 * 126 people (over 4,000 color images).
 * http://www2.ece.ohio-state.edu/~aleix/ARdatabase.html (Note : download les 10 CD)
* A quelle heure on se fait un database de nous pour l'application ? Différentes luminosités / faces.

## Notebooks & Docs

* FaceRecogTool : is a matlab implementation of the paper with the database of the article
* csv_saved 
* impFaceRecognition-master :is a matlab implementation of the paper all the .m files.
* presentation : the repository for presentation files
* Add_Noise : notebook where we implement the functions dedicated to noise the image
* Add_Noise_py2 : same notebook but for a python 2 implementation (problem with class build in python 3 used for python 2)
* Face_PCA_SVM : Implementation of PCA + SVM on occluded image
* VGG16Face : CNN work on the database
* fetures_reduction : implement eigenfaces,fisherfaces and randomfaces for features reduction
* implementation_basique_yale : the python implementation of the paper


# arborescence

.
+-- _README.md
+-- _carnet_bord.md
+-- _implementation_basique_yale.ipynb
+-- _util_facerecog.py >> Useless
+-- _util_facerecog.pyc >> Useless
+-- _data
|   +-- _CroppedYale
|       +-- _...
|   +-- _resources
|       +-- _tuto_facerec_python.pdf
+-- _data
|   +-- footer.html
|   +-- header.html
+-- _FaceRecogTool
+-- _impFaceRecognition-master

> Répertoires où l'on trouve les différentes images

## Liens utiles

* http://www.bytefish.de/pdf/facerec_python.pdf > Explique comment manipuler des images .pgm avec *python* et faire de la reconnaissance faciale (Eigenfaces, Fisherfaces ... etc )
* http://cvxopt.org/examples/mlbook/l1.html?highlight=l1#l1 > Librairie avec minimisation L1, à voir si c'est performant ! > Inutile en fait ...
* https://github.com/kastur/ECGCS/blob/master/matlab/l1magic-1.1/Optimization/l1eq_pd.m > Résolution de l'optimisation de notre problème, en matlab ...
* https://en.wikipedia.org/wiki/Basis_pursuit_denoising > Explication du problème d'optimisation à résoudre (BPD). On y retrouve notre fonction à minimiser.
* https://fr.slideshare.net/gpeyre/signal-processing-course-sparse-regularization-of-inverse-problems > Slides sur les problèmes inverses, intéressant pour comprendre la théorie derrière la minimisation (mais pas grand chose à voir avec notre sujet ...)
* https://web.stanford.edu/class/ee364b/projects/2015projects/reports/pinto_report.pdf > 

## Todo

* Améliorer l'optimisayion du problème : 
  * on fait du PBDN, faut-il faire du PB tout court (méthode incrémentale) ?
  * Trouver le meilleur lambda (alpha dans le code). Métrique : SCI ? cf article
* Faire l'étude avec différentes méthodes de réduction de dimension + différentes méthodes d'occlusion. Comparer les résultats.
* Implémenter la méthode PCA + SVM et observer les résultats => Les comparer avec nos résultats classiques
* S'inspirer des notebooks envoyés par Chainais pour les représentations de nos résultats (Inverse problems)


* Comparaison de l'algorithme SRC en fonction des différentes dimensions. Faire de même pour SVM
* Comparaison de SRC et SVM pour différents niveaux de bruitage
* Oocclusion avec un carré noir + Visualisation de l'erreur (résidu ?)
* Outlier rejection
* Partition de l'image
* Normaliser