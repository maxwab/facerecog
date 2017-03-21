# facerecog
A Python implementation of Face Recognition using Sparse Representation from the original paper:

John Wright et al, "Robust Face Recognition via Sparse Representation", PAMI 2009.

## Datasets utilisés

* Extended Yale B Database
** 16128 images of 28 human subjects under 9 poses and 64 illumination conditions.
** http://vision.ucsd.edu/~leekc/ExtYaleDatabase/download.html
* AR Face Database
** 126 people (over 4,000 color images).
** http://www2.ece.ohio-state.edu/~aleix/ARdatabase.html (Note : download les 10 CD)
* A quelle heure on se fait un database de nous pour l'application ? Différentes luminosités / faces.

## Organisation des dossiers

* /
fichiers de base (.py)

* /data/
./yale/
./ar/
./perso/

> Répertoires où l'on trouve les différentes images

## Liens utiles

* http://www.bytefish.de/pdf/facerec_python.pdf > Explique comment manipuler des images .pgm avec *python* et faire de la reconnaissance faciale (Eigenfaces, Fisherfaces ... etc )
* http://cvxopt.org/examples/mlbook/l1.html?highlight=l1#l1 > Librairie avec minimisation L1, à voir si c'est performant !
.nggj,v,,,,

