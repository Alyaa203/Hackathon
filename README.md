# Hackathon Rock Climbing Robots

Travail de l'équipe pour le hackathon [Rock Climbing Robots](https://rock-climbing-robots.github.io/) : un robot
Microban qui grimpe un mur instrumenté, évalué en simulation (MuJoCo) et sur le vrai robot.

## Contenu

| Dossier | Quoi |
|---|---|
| [`perception/`](perception/) | **Notre code** : perception par marqueurs ArUco (règlement, section 6.1) |
| [`perception/doc/perception_aruco.pdf`](perception/doc/perception_aruco.pdf) | Le document qui explique tout en détail (schémas, commandes, résultats) |
| `simulator/` | Le simulateur des organisateurs ([rock-climbing-robots/simulator](https://github.com/rock-climbing-robots/simulator)), sous-module |
| `microban/` | Le programme du robot ([Rhoban/microban](https://github.com/Rhoban/microban), branche `dev_v2`), sous-module |

### `perception/`

| Fichier | Rôle |
|---|---|
| `perception.py` | Le traitement d'image : détection des marqueurs, pose des caméras (marqueurs 1 à 4 du mur), pose du robot (marqueur 5, triangulation avec les deux caméras, inclinaison par l'IMU), détection des prises (par leur couleur, calées sur la grille de trous). Fonctionne sur des images simulées ou réelles. |
| `robot_marker.py` | Le marqueur ID 5 du robot, comme le donne la fiche des organisateurs (code de 42 mm, autocollant à plat sur le dos, centre en (−33, 0, 54,75) mm dans le repère du tronc), et son ajout au simulateur (`MarkedSim`). |
| `demo.py` | Démonstration dans le simulateur, comparée à la vérité du simulateur, avec une vidéo (`out/demo.mp4`) et une fenêtre en direct (`--show`). |
| `live.py` | Le jour J : les deux caméras USB (ou des images), en 1920 × 1080 : position du robot et prises en direct. |
| `doc/` | Le document LaTeX, son PDF et ses images (`make_figures.py` les régénère). |

Résultats dans le simulateur : caméras localisées à environ 2 mm près ; robot à environ 1 mm près en moyenne
(4 mm au pire) avec les deux caméras en 1920 × 1080 et l'IMU ; prises toutes trouvées sur le mur vide (N1, N2, N3),
sans fausse prise. Sur une photo du vrai mur : les 52 prises trouvées, avec leur couleur.

## Installation

Il faut `git`, `git-lfs` et [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone --recurse-submodules https://github.com/Alyaa203/Hackathon
cd Hackathon/simulator
uv sync
```

Déjà cloné sans `--recurse-submodules` ? `git submodule update --init` récupère `simulator` et `microban`.

## Commandes

Toujours depuis `simulator/` (OpenCV est ajouté à la volée par `--with`) :

```bash
# démonstration dans le simulateur (mur N1, N2 ou N3), avec une fenêtre ; puis la vidéo
uv run --with opencv-contrib-python python ../perception/demo.py N2 --show
open ../perception/out/demo.mp4
# les caméras placées au hasard (le nombre choisit le placement) ; l'autre résolution
uv run --with opencv-contrib-python python ../perception/demo.py N2 --shuffle 3
uv run --with opencv-contrib-python python ../perception/demo.py N2 --res 1280x720
# les prises trouvées, dans l'éditeur de mur du simulateur
uv run climbing-sim-wall-editor ../perception/out/holds.json
# le jour J : les deux caméras USB, gauche puis droite
# (touches : c recalibrer, h liste des prises -> out/holds.json, s sauvegarder, q quitter)
uv run --with opencv-contrib-python python ../perception/live.py 0 1
```

Sur macOS, la fenêtre 3D de MuJoCo demande `mjpython` au lieu de `python` (`uv run mjpython examples/view.py`).

## Ajouter un script à la simu :
dans `microban/src/moves/` créer une classe `NomDeLaClasseMove` (bien avoir `Move` à la fin) et ajouter un shortcut et le nom de la classe (en snake et sans le 'Move' donc ici `nom_de_la_casse`) à la fin de `constants.py`.  
Ensuite lancer la simu avec `make sim` et dans la fenêtre du simulateur apuyer sur la touche pour lancer le script.

## À faire

- [x] Pose des caméras à partir des 4 marqueurs du mur
- [x] Pose du robot avec son marqueur ID 5 (fiche officielle, 42 mm)
- [x] Détection des prises dans les images
- [x] Document PDF à jour
- [ ] Programme de grimpe
- [ ] Relier l'IMU du vrai robot à `live.py`
- [ ] Essais avec les vraies caméras et le vrai robot

## Licence

Le simulateur et le programme du robot sont sous GPL-3.0 (voir leurs dépôts) ; notre code les utilise.
