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
| `perception.py` | Le traitement d'image : détection des marqueurs, pose des caméras (marqueurs 1 à 4 du mur), pose du robot (marqueur 5, triangulation avec les deux caméras). Fonctionne sur des images simulées ou réelles. |
| `robot_marker.py` | Le marqueur ID 5 sur le dos du robot : ses dimensions, sa position sur le tronc, et son ajout au simulateur (`MarkedSim`). |
| `demo.py` | Démonstration dans le simulateur, comparée à la vérité du simulateur. |
| `live.py` | Le jour J : les deux caméras USB (ou des images), position du robot en direct. |
| `doc/` | Le document LaTeX, son PDF et ses images (`make_figures.py` les régénère). |

Résultats dans le simulateur : caméras localisées à environ 2 mm près, robot à 1 à 3 mm près (en moyenne) avec les
deux caméras.

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
# démonstration dans le simulateur (mur N1, N2 ou N3)
uv run --with opencv-contrib-python python ../perception/demo.py N1
# les caméras placées au hasard (le nombre choisit le placement)
uv run --with opencv-contrib-python python ../perception/demo.py N2 --shuffle 3
# le jour J : les deux caméras USB, gauche puis droite (touches : c recalibrer, s sauvegarder, q quitter)
uv run --with opencv-contrib-python python ../perception/live.py 0 1
```

Sur macOS, la fenêtre 3D de MuJoCo demande `mjpython` au lieu de `python` (`uv run mjpython examples/view.py`).

## À faire

- [x] Pose des caméras à partir des 4 marqueurs du mur
- [x] Pose du robot avec son marqueur ID 5
- [ ] Détection des prises dans les images
- [ ] Programme de grimpe
- [ ] Essais avec les vraies caméras et le vrai robot

## Licence

Le simulateur et le programme du robot sont sous GPL-3.0 (voir leurs dépôts) ; notre code les utilise.
