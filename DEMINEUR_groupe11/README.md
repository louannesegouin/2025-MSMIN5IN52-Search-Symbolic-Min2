# Démineur IA avec Solveur CSP

Ce projet est une implémentation du jeu classique Démineur en Python. Il intègre une interface graphique moderne via Pygame et une intelligence artificielle capable de résoudre les grilles.

L'IA utilise la programmation par contraintes (CSP) via la bibliothèque Google OR-Tools pour déterminer de manière certaine les coups sûrs et les mines. Lorsque la logique pure ne suffit pas, elle calcule les probabilités pour jouer le coup statistiquement le plus sûr.

## Fonctionnalités

* Interface graphique fluide avec Pygame.
* Trois niveaux de difficulté : Débutant, Intermédiaire, Expert.
* Solveur IA intégré activable à la demande.
* Gestion des probabilités pour les situations incertaines.

## Prérequis

* Python 3.8 ou supérieur
* Pip (gestionnaire de paquets Python)

## Installation

* Assurez-vous d'avoir Python installé sur votre machine.
* Installez les dépendances nécessaires (Pygame et OR-Tools) en exécutant la commande suivante dans votre terminal :

```
    pip install pygame ortools  
```

## Lancement et Tests

Pour lancer le jeu, exécutez la commande suivante dans le terminal :

```
    python demineur.py  
```

### Contrôles en jeu

* **Clic Gauche** : Révéler une case.
* **Clic Droit** : Placer ou retirer un drapeau.
* **Barre Espace** : Demander à l'IA de jouer le prochain coup.
* **Touche R** : Réinitialiser la partie actuelle.
* **Touche ECHAP** : Retourner au menu principal ou quitter.

### Fonctionnement de l'IA

Lorsque vous appuyez sur la barre Espace :

* L'IA analyse la grille visible.
* Elle crée un modèle de contraintes pour identifier formellement les mines et les cases sûres.
* Si des coups sûrs sont trouvés, ils sont joués (révélation ou drapeau).
* Si aucune certitude n'est possible, l'IA calcule la probabilité de présence de mine pour chaque case frontalière et joue celle ayant le risque le plus faible.

## Structure du Code

* **MinesweeperGame** : Gère la logique interne du jeu (grille, règles, victoire/défaite).
* **CSPSolver** : Contient la logique de l'IA, l'intégration avec OR-Tools et le calcul de probabilités.
* **main_menu** / **game_loop** : Gèrent l'affichage et les interactions utilisateur via Pygame.

## Auteurs

- Arthur ADBELLI
- Lou-Anne SEGOUIN
