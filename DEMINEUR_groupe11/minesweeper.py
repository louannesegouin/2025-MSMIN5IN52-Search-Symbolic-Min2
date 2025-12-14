import pygame
import random
import math
# On utilise OR-Tools de Google pour résoudre les contraintes (moteur de l'IA)
from ortools.sat.python import cp_model

# --- CONFIG ET COULEURS ---
BG_COLOR = (26, 32, 44)
HEADER_COLOR = (45, 55, 72)
CELL_HIDDEN = (74, 85, 104)
CELL_REVEALED = (237, 242, 247)
CELL_HOVER = (90, 103, 126)
BORDER_COLOR = (203, 213, 224)
MINE_COLOR = (220, 38, 38)
FLAG_COLOR = (245, 101, 101)
GOLD = (251, 191, 36)
TEXT_LIGHT = (247, 250, 252)
TEXT_DARK = (26, 32, 44)
ACCENT_BLUE = (56, 178, 172)
SUCCESS_GREEN = (72, 187, 120)

NUMBER_COLORS = [
    (59, 130, 246), (34, 197, 94), (239, 68, 68), (139, 92, 246),
    (249, 115, 22), (236, 72, 153), (20, 184, 166), (100, 116, 139)
]

CELL_SIZE = 40
MARGIN = 5
TOOLBAR_HEIGHT = 100


# --- LOGIQUE DU JEU ---
class MinesweeperGame:
    def __init__(self, rows, cols, mines):
        self.rows = rows
        self.cols = cols
        self.total_mines = mines
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]
        self.visible = [[False for _ in range(cols)] for _ in range(rows)]
        self.flags = [[False for _ in range(cols)] for _ in range(rows)]
        self.game_over = False
        self.win = False
        self.first_click = True

    # On génère la grille seulement après le premier clic pour être sûr de pas tomber sur une mine direct
    def _generate_grid(self, exclude_r=None, exclude_c=None):
        count = 0
        while count < self.total_mines:
            r, c = random.randint(
                0, self.rows - 1), random.randint(0, self.cols - 1)
            # On vérifie qu'il n'y a pas déjà une mine et qu'on n'est pas sur la case de départ
            if self.grid[r][c] != -1 and not (r == exclude_r and c == exclude_c):
                self.grid[r][c] = -1
                count += 1

        # On calcule les chiffres pour chaque case (nombre de mines voisines)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == -1:
                    continue
                self.grid[r][c] = sum(1 for nr, nc in self._get_neighbors(r, c)
                                      if self.grid[nr][nc] == -1)

    # Petit utilitaire pour chopper les 8 cases autour (haut, bas, diago...)
    def _get_neighbors(self, r, c):
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                   (0, 1), (1, -1), (1, 0), (1, 1)]
        neighbors = []
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbors.append((nr, nc))
        return neighbors

    # Action de cliquer sur une case
    def reveal(self, r, c):
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return
        # Si c'est déjà visible, flaggé ou fini, on fait rien
        if self.visible[r][c] or self.flags[r][c] or self.game_over:
            return

        if self.first_click:
            self._generate_grid(exclude_r=r, exclude_c=c)
            self.first_click = False

        self.visible[r][c] = True

        if self.grid[r][c] == -1:
            # une mine -> Perdu
            self.game_over = True
            self.win = False
        else:
            # Si c'est un 0 (pas de mine autour), on ouvre tout autour récursivement (flood fill)
            if self.grid[r][c] == 0:
                for nr, nc in self._get_neighbors(r, c):
                    self.reveal(nr, nc)
            self._check_victory()

    # On vérifie si toutes les cases sans mines sont ouvertes
    def _check_victory(self):
        count_visible = sum(row.count(True) for row in self.visible)
        total_safe_cells = (self.rows * self.cols) - self.total_mines

        if count_visible == total_safe_cells:
            self.win = True
            self.game_over = True
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.grid[r][c] == -1:
                        self.flags[r][c] = True

    def toggle_flag(self, r, c):
        if not self.visible[r][c] and not self.game_over:
            self.flags[r][c] = not self.flags[r][c]

    # -2 = drapeau, -1 = inconnu, >=0 = chiffre révélé
    def get_view_for_ai(self):
        ai_grid = [[-2 for _ in range(self.cols)] for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                if self.visible[r][c]:
                    ai_grid[r][c] = self.grid[r][c]
                elif self.flags[r][c]:
                    ai_grid[r][c] = -2
                else:
                    ai_grid[r][c] = -1
        return ai_grid


# --- CONFIGURATION DE L'IA ---
class CSPSolver:
    def __init__(self, game):
        self.game = game
        self.last_probabilities = []

    # On prépare le problème pour le solveur CP-SAT
    def _create_model(self, grid_view):
        model = cp_model.CpModel()
        vars = {}
        rows, cols = len(grid_view), len(grid_view[0])
        unknowns = set()
        open_numbered = []

        # On repère les cases inconnues qui touchent des chiffres
        for r in range(rows):
            for c in range(cols):
                if grid_view[r][c] >= 0:
                    open_numbered.append((r, c))
                    for nr, nc in self.game._get_neighbors(r, c):
                        if not self.game.visible[nr][nc]:
                            unknowns.add((nr, nc))

        # On crée une variable booléenne (0 ou 1) pour chaque case inconnue
        for (r, c) in unknowns:
            vars[(r, c)] = model.NewBoolVar(f'cell_{r}_{c}')

        # On ajoute les contraintes : Somme des voisins = Valeur de la case
        for r, c in open_numbered:
            val = grid_view[r][c]
            neighbors = self.game._get_neighbors(r, c)
            current_vars = [vars.get((nr, nc))
                            for nr, nc in neighbors if (nr, nc) in vars]

            if current_vars:
                model.Add(sum(current_vars) == val)

        return model, vars, list(unknowns)

    # Fonction principale de l'IA
    def find_safe_moves(self):
        grid_view = self.game.get_view_for_ai()
        _, _, unknowns = self._create_model(grid_view)

        if not unknowns:
            return self._guess_random()

        moves = []
        solver = cp_model.CpSolver()

        # Pour chaque case inconnue, on teste si elle peut être une mine ou pas
        for (r, c) in unknowns:
            # Test 1 : Est-ce qu'il est impossible que ce soit sûr ? (Donc c'est une mine)
            model_safe, vars_safe, _ = self._create_model(grid_view)
            model_safe.Add(vars_safe[(r, c)] == 0)  # Supposons que c'est safe
            if solver.Solve(model_safe) == cp_model.INFEASIBLE:
                moves.append((r, c, 'FLAG'))  # Si impossible c'est une mine
                continue

            # Test 2 : Est-ce qu'il est impossible que ce soit une mine ? (Donc c'est safe)
            model_mine, vars_mine, _ = self._create_model(grid_view)
            # Supposons que c'est une mine
            model_mine.Add(vars_mine[(r, c)] == 1)
            if solver.Solve(model_mine) == cp_model.INFEASIBLE:
                moves.append((r, c, 'REVEAL'))  # Impossible= sûr

        # Si le solveur logique ne trouve rien, on passe aux probabilités
        if not moves:
            self.last_probabilities = self._compute_probabilities()
            prob_move = self._choose_safest_from_probs(self.last_probabilities)
            return [prob_move] if prob_move else self._guess_random()

        return moves

    def _guess_random(self):
        # Choix aléatoire quand on commence ou qu'on est perdu
        choices = [(r, c) for r in range(self.game.rows) for c in range(
            self.game.cols) if not self.game.visible[r][c] and not self.game.flags[r][c]]
        return [(*random.choice(choices), 'REVEAL')] if choices else []

    # On sépare les groupes de cases interconnectées et on compte toutes les combinaisons valides
    def _compute_probabilities(self):
        grid = self.game.get_view_for_ai()
        rows, cols = self.game.rows, self.game.cols
        frontier = set()
        numbered = []

        # On identifie la frontière (cases inconnues à côté de chiffres)
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] >= 0:
                    numbered.append((r, c))
                    for nr, nc in self.game._get_neighbors(r, c):
                        if not self.game.visible[nr][nc] and not self.game.flags[nr][nc]:
                            frontier.add((nr, nc))
        if not frontier:
            return []

        # On map les chiffres à leurs voisins inconnus
        neighbor_to_unknowns = {}
        for r, c in numbered:
            if grid[r][c] >= 0:
                lst = [(nr, nc) for nr, nc in self.game._get_neighbors(
                    r, c) if (nr, nc) in frontier]
                if lst:
                    neighbor_to_unknowns[(r, c)] = lst

        # On construit un graphe pour trouver les composantes connexes
        adj = {u: set() for u in frontier}
        for _, unk_list in neighbor_to_unknowns.items():
            for i in range(len(unk_list)):
                for j in range(i + 1, len(unk_list)):
                    adj[unk_list[i]].add(unk_list[j])
                    adj[unk_list[j]].add(unk_list[i])

        # Parcours pour séparer les groupes indépendants
        components, seen = [], set()
        for u in frontier:
            if u not in seen:
                comp, stack = [], [u]
                seen.add(u)
                while stack:
                    cur = stack.pop()
                    comp.append(cur)
                    for v in adj[cur]:
                        if v not in seen:
                            seen.add(v)
                            stack.append(v)
                components.append(comp)

        MAX_ENUM_SIZE = 18  # Limite pour éviter que ça explose si trop de cases
        cell_prob_mine = {u: 0.0 for u in frontier}
        cell_prob_count = {u: 0 for u in frontier}

        # Fonction locale pour vérifier les contraintes
        def constraints_for_cells(cells):
            cons, cell_set = [], set(cells)
            for (nr, nc), unk_list in neighbor_to_unknowns.items():
                sub = [x for x in unk_list if x in cell_set]
                if not sub:
                    continue

                total_required = grid[nr][nc]
                flagged = sum(1 for rr, cc in self.game._get_neighbors(
                    nr, nc) if self.game.flags[rr][cc])
                cons.append((sub, max(0, total_required-flagged)))
            return cons

        # Backtracking : on essaie toutes les possibilités (0 ou 1) pour compter celles qui marchent
        def enumerate_component(cells):
            cons = constraints_for_cells(cells)
            n = len(cells)
            if n == 0 or n > MAX_ENUM_SIZE:
                return None

            idx_map = {cells[i]: i for i in range(n)}
            mine_counts, total_assignments = [0] * n, 0
            cons_idx = [([idx_map[u] for u in unk_list], req)
                        for unk_list, req in cons]
            assign, cur_cnt = [0] * n, [0] * len(cons_idx)

            # Vérifie si on peut encore satisfaire les contraintes
            def feasible(vi, val):
                for k, (indices, required) in enumerate(cons_idx):
                    if vi in indices:
                        new_cnt = cur_cnt[k] + val
                        remaining_slots = sum(1 for idx in indices if idx > vi)
                        if new_cnt > required or new_cnt + remaining_slots < required:
                            return False
                return True

            def set_vi(vi, val, sign):
                assign[vi] = val
                for k, (indices, _) in enumerate(cons_idx):
                    if vi in indices:
                        cur_cnt[k] += val * sign

            # Récursion
            def backtrack(vi=0):
                nonlocal total_assignments
                if vi == n:
                    if all(cur_cnt[k] == req for k, (_, req) in enumerate(cons_idx)):
                        total_assignments += 1
                        for i in range(n):
                            mine_counts[i] += assign[i]
                    return
                for val in (0, 1):
                    if feasible(vi, val):
                        set_vi(vi, val, 1)
                        backtrack(vi + 1)
                        set_vi(vi, val, -1)

            backtrack()
            # Retourne la probabilité d'avoir une mine
            return {cells[i]: mine_counts[i] / total_assignments for i in range(n)} if total_assignments > 0 else None

        # On lance le calcul pour chaque groupe
        for comp in components:
            probs = enumerate_component(comp)
            for u in comp:
                cell_prob_mine[u] += probs.get(u, 0.5) if probs else 0.5
                cell_prob_count[u] += 1

        # On trie pour avoir la proba la plus faible en premier
        averaged = sorted([(u, cell_prob_mine[u]/cell_prob_count[u])
                          for u in frontier if cell_prob_count[u] > 0], key=lambda x: x[1])
        return averaged

    def _choose_safest_from_probs(self, probs):
        if not probs:
            return None
        # On prend celui avec la proba de mine la plus basse
        (br, bc), p = probs[0]
        print(f"IA (proba) joue: REVEAL ({br},{bc}) avec P(mine)={p:.3f}")
        return (br, bc, 'REVEAL')


# --- FONCTIONS D'AFFICHAGE ---

def draw_text(screen, text, size, color, x, y, bold=False):
    font = pygame.font.SysFont(
        "segoeui" if "segoeui" in pygame.font.get_fonts() else "arial", size, bold)
    img = font.render(text, True, color)
    screen.blit(img, (x - img.get_width() // 2, y - img.get_height() // 2))


def draw_text_left(screen, text, size, color, x, y, bold=False):
    font = pygame.font.SysFont(
        "segoeui" if "segoeui" in pygame.font.get_fonts() else "arial", size, bold)
    img = font.render(text, True, color)
    screen.blit(img, (x, y))


def draw_rounded_rect(screen, color, rect, radius=8):
    pygame.draw.rect(screen, color, rect, border_radius=radius)


def draw_mine(screen, center_x, center_y, size=12):
    pygame.draw.circle(screen, MINE_COLOR, (center_x, center_y), size)
    for angle in range(0, 360, 45):
        rad = math.radians(angle)
        x, y = center_x + int(math.cos(rad) * size *
                              1.3), center_y + int(math.sin(rad) * size * 1.3)
        pygame.draw.line(screen, MINE_COLOR, (center_x, center_y), (x, y), 3)
    pygame.draw.circle(screen, TEXT_LIGHT, (center_x - 3, center_y - 3), 3)


def draw_flag(screen, center_x, center_y, size=14):
    pygame.draw.line(screen, TEXT_DARK, (center_x - size//3, center_y +
                     size//1.5), (center_x - size//3, center_y - size//1.5), 3)
    points = [(center_x - size//3, center_y - size//1.5), (center_x +
                                                           size, center_y - size//3), (center_x - size//3, center_y)]
    pygame.draw.polygon(screen, FLAG_COLOR, points)


# --- MENU PRINCIPAL ---
def main_menu():
    pygame.init()
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Démineur IA - Choix du niveau")

    # Choix du niveau
    levels = {
        "Débutant": {"size": (9, 9), "mines": 10},
        "Intermédiaire": {"size": (16, 16), "mines": 40},
        "Expert": {"size": (30, 15), "mines": 99}
    }

    # Création des boutons
    buttons = []
    button_y_start = 250
    for i, (level, config) in enumerate(levels.items()):
        rect = pygame.Rect(screen_width // 2 - 150,
                           button_y_start + i * 80, 300, 60)
        buttons.append({"rect": rect, "text": level, "config": config})

    running = True
    while running:
        screen.fill(BG_COLOR)

        draw_text(screen, "DÉMINEUR IA", 50, TEXT_LIGHT,
                  screen_width // 2, 100, bold=True)
        draw_text(screen, "Choisissez un niveau", 24,
                  ACCENT_BLUE, screen_width // 2, 160)

        mouse_pos = pygame.mouse.get_pos()
        for button in buttons:
            # Effet de survol sur les boutons
            color = CELL_HOVER if button["rect"].collidepoint(
                mouse_pos) else HEADER_COLOR
            draw_rounded_rect(screen, color, button["rect"])
            draw_text(screen, button["text"], 22, TEXT_LIGHT,
                      button["rect"].centerx, button["rect"].centery, bold=True)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Si on clique sur un bouton, on lance le jeu
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    for button in buttons:
                        if button["rect"].collidepoint(mouse_pos):
                            cols, rows = button["config"]["size"]
                            mines = button["config"]["mines"]
                            game_loop(rows, cols, mines)
                            # Quand on quitte le jeu, on revient au menu
                            screen = pygame.display.set_mode(
                                (screen_width, screen_height))
                            pygame.display.set_caption(
                                "Démineur IA - Choix du niveau")

        pygame.display.flip()

    pygame.quit()


# --- BOUCLE DE JEU ---
def game_loop(rows, cols, num_mines):
    # Calcul dynamique de la taille de la fenêtre
    screen_width = MARGIN + (CELL_SIZE + MARGIN) * cols
    screen_height = MARGIN + (CELL_SIZE + MARGIN) * rows + TOOLBAR_HEIGHT
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Démineur IA - CSP Solver")
    clock = pygame.time.Clock()

    # Initialisation du jeu et de l'IA
    game = MinesweeperGame(rows, cols, num_mines)
    ai = CSPSolver(game)

    hover_cell = None
    running = True
    while running:
        screen.fill(BG_COLOR)

        # Gestion de la souris pour savoir sur quelle case on est
        mouse_pos = pygame.mouse.get_pos()
        mouse_c = (mouse_pos[0] - MARGIN) // (CELL_SIZE + MARGIN)
        mouse_r = (mouse_pos[1] - (TOOLBAR_HEIGHT + MARGIN)
                   ) // (CELL_SIZE + MARGIN)

        hover_cell = (
            mouse_r, mouse_c) if 0 <= mouse_r < rows and 0 <= mouse_c < cols else None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Gestion des clics joueur (Gauche = Révéler, Droit = Flag)
            if event.type == pygame.MOUSEBUTTONDOWN and not game.game_over:
                if hover_cell:
                    r, c = hover_cell
                    if event.button == 1:
                        game.reveal(r, c)
                    elif event.button == 3:
                        game.toggle_flag(r, c)

            if event.type == pygame.KEYDOWN:
                # TOUCHE ESPACE : L'IA joue
                if event.key == pygame.K_SPACE and not game.game_over:
                    moves = ai.find_safe_moves()
                    print(f"IA suggère {len(moves)} coups.")
                    for (r, c, action) in moves:
                        if action == 'REVEAL':
                            game.reveal(r, c)
                        elif action == 'FLAG' and not game.flags[r][c]:
                            game.toggle_flag(r, c)

                # Reset avec 'R'
                if event.key == pygame.K_r:
                    game = MinesweeperGame(rows, cols, num_mines)
                    ai = CSPSolver(game)
                    print("Nouvelle partie !")

                if event.key == pygame.K_ESCAPE:
                    running = False

        # --- DESSIN DE L'INTERFACE ---
        header_rect = pygame.Rect(0, 0, screen_width, TOOLBAR_HEIGHT)
        draw_rounded_rect(screen, HEADER_COLOR, header_rect, radius=0)

        flags_placed = sum(row.count(True) for row in game.flags)
        draw_text(screen, "DÉMINEUR IA", 32, TEXT_LIGHT,
                  screen_width // 2, 25, bold=True)
        draw_text_left(
            screen, f"💣 {game.total_mines - flags_placed}", 22, ACCENT_BLUE, 20, 55, bold=True)

        if not game.game_over:
            draw_text(screen, "[ESPACE] IA | [R] Reset | [ECHAP] Menu",
                      16, TEXT_LIGHT, screen_width // 2, 70)
        else:
            draw_text(screen, "[R] Reset | [ECHAP] Menu", 16,
                      SUCCESS_GREEN if game.win else FLAG_COLOR, screen_width // 2, 70, bold=True)

        # Dessin de la grille
        for r in range(rows):
            for c in range(cols):
                rect = pygame.Rect(MARGIN + c * (CELL_SIZE + MARGIN), TOOLBAR_HEIGHT +
                                   MARGIN + r * (CELL_SIZE + MARGIN), CELL_SIZE, CELL_SIZE)
                center_x, center_y = rect.centerx, rect.centery
                is_hover = (hover_cell == (r, c)) and not game.game_over

                if game.visible[r][c]:
                    draw_rounded_rect(screen, CELL_REVEALED, rect, radius=6)
                    if game.grid[r][c] == -1:
                        draw_mine(screen, center_x, center_y,
                                  size=CELL_SIZE // 3)
                    elif game.grid[r][c] > 0:
                        val = game.grid[r][c]
                        color = NUMBER_COLORS[val - 1]
                        draw_text(screen, str(val), 20, color,
                                  center_x, center_y, bold=True)
                else:
                    draw_rounded_rect(
                        screen, CELL_HOVER if is_hover else CELL_HIDDEN, rect, radius=6)
                    if game.flags[r][c]:
                        draw_flag(screen, center_x, center_y,
                                  size=CELL_SIZE // 3)

        # Overlay Game Over / Victoire
        if game.game_over:
            overlay = pygame.Surface(
                (screen_width, screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            screen.blit(overlay, (0, 0))
            status_text = "🎉 VICTOIRE ! 🎉" if game.win else "💥 GAME OVER 💥"
            status_color = GOLD if game.win else FLAG_COLOR
            subtitle = "Toutes les mines trouvées !" if game.win else "Vous avez touché une mine..."
            draw_text(screen, status_text, 40, status_color,
                      screen_width // 2, screen_height // 2 - 20, bold=True)
            draw_text(screen, subtitle, 20, TEXT_LIGHT,
                      screen_width // 2, screen_height // 2 + 30)

        pygame.display.flip()
        clock.tick(60)


# Point d'entrée
if __name__ == "__main__":
    main_menu()
