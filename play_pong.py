import os
import numpy as np
import torch
import pygame
from stable_baselines3 import PPO

# --- CONFIGURATION ---
WIDTH, HEIGHT = 600, 400
PADDLE_WIDTH, PADDLE_HEIGHT = 12, 60
BALL_SIZE = 12
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)

class GameState:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.player_y = HEIGHT // 2 - PADDLE_HEIGHT // 2
        self.nn_y = HEIGHT // 2 - PADDLE_HEIGHT // 2
        self.ball_x = WIDTH // 2
        self.ball_y = HEIGHT // 2
        self.ball_dx = 5 if np.random.rand() > 0.5 else -5
        self.ball_dy = np.random.uniform(-3, 3)
        self.player_score = 0
        self.nn_score = 0

    def update(self, player_action, nn_action):
        # 1. Move Player (Left Paddle) -> 2: UP, 3: DOWN
        if player_action == 2:
            self.player_y = max(0, self.player_y - 6)
        elif player_action == 3:
            self.player_y = min(HEIGHT - PADDLE_HEIGHT, self.player_y + 6)

        # 2. Move NN (Right Paddle) -> 2: UP, 3: DOWN
        if nn_action == 2:
            self.nn_y = max(0, self.nn_y - 6)
        elif nn_action == 3:
            self.nn_y = min(HEIGHT - PADDLE_HEIGHT, self.nn_y + 6)

        # 3. Move Ball
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # Wall bounces (Top/Bottom)
        if self.ball_y <= 0 or self.ball_y >= HEIGHT - BALL_SIZE:
            self.ball_dy *= -1

        # Paddle Collisions (Left / Player)
        if (self.ball_x <= PADDLE_WIDTH + 10 and 
            self.player_y <= self.ball_y <= self.player_y + PADDLE_HEIGHT):
            self.ball_dx *= -1.1  # Speed up slightly on hits
            relative_intersect_y = (self.player_y + (PADDLE_HEIGHT / 2)) - self.ball_y
            self.ball_dy = -(relative_intersect_y / (PADDLE_HEIGHT / 2)) * 5

        # Paddle Collisions (Right / NN)
        if (self.ball_x >= WIDTH - PADDLE_WIDTH - 10 - BALL_SIZE and 
            self.nn_y <= self.ball_y <= self.nn_y + PADDLE_HEIGHT):
            self.ball_dx *= -1.1
            relative_intersect_y = (self.nn_y + (PADDLE_HEIGHT / 2)) - self.ball_y
            self.ball_dy = -(relative_intersect_y / (PADDLE_HEIGHT / 2)) * 5

        # Scoring
        if self.ball_x < 0:
            self.nn_score += 1
            self.reset_ball()
        elif self.ball_x > WIDTH:
            self.player_score += 1
            self.reset_ball()

    def reset_ball(self):
        self.ball_x = WIDTH // 2
        self.ball_y = HEIGHT // 2
        self.ball_dx = 5 if np.random.rand() > 0.5 else -5
        self.ball_dy = np.random.uniform(-3, 3)

    def generate_atari_frame(self):
        """Generates a simplified 84x84 grayscale representation for the NN."""
        frame = np.zeros((84, 84), dtype=np.uint8)
        scale_x = 84 / WIDTH
        scale_y = 84 / HEIGHT

        # Draw left paddle (Value 142 mimicking Atari surface values)
        py1, py2 = int(self.player_y * scale_y), int((self.player_y + PADDLE_HEIGHT) * scale_y)
        px1, px2 = int(10 * scale_x), int((10 + PADDLE_WIDTH) * scale_x)
        frame[py1:py2, px1:px2] = 142

        # Draw right paddle
        ny1, ny2 = int(self.nn_y * scale_y), int((self.nn_y + PADDLE_HEIGHT) * scale_y)
        nx1, nx2 = int((WIDTH - 10 - PADDLE_WIDTH) * scale_x), int((WIDTH - 10) * scale_x)
        frame[ny1:ny2, nx1:nx2] = 142

        # Draw ball
        bx, by = int(self.ball_x * scale_x), int(self.ball_y * scale_y)
        frame[max(0, by-1):min(84, by+2), max(0, bx-1):min(84, bx+2)] = 255
        
        return frame

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Human (Left keys) vs. Trained NN (Right)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)

    # Load Model Weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists("ppo_pong_model_v1.zip"):
        print("❌ Model weights file 'ppo_pong_model_v1.zip' not found.")
        return
    model = PPO.load("ppo_pong_model_v1.zip", device=device)

    game = GameState()
    
    # Initialize a 4-frame buffer history for the VecFrameStack imitation
    initial_frame = game.generate_atari_frame()
    frame_stack = np.stack([initial_frame] * 4, axis=0)

    running = True
    while running:
        clock.tick(FPS)
        
        # 1. Handle Window Closing Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 2. Process Human Keyboard Input
        keys = pygame.key.get_pressed()
        player_action = 0  # Default: 0 = Stay still
        
        if keys[pygame.K_UP]:
            player_action = 2   # Atari mapping for UP
        elif keys[pygame.K_DOWN]:
            player_action = 3   # Atari mapping for DOWN

        # Escape route via 'Escape' key
        if keys[pygame.K_ESCAPE]:
            running = False

        # 3. Get AI Prediction
        nn_input = np.expand_dims(frame_stack, axis=0)
        nn_action, _ = model.predict(nn_input, deterministic=True)
        nn_action = nn_action[0]

        # 4. Update game math
        game.update(player_action, nn_action)

        # 5. Generate the latest observation frame and cycle the stack
        new_frame = game.generate_atari_frame()
        frame_stack = np.roll(frame_stack, shift=-1, axis=0)
        frame_stack[-1] = new_frame

        # 6. Render Screen
        screen.fill(BLACK)
        
        # Center divider line
        for y in range(0, HEIGHT, 20):
            if (y // 20) % 2 == 0:
                pygame.draw.rect(screen, GRAY, (WIDTH // 2 - 2, y, 4, 10))

        # Draw Paddles and Ball
        pygame.draw.rect(screen, WHITE, (10, game.player_y, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.rect(screen, WHITE, (WIDTH - 10 - PADDLE_WIDTH, game.nn_y, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.rect(screen, WHITE, (game.ball_x, game.ball_y, BALL_SIZE, BALL_SIZE))

        # Draw Scoreboard
        score_text = font.render(f"{game.player_score}   {game.nn_score}", True, WHITE)
        screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 20))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()