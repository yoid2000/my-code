import pygame
import sys
from datetime import datetime
import os

# Initialize pygame
pygame.init()

# Position window at top center of screen
os.environ['SDL_VIDEO_WINDOW_POS'] = f"{(pygame.display.Info().current_w - 600) // 2},0"

# Set up the display
WIDTH, HEIGHT = 600, 300
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.NOFRAME)
#screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Clock")


# Colors
BLACK = (0, 0, 0)
DARK_RED = (120, 0, 0)

# Font
font = pygame.font.SysFont('Arial', 120, bold=True)

# Clock
clock = pygame.time.Clock()

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Fill screen with black
    screen.fill(BLACK)
    
    # Get current time in 24-hour format
    current_time = datetime.now().strftime('%H:%M')
    
    # Render the time
    time_surface = font.render(current_time, True, DARK_RED)
    time_rect = time_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    
    # Draw the time
    screen.blit(time_surface, time_rect)
    
    # Update display
    pygame.display.flip()
    
    # Control frame rate
    clock.tick(1)  # Update once per second

pygame.quit()
sys.exit()
