

import pygame
import sys
import math


pygame.init()

WIDTH, HEIGHT = 1400, 900
FPS = 60
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Simple Robotic Hand BCI")
clock = pygame.time.Clock()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (50, 150, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 100)
GRAY = (100, 100, 100)

font = pygame.font.SysFont(None, 36)

class SimpleHand:
    def __init__(self):
        self.state = "CLOSED"
    def update(self, state=None):
        
        if state is not None:
            if state == "CLOSED":
                self.state = "CLOSED"
            elif state == "OPEN":
                self.state = "OPEN"
     

    def set_state(self, state):
        if state == "open":
            self.state = "OPEN"
        elif state == "closed":
            self.state = "CLOSED"


    def draw(self, surface, width, height):
        screen.fill((180, 180, 245))
        
        cx = width // 2
        cy = height // 2
        
        # Draw wrist
        pygame.draw.rect(surface, GRAY, (cx - 60, cy + 100, 120, 60), border_radius=10)
        
        # Draw palm
        pygame.draw.ellipse(surface, (200, 200, 220), (cx - 100, cy, 200, 120))
        
        # Draw fingers based on state
        if self.state == "OPEN":
            color = BLUE
            for i in range(5):
                x = cx - 80 + i * 40
                pygame.draw.line(surface, color, (x, cy), (x, cy - 100), 10)
                
        elif self.state == "HALF":
            color = GREEN
            for i in range(5):
                x = cx - 80 + i * 40
                pygame.draw.line(surface, color, (x, cy), (x + 20, cy - 50), 10)
                
        else:  # CLOSED
            color = RED
            for i in range(5):
                x = cx - 80 + i * 40
                pygame.draw.line(surface, color, (x, cy), (x + 40, cy), 10)
        
        # Draw info
        state_text = font.render(f"Hand State: {self.state}", True, BLACK)
        surface.blit(state_text, (cx - 100, 50))


def run_simulator(hand, clf, X_scaled):

    global screen, WIDTH, HEIGHT
    running = True
    
    prediction_buffer = []
    is_hand_open = True
    last_toggle_time = 0
    refractory_period = 2.0
    current_index = 0
    last_update_time = 0
    update_interval = 0.5
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                WIDTH, HEIGHT = event.size
                screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        
        if X_scaled is not None and current_index < len(X_scaled):
            if time.time() - last_update_time > update_interval:
                last_update_time = time.time()
                
                try:
                    prediction = clf.predict([X_scaled[current_index]])[0]
                    
                    prediction_buffer.append(prediction)
                    if len(prediction_buffer) > 5:
                        prediction_buffer.pop(0)
                        
                    activity_ratio = sum(prediction_buffer) / len(prediction_buffer)
                    
      
                    if (time.time() - last_toggle_time) > refractory_period:
                        if activity_ratio >= 0.6: # Threshold
                            # Toggle
                            is_hand_open = not is_hand_open
                            last_toggle_time = time.time()
                            prediction_buffer = [] 
                    
                    current_index = (current_index + 1) % len(X_scaled)
                except Exception as e:
                    print(f"Prediction error: {e}")

      
        hand.update("OPEN" if is_hand_open else "CLOSED")
        
     
        hand.draw(screen, WIDTH, HEIGHT)
        
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    import time
    
    # إنشاء اليد
    hand = SimpleHand()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_o:
                    hand.state = "OPEN"
                elif event.key == pygame.K_h:
                    hand.state = "HALF"
                elif event.key == pygame.K_c:
                    hand.state = "CLOSED"
                elif event.key == pygame.K_q:
                    running = False
        
        hand.draw(screen, WIDTH, HEIGHT)
        
        inst_text = font.render("Press: O=Open  H=Half  C=Close  Q=Quit", True, BLACK)
        screen.blit(inst_text, (WIDTH//2 - 200, HEIGHT - 100))
        
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()