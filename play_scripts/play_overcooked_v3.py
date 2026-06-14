#!/usr/bin/env python3
"""Interactive Overcooked V3 player with keyboard controls."""

import jax
import pygame
import numpy as np
from jaxmarl import make
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer

# Keyboard mappings for Agent 0 (WASD + Space)
AGENT0_KEYS = {
    pygame.K_w: 3,      # up
    pygame.K_s: 1,      # down
    pygame.K_a: 2,      # left
    pygame.K_d: 0,      # right
    pygame.K_SPACE: 5,  # interact
}

# Keyboard mappings for Agent 1 (Arrow keys + Enter)
AGENT1_KEYS = {
    pygame.K_UP: 3,     # up
    pygame.K_DOWN: 1,   # down
    pygame.K_LEFT: 2,   # left
    pygame.K_RIGHT: 0,  # right
    pygame.K_RETURN: 5, # interact
}


def main():
    print("=" * 50)
    print("  OVERCOOKED V3 - Interactive Mode")
    print("=" * 50)
    print()
    print("Controls:")
    print("  Agent 0 (Blue):  WASD to move, SPACE to interact")
    print("  Agent 1 (Green): Arrow keys to move, ENTER to interact")
    print()
    print("  R = Reset")
    print("  Q or ESC = Quit")
    print()
    print("Goal: Pick up onions, put 3 in the pot, wait for cooking,")
    print("      then pick up soup with a plate and deliver to green zone!")
    print()
    print("Pot timing: Cook=20 steps, Burn window=10 steps")
    print("  - Green bar = cooking progress")
    print("  - Orange bar = burning window (pick up before it empties!)")
    print("=" * 50)

    # Create environment
    env = make('overcooked_v3', layout='moving_wall_bounce_demo', pot_cook_time=20, pot_burn_time=10)
    viz = OvercookedV3Visualizer(env, tile_size=48)  # Larger tiles for visibility

    # Initialize pygame
    pygame.init()

    # Calculate window size
    width = env.width * 48
    height = env.height * 48
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Overcooked V3 - Press Q to quit")
    clock = pygame.time.Clock()

    # Initialize game state
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    obs, state = env.reset(subkey)

    total_reward = 0
    step_count = 0
    running = True

    while running:
        # Handle events
        agent0_action = 4  # stay
        agent1_action = 4  # stay

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Reset
                    key, subkey = jax.random.split(key)
                    obs, state = env.reset(subkey)
                    total_reward = 0
                    step_count = 0
                    print("\n--- Game Reset ---\n")

        # Get current key states for continuous input
        keys = pygame.key.get_pressed()

        # Agent 0 controls
        for k, action in AGENT0_KEYS.items():
            if keys[k]:
                agent0_action = action
                break

        # Agent 1 controls
        for k, action in AGENT1_KEYS.items():
            if keys[k]:
                agent1_action = action
                break

        # Step environment
        actions = {'agent_0': agent0_action, 'agent_1': agent1_action}
        key, subkey = jax.random.split(key)
        obs, state, rewards, dones, info = env.step(subkey, state, actions)

        step_count += 1
        reward = rewards['agent_0']
        total_reward += reward

        if reward > 0:
            print(f"🎉 DELIVERY! +{reward:.0f} points! (Total: {total_reward:.0f})")

        # Render
        img = viz.render_state(state)
        img_np = np.array(img)

        # Convert to pygame surface (need to transpose for pygame)
        surf = pygame.surfarray.make_surface(img_np.swapaxes(0, 1))
        screen.blit(surf, (0, 0))

        # Draw HUD
        font = pygame.font.Font(None, 24)
        hud_text = f"Step: {step_count}  Score: {total_reward:.0f}"
        text_surf = font.render(hud_text, True, (255, 255, 255))
        screen.blit(text_surf, (5, 5))

        pygame.display.flip()
        clock.tick(10)  # 10 FPS for playable speed

    pygame.quit()
    print(f"\nGame Over! Final Score: {total_reward:.0f} in {step_count} steps")


if __name__ == "__main__":
    main()
