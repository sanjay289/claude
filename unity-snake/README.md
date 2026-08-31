# Unity Snake

A minimal, classic Snake game for the Unity engine. The entire game lives in one
script — `Assets/Scripts/SnakeGame.cs` — which builds its own camera and objects
at runtime, so there is no scene to wire up.

## Requirements

Unity 2020.3 or newer (uses the built-in `Input` manager and `OnGUI`, no
packages). Any render pipeline works; the default built-in pipeline is simplest.

## Setup

1. Open Unity Hub → **New project** → 2D (Built-in Render Pipeline) → create it.
2. Copy this repo's `Assets/Scripts/SnakeGame.cs` into the project's `Assets/`
   folder (or copy the whole `Assets/` folder over).
3. In the default `SampleScene`, create an empty GameObject
   (**GameObject → Create Empty**), name it `Game`.
4. Select it, **Add Component → Snake Game**.
5. Press **Play**.

## Controls

| Key         | Action              |
|-------------|---------------------|
| Arrow keys  | Steer the snake     |
| Space       | Restart after death |

## Tunables

Exposed on the component in the Inspector:

- **Width / Height** — board size in cells (default 20 × 20).
- **Step Interval** — seconds between moves; lower is faster (default `0.12`).

## How it works

- `cells` is a `List<Vector2Int>` — the snake body, head at index 0. This is the
  single source of truth for game state.
- Each step inserts a new head cell; if it didn't eat, the tail cell is removed.
- Wall or self collision (`cells.Contains(head)`) ends the game.
- `segments` is a pool of `Quad` transforms re-positioned to match `cells` each
  step, so no objects are created or destroyed during normal play.
- Food is placed on a uniformly random free cell; filling the board is a win.

## Not included

Sound, menus, high-score persistence, touch/gamepad input. Add them as separate
components so this file stays the small readable core.
