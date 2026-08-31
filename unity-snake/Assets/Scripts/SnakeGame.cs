using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Classic grid-based Snake in one self-contained component.
///
/// Setup: create an empty scene, add an empty GameObject, attach this script,
/// press Play. The script builds its own camera and quads at runtime.
///
/// Controls: arrow keys to steer, Space to restart after game over.
/// </summary>
public class SnakeGame : MonoBehaviour
{
    [Header("Board")]
    public int width = 20;
    public int height = 20;

    [Header("Speed")]
    [Tooltip("Seconds between snake steps. Lower is faster.")]
    public float stepInterval = 0.12f;

    static readonly Color BackgroundColor = new Color(0.10f, 0.10f, 0.12f);
    static readonly Color SnakeColor = new Color(0.40f, 0.85f, 0.40f);
    static readonly Color FoodColor = new Color(0.90f, 0.30f, 0.30f);

    readonly List<Vector2Int> cells = new List<Vector2Int>();   // [0] is the head
    readonly List<Transform> segments = new List<Transform>();  // reusable quad pool

    Transform snakeRoot;
    Transform food;
    Vector2Int direction = Vector2Int.right;
    Vector2Int pendingDirection = Vector2Int.right;
    Vector2Int foodCell;
    float stepTimer;
    int score;
    bool gameOver;

    void Start()
    {
        SetupCamera();
        NewGame();
    }

    void Update()
    {
        ReadInput();

        if (gameOver)
        {
            if (Input.GetKeyDown(KeyCode.Space)) NewGame();
            return;
        }

        stepTimer += Time.deltaTime;
        if (stepTimer >= stepInterval)
        {
            stepTimer -= stepInterval;
            Step();
        }
    }

    void SetupCamera()
    {
        Camera cam = Camera.main;
        if (cam == null)
        {
            var go = new GameObject("Main Camera") { tag = "MainCamera" };
            cam = go.AddComponent<Camera>();
        }

        cam.orthographic = true;
        cam.orthographicSize = Mathf.Max(width, height) / 2f + 1f;
        cam.transform.position = new Vector3(width / 2f - 0.5f, height / 2f - 0.5f, -10f);
        cam.backgroundColor = BackgroundColor;
    }

    void NewGame()
    {
        if (snakeRoot != null) Destroy(snakeRoot.gameObject);
        snakeRoot = new GameObject("Snake").transform;
        segments.Clear();
        cells.Clear();

        var start = new Vector2Int(width / 2, height / 2);
        for (int i = 0; i < 3; i++)
            cells.Add(new Vector2Int(start.x - i, start.y));

        direction = pendingDirection = Vector2Int.right;
        score = 0;
        gameOver = false;
        stepTimer = 0f;

        EnsureFood();
        PlaceFood();
        SyncSegments();
    }

    void ReadInput()
    {
        if (Input.GetKeyDown(KeyCode.UpArrow) && direction != Vector2Int.down)
            pendingDirection = Vector2Int.up;
        else if (Input.GetKeyDown(KeyCode.DownArrow) && direction != Vector2Int.up)
            pendingDirection = Vector2Int.down;
        else if (Input.GetKeyDown(KeyCode.LeftArrow) && direction != Vector2Int.right)
            pendingDirection = Vector2Int.left;
        else if (Input.GetKeyDown(KeyCode.RightArrow) && direction != Vector2Int.left)
            pendingDirection = Vector2Int.right;
    }

    void Step()
    {
        direction = pendingDirection;
        Vector2Int head = cells[0] + direction;

        bool hitWall = head.x < 0 || head.x >= width || head.y < 0 || head.y >= height;
        if (hitWall || cells.Contains(head))
        {
            gameOver = true;
            return;
        }

        cells.Insert(0, head);

        if (head == foodCell)
        {
            score++;
            PlaceFood();
        }
        else
        {
            cells.RemoveAt(cells.Count - 1);
        }

        SyncSegments();
    }

    void EnsureFood()
    {
        if (food != null) return;
        food = GameObject.CreatePrimitive(PrimitiveType.Quad).transform;
        food.name = "Food";
        food.GetComponent<Renderer>().material.color = FoodColor;
    }

    void PlaceFood()
    {
        var freeCells = new List<Vector2Int>();
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
            {
                var cell = new Vector2Int(x, y);
                if (!cells.Contains(cell)) freeCells.Add(cell);
            }

        if (freeCells.Count == 0)
        {
            gameOver = true; // board full: you win
            return;
        }

        foodCell = freeCells[Random.Range(0, freeCells.Count)];
        food.position = new Vector3(foodCell.x, foodCell.y, 0f);
    }

    void SyncSegments()
    {
        while (segments.Count < cells.Count)
        {
            var quad = GameObject.CreatePrimitive(PrimitiveType.Quad).transform;
            quad.name = "Segment";
            quad.SetParent(snakeRoot);
            quad.localScale = Vector3.one * 0.9f;
            quad.GetComponent<Renderer>().material.color = SnakeColor;
            segments.Add(quad);
        }

        for (int i = 0; i < segments.Count; i++)
        {
            bool used = i < cells.Count;
            segments[i].gameObject.SetActive(used);
            if (used)
                segments[i].position = new Vector3(cells[i].x, cells[i].y, 0f);
        }
    }

    void OnGUI()
    {
        GUI.color = Color.white;
        GUI.Label(new Rect(10, 10, 200, 20), "Score: " + score);
        if (gameOver)
            GUI.Label(new Rect(10, 30, 320, 20), "Game Over — press Space to restart");
    }
}
