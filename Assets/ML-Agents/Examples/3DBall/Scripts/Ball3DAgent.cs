using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

/// <summary>
/// AgentScopeBalancer — Unity ML-Agents controller for the 3DBall balancing task.
///
/// ─────────────────────────────────────────────────────────────────────────────
/// TASK:  Keep a rigid-body ball balanced on a tilting platform.
///
/// OBSERVATIONS (Space Size = 8):
///   [0]   Platform rotation (quaternion Z component)
///   [1]   Platform rotation (quaternion X component)
///   [2-4] Ball position relative to platform (X, Y, Z)
///   [5-7] Ball linear velocity (X, Y, Z)
///
/// ACTIONS (Continuous Size = 2):
///   [0]   Tilt along Z-axis (horizontal / left-right)
///   [1]   Tilt along X-axis (vertical / forward-back)
///
/// REWARDS:
///   +0.1  per step survived (living reward)
///   -1.0  terminal penalty on ball drop (SetReward override)
///
/// SETUP:
///   1. Attach this script to the Platform (Cube, Scale 5×0.2×5)
///   2. Create a Sphere with Rigidbody → drag into the "ball" field
///   3. Add Decision Requester (Period = 5) + Behavior Parameters:
///        Behavior Name: Ball3DBrain | Obs Size: 8 | Continuous: 2
///   4. Press Play → run:  python run_agent.py
/// ─────────────────────────────────────────────────────────────────────────────
/// </summary>
public class AgentScopeBalancer : Agent
{
    // ── Inspector ────────────────────────────────────────────────────────────

    [Header("Scene References")]
    [Tooltip("The ball GameObject that the platform must balance.")]
    public GameObject ball;

    [Header("Physics Tuning")]
    [Tooltip("Tilt multiplier applied to each continuous action.")]
    [Range(0.5f, 5f)]
    public float tiltMultiplier = 2f;

    [Tooltip("Random tilt range (degrees) applied to the platform on episode reset.")]
    [Range(0f, 15f)]
    public float resetTiltRange = 5f;

    [Tooltip("Height above platform centre where the ball spawns on reset.")]
    [Range(0.5f, 3f)]
    public float ballSpawnHeight = 1.5f;

    [Header("Episode Termination")]
    [Tooltip("Horizontal distance from centre beyond which the ball is 'fallen'.")]
    [Range(1f, 8f)]
    public float fallDistanceThreshold = 3.5f;

    [Tooltip("Height below platform beyond which the ball is 'dropped'.")]
    [Range(0.5f, 3f)]
    public float fallHeightThreshold = 1f;

    // ── Private ──────────────────────────────────────────────────────────────

    private Rigidbody ballRb;

    // ── Lifecycle ────────────────────────────────────────────────────────────

    void Start()
    {
        ballRb = ball.GetComponent<Rigidbody>();

        if (ballRb == null)
            Debug.LogError("[AgentScopeBalancer] Ball is missing a Rigidbody component!");
    }

    /// <summary>
    /// Called at the start of every episode.
    /// Resets platform with a small random tilt and repositions the ball.
    /// </summary>
    public override void OnEpisodeBegin()
    {
        // Reset platform with a slight random tilt to prevent trajectory memorisation.
        transform.rotation = Quaternion.Euler(
            Random.Range(-resetTiltRange, resetTiltRange),
            0f,
            Random.Range(-resetTiltRange, resetTiltRange)
        );

        // Reset ball: zero velocity, reposition above platform centre.
        ballRb.linearVelocity = Vector3.zero;
        ballRb.angularVelocity = Vector3.zero;
        ball.transform.position = new Vector3(
            transform.position.x,
            transform.position.y + ballSpawnHeight,
            transform.position.z
        );
    }

    /// <summary>
    /// Collects 8 observations the neural network uses to decide actions.
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        // Platform tilt — raw quaternion components (stable, no gimbal lock).
        sensor.AddObservation(transform.rotation.z);                        // [0]
        sensor.AddObservation(transform.rotation.x);                        // [1]

        // Ball position relative to platform centre (3 values).
        sensor.AddObservation(ball.transform.position - transform.position); // [2-4]

        // Ball velocity (3 values).
        sensor.AddObservation(ballRb.linearVelocity);                       // [5-7]
    }

    /// <summary>
    /// Receives 2 continuous actions from the policy and tilts the platform.
    /// </summary>
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var continuousActions = actionBuffers.ContinuousActions;
        float tiltZ = tiltMultiplier * Mathf.Clamp(continuousActions[0], -1f, 1f);
        float tiltX = tiltMultiplier * Mathf.Clamp(continuousActions[1], -1f, 1f);

        // Apply tilt — separate axis rotations for clean control.
        transform.Rotate(new Vector3(0, 0, 1), tiltZ);
        transform.Rotate(new Vector3(1, 0, 0), tiltX);

        // ── Reward ───────────────────────────────────────────────────────────

        // Living reward: +0.1 per step survived.
        AddReward(0.1f);

        // Failure check: ball fell off the platform.
        bool fallen =
            ball.transform.position.y < transform.position.y - fallHeightThreshold ||
            Mathf.Abs(ball.transform.position.x) > fallDistanceThreshold ||
            Mathf.Abs(ball.transform.position.z) > fallDistanceThreshold;

        if (fallen)
        {
            SetReward(-1f);  // Override cumulative reward with terminal penalty.
            EndEpisode();
        }
    }

    /// <summary>
    /// Manual keyboard control for testing (WASD / Arrow Keys).
    /// Set Behavior Type → Heuristic Only in the Inspector to use.
    /// </summary>
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Horizontal"); // Tilt Z
        continuousActionsOut[1] = Input.GetAxis("Vertical");   // Tilt X
    }

    // ── Editor Gizmos ────────────────────────────────────────────────────────

#if UNITY_EDITOR
    private void OnDrawGizmos()
    {
        if (ball == null) return;

        // Yellow boundary box showing the fall threshold.
        Gizmos.color = Color.yellow;
        Gizmos.DrawWireCube(
            transform.position,
            new Vector3(fallDistanceThreshold * 2f, 0.1f, fallDistanceThreshold * 2f)
        );

        // Green line from platform centre to ball.
        Gizmos.color = Color.green;
        Gizmos.DrawLine(transform.position, ball.transform.position);
    }
#endif
}
  