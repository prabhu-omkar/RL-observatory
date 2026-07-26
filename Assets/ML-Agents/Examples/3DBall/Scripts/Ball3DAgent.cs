using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

/// <summary>
/// Ball3DAgent -- Unity ML-Agents C# controller for the 3DBall platform balancing task.
///
/// TASK DESCRIPTION
/// -------------------------------------------------------------------------------
/// A rigid-body ball is placed on top of a flat platform. The agent's job is
/// to keep the ball balanced by tilting the platform along its X and Z axes.
/// The episode ends when the ball falls off the platform, or when the
/// configurable max step count is reached (success / timeout).
///
/// OBSERVATION SPACE  (8 continuous values)
/// -------------------------------------------------------------------------------
///  [0]   Platform rotation X (euler, normalised to +/-1)
///  [1]   Platform rotation Z (euler, normalised to +/-1)
///  [2]   Ball X-position relative to platform centre
///  [3]   Ball Y-position relative to platform centre
///  [4]   Ball Z-position relative to platform centre
///  [5]   Ball X-velocity
///  [6]   Ball Y-velocity
///  [7]   Ball Z-velocity
///
/// ACTION SPACE  (2 continuous actions, each in [-1, 1])
/// -------------------------------------------------------------------------------
///  [0]   Tilt force along Z-axis (controls left / right lean)
///  [1]   Tilt force along X-axis (controls forward / backward lean)
///
/// REWARD STRUCTURE
/// -------------------------------------------------------------------------------
///  +0.1   per step  -- living reward to encourage survival
///  -1.0   terminal  -- ball falls off platform (episode failure)
///
/// SETUP INSTRUCTIONS  (see README.md for full walkthrough)
/// -------------------------------------------------------------------------------
///  1. Import the ML-Agents Unity Package (Package Manager -> Add by name ->
///     com.unity.ml-agents).
///  2. Create a new scene or open an existing one.
///  3. Add a Plane (or Cube scaled to a flat slab) -- this is the Platform.
///  4. Add a Sphere -- this is the Ball. Add a Rigidbody to it.
///  5. Attach THIS script to the Platform GameObject.
///  6. Drag the Ball into the "Ball" field in the Inspector.
///  7. Add a Decision Requester component (set Decision Period = 5).
///  8. Add a Behavior Parameters component and configure it:
///       Behavior Name     : Ball3DBrain
///       Vector Observation: Space Size = 8
///       Continuous Actions: 2
///       Behavior Type     : Default (for training via Python)
///  9. Press Play in Unity, then run:  python run_agent.py
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class Ball3DAgent : Agent
{
    // --- Inspector-exposed fields -------------------------------------------

    [Header("Scene References")]
    [Tooltip("The ball Rigidbody that the platform must balance.")]
    public Rigidbody ball;

    [Tooltip("The platform Transform (usually this GameObject).")]
    public Transform platform;

    [Header("Physics Settings")]
    [Tooltip("Maximum tilt angle applied per action step (degrees).")]
    [Range(1f, 30f)]
    public float tiltForce = 10f;

    [Tooltip("How fast the platform rotates back towards neutral when no action is applied.")]
    [Range(0f, 5f)]
    public float resetSpeed = 1f;

    [Header("Ball Spawn Randomisation")]
    [Tooltip("Random offset radius for ball starting position on reset.")]
    [Range(0f, 2f)]
    public float spawnRadius = 1.5f;

    [Tooltip("Random linear velocity magnitude given to ball on reset.")]
    [Range(0f, 2f)]
    public float spawnVelocity = 0.5f;

    [Header("Episode Termination")]
    [Tooltip("Distance from platform centre beyond which the ball is considered fallen.")]
    [Range(1f, 8f)]
    public float fallThreshold = 3.0f;

    [Tooltip("Height below which the ball is considered dropped off the platform.")]
    public float dropHeightThreshold = -2f;

    // --- Private state -------------------------------------------------------

    private Vector3 _initialBallPosition;
    private Rigidbody _platformRb;

    // --- ML-Agents lifecycle overrides --------------------------------------

    /// <summary>Called once by ML-Agents when the scene loads.</summary>
    public override void Initialize()
    {
        _platformRb = GetComponent<Rigidbody>();

        // Make the platform kinematic -- physics only apply to the ball.
        if (_platformRb != null)
            _platformRb.isKinematic = true;

        // Cache the ball local rest position above the platform centre.
        if (ball != null)
            _initialBallPosition = ball.transform.localPosition;

        // Validate Inspector wiring.
        if (ball == null)
            Debug.LogError("[Ball3DAgent] Ball Rigidbody is not assigned in the Inspector!");
    }

    /// <summary>
    /// Called at the beginning of every episode.
    /// Resets platform rotation and randomises ball start position / velocity.
    /// </summary>
    public override void OnEpisodeBegin()
    {
        // Reset platform to flat (no tilt).
        transform.rotation = Quaternion.identity;

        if (ball == null) return;

        // Randomise ball start position within a disc above the platform.
        float rx = Random.Range(-spawnRadius, spawnRadius);
        float rz = Random.Range(-spawnRadius, spawnRadius);
        ball.transform.localPosition = _initialBallPosition + new Vector3(rx, 0f, rz);

        // Zero existing velocity, then apply a small random impulse.
        ball.linearVelocity = Vector3.zero;
        ball.angularVelocity = Vector3.zero;

        Vector3 randomImpulse = new Vector3(
            Random.Range(-spawnVelocity, spawnVelocity),
            0f,
            Random.Range(-spawnVelocity, spawnVelocity)
        );
        ball.AddForce(randomImpulse, ForceMode.VelocityChange);
    }

    /// <summary>
    /// Called every Decision Period (set on DecisionRequester).
    /// Appends all 8 observation values to the sensor buffer.
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        if (ball == null)
        {
            // Pad observations with zeros if ball reference is missing.
            sensor.AddObservation(new float[8]);
            return;
        }

        // Platform tilt (normalise euler angles from [0,360] to [-1,1] range).
        sensor.AddObservation(NormaliseAngle(transform.rotation.eulerAngles.x));
        sensor.AddObservation(NormaliseAngle(transform.rotation.eulerAngles.z));

        // Ball position relative to platform centre.
        Vector3 relativePos = ball.transform.localPosition;
        sensor.AddObservation(relativePos.x);
        sensor.AddObservation(relativePos.y);
        sensor.AddObservation(relativePos.z);

        // Ball velocity.
        sensor.AddObservation(ball.linearVelocity.x);
        sensor.AddObservation(ball.linearVelocity.y);
        sensor.AddObservation(ball.linearVelocity.z);
    }

    /// <summary>
    /// Called when the Python policy (or heuristic) sends an action.
    /// actions.ContinuousActions[0] = Z-axis tilt
    /// actions.ContinuousActions[1] = X-axis tilt
    /// </summary>
    public override void OnActionReceived(ActionBuffers actions)
    {
        if (ball == null) return;

        float actionZ = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float actionX = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

        // Apply tilt as a direct rotation delta (scaled by tiltForce and dt).
        transform.Rotate(
            new Vector3(actionX * tiltForce, 0f, actionZ * tiltForce) * Time.fixedDeltaTime
        );

        // --- Reward Structure ------------------------------------------------

        // Living reward -- encourage the agent to keep the ball on the platform.
        AddReward(0.1f);

        // Check if ball has fallen off.
        bool fallenOff =
            Mathf.Abs(ball.transform.localPosition.x) > fallThreshold ||
            Mathf.Abs(ball.transform.localPosition.z) > fallThreshold ||
            ball.transform.position.y < (transform.position.y + dropHeightThreshold);

        if (fallenOff)
        {
            // Penalty and episode termination.
            AddReward(-1f);
            EndEpisode();
        }
    }

    /// <summary>
    /// Manual control for testing without ML-Agents (keyboard fallback).
    /// WASD / Arrow keys tilt the platform.
    /// </summary>
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;

        // Horizontal input = Z-axis tilt.
        continuousActions[0] = -Input.GetAxis("Horizontal");
        // Vertical input   = X-axis tilt.
        continuousActions[1] =  Input.GetAxis("Vertical");
    }

    // --- Helper methods ------------------------------------------------------

    /// <summary>
    /// Normalises a Unity euler angle (0-360) to the range [-1, 1].
    /// </summary>
    private float NormaliseAngle(float angle)
    {
        // Wrap [0,360] to [-180,180] first.
        if (angle > 180f) angle -= 360f;
        // Normalise to [-1, 1] assuming max meaningful tilt is 90 degrees.
        return angle / 90f;
    }

    // --- Gizmo visualisation (Editor only) -----------------------------------

#if UNITY_EDITOR
    private void OnDrawGizmos()
    {
        if (ball == null) return;

        // Draw fall-threshold boundary as a wire cube in the Scene view.
        Gizmos.color = Color.yellow;
        Gizmos.matrix = transform.localToWorldMatrix;
        Gizmos.DrawWireCube(
            Vector3.zero,
            new Vector3(fallThreshold * 2f, 0.05f, fallThreshold * 2f)
        );

        // Draw a line from platform centre to ball.
        Gizmos.color = Color.green;
        Gizmos.DrawLine(
            transform.position,
            ball.transform.position
        );
    }
#endif
}
