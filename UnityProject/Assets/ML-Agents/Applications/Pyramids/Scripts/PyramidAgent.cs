using System;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;
using MLAgents;

public class PyramidAgent : Agent
{
    public GameObject area;
    private PyramidArea m_MyArea;
    private Rigidbody m_AgentRb;
    private RayPerception m_RayPer;
    private PyramidSwitch m_SwitchLogic;
    public GameObject areaSwitch;
    public bool useVectorObs;
    private List<String> MEMORY = new List<String>();
    private const int POSITION_ROUNDING = 0;
    private const int POSITION_DECIMAL_ROUNDING = 0;

    public override void InitializeAgent()
    {
        base.InitializeAgent();
        m_AgentRb = GetComponent<Rigidbody>();
        m_MyArea = area.GetComponent<PyramidArea>();
        m_RayPer = GetComponent<RayPerception>();
        m_SwitchLogic = areaSwitch.GetComponent<PyramidSwitch>();
    }

    public override void CollectObservations()
    {
        if (useVectorObs)
        {
        /*
            const float rayDistance = 35f;
            float[] rayAngles = { 20f, 90f, 160f, 45f, 135f, 70f, 110f };
            float[] rayAngles1 = { 25f, 95f, 165f, 50f, 140f, 75f, 115f };
            float[] rayAngles2 = { 15f, 85f, 155f, 40f, 130f, 65f, 105f };

            string[] detectableObjects = { "block", "wall", "goal", "switchOff", "switchOn", "stone" };
            AddVectorObs(m_RayPer.Perceive(rayDistance, rayAngles, detectableObjects, 0f, 0f));
            AddVectorObs(m_RayPer.Perceive(rayDistance, rayAngles1, detectableObjects, 0f, 5f));
            AddVectorObs(m_RayPer.Perceive(rayDistance, rayAngles2, detectableObjects, 0f, 10f));
            AddVectorObs(m_SwitchLogic.GetState());
            AddVectorObs(transform.InverseTransformDirection(m_AgentRb.velocity));
            */

            AddVectorObs(transform.position);
        }
    }

    public void MoveAgent(float[] act)
    {
        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        var action = Mathf.FloorToInt(act[0]);
        switch (action)
        {
            case 1:
                dirToGo = transform.forward * 1f;
                break;
            case 2:
                dirToGo = transform.forward * -1f;
                break;
            case 3:
                rotateDir = transform.up * 1f;
                break;
            case 4:
                rotateDir = transform.up * -1f;
                break;
        }
        transform.Rotate(rotateDir, Time.deltaTime * 200f);
        m_AgentRb.AddForce(dirToGo * 2f, ForceMode.VelocityChange);
    }

    private float count_position_occurrences(String current_position)
    {
        int occurences = 0;
        foreach (String m in MEMORY)
        {
            if (m == current_position) {
                occurences++;
            }
        }

        return occurences;
    }

    public override void AgentAction(float[] vectorAction)
    {
        // Calculate exploration reward
        float exploration_reward = 0;
        String current_position = String.Format(
            "{0}:{1}",
            Math.Round(transform.position.x * Math.Pow(10, POSITION_DECIMAL_ROUNDING), POSITION_ROUNDING),
            Math.Round(transform.position.z * Math.Pow(10, POSITION_DECIMAL_ROUNDING), POSITION_ROUNDING)
        );
        float position_occurrences = count_position_occurrences(current_position);

        MEMORY.Add(current_position);

        if(position_occurrences > 0){
            exploration_reward += 0.1f / position_occurrences;
        } else {
            exploration_reward += 1;
        }

        AddReward(exploration_reward);
        MoveAgent(vectorAction);
    }

    public override float[] Heuristic()
    {
        if (Input.GetKey(KeyCode.D))
        {
            return new float[] { 3 };
        }
        if (Input.GetKey(KeyCode.W))
        {
            return new float[] { 1 };
        }
        if (Input.GetKey(KeyCode.A))
        {
            return new float[] { 4 };
        }
        if (Input.GetKey(KeyCode.S))
        {
            return new float[] { 2 };
        }
        return new float[] { 0 };
    }

    public override void AgentReset()
    {
        var enumerable = Enumerable.Range(0, 9).OrderBy(x => Guid.NewGuid()).Take(9);
        var items = enumerable.ToArray();

        m_MyArea.CleanPyramidArea();

        m_AgentRb.velocity = Vector3.zero;
        m_MyArea.PlaceObject(gameObject, items[0]);
        transform.rotation = Quaternion.Euler(new Vector3(0f, Random.Range(0, 360)));

        m_SwitchLogic.ResetSwitch(items[1], items[2]);
        m_MyArea.CreateStonePyramid(1, items[3]);
        m_MyArea.CreateStonePyramid(1, items[4]);
        m_MyArea.CreateStonePyramid(1, items[5]);
        m_MyArea.CreateStonePyramid(1, items[6]);
        m_MyArea.CreateStonePyramid(1, items[7]);
        m_MyArea.CreateStonePyramid(1, items[8]);

        MEMORY.Clear();
    }

    private void OnCollisionEnter(Collision collision)
    {

        /*
        if (collision.gameObject.CompareTag("goal"))
        {
            SetReward(2f);
            Done();
        }
        */
    }

    public override void AgentOnDone()
    {
    }
}
