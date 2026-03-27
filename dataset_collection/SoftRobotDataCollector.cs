using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using Newtonsoft.Json;
using UnityEngine;

/// <summary>
/// Soft Growing Robot — TCP Trajectory Dataset Collection Controller (Network I)
///
/// Receives angle sequences from collect_traj_data.py, grows the robot
/// segment-by-segment, records tip coordinates, and appends the episode
/// trajectory to coordinates_new.json.
///
/// Protocol:
///   Python → Unity : SEQ:{n}:START | SEQ:{n}:{angle} | SEQ:{n}:END
///   Unity  → Python: READY | STATE:{x},{z},{rotY}
///                    ACK:{n}:START_DONE | SEGMENT_DONE | END_DONE
/// </summary>
public class SoftRobotDataCollector : MonoBehaviour
{
    #region ── Inspector Fields ──────────────────────────────────────

    [Header("TCP")]
    public int port = 5555;

    [Header("Robot")]
    [NonSerialized] public float segmentLength   = 0.2f;
    [NonSerialized] public int   initialSegments = 2;
    [NonSerialized] public int   ini_theta       = 0;

    [Header("Target points (visualisation only)")]
    [NonSerialized] public double t_x1 = 1.599;
    [NonSerialized] public double t_z1 = 4.317;
    [NonSerialized] public double t_x2 = -1.38;
    [NonSerialized] public double t_z2 = 4.313;

    [Header("Scene objects (auto-created if null)")]
    public GameObject groundPrefab;
    public List<GameObject> cubeObjects = new List<GameObject>();

    #endregion

    #region ── Private Fields ────────────────────────────────────────

    private TcpListener    _listener;
    private TcpClient      _client;
    private NetworkStream  _stream;
    private Thread         _listenThread;
    private bool           _running = false;

    private readonly Queue<string> _cmdQueue = new Queue<string>();
    private readonly object        _lock     = new object();
    private bool _busy    = false;
    private int  _lastSeq = -1;

    private GameObject             _head;
    private List<GameObject>       _segments = new List<GameObject>();
    private List<ConfigurableJoint>_joints   = new List<ConfigurableJoint>();
    private List<GameObject>       _pool     = new List<GameObject>();

    private List<float[]> _coords = new List<float[]>();
    private string _jsonPath;

    private static PhysicMaterial       _mat;
    private static SoftJointLimitSpring _angXSpring, _angYZSpring;
    private static JointDrive           _angXDrive,  _angYZDrive;
    private static bool                 _physicsReady = false;

    private string _username = System.Environment.UserName;

    #endregion

    #region ── Unity Lifecycle ───────────────────────────────────────

    void Start()
    {
        InitSharedPhysics(); SetupPath(); BuildScene(); StartServer();
        Debug.Log($"[Collector] Started on port {port}");
    }

    void Update()            => DrainQueue();
    void OnApplicationQuit() => Shutdown();
    void OnDestroy()         => Shutdown();

    #endregion

    #region ── Initialisation ────────────────────────────────────────

    void InitSharedPhysics()
    {
        if (_mat == null)
            _mat = new PhysicMaterial
            {
                bounciness = 0.5f, frictionCombine = PhysicMaterialCombine.Minimum,
                dynamicFriction = 0.5f, staticFriction = 0.5f
            };
        if (!_physicsReady)
        {
            _physicsReady = true;
            _angXSpring  = new SoftJointLimitSpring { spring = 100f, damper = 30f };
            _angYZSpring = new SoftJointLimitSpring { spring = 100f, damper = 30f };
            _angXDrive   = new JointDrive { positionSpring = 100f, positionDamper = 30f, maximumForce = float.MaxValue };
            _angYZDrive  = new JointDrive { positionSpring = 100f, positionDamper = 30f, maximumForce = float.MaxValue };
        }
    }

    void SetupPath()
    {
        string baseDir = Application.persistentDataPath;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            if (_username == "ayaka") baseDir = @"C:\Users\ayaka\PycharmProjects\pythonProject1\diffusion";
            else if (_username == "B610") baseDir = @"C:\Users\B610\PycharmProjects\PythonProject\diffusion";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            baseDir = "/media/dockang/D/haoran/pythonProject1/diffusion";

        _jsonPath = Path.Combine(baseDir, "coordinates_new.json");
        Debug.Log($"[Collector] JSON path: {_jsonPath}");
    }

    void BuildScene()
    {
        if (groundPrefab == null)
        {
            groundPrefab = GameObject.CreatePrimitive(PrimitiveType.Cube);
            groundPrefab.transform.localScale = new Vector3(10f, 0.1f, 10f);
            groundPrefab.transform.position   = new Vector3(2.09f, -0.182f, 0f);
            groundPrefab.GetComponent<Renderer>().material.color = Color.gray * 0.5f;
            groundPrefab.GetComponent<Collider>().material       = _mat;
        }
        SpawnCube(new Vector3(0.352f, 0.214f, 1.762f),
                  new Vector3(0f, 63.81f, 0f),
                  new Vector3(2.01f, 0.72f, 0.49f));
        SpawnSphere((float)t_x1, (float)t_z1, Color.red,   "Target1");
        SpawnSphere((float)t_x2, (float)t_z2, Color.green, "Target2");
    }

    void SpawnCube(Vector3 pos, Vector3 eulerRot, Vector3 scale)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.transform.SetPositionAndRotation(pos, Quaternion.Euler(eulerRot));
        go.transform.localScale = scale; go.tag = "RobotCube"; cubeObjects.Add(go);
    }

    void SpawnSphere(float x, float z, Color color, string name)
    {
        var s = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        Destroy(s.GetComponent<Collider>());
        s.transform.position   = new Vector3(x, 0.1f, z);
        s.transform.localScale = Vector3.one * 0.07f;
        s.GetComponent<Renderer>().material.color = color; s.name = name;
    }

    #endregion

    #region ── TCP Server ────────────────────────────────────────────

    void StartServer()
    {
        _running = true;
        _listenThread = new Thread(ListenLoop) { IsBackground = true };
        _listenThread.Start();
    }

    void Shutdown()
    {
        _running = false;
        try { _stream?.Close(); } catch { }
        try { _client?.Close(); } catch { }
        try { _listener?.Stop(); } catch { }
    }

    void ListenLoop()
    {
        try
        {
            _listener = new TcpListener(IPAddress.Any, port); _listener.Start();
            _client = _listener.AcceptTcpClient(); _stream = _client.GetStream();
            Debug.Log("[Collector] Python connected"); Send("READY");

            var buf = new byte[4096]; var sb = new StringBuilder();
            while (_running && _client.Connected)
            {
                int n = _stream.Read(buf, 0, buf.Length); if (n == 0) break;
                sb.Append(Encoding.UTF8.GetString(buf, 0, n));
                string[] lines = sb.ToString().Split('\n');
                for (int i = 0; i < lines.Length - 1; i++)
                {
                    string l = lines[i].Trim();
                    if (!string.IsNullOrEmpty(l)) lock (_lock) { _cmdQueue.Enqueue(l); }
                }
                sb.Clear(); sb.Append(lines[lines.Length - 1]);
            }
        }
        catch (Exception e) { Debug.LogError($"[Collector] Listen: {e.Message}"); }
    }

    void Send(string msg)
    {
        if (_stream == null || !_stream.CanWrite) return;
        try
        {
            byte[] d = Encoding.UTF8.GetBytes(msg + "\n");
            _stream.Write(d, 0, d.Length); _stream.Flush();
        }
        catch (Exception e) { Debug.LogError($"[Collector] Send: {e.Message}"); }
    }

    void DrainQueue()
    {
        if (_busy) return;
        string cmd = null;
        lock (_lock) { if (_cmdQueue.Count > 0) cmd = _cmdQueue.Dequeue(); }
        if (cmd != null) Dispatch(cmd);
    }

    #endregion

    #region ── Command Dispatch ──────────────────────────────────────

    void Dispatch(string raw)
    {
        if (!raw.StartsWith("SEQ:")) return;
        string[] parts = raw.Split(new[] { ':' }, 3);
        if (parts.Length != 3 || !int.TryParse(parts[1], out int seq)) return;
        if (seq <= _lastSeq) { Debug.LogWarning($"[Collector] Duplicate #{seq}"); return; }
        _lastSeq = seq; _busy = true;
        string payload = parts[2];

        if (payload == "START")
        {
            ResetScene(); _coords.Clear(); InitRobot();
            RecordAndSend(0f); Send($"ACK:{seq}:START_DONE"); _busy = false; return;
        }
        if (payload == "END")
        {
            AppendToJson(); Send($"ACK:{seq}:END_DONE"); _busy = false; return;
        }

        if (!float.TryParse(payload,
                System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture,
                out float angle))
        {
            Send($"ACK:{seq}:ERROR_INVALID_ANGLE"); _busy = false; return;
        }

        if (_head == null) { InitRobot(); RecordAndSend(0f); }
        float target = GetHeadYRot() + angle;
        _head = (Mathf.Abs(angle) < 0.01f) ? GrowStraight(_head, target) : GrowBend(_head, target);
        RecordAndSend(angle);
        Send($"ACK:{seq}:SEGMENT_DONE"); _busy = false;
    }

    void RecordAndSend(float angle)
    {
        if (_head == null) return;
        float x = _head.transform.position.x;
        float z = _head.transform.position.z;
        _coords.Add(new[] { x, z, angle });
        Send($"STATE:{x:F4},{z:F4},{GetHeadYRot():F4}");
    }

    #endregion

    #region ── Robot Growth ──────────────────────────────────────────

    void InitRobot()
    {
        _head = GetFromPool(Vector3.zero, Quaternion.Euler(90, 0, ini_theta), true);
        for (int i = 1; i < initialSegments; i++)
            _head = GrowStraight(_head, ini_theta);
    }

    GameObject GrowStraight(GameObject prev, float theta)
    {
        Vector3 top = prev.transform.position + prev.transform.up * segmentLength;
        Vector3 pos = top - prev.transform.up * 0.1f;
        var seg = GetFromPool(pos, Quaternion.Euler(90, 0, theta), false);
        IgnoreSegmentCollisions(seg);
        AttachJoint(seg, prev, Vector3.up * (-5f * segmentLength / 2f));
        return seg;
    }

    GameObject GrowBend(GameObject prev, float theta)
    {
        Vector3    top   = prev.transform.position + prev.transform.up * segmentLength / 4f;
        float      r     = prev.transform.localScale.x / 2f;
        Vector3    rotUp = Quaternion.Euler(0, theta, 0) * prev.transform.up;
        Vector3    pos   = top + rotUp * (segmentLength - 2f * r);
        Quaternion segRot = Quaternion.Euler(90, 0, theta);
        Vector3    segUp  = segRot * Vector3.up;
        pos += top - (pos - segUp * segmentLength / 4f);
        var seg = GetFromPool(pos, segRot, false);
        IgnoreSegmentCollisions(seg);
        AttachJoint(seg, prev, seg.transform.position - segUp * segmentLength / 4f);
        return seg;
    }

    void AttachJoint(GameObject seg, GameObject prev, Vector3 anchor)
    {
        var j = seg.AddComponent<ConfigurableJoint>();
        j.connectedBody = prev.GetComponent<Rigidbody>();
        j.targetRotation = Quaternion.identity;
        j.xMotion = j.yMotion = j.zMotion = ConfigurableJointMotion.Locked;
        j.angularXMotion = j.angularYMotion = ConfigurableJointMotion.Locked;
        j.angularZMotion = ConfigurableJointMotion.Free;
        j.angularXLimitSpring = _angXSpring; j.angularYZLimitSpring = _angYZSpring;
        j.angularXDrive = _angXDrive;        j.angularYZDrive = _angYZDrive;
        j.anchor = anchor; _joints.Add(j);
    }

    void IgnoreSegmentCollisions(GameObject newSeg)
    {
        var col = newSeg.GetComponent<Collider>();
        foreach (var s in _segments)
            if (s != null && s != newSeg)
                Physics.IgnoreCollision(s.GetComponent<Collider>(), col);
    }

    float GetHeadYRot() =>
        _head == null ? 0f :
        Vector3.SignedAngle(_head.transform.up, Vector3.forward, Vector3.up);

    #endregion

    #region ── Object Pool ───────────────────────────────────────────

    void ResetScene()
    {
        foreach (var j in _joints) if (j) Destroy(j);
        _joints.Clear();
        foreach (var seg in _segments)
        {
            if (seg == null) continue;
            var rb = seg.GetComponent<Rigidbody>();
            if (rb) { rb.velocity = rb.angularVelocity = Vector3.zero; }
            seg.SetActive(false); _pool.Add(seg);
        }
        _segments.Clear(); _head = null;
    }

    GameObject GetFromPool(Vector3 pos, Quaternion rot, bool kinematic)
    {
        GameObject cap = null;
        while (_pool.Count > 0 && cap == null)
        {
            cap = _pool[_pool.Count - 1]; _pool.RemoveAt(_pool.Count - 1);
        }
        if (cap == null)
        {
            cap = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            cap.tag = "RobotSegment";
            cap.GetComponent<Collider>().material = _mat;
            cap.AddComponent<Rigidbody>().mass    = 0.001f;
        }
        cap.transform.position   = pos; cap.transform.rotation = rot;
        cap.transform.localScale = new Vector3(0.182f, segmentLength / 2f, 0.182f);
        cap.SetActive(true);
        var rb = cap.GetComponent<Rigidbody>();
        rb.velocity = rb.angularVelocity = Vector3.zero;
        rb.isKinematic = kinematic; rb.mass = 0.001f;
        _segments.Add(cap); return cap;
    }

    #endregion

    #region ── Dataset Saving ────────────────────────────────────────

    /// <summary>
    /// Append episode tip trajectory to the JSON dataset file.
    /// Format: [{"timestamp": "...", "coordinates": [[x, z, angle], ...]}, ...]
    /// </summary>
    void AppendToJson()
    {
        try
        {
            var entry = new { timestamp = DateTime.UtcNow.ToString("o"), coordinates = _coords };
            var all   = new List<object>();
            if (File.Exists(_jsonPath))
            {
                try
                {
                    var prev = JsonConvert.DeserializeObject<List<object>>(File.ReadAllText(_jsonPath));
                    if (prev != null) all.AddRange(prev);
                }
                catch { }
            }
            all.Add(entry);
            File.WriteAllText(_jsonPath, JsonConvert.SerializeObject(all, Formatting.Indented));
            Debug.Log($"[Collector] Saved {_coords.Count} waypoints → {_jsonPath}");
        }
        catch (Exception e) { Debug.LogError($"[Collector] AppendToJson: {e.Message}"); }
    }

    #endregion
}
