using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>
/// Soft Growing Robot — TCP Offline RL Dataset Collection Controller
///
/// Runs two parallel scenes simultaneously:
///   Original scene : no rotation applied (clean reference)
///   Comparison scene: same growth + additive Gaussian noise + rotation actions
///
/// After each segment addition or rotation, a 6-D state vector is pushed to Python:
///   STATE_VECTOR:{robot_length},{end_x},{end_z},{growth_angle},{comp_x},{comp_z}
///
/// Communication protocol (same as SoftRobotDataCollector.cs):
///   Python → Unity : SEQ:{n}:START          reset both scenes
///   Python → Unity : SEQ:{n}:{angle}        add one segment
///   Python → Unity : SEQ:{n}:ROTATE:{a}:{p} rotate comparison scene at pivot p
///   Python → Unity : SEQ:{n}:END            end episode, save JSON
///
///   Unity → Python : READY
///   Unity → Python : STATE_VECTOR:{l},{ex},{ez},{θ},{ecx},{ecz}
///   Unity → Python : ACK:{n}:START_DONE / SEGMENT_DONE / ROTATE_DONE / END_DONE
/// </summary>
public class SoftRobotRLDataCollector : MonoBehaviour
{
    #region ── Inspector Fields ──────────────────────────────────────

    [Header("TCP")]
    public int port = 5555;

    [Header("Robot")]
    [NonSerialized] public float segmentLength = 0.2f;
    [NonSerialized] public int   initialSegments = 2;
    [NonSerialized] public int   ini_theta = 0;

    [Header("Comparison scene")]
    public float comparisonOffsetX = 5.0f;    // X offset of the comparison scene
    public float gaussianNoiseStd  = 2.0f;    // Gaussian noise std-dev (degrees)

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

    // TCP
    private TcpListener   _listener;
    private TcpClient     _client;
    private NetworkStream _stream;
    private Thread        _listenThread;
    private bool          _running = false;

    private readonly Queue<string> _cmdQueue = new Queue<string>();
    private readonly object        _lock     = new object();
    private bool _busy    = false;
    private int  _lastSeq = -1;

    // Original scene
    private GameObject             _head;
    private List<GameObject>       _segments = new List<GameObject>();
    private List<ConfigurableJoint>_joints   = new List<ConfigurableJoint>();

    // Comparison scene
    private GameObject             _headComp;
    private List<GameObject>       _segmentsComp = new List<GameObject>();
    private List<ConfigurableJoint>_jointsComp   = new List<ConfigurableJoint>();
    private List<GameObject>       _cubesComp    = new List<GameObject>();

    // Per-episode coordinate logs
    private string _growthLog, _rotLog;
    private string _growthLogComp, _rotLogComp;

    // Output JSON paths
    private string _jsonPath, _jsonPathComp;

    // Shared physics
    private static PhysicMaterial        _mat;
    private static SoftJointLimitSpring  _angXSpring, _angYZSpring;
    private static JointDrive            _angXDrive,  _angYZDrive;
    private static bool                  _physReady = false;

    private string _username = Environment.UserName;
    private System.Random _rng = new System.Random();

    private bool _endRunning = false;

    #endregion

    #region ── Unity Lifecycle ───────────────────────────────────────

    void Start()
    {
        InitSharedPhysics();
        SetupPaths();
        RemoveExtraAudioListeners();
        VisualizeTarget(0f, 0f);
        VisualizeTarget(comparisonOffsetX, 0f);
        CreateGround();
        StartServer();
        Debug.Log($"[Collector] Started on port {port}  compOffset={comparisonOffsetX}");
    }

    void Update()        => DrainQueue();
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
        if (!_physReady)
        {
            _physReady  = true;
            _angXSpring = new SoftJointLimitSpring { spring = 100f, damper = 30f };
            _angYZSpring = new SoftJointLimitSpring { spring = 100f, damper = 30f };
            _angXDrive  = new JointDrive { positionSpring = 100f, positionDamper = 30f, maximumForce = float.MaxValue };
            _angYZDrive = new JointDrive { positionSpring = 100f, positionDamper = 30f, maximumForce = float.MaxValue };
        }
    }

    void SetupPaths()
    {
        string baseDir = Application.persistentDataPath;
        if (Application.platform == RuntimePlatform.WindowsEditor ||
            Application.platform == RuntimePlatform.WindowsPlayer)
        {
            if (_username == "ayaka") baseDir = @"C:\Users\ayaka\PycharmProjects\pythonProject1\diffusion";
            else if (_username == "B610") baseDir = @"C:\Users\B610\PycharmProjects\PythonProject\diffusion";
        }
        else if (Application.platform == RuntimePlatform.LinuxEditor ||
                 Application.platform == RuntimePlatform.LinuxPlayer)
            baseDir = "/media/dockang/D/haoran/pythonProject1/diffusion";

        _jsonPath     = Path.Combine(baseDir, "coordinates_new.json");
        _jsonPathComp = Path.Combine(baseDir, "coordinates_new_comparison.json");
        _growthLog     = Path.Combine(baseDir, "growth_orig.txt");
        _rotLog        = Path.Combine(baseDir, "rotation_orig.txt");
        _growthLogComp = Path.Combine(baseDir, "growth_comp.txt");
        _rotLogComp    = Path.Combine(baseDir, "rotation_comp.txt");
        Debug.Log($"[Collector] JSON: {_jsonPath}  |  {_jsonPathComp}");
    }

    void RemoveExtraAudioListeners()
    {
        var listeners = FindObjectsOfType<AudioListener>();
        for (int i = 1; i < listeners.Length; i++) Destroy(listeners[i]);
    }

    void CreateGround()
    {
        if (groundPrefab != null) return;
        groundPrefab = GameObject.CreatePrimitive(PrimitiveType.Cube);
        groundPrefab.transform.localScale = new Vector3(30f, 0.1f, 20f);
        groundPrefab.transform.position   = new Vector3(2.09f + comparisonOffsetX / 2f, -0.182f, 0f);
        groundPrefab.GetComponent<Renderer>().material.color = Color.gray * 0.5f;
        groundPrefab.GetComponent<Collider>().material       = _mat;
    }

    void VisualizeTarget(double x, double z)
    {
        var s = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        Destroy(s.GetComponent<Collider>());
        s.transform.position   = new Vector3((float)x, 0.1f, (float)z);
        s.transform.localScale = Vector3.one * 0.07f;
        s.GetComponent<Renderer>().material.color = Color.red;
    }

    #endregion

    #region ── TCP Server ────────────────────────────────────────────

    void StartServer()
    {
        _running      = true;
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
            _listener = new TcpListener(IPAddress.Any, port);
            _listener.Start();
            _client = _listener.AcceptTcpClient();
            _stream = _client.GetStream();
            Debug.Log("[Collector] Python connected");
            Send("READY");

            var buf = new byte[4096]; var sb = new StringBuilder();
            while (_running && _client.Connected)
            {
                int n = _stream.Read(buf, 0, buf.Length);
                if (n == 0) break;
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
        if (seq <= _lastSeq) return;
        _lastSeq = seq; _busy = true;
        string payload = parts[2];

        if (payload == "START")
        {
            ResetScene();
            if (File.Exists(_growthLog)) File.Delete(_growthLog);
            if (File.Exists(_growthLogComp)) File.Delete(_growthLogComp);
            InitRobot(false); InitRobot(true);
            WriteLog(0f, _head, true, false); WriteLog(0f, _headComp, true, true);
            Send($"ACK:{seq}:START_DONE"); _busy = false; return;
        }

        if (payload == "END")
        {
            StartCoroutine(EndSequence(seq)); return;
        }

        if (payload.StartsWith("ROTATE:"))
        {
            string[] rp = payload.Substring(7).Split(':');
            if (rp.Length == 2 && float.TryParse(rp[0], out float ang) && int.TryParse(rp[1], out int pivot))
                StartCoroutine(RotateComparison(ang, pivot, seq));
            else { Send($"ACK:{seq}:ERROR_INVALID_ROTATE"); _busy = false; }
            return;
        }

        if (!float.TryParse(payload, System.Globalization.NumberStyles.Float,
                              System.Globalization.CultureInfo.InvariantCulture, out float angle))
        {
            Send($"ACK:{seq}:ERROR_INVALID_ANGLE"); _busy = false; return;
        }

        // Add segment — original scene (clean)
        float rot0 = GetYRot(_head);
        _head = (Mathf.Abs(angle) < 0.01f)
            ? GrowStraight(_head, rot0 + angle, false)
            : GrowBend(_head, rot0 + angle, false);
        WriteLog(angle, _head, true, false);

        // Add segment — comparison scene (with noise)
        float noisy = (Mathf.Abs(angle) >= 0.01f) ? angle + (float)GaussianNoise(0, gaussianNoiseStd) : 0f;
        float rotC  = GetYRot(_headComp);
        _headComp = (Mathf.Abs(noisy) < 0.01f)
            ? GrowStraight(_headComp, rotC + noisy, true)
            : GrowBend(_headComp, rotC + noisy, true);
        WriteLog(noisy, _headComp, true, true);

        SendStateVector();
        Send($"ACK:{seq}:SEGMENT_DONE"); _busy = false;
    }

    IEnumerator RotateComparison(float angleDeg, int pivotIdx, int seq)
    {
        if (File.Exists(_rotLogComp)) File.Delete(_rotLogComp);

        int N = _segmentsComp.Count;
        if (pivotIdx < 2 || pivotIdx >= N) { Send($"ACK:{seq}:ERROR_PIVOT_OUT_OF_RANGE"); _busy = false; yield break; }

        int ci = pivotIdx - 1, fi = ci - 1, ti = ci + 1;
        if (fi < 0 || ti >= N) { Send($"ACK:{seq}:ERROR_PIVOT_OUT_OF_RANGE"); _busy = false; yield break; }

        Vector3 p1 = _segmentsComp[fi].transform.position;
        Vector3 p2 = _segmentsComp[ci].transform.position;
        Vector3 p3 = _segmentsComp[ti].transform.position;

        // Lock affected segments
        for (int i = fi; i < N; i++)
        {
            var rb = _segmentsComp[i].GetComponent<Rigidbody>();
            if (rb) { rb.isKinematic = true; rb.velocity = rb.angularVelocity = Vector3.zero; }
            var j = _segmentsComp[i].GetComponent<ConfigurableJoint>();
            if (j) j.connectedBody = null;
        }

        // Save original transforms
        var origPos = new Vector3[N]; var origRot = new Quaternion[N];
        for (int i = 0; i < N; i++)
        {
            origPos[i] = _segmentsComp[i].transform.position;
            origRot[i] = _segmentsComp[i].transform.rotation;
        }

        float perJoint = angleDeg / 3f;
        int   steps    = Mathf.CeilToInt(0.8f / Time.fixedDeltaTime);
        for (int s = 0; s < steps; s++)
        {
            float t = (float)(s + 1) / steps;
            Quaternion r1 = Quaternion.AngleAxis(perJoint * t, Vector3.up);
            Quaternion r2 = Quaternion.AngleAxis(perJoint * t, Vector3.up);
            Quaternion r3 = Quaternion.AngleAxis(perJoint * t, Vector3.up);

            var tp = new Vector3[N]; var tr = new Quaternion[N];
            for (int i = 0; i < fi; i++) { tp[i] = _segmentsComp[i].transform.position; tr[i] = _segmentsComp[i].transform.rotation; }
            for (int i = fi; i < N; i++) { tp[i] = p1 + r1 * (origPos[i] - p1); tr[i] = r1 * origRot[i]; }
            Vector3 rp2 = p1 + r1 * (p2 - p1);
            for (int i = ci; i < N; i++) { tp[i] = rp2 + r2 * (tp[i] - rp2); tr[i] = r2 * tr[i]; }
            Vector3 rp3 = rp2 + r2 * (p3 - p2);
            for (int i = ti; i < N; i++) { tp[i] = rp3 + r3 * (tp[i] - rp3); tr[i] = r3 * tr[i]; }
            for (int i = fi; i < N; i++) { _segmentsComp[i].transform.position = tp[i]; _segmentsComp[i].transform.rotation = tr[i]; }
            yield return new WaitForFixedUpdate();
        }

        // Unlock
        for (int i = fi; i < N; i++)
        {
            var rb = _segmentsComp[i].GetComponent<Rigidbody>();
            if (rb) { rb.isKinematic = false; rb.velocity = rb.angularVelocity = Vector3.zero; }
        }
        yield return new WaitForFixedUpdate();

        WriteLog(angleDeg, _headComp, false, true);
        WriteLog(0f,       _head,     false, false);
        SendStateVector();
        Send($"ACK:{seq}:ROTATE_DONE"); _busy = false;
    }

    IEnumerator EndSequence(int seq)
    {
        if (_endRunning) { Send($"ACK:{seq}:END_ALREADY_RUNNING"); _busy = false; yield break; }
        _endRunning = true;
        yield return new WaitForSeconds(0.5f);

        ConvertToJson(_growthLog, _rotLog, _jsonPath);
        ConvertToJson(_growthLogComp, _rotLogComp, _jsonPathComp);
        yield return new WaitForSeconds(0.3f);

        ResetScene();
        Send($"ACK:{seq}:END_DONE"); _busy = false; _endRunning = false;
    }

    #endregion

    #region ── State Vector ──────────────────────────────────────────

    void SendStateVector()
    {
        float len = 0f;
        for (int i = 1; i < _segments.Count; i++)
            len += Vector3.Distance(_segments[i-1].transform.position, _segments[i].transform.position);

        float ex = _head     != null ? _head.transform.position.x     : 0f;
        float ez = _head     != null ? _head.transform.position.z     : 0f;
        float cx = _headComp != null ? _headComp.transform.position.x - comparisonOffsetX : 0f;
        float cz = _headComp != null ? _headComp.transform.position.z : 0f;

        float growthAngle = 0f;
        if (_segments.Count >= 2)
        {
            Vector3 dir = _segments[_segments.Count-1].transform.position -
                          _segments[_segments.Count-2].transform.position;
            growthAngle = Mathf.Atan2(dir.z, dir.x) * Mathf.Rad2Deg;
        }

        Send($"STATE_VECTOR:{len:F4},{ex:F4},{ez:F4},{growthAngle:F4},{cx:F4},{cz:F4}");
    }

    #endregion

    #region ── Robot Growth ──────────────────────────────────────────

    void InitRobot(bool isComp)
    {
        Vector3 origin = isComp ? new Vector3(comparisonOffsetX, 0, 0) : Vector3.zero;
        var head = MakeCapsule(origin, Quaternion.Euler(90, 0, ini_theta), true, isComp);
        if (isComp) _headComp = head; else _head = head;
        for (int i = 1; i < initialSegments; i++)
        {
            if (isComp) _headComp = GrowStraight(_headComp, ini_theta, true);
            else         _head     = GrowStraight(_head,     ini_theta, false);
        }
    }

    GameObject GrowStraight(GameObject prev, float theta, bool isComp)
    {
        Vector3 top = prev.transform.position + prev.transform.up * segmentLength;
        Vector3 pos = top - prev.transform.up * 0.1f;
        var seg = MakeCapsule(pos, Quaternion.Euler(90, 0, theta), false, isComp);
        IgnoreIntraCollisions(seg, isComp);
        AttachJoint(seg, prev, Vector3.up * (-5f * segmentLength / 2f), isComp);
        return seg;
    }

    GameObject GrowBend(GameObject prev, float theta, bool isComp)
    {
        Vector3   top    = prev.transform.position + prev.transform.up * segmentLength / 4f;
        float     r      = prev.transform.localScale.x / 2f;
        Vector3   rotUp  = Quaternion.Euler(0, theta, 0) * prev.transform.up;
        Vector3   pos    = top + rotUp * (segmentLength - 2f * r);
        Quaternion segRot = Quaternion.Euler(90, 0, theta);
        Vector3    segUp  = segRot * Vector3.up;
        pos += top - (pos - segUp * segmentLength / 4f);

        var seg = MakeCapsule(pos, segRot, false, isComp);
        IgnoreIntraCollisions(seg, isComp);
        Vector3 anchor = seg.transform.position - segUp * segmentLength / 4f;
        AttachJoint(seg, prev, anchor, isComp);
        return seg;
    }

    void AttachJoint(GameObject seg, GameObject prev, Vector3 anchor, bool isComp)
    {
        var j = seg.AddComponent<ConfigurableJoint>();
        j.connectedBody = prev.GetComponent<Rigidbody>();
        j.xMotion = j.yMotion = j.zMotion = ConfigurableJointMotion.Locked;
        j.angularXMotion = j.angularYMotion = ConfigurableJointMotion.Locked;
        j.angularZMotion = ConfigurableJointMotion.Free;
        j.angularXLimitSpring = _angXSpring; j.angularYZLimitSpring = _angYZSpring;
        j.angularXDrive = _angXDrive;        j.angularYZDrive = _angYZDrive;
        j.anchor = anchor;
        if (isComp) _jointsComp.Add(j); else _joints.Add(j);
    }

    void IgnoreIntraCollisions(GameObject newSeg, bool isComp)
    {
        var col  = newSeg.GetComponent<Collider>();
        var list = isComp ? _segmentsComp : _segments;
        foreach (var s in list)
            if (s != null && s != newSeg) Physics.IgnoreCollision(s.GetComponent<Collider>(), col);
    }

    float GetYRot(GameObject seg)
    {
        if (seg == null) return 0f;
        return Vector3.SignedAngle(seg.transform.up, Vector3.forward, Vector3.up);
    }

    GameObject MakeCapsule(Vector3 pos, Quaternion rot, bool kinematic, bool isComp)
    {
        var cap = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        cap.transform.position = pos; cap.transform.rotation = rot;
        cap.transform.localScale = new Vector3(0.182f, segmentLength / 2f, 0.182f);
        cap.GetComponent<Collider>().material = _mat;
        var rb = cap.AddComponent<Rigidbody>(); rb.mass = 0.001f; rb.isKinematic = kinematic;
        cap.tag = isComp ? "RobotSegmentComparison" : "RobotSegment";
        if (isComp) { cap.GetComponent<Renderer>().material.color = new Color(0.7f, 0.9f, 1f); _segmentsComp.Add(cap); }
        else         _segments.Add(cap);
        return cap;
    }

    void ResetScene()
    {
        foreach (var j in _joints)     if (j) Destroy(j);
        foreach (var j in _jointsComp) if (j) Destroy(j);
        _joints.Clear(); _jointsComp.Clear();

        foreach (var tag in new[] { "RobotSegment", "RobotSegmentComparison" })
            foreach (var go in GameObject.FindGameObjectsWithTag(tag)) Destroy(go);
        _segments.Clear(); _segmentsComp.Clear();
        _head = _headComp = null;
    }

    #endregion

    #region ── File I/O ─────────────────────────────────────────────

    void WriteLog(float angle, GameObject seg, bool isGrowth, bool isComp)
    {
        if (seg == null) return;
        float x = seg.transform.position.x;
        float z = seg.transform.position.z;
        string path = isComp
            ? (isGrowth ? _growthLogComp : _rotLogComp)
            : (isGrowth ? _growthLog     : _rotLog);
        try { File.AppendAllText(path, $"{x} {z} {angle}\n"); }
        catch (Exception e) { Debug.LogError($"[Collector] WriteLog: {e.Message}"); }
    }

    /// <summary>
    /// Append episode coordinates to the dataset JSON.
    /// Output format: [{"timestamp": "...", "growth_coordinates": [[x,z,a],...],
    ///                                        "rotation_coordinates": [[x,z,a],...]}, ...]
    /// </summary>
    static void ConvertToJson(string growthTxt, string rotTxt, string jsonPath)
    {
        try
        {
            List<float[]> ReadCoords(string path)
            {
                var list = new List<float[]>();
                if (!File.Exists(path)) return list;
                foreach (var line in File.ReadAllLines(path))
                {
                    var p = line.Trim().Split(new[]{' ','\t'}, StringSplitOptions.RemoveEmptyEntries);
                    if (p.Length == 3 &&
                        float.TryParse(p[0], out float x) &&
                        float.TryParse(p[1], out float z) &&
                        float.TryParse(p[2], out float a))
                        list.Add(new[]{x, z, a});
                }
                return list;
            }

            var growth = ReadCoords(growthTxt);
            var rotation = ReadCoords(rotTxt);

            var sb = new StringBuilder();
            sb.Append("{\n  \"timestamp\": \"").Append(DateTime.UtcNow.ToString("o")).Append("\",\n");

            void AppendArr(string key, List<float[]> data) {
                sb.Append($"  \"{key}\": [");
                if (data.Count > 0) {
                    sb.Append("\n");
                    for (int i = 0; i < data.Count; i++) {
                        sb.Append($"    [{data[i][0]}, {data[i][1]}, {data[i][2]}]");
                        if (i < data.Count - 1) sb.Append(",");
                        sb.Append("\n");
                    }
                    sb.Append("  ");
                }
                sb.Append("]");
            }
            AppendArr("growth_coordinates", growth); sb.Append(",\n");
            AppendArr("rotation_coordinates", rotation); sb.Append("\n}");

            string entry = sb.ToString();
            string finalJson;
            if (File.Exists(jsonPath))
            {
                string existing = File.ReadAllText(jsonPath).Trim();
                if (existing.StartsWith("[") && existing.EndsWith("]"))
                    finalJson = existing.Substring(0, existing.Length - 1).TrimEnd() +
                                (existing.Length > 2 ? ",\n  " : "\n  ") + entry + "\n]";
                else
                    finalJson = "[\n  " + entry + "\n]";
            }
            else
                finalJson = "[\n  " + entry + "\n]";

            File.WriteAllText(jsonPath, finalJson);
            Debug.Log($"[Collector] Saved → {jsonPath}  (growth={growth.Count}, rotation={rotation.Count})");
        }
        catch (Exception e) { Debug.LogError($"[Collector] ConvertToJson: {e.Message}"); }
    }

    double GaussianNoise(double mean, double std)
    {
        double u1 = 1.0 - _rng.NextDouble(), u2 = 1.0 - _rng.NextDouble();
        return mean + std * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }

    #endregion
}
