# Rheidos Houdini Debugger (debugpy) — User Documentation

## Tutorials

### Tutorial 1: First successful breakpoint (Python SOP)

**Goal:** enable debugging on a node, attach VS Code, hit a breakpoint in your external Python code.

#### Prerequisites

* Houdini is running.
* You are using a Rheidos-enabled Python SOP node template (or HDA) that includes the **Debug** parameters.
* Your cook logic is in an external `.py` file (recommended), e.g. `my_project/houdini_nodes/poisson_demo.py`.

#### Step 1 — Install debugpy into Houdini’s Python (once)

1. Open Houdini.

2. Open **Python Shell** and run:

   ```python
   import sys
   print(sys.executable)
   ```

3. Copy the printed path (this is Houdini’s Python executable).

4. In a terminal, run:

   ```bash
   <PASTE_SYS_EXECUTABLE> -m pip install --user debugpy
   ```

If this fails because pip isn’t available, you’ll need to set up pip for that Houdini Python build (or install debugpy into the Houdini Python environment using your standard studio setup).

#### Step 2 — Put a breakpoint in your external module

Open your module (example):

```python
# my_project/houdini_nodes/poisson_demo.py

def cook(node, geo):
    x = 42  # <-- put breakpoint here
    ...
```

Set a breakpoint on that line in VS Code.

#### Step 3 — Enable debug on the node in Houdini

1. Select the Python SOP node.
2. Open the **Debug** folder.
3. Turn on:

   * **Enable Debugging**
4. Leave defaults:

   * Host: (implicitly localhost)
   * Port: `5678`
   * Port Strategy: `Fallback` (recommended)

Now force a cook (e.g., move the timeline slightly or touch a parameter that triggers cooking).

You should see a single console message similar to:

* “Debug server started on 127.0.0.1:5678”
* “Owner: /obj/…/python1”

#### Step 4 — Attach VS Code to Houdini

1. In VS Code, create or open `.vscode/launch.json`.

2. Add:

   ```json
   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "Attach to Houdini (Rheidos)",
         "type": "python",
         "request": "attach",
         "connect": { "host": "127.0.0.1", "port": 5678 }
       }
     ]
   }
   ```

3. Start the debugger using **Attach to Houdini (Rheidos)**.

#### Step 5 — Recook to hit the breakpoint

Trigger another cook (scrub timeline, toggle a parm, etc.).
When the cook reaches your breakpoint, VS Code should pause and you can inspect variables, step, etc.

✅ You have successfully attached and hit a breakpoint.

---

### Tutorial 2: “Break on next cook” workflow (fastest way to pause)

**Goal:** attach debugger first, then force Houdini to pause inside the next cook without needing an existing breakpoint.

1. Enable debugging on the node (same as Tutorial 1).
2. Attach VS Code.
3. In Houdini node Debug folder, click:

   * **Break on Next Cook**
4. Trigger a cook.

If VS Code is attached, it will pause immediately at the framework “breakpoint injection” site, and you can then step forward into your cook code.

---

## How-to Guides

### How to debug a Solver SOP safely

Solver SOPs may cook every frame and sometimes per substep. The debugging feature is designed to be safe (non-blocking), but you still want good habits.

1. Enable debugging on the Solver SOP node (or the node you want to “own” debugging).
2. Attach VS Code.
3. Use **Break on Next Cook** right before the moment you want to inspect.
4. Start playing the timeline or step frames.

**Tip:** Prefer “break on next cook” over placing a breakpoint deep inside a tight per-frame loop, because you can pause early and then selectively step.

---

### How to debug without spamming cooks

If your node is cooking constantly:

* Don’t rely on print spam.
* Attach VS Code and use:

  * **Break on Next Cook**
  * or a single targeted breakpoint early in `cook()`.

Then inspect state and step.

---

### How to work with multiple nodes (ownership)

Only one node “owns” the debug server per Houdini process.

**To make a node the owner:**

1. Enable debugging on that node first.
2. If another node already owns debugging, use:

   * **Take Ownership** (if your node provides it), then recook.

**Good practice:** pick one “debug owner” node in your network and keep it consistent.

---

### How to fix “port already in use”

Symptoms:

* The debug server fails to start.
* You see a one-time message indicating the port is unavailable.

Fix:

1. In the node Debug folder, change **Port** to another number (e.g., 5679).
2. Or change **Port Strategy** to `Fallback` or `Auto`.
3. Recook.
4. Update VS Code attach port to match the printed port.

---

### How to debug headless (`hbatch`) / no UI

This feature never blocks, so it can run headless.

To enable debugging headless you typically rely on environment variables used by your studio pipeline, for example:

* enable debug globally
* set port explicitly

Then attach from VS Code to the headless process. (In practice, you’ll often SSH tunnel the port.)

---

### How to attach from a remote machine (safe way)

By default, the debugger binds to localhost only. That’s intentional.

Preferred method: **SSH port forwarding**

1. Run Houdini on the remote machine.

2. SSH tunnel port 5678:

   ```bash
   ssh -L 5678:127.0.0.1:5678 user@remote
   ```

3. Attach VS Code to `127.0.0.1:5678` on your local machine.

This avoids exposing a debug port to your whole network.

---

## Reference

### Node Debug Parameters (expected behavior)

**Enable Debugging**

* Starts (or reuses) a debug server in the Houdini process.
* Does not pause execution.

**Port**

* The preferred port used for debug server.

**Port Strategy**

* `Fixed`: use only the chosen port; fail if unavailable.
* `Fallback`: try port, then nearby ports, then a free port.
* `Auto`: pick a free port immediately.

**Take Ownership**

* Makes this node the “owner” for debug messaging and UI.
* Does not restart the server (ownership is about control/clarity, not disruption).

**Break on Next Cook**

* Sets a session flag that will trigger a breakpoint **only if** the debugger is attached.
* If debugger is not attached, nothing blocks; the break request will remain pending (implementation-defined, but typical behavior is “break when attached”).

### Environment variables (if supported by your build)

Common patterns used by studios / pipelines:

* `RHEDIOS_DEBUG=1`
  Enables debug server start (equivalent to turning on Enable Debugging).
* `RHEDIOS_DEBUG_PORT=5678`
  Preferred port.

(Exact env var names are project-defined; use the ones your Rheidos build documents.)

---

## Explanation

### What is actually happening under the hood?

Houdini runs Python inside its own process. Rheidos starts a small debug server (via `debugpy`) inside that same process, listening on `127.0.0.1:<port>`. VS Code attaches to that server, so when your code runs during a cook, VS Code can pause execution, inspect variables, and step through the code.

### Why there is no “wait for debugger”

Blocking inside a cook is dangerous:

* it can look like Houdini is frozen,
* it can deadlock networks or per-frame cooking,
* it can cause confusion about which node is responsible.

So Rheidos chooses a safer pattern:

* start server non-blocking,
* let you attach whenever,
* let you trigger breaks intentionally (**Break on Next Cook**) or via normal breakpoints.

### Why ownership exists

If multiple nodes can start or reconfigure the debug server, debugging becomes unpredictable and confusing. Ownership makes it deterministic: one node is the “control panel,” and other nodes don’t fight over ports and logs.

### What you can debug vs what you can’t

You can debug:

* Python-level scheduling logic,
* registry/resource updates,
* data marshaling,
* your cook/setup/step functions.

You generally can’t step through:

* Taichi kernels as if they are Python code.
  You debug kernels by inspecting inputs/outputs, reducing kernels, and using Taichi’s own debugging/profiling tools.

---

## Quick “Do / Don’t” Checklist

**Do**

* Keep node code as a thin shim and put logic in external `.py` files.
* Attach VS Code first, then use **Break on Next Cook** for predictable pauses.
* Use `Fallback` port strategy if you run multiple Houdinis.

**Don’t**

* Expect “Enable Debugging” to pause execution by itself.
* Bind to non-localhost unless you explicitly understand the security implications.
* Put breakpoints inside super-tight per-point loops unless you enjoy suffering.

---