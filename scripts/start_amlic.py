#!/usr/bin/env python3
"""
start_amlic.py — AMLIC session startup orchestrator.

Spins up both VMs (prefill L4 on hpmlproj, decode T4 on amlic-proj) in
parallel using the GPU availability checker, then starts all AMLIC services
via SSH and writes discovered IPs to .env.

Usage:
    python scripts/start_amlic.py              # spin up + start services
    python scripts/start_amlic.py --dry-run    # find VMs only, don't start services
    python scripts/start_amlic.py --stop       # stop both VMs to save cost
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
ENV_FILE     = PROJECT_ROOT / ".env"
STATE_FILE   = PROJECT_ROOT / ".amlic_state.json"

PREFILL = {
    "name":     "prefill-vm",
    "gpu_type": "nvidia-l4",
    "project":  "hpmlproj",
    "port":     8100,
    "cost_hr":  0.71 + 0.22,   # L4 GPU + g2-standard-4 machine
    "use_iap":  False,
}
DECODE = {
    "name":     "decode-vm",
    "gpu_type": "nvidia-tesla-t4",
    "project":  "amlic-proj",
    "port":     8200,
    "cost_hr":  0.35 + 0.19,   # T4 GPU + n1-standard-4 machine
    "use_iap":  True,
}

COLLOCATED_PORT = 8000
PROXY_PORT      = 9000


# ── Preflight checks ──────────────────────────────────────────────────────────

def check_prereqs():
    try:
        subprocess.run(["gcloud", "--version"], capture_output=True, timeout=10, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        sys.exit("ERROR: gcloud CLI not found or not working. Install the Google Cloud SDK.")

    result = subprocess.run(
        ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=json"],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0 or result.stdout.strip() in ("", "[]"):
        sys.exit("ERROR: gcloud not authenticated. Run:  gcloud auth login")

    if not ENV_FILE.exists():
        sys.exit(f"ERROR: .env not found at {ENV_FILE}. Copy .env.example and fill in values.")


# ── VM provisioning ───────────────────────────────────────────────────────────

def find_existing_vm(vm: dict) -> tuple[str | None, str | None]:
    """Look up an existing VM by name. Returns (zone, status) or (None, None)."""
    result = subprocess.run(
        [
            "gcloud", "compute", "instances", "list",
            f"--filter=name={vm['name']}",
            f"--project={vm['project']}",
            "--format=json",
        ],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None, None
    try:
        instances = json.loads(result.stdout)
        if instances:
            zone   = instances[0]["zone"].split("/")[-1]
            status = instances[0].get("status", "UNKNOWN")
            return zone, status
    except (json.JSONDecodeError, KeyError, IndexError):
        pass
    return None, None


def ensure_vm_running(vm: dict, zone: str, status: str):
    """Start a VM that exists but is not yet RUNNING."""
    if status == "RUNNING":
        return
    print(f"[{vm['name']}] Status={status}, starting VM in {zone}...")
    result = subprocess.run(
        [
            "gcloud", "compute", "instances", "start", vm["name"],
            f"--zone={zone}", f"--project={vm['project']}", "--quiet",
        ],
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to start {vm['name']}: {result.stderr.strip() or result.stdout.strip()}"
        )


def run_checker(vm: dict) -> tuple[str | None, str, int]:
    """
    Provision one VM:
      1. If VM already exists (any status) — start it if stopped, return its zone.
      2. Otherwise — run the GPU checker to find a zone with capacity and create it.
    Returns (zone, stdout, returncode).
    """
    zone, status = find_existing_vm(vm)
    if zone:
        print(f"[{vm['name']}] Found existing VM in {zone} (status={status}).")
        try:
            ensure_vm_running(vm, zone, status)
        except RuntimeError as e:
            return None, str(e), 1
        return zone, f"WINNER_ZONE={zone}\n", 0

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "krv2123_gcp_gpu_checker.py"),
        "--method",   "B",
        "--keep",
        "--gpu-type", vm["gpu_type"],
        "--vm-name",  vm["name"],
        "--project",  vm["project"],
        "--price-limit", "2.00",
    ]
    print(f"[{vm['name']}] No existing VM found. Searching for {vm['gpu_type']} in project {vm['project']}...")
    result = subprocess.run(cmd, capture_output=False, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    zone = None
    for line in result.stdout.splitlines():
        if line.startswith("WINNER_ZONE="):
            zone = line.split("=", 1)[1].strip()
            break

    return zone, result.stdout, result.returncode


# ── Tailscale IP discovery ────────────────────────────────────────────────────

def get_tailscale_ip(vm: dict, zone: str) -> str:
    """SSH into a VM and retrieve its Tailscale IP."""
    cmd = [
        "gcloud", "compute", "ssh", vm["name"],
        f"--zone={zone}",
        f"--project={vm['project']}",
        "--command=tailscale ip -4",
    ]
    if vm["use_iap"]:
        cmd.append("--tunnel-through-iap")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(
            f"SSH to {vm['name']} failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    ip = result.stdout.strip()
    if not ip:
        raise RuntimeError(
            f"Tailscale not running on {vm['name']}. "
            "SSH in and run: curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up"
        )
    return ip


# ── .env update ───────────────────────────────────────────────────────────────

def update_env(prefill_ip: str, decode_ip: str):
    """Replace VM_PREFILL_IP, VM_DECODE_IP, REDIS_HOST in .env in-place."""
    text = ENV_FILE.read_text()
    subs = {
        r"^VM_PREFILL_IP=.*":  f"VM_PREFILL_IP={prefill_ip}",
        r"^VM_DECODE_IP=.*":   f"VM_DECODE_IP={decode_ip}",
        r"^REDIS_HOST=.*":     f"REDIS_HOST={prefill_ip}",
    }
    for pattern, replacement in subs.items():
        new_text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
        if new_text == text:
            text += f"\n{replacement}"   # add if line not present
        else:
            text = new_text
    ENV_FILE.write_text(text)


# ── SSH service launcher ──────────────────────────────────────────────────────

def ssh_run(vm: dict, zone: str, command: str):
    """Run a command on a remote VM via gcloud compute ssh."""
    cmd = [
        "gcloud", "compute", "ssh", vm["name"],
        f"--zone={zone}",
        f"--project={vm['project']}",
        f"--command={command}",
    ]
    if vm["use_iap"]:
        cmd.append("--tunnel-through-iap")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed on {vm['name']}:\n{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout


def wait_for_health(name: str, ip: str, port: int, timeout: int = 300, interval: int = 10):
    """Poll http://{ip}:{port}/health until 200 or timeout (seconds)."""
    url = f"http://{ip}:{port}/health"
    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    print(f"[Health] {name} ready after {attempt} poll(s).")
                    return
        except Exception:
            pass
        remaining = int(deadline - time.time())
        print(f"[Health] {name} not ready (attempt {attempt}, {remaining}s left) — retrying in {interval}s...")
        time.sleep(interval)
    raise RuntimeError(
        f"{name} health check timed out after {timeout}s. "
        f"Check logs: sudo docker logs -f amlic-{'prefill' if 'prefill' in name.lower() else 'decode'}"
    )


def start_services(prefill_zone: str, decode_zone: str, prefill_ip: str, decode_ip: str):
    """Start Redis, prefill, decode, and proxy in the correct order."""
    print("\n[Services] Starting Redis + prefill vLLM on prefill VM...")
    ssh_run(
        PREFILL, prefill_zone,
        "cd ~/amlic && git pull --ff-only && "
        "bash infra/start_redis.sh && "
        "bash infra/start_prefill_lmcache.sh",
    )
    print("[Services] Prefill container launched (detached). Waiting for /health...")
    wait_for_health("Prefill", prefill_ip, PREFILL["port"])

    print("[Services] Starting decode vLLM on decode VM...")
    ssh_run(
        DECODE, decode_zone,
        "cd ~/amlic && git pull --ff-only && "
        "bash infra/start_decode_lmcache.sh",
    )
    print("[Services] Decode container launched (detached). Waiting for /health...")
    wait_for_health("Decode", decode_ip, DECODE["port"])

    print("[Services] Starting disaggregated proxy locally...")
    proxy_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "infra" / "proxy_server.py"),
        "--prefiller-host", prefill_ip, "--prefiller-port", str(PREFILL["port"]),
        "--decoder-host",   decode_ip,  "--decoder-port",   str(DECODE["port"]),
        "--port", str(PROXY_PORT),
    ]
    subprocess.Popen(
        proxy_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(1)
    print(f"[Services] Proxy running on localhost:{PROXY_PORT} (background PID detached).")


# ── VM teardown ───────────────────────────────────────────────────────────────

def stop_vms():
    if not STATE_FILE.exists():
        sys.exit(
            "ERROR: .amlic_state.json not found. Cannot determine VM zones.\n"
            "If VMs are running, stop them manually via GCP console."
        )
    state = json.loads(STATE_FILE.read_text())
    prefill_zone = state.get("prefill_zone")
    decode_zone  = state.get("decode_zone")
    if not prefill_zone or not decode_zone:
        sys.exit("ERROR: State file is missing zone info.")

    print(f"Stopping {PREFILL['name']} in {prefill_zone} ({PREFILL['project']})...")
    subprocess.run([
        "gcloud", "compute", "instances", "stop", PREFILL["name"],
        f"--zone={prefill_zone}", f"--project={PREFILL['project']}", "--quiet",
    ], check=False)

    print(f"Stopping {DECODE['name']} in {decode_zone} ({DECODE['project']})...")
    subprocess.run([
        "gcloud", "compute", "instances", "stop", DECODE["name"],
        f"--zone={decode_zone}", f"--project={DECODE['project']}", "--quiet",
    ], check=False)

    print("Both VMs stopped. Cost billing ceased.")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(prefill_ip: str, prefill_zone: str, decode_ip: str, decode_zone: str):
    total_cost = PREFILL["cost_hr"] + DECODE["cost_hr"]
    print(f"""
============================================
  AMLIC Session Ready
  Prefill (L4):  {prefill_ip:<15}  [{prefill_zone}]  hpmlproj
  Decode  (T4):  {decode_ip:<15}  [{decode_zone}]  amlic-proj
  Redis:         {prefill_ip}:6379
  Proxy:         localhost:{PROXY_PORT}
  Collocated:    {prefill_ip}:{COLLOCATED_PORT}
============================================
  Cost: ~${total_cost:.2f}/hr while running
  Stop:  python scripts/start_amlic.py --stop
============================================""")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="AMLIC session startup orchestrator")
    ap.add_argument("--dry-run", action="store_true",
                    help="Find and provision VMs but do not start services")
    ap.add_argument("--stop", action="store_true",
                    help="Stop both VMs to save cost")
    args = ap.parse_args()

    if args.stop:
        stop_vms()
        return

    check_prereqs()

    # ── Provision both VMs in parallel ────────────────────────────────────────
    print("Provisioning AMLIC VMs in parallel (this may take several minutes)...\n")
    with ThreadPoolExecutor(max_workers=2) as ex:
        prefill_future = ex.submit(run_checker, PREFILL)
        decode_future  = ex.submit(run_checker, DECODE)
        prefill_zone, prefill_out, prefill_rc = prefill_future.result()
        decode_zone,  decode_out,  decode_rc  = decode_future.result()

    if not prefill_zone:
        print(f"ERROR: Failed to provision {PREFILL['name']} (L4 on {PREFILL['project']}).")
        print(prefill_out[-3000:])
        sys.exit(1)
    if not decode_zone:
        print(f"ERROR: Failed to provision {DECODE['name']} (T4 on {DECODE['project']}).")
        print(decode_out[-3000:])
        sys.exit(1)

    print(f"[OK] {PREFILL['name']} ready in {prefill_zone} ({PREFILL['project']})")
    print(f"[OK] {DECODE['name']}  ready in {decode_zone}  ({DECODE['project']})")

    # ── Persist zones for --stop ───────────────────────────────────────────────
    STATE_FILE.write_text(json.dumps({
        "prefill_zone": prefill_zone,
        "decode_zone":  decode_zone,
    }, indent=2))

    # ── Discover Tailscale IPs ─────────────────────────────────────────────────
    print("\nFetching Tailscale IPs via SSH...")
    try:
        prefill_ip = get_tailscale_ip(PREFILL, prefill_zone)
        decode_ip  = get_tailscale_ip(DECODE,  decode_zone)
    except RuntimeError as e:
        sys.exit(f"ERROR: {e}")

    print(f"  Prefill IP: {prefill_ip}")
    print(f"  Decode IP:  {decode_ip}")

    # ── Update .env ───────────────────────────────────────────────────────────
    update_env(prefill_ip, decode_ip)
    print(f"  .env updated with discovered IPs.")

    if args.dry_run:
        print_summary(prefill_ip, prefill_zone, decode_ip, decode_zone)
        print("(--dry-run: services not started)")
        return

    # ── Start services ────────────────────────────────────────────────────────
    start_services(prefill_zone, decode_zone, prefill_ip, decode_ip)

    print_summary(prefill_ip, prefill_zone, decode_ip, decode_zone)


if __name__ == "__main__":
    main()
