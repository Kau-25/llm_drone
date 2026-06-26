"""
MAVLink Drone MCP Server (mavsdk)
Connects to a drone via mavsdk and exposes flight control tools for Claude.

Connection is configured via a .env file in the same directory:
    MAVLINK_ADDRESS=192.168.1.10
    MAVLINK_PORT=14540
    MAVLINK_PROTOCOL=udp
"""

import asyncio
import math
import os
import sys
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from dotenv import load_dotenv
from mavsdk import System
from mcp.server.fastmcp import FastMCP, Context

import base64
import time
from datetime import datetime
from pathlib import Path
load_dotenv()

# ─────────────────────────────────────────────
# Drone connection state (shared across tools)
# ─────────────────────────────────────────────

@dataclass
class DroneState:
    drone: System
    connected: asyncio.Event = field(default_factory=asyncio.Event)
    gps_ready: asyncio.Event = field(default_factory=asyncio.Event)


_drone_state: DroneState | None = None


async def _connect_in_background(state: DroneState, address: str, port: str, protocol: str):
    """Connect to the drone and wait for heartbeat + GPS lock in the background."""
    connection_string = f"{protocol}://{address}:{port}"
    print(f"Connecting to drone at {connection_string}...", file=sys.stderr)

    await state.drone.connect(system_address=connection_string)

    # Wait for heartbeat
    async for conn in state.drone.core.connection_state():
        if conn.is_connected:
            print(f"✓ Drone connected at {connection_string}", file=sys.stderr)
            state.connected.set()
            break

    # Wait for GPS lock
    print("Waiting for GPS lock...", file=sys.stderr)
    async for health in state.drone.telemetry.health():
        if health.is_global_position_ok or health.is_home_position_ok:
            print("✓ GPS lock acquired — drone is ready", file=sys.stderr)
            state.gps_ready.set()
            break


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[DroneState]:
    global _drone_state

    address  = os.environ.get("MAVLINK_ADDRESS", "")
    port     = os.environ.get("MAVLINK_PORT", "14540")
    protocol = os.environ.get("MAVLINK_PROTOCOL", "udp").lower()

    # Map legacy protocol names to current mavsdk equivalents
    if protocol == "tcp":
        protocol = "tcpout"
    elif protocol == "udp":
        # udp:// is deprecated; use udpout when a specific host address is given
        protocol = "udpout"

    if not address:
        raise ValueError("MAVLINK_ADDRESS is not set in your .env file")

    drone = System()
    _drone_state = DroneState(drone=drone)

    # Connect without blocking server startup
    asyncio.create_task(_connect_in_background(_drone_state, address, port, protocol))

    yield _drone_state


# ─────────────────────────────────────────────
# MCP server
# ─────────────────────────────────────────────

mcp = FastMCP("drone", lifespan=lifespan)


async def _wait_connected(state: DroneState, timeout: float = 30.0) -> bool:
    """Wait for MAVLink heartbeat (does not require GPS)."""
    try:
        await asyncio.wait_for(state.connected.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False


async def _wait_gps(state: DroneState, timeout: float = 120.0) -> bool:
    """Wait for MAVLink heartbeat + GPS lock."""
    try:
        await asyncio.wait_for(state.gps_ready.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False


# ─────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────

@mcp.tool()
async def get_status(ctx: Context) -> dict:
    """
    Return the current connection status, GPS lock, armed state,
    flight mode, and position of the drone.
    """
    state: DroneState = ctx.request_context.lifespan_context

    if not await _wait_connected(state):
        return {"error": "Could not reach drone — check MAVLINK_ADDRESS in .env"}

    drone = state.drone

    # Basic connectivity info
    result: dict = {
        "connected": state.connected.is_set(),
        "gps_ready": state.gps_ready.is_set(),
    }

    # Armed state
    async for armed in drone.telemetry.armed():
        result["armed"] = armed
        break

    # Flight mode
    async for mode in drone.telemetry.flight_mode():
        result["flight_mode"] = str(mode)
        break

    # Position (only if GPS ready)
    if state.gps_ready.is_set():
        async for pos in drone.telemetry.position():
            result["position"] = {
                "latitude_deg": pos.latitude_deg,
                "longitude_deg": pos.longitude_deg,
                "altitude_agl_m": round(pos.relative_altitude_m, 2),
                "altitude_msl_m": round(pos.absolute_altitude_m, 2),
            }
            break

    return result


@mcp.tool()
async def arm(ctx: Context) -> dict:
    """
    Arm the drone motors.
    The drone must be on the ground with a GPS fix before arming.
    """
    state: DroneState = ctx.request_context.lifespan_context

    if not await _wait_gps(state):
        return {"error": "GPS not ready — cannot arm safely"}

    try:
        await state.drone.action.arm()
        return {"status": "armed"}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def disarm(ctx: Context) -> dict:
    """
    Disarm the drone motors.
    Only safe to call when the drone is on the ground.
    """
    state: DroneState = ctx.request_context.lifespan_context

    if not await _wait_connected(state):
        return {"error": "Drone not connected"}

    try:
        await state.drone.action.disarm()
        return {"status": "disarmed"}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def takeoff(ctx: Context, altitude_m: float = 3.0) -> dict:
    """
    Arm the drone (if needed) and take off to the given altitude above ground.

    Args:
        altitude_m: Target altitude in metres AGL. Must be between 1 and 120.
    """
    state: DroneState = ctx.request_context.lifespan_context

    if not await _wait_gps(state):
        return {"error": "GPS not ready — cannot take off"}

    if not (1.0 <= altitude_m <= 120.0):
        return {"error": f"altitude_m must be between 1 and 120, got {altitude_m}"}

    drone = state.drone

    try:
        await drone.action.set_takeoff_altitude(altitude_m)
        await drone.action.arm()
        await drone.action.takeoff()
        return {
            "status": "taking_off",
            "target_altitude_m": altitude_m,
            "tip": "Call get_status() to monitor altitude, or wait_for_altitude() to block until arrived",
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def wait_for_altitude(ctx: Context, target_m: float, tolerance_m: float = 0.5, timeout_s: float = 60.0) -> dict:
    """
    Block until the drone reaches the target altitude (±tolerance).
    Useful after takeoff() or go_to_altitude() to confirm the drone arrived.

    Args:
        target_m:    Expected altitude in metres AGL.
        tolerance_m: Acceptable deviation in metres (default 0.5).
        timeout_s:   Max seconds to wait before giving up (default 60).
    """
    state: DroneState = ctx.request_context.lifespan_context

    if not await _wait_connected(state):
        return {"error": "Drone not connected"}

    deadline = asyncio.get_event_loop().time() + timeout_s

    async for pos in state.drone.telemetry.position():
        current = pos.relative_altitude_m
        if abs(current - target_m) <= tolerance_m:
            return {
                "status": "reached",
                "current_altitude_m": round(current, 2),
                "target_altitude_m": target_m,
            }
        if asyncio.get_event_loop().time() >= deadline:
            return {
                "status": "timeout",
                "current_altitude_m": round(current, 2),
                "target_altitude_m": target_m,
            }
        await asyncio.sleep(0.5)

    return {"error": "Telemetry stream ended unexpectedly"}


@mcp.tool()
async def go_to_altitude(ctx: Context, altitude_m: float) -> dict:
    """
    Change the drone's altitude while hovering at its current lat/lon.
    The drone must already be airborne.

    Args:
        altitude_m: Target altitude in metres AGL. Must be between 1 and 120.
    """
    state: DroneState = ctx.request_context.lifespan_context

    if not await _wait_gps(state):
        return {"error": "GPS not ready"}

    if not (1.0 <= altitude_m <= 120.0):
        return {"error": f"altitude_m must be between 1 and 120, got {altitude_m}"}

    drone = state.drone

    try:
        # Snapshot current lat/lon, compute new absolute altitude
        async for pos in drone.telemetry.position():
            current_lat = pos.latitude_deg
            current_lon = pos.longitude_deg
            home_alt    = pos.absolute_altitude_m - pos.relative_altitude_m
            target_msl  = home_alt + altitude_m
            break

        await drone.action.goto_location(current_lat, current_lon, target_msl, math.nan)

        return {
            "status": "climbing" if altitude_m > pos.relative_altitude_m else "descending",
            "target_altitude_agl_m": altitude_m,
            "target_altitude_msl_m": round(target_msl, 2),
            "tip": "Call wait_for_altitude() to block until the drone arrives",
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def land(ctx: Context) -> dict:
    """
    Land the drone at its current position.
    The flight controller switches to LAND mode and descends automatically.
    """
    state: DroneState = ctx.request_context.lifespan_context

    if not await _wait_connected(state):
        return {"error": "Drone not connected"}

    try:
        await state.drone.action.land()
        return {
            "status": "landing",
            "tip": "Call get_status() to monitor altitude until the drone is on the ground",
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def takeoff_and_hold(ctx: Context, altitude_m: float = 2.0) -> dict:
    """
    Arm, take off to the given altitude, and immediately hold position.
    Prevents ArduPilot from landing due to OFFBOARD setpoint timeout.

    Args:
        altitude_m: Target altitude in metres AGL. Must be between 1 and 120.
    """
    state: DroneState = ctx.request_context.lifespan_context

    if not await _wait_gps(state):
        return {"error": "GPS not ready — cannot take off"}

    if not (1.0 <= altitude_m <= 120.0):
        return {"error": f"altitude_m must be between 1 and 120, got {altitude_m}"}

    drone = state.drone

    try:
        # Take off
        await drone.action.set_takeoff_altitude(altitude_m)
        await drone.action.arm()
        await drone.action.takeoff()

        # Wait until target altitude reached
        deadline = asyncio.get_event_loop().time() + 30.0
        async for pos in drone.telemetry.position():
            if abs(pos.relative_altitude_m - altitude_m) <= 0.5:
                break
            if asyncio.get_event_loop().time() >= deadline:
                return {"error": "Timed out waiting to reach altitude", "altitude_m": round(pos.relative_altitude_m, 2)}
            await asyncio.sleep(0.2)

        # Snapshot current position and hold it
        async for pos in drone.telemetry.position():
            lat = pos.latitude_deg
            lon = pos.longitude_deg
            alt_msl = pos.absolute_altitude_m
            break

        await drone.action.goto_location(lat, lon, alt_msl, math.nan)

        return {
            "status": "holding",
            "altitude_agl_m": round(pos.relative_altitude_m, 2),
            "latitude_deg": lat,
            "longitude_deg": lon,
        }

    except Exception as e:
        return {"error": str(e)}
    
@mcp.tool()
async def return_to_launch(ctx: Context) -> dict:
    """
    Command the drone to fly back to its launch point and land (RTL mode).
    """
    state: DroneState = ctx.request_context.lifespan_context

    if not await _wait_connected(state):
        return {"error": "Drone not connected"}

    try:
        await state.drone.action.return_to_launch()
        return {"status": "returning_to_launch"}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def kill_motors(ctx: Context) -> dict:
    """
    EMERGENCY ONLY: Immediately cut power to all motors.
    The drone will fall. Only use to prevent a worse outcome.
    """
    state: DroneState = ctx.request_context.lifespan_context

    if not await _wait_connected(state):
        return {"error": "Drone not connected"}

    try:
        await state.drone.action.kill()
        return {"status": "motors_killed", "warning": "Drone has lost power and will fall"}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
