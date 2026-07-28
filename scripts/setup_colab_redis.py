#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

from redis import Redis


COLAB_REDIS_URL = "redis://localhost:6379"
REDIS_MINOR_VERSION = (8, 6)
REQUIRED_MODULES = {"search"}
REQUIRED_COMMANDS = {"FT.HYBRID"}


def run_checked(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, **kwargs)


def inspect_redis(redis_url: str = COLAB_REDIS_URL) -> dict[str, Any]:
    state: dict[str, Any] = {
        "ready": False,
        "version": "unavailable",
        "modules": set(),
        "commands": set(),
    }
    try:
        client = Redis.from_url(
            redis_url,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        client.ping()

        version = client.info("server")["redis_version"]
        version_parts = tuple(int(part) for part in version.split(".")[:2])

        module_names = set()
        for module in client.module_list():
            name = module.get(b"name", module.get("name"))
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            if name:
                module_names.add(str(name).lower())

        available_commands = set()
        for command in REQUIRED_COMMANDS:
            try:
                command_info = client.execute_command("COMMAND", "INFO", command)
            except Exception:
                command_info = None
            if command_info:
                available_commands.add(command)

        state.update(
            {
                "version": version,
                "modules": module_names,
                "commands": available_commands,
            }
        )
        state["ready"] = (
            version_parts == REDIS_MINOR_VERSION
            and REQUIRED_MODULES <= module_names
            and REQUIRED_COMMANDS <= available_commands
        )
    except Exception:
        pass
    return state


def install_redis_86() -> None:
    apt_environment = {**os.environ, "DEBIAN_FRONTEND": "noninteractive"}
    run_checked(
        ["sudo", "apt-get", "update", "-qq"],
        env=apt_environment,
        stdout=subprocess.DEVNULL,
    )
    run_checked(
        [
            "sudo",
            "apt-get",
            "install",
            "-y",
            "-qq",
            "ca-certificates",
            "lsb-release",
            "curl",
            "gpg",
        ],
        env=apt_environment,
        stdout=subprocess.DEVNULL,
    )

    key_download = Path("/tmp/redis-archive-keyring.asc")
    run_checked(
        ["curl", "-fsSL", "https://packages.redis.io/gpg", "-o", str(key_download)]
    )
    run_checked(
        [
            "sudo",
            "gpg",
            "--batch",
            "--yes",
            "--dearmor",
            "-o",
            "/usr/share/keyrings/redis-archive-keyring.gpg",
            str(key_download),
        ]
    )
    run_checked(
        ["sudo", "chmod", "644", "/usr/share/keyrings/redis-archive-keyring.gpg"]
    )

    ubuntu_codename = subprocess.run(
        ["lsb_release", "-cs"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    repository_line = (
        "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] "
        f"https://packages.redis.io/deb {ubuntu_codename} main\n"
    )
    run_checked(
        ["sudo", "tee", "/etc/apt/sources.list.d/redis.list"],
        input=repository_line,
        text=True,
        stdout=subprocess.DEVNULL,
    )

    redis_pin = """Package: redis redis-server redis-sentinel redis-tools
Pin: version 6:8.6.*
Pin-Priority: 1001
"""
    run_checked(
        ["sudo", "tee", "/etc/apt/preferences.d/redis-8-6"],
        input=redis_pin,
        text=True,
        stdout=subprocess.DEVNULL,
    )
    run_checked(
        ["sudo", "apt-get", "update", "-qq"],
        env=apt_environment,
        stdout=subprocess.DEVNULL,
    )
    run_checked(
        [
            "sudo",
            "apt-get",
            "install",
            "-y",
            "-qq",
            "--allow-downgrades",
            "redis",
        ],
        env=apt_environment,
        stdout=subprocess.DEVNULL,
    )


def start_redis_with_required_modules() -> None:
    subprocess.run(
        ["redis-cli", "shutdown", "nosave"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    redis_server = shutil.which("redis-server")
    module_paths = [
        Path("/usr/lib/redis/modules/redisearch.so"),
    ]
    missing_paths = [str(path) for path in module_paths if not path.is_file()]
    if redis_server is None or missing_paths:
        raise RuntimeError(
            "Redis 8.6 modules were not installed as expected; "
            f"redis_server={redis_server}, missing_modules={missing_paths}"
        )

    command = [redis_server, "--daemonize", "yes"]
    for module_path in module_paths:
        command.extend(["--loadmodule", str(module_path)])
    run_checked(command)


def setup_colab_redis(redis_url: str = COLAB_REDIS_URL) -> dict[str, Any]:
    state = inspect_redis(redis_url)
    if not state["ready"]:
        install_redis_86()
        state = inspect_redis(redis_url)

    if not state["ready"]:
        start_redis_with_required_modules()
        for _ in range(40):
            state = inspect_redis(redis_url)
            if state["ready"]:
                break
            time.sleep(0.25)

    if not state["ready"]:
        raise RuntimeError(
            "Redis 8.6 with Search and FT.HYBRID is required; "
            f"version={state['version']}, "
            f"modules={sorted(state['modules'])}, "
            f"commands={sorted(state['commands'])}"
        )
    return state


def main() -> None:
    state = setup_colab_redis()
    print(f"Redis {state['version']} is ready at {COLAB_REDIS_URL}.")
    print(f"Loaded modules: {', '.join(sorted(state['modules']))}")
    print(f"Verified commands: {', '.join(sorted(state['commands']))}")


if __name__ == "__main__":
    main()
