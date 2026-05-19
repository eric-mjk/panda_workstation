#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid


DEFAULT_SERVER_URL = "http://127.0.0.1:5000"
GRASP_POSE_ROUTE = "/grasp_pose"


class ThinkGraspClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class GraspPoseResult:
    xyz: List[float]
    rot: List[List[float]]
    dep: Union[float, List[float], int]
    raw: Dict[str, Any]


class GraspPoseClient:
    def __init__(self, server_url: str = DEFAULT_SERVER_URL, timeout_sec: float = 300.0):
        self.server_url = server_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def get_grasp_pose(
        self,
        image_path: Union[str, Path],
        depth_path: Union[str, Path],
        text_path: Union[str, Path],
    ) -> GraspPoseResult:
        image_path = Path(image_path).expanduser().resolve()
        depth_path = Path(depth_path).expanduser().resolve()
        text_path = Path(text_path).expanduser().resolve()

        for path in (image_path, depth_path, text_path):
            if not path.is_file():
                raise ThinkGraspClientError(f"File does not exist: {path}")

        response = self._post_files(
            GRASP_POSE_ROUTE,
            files={
                "image": image_path,
                "depth": depth_path,
                "text": text_path,
            },
        )

        if "error" in response:
            raise ThinkGraspClientError(str(response["error"]))

        missing = {"xyz", "rot", "dep"} - response.keys()
        if missing:
            raise ThinkGraspClientError(
                f"Server response is missing expected field(s): {sorted(missing)}"
            )

        return GraspPoseResult(
            xyz=response["xyz"],
            rot=response["rot"],
            dep=response["dep"],
            raw=response,
        )

    def _post_files(self, route: str, files: Dict[str, Path]) -> Dict[str, Any]:
        url = f"{self.server_url}{route}"
        boundary = f"----thinkgrasp-{uuid.uuid4().hex}"
        body = bytearray()

        for field_name, path in files.items():
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(
                (
                    f'Content-Disposition: form-data; name="{field_name}"; '
                    f'filename="{path.name}"\r\n'
                    "Content-Type: application/octet-stream\r\n\r\n"
                ).encode()
            )
            body.extend(path.read_bytes())
            body.extend(b"\r\n")

        body.extend(f"--{boundary}--\r\n".encode())

        request = Request(
            url,
            data=bytes(body),
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_sec) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise ThinkGraspClientError(
                f"Server returned HTTP {exc.code}: {error_body}"
            ) from exc
        except URLError as exc:
            raise ThinkGraspClientError(
                f"Could not connect to ThinkGrasp server at {url}: {exc.reason}"
            ) from exc

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise ThinkGraspClientError(
                f"Server returned non-JSON response: {response_body[:500]}"
            ) from exc

        if not isinstance(parsed, dict):
            raise ThinkGraspClientError(f"Expected JSON object, got {type(parsed).__name__}")

        return parsed
