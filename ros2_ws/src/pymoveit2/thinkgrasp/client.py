#!/usr/bin/env python3

"""Client helpers for the ThinkGrasp Flask server in realarm.py."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_SERVER_URL = "http://127.0.0.1:5000"
GRASP_POSE_ROUTE = "/grasp_pose"


class ThinkGraspClientError(RuntimeError):
    """Raised when the Flask server cannot complete a request."""


@dataclass(frozen=True)
class GraspPoseResult:
    """Parsed response from realarm.py's /grasp_pose endpoint."""

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
        payload = {
            "image_path": str(Path(image_path).expanduser().resolve()),
            "depth_path": str(Path(depth_path).expanduser().resolve()),
            "text_path": str(Path(text_path).expanduser().resolve()),
        }
        self._validate_payload_paths(payload)

        response = self._post_json(GRASP_POSE_ROUTE, payload)
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

    def _post_json(self, route: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.server_url}{route}"
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_sec) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            message = self._extract_error_message(error_body) or error_body
            raise ThinkGraspClientError(
                f"Server returned HTTP {exc.code}: {message}"
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
            raise ThinkGraspClientError(f"Expected JSON object, got: {type(parsed).__name__}")
        return parsed

    @staticmethod
    def _validate_payload_paths(payload: Dict[str, str]) -> None:
        for field_name, path_text in payload.items():
            path = Path(path_text)
            if not path.is_file():
                raise ThinkGraspClientError(f"{field_name} does not exist: {path}")

    @staticmethod
    def _extract_error_message(body: str) -> Optional[str]:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return None

        if isinstance(parsed, dict) and "error" in parsed:
            return str(parsed["error"])
        return None
