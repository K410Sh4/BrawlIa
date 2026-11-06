"""Movimento de barra/corpo com câmera estimada por fluxo óptico e RANSAC.

Movimento e imobilidade são evidências; nunca provas de caixa ou personagem.
"""

from __future__ import annotations

import cv2
import numpy as np


class MotionEvidence:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = None
        self.objects = []
        self.last_time = None
        self.original_size = None
        self._prepared = None

    def prepare(self, frame, now):
        if self._prepared is not None and self._prepared["now"] == now:
            return self._prepared["summary"]
        h, w = frame.shape[:2]
        small_w = min(640, w)
        scale = small_w / w
        gray = cv2.cvtColor(
            cv2.resize(frame, (small_w, round(h * scale))), cv2.COLOR_BGR2GRAY
        )
        old = new = affine = None
        dt = 0.0
        summary = {"valid": False, "reason": "first_frame_or_reset", "inliers": 0}
        if (
            self.previous is not None
            and self.original_size == (w, h)
            and 0 < now - self.last_time <= 0.5
        ):
            dt = now - self.last_time
            mask = np.zeros_like(gray)
            mask[round(gray.shape[0] * 0.16) : round(gray.shape[0] * 0.92)] = 255
            pts = cv2.goodFeaturesToTrack(
                self.previous,
                maxCorners=600,
                qualityLevel=0.025,
                minDistance=6,
                mask=mask,
                blockSize=5,
            )
            if pts is not None and len(pts) >= 16:
                flowed, status, error = cv2.calcOpticalFlowPyrLK(
                    self.previous,
                    gray,
                    pts,
                    None,
                    winSize=(21, 21),
                    maxLevel=2,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        15,
                        0.03,
                    ),
                )
                if flowed is not None:
                    good = (status.ravel() > 0) & (error.ravel() < 25)
                    old = pts.reshape(-1, 2)[good]
                    new = flowed.reshape(-1, 2)[good]
                    if len(old) >= 16:
                        affine, inliers = cv2.estimateAffinePartial2D(
                            old,
                            new,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=1.5,
                            maxIters=200,
                            confidence=0.98,
                        )
                        count = int(inliers.sum()) if inliers is not None else 0
                        if (
                            affine is not None
                            and count >= 14
                            and count / len(old) >= 0.55
                            and 0.96 <= np.hypot(*affine[0, :2]) <= 1.04
                        ):
                            summary = {
                                "valid": True,
                                "reason": "sparse_flow_ransac",
                                "inliers": count,
                                "inlier_fraction": round(count / len(old), 3),
                                "translation_px": [
                                    round(float(v / scale), 3) for v in affine[:, 2]
                                ],
                                "affine_thumbnail": affine.tolist(),
                            }
                            original_affine = affine.copy()
                            original_affine[:, 2] /= scale
                            summary["affine_original"] = original_affine.tolist()
                            summary["resolution"] = [w, h]
                        else:
                            summary["reason"] = "insufficient_static_consensus"
                    else:
                        summary["reason"] = "insufficient_flow_points"
            else:
                summary["reason"] = "insufficient_texture"
        self._prepared = {
            "now": now,
            "summary": summary,
            "gray": gray,
            "old": old,
            "new": new,
            "affine": affine,
            "dt": dt,
            "scale": scale,
            "size": (w, h),
        }
        return summary

    def update(self, frame, objects, now):
        summary = self.prepare(frame, now)
        prepared = self._prepared
        gray = prepared["gray"]
        w, h = prepared["size"]
        current = []
        for item in objects:
            item["motion_evidence"] = {
                "camera_valid": False,
                "body_bar_consistent": False,
                "classification": "insufficient_evidence",
            }
        if summary["valid"]:
            self._associate(
                objects,
                prepared["old"],
                prepared["new"],
                prepared["affine"],
                prepared["dt"],
                prepared["scale"],
                w,
                current,
            )
        if not current:
            current = [
                {**item, "motion_hits": 1, "motion_since": now} for item in objects
            ]
        self.previous = gray
        self.objects = current
        self.last_time = now
        self.original_size = (w, h)
        self._prepared = None
        return summary

    def _associate(self, objects, old, new, affine, dt, scale, width, current):
        used = set()
        for item in objects:
            center = np.array(item["center"], float)
            pairs = []
            for index, prev in enumerate(self.objects):
                # Em sessão, o tracker principal decide a identidade; não faça uma segunda atribuição independente.
                if (
                    "track_id" in item
                    and "track_id" in prev
                    and item["track_id"] != prev["track_id"]
                ):
                    continue
                if index in used or (prev["kind"] == "jogador_candidato") != (
                    item["kind"] == "jogador_candidato"
                ):
                    continue
                expected = affine @ np.r_[np.array(prev["center"]) * scale, 1] / scale
                distance = float(np.linalg.norm(center - expected))
                if distance < (30 + 600 * dt) * width / 1280:
                    pairs.append((distance, index, prev, expected))
            if not pairs:
                current.append(
                    {**item, "motion_hits": 1, "motion_since": self.last_time + dt}
                )
                continue
            _, index, prev, expected = min(pairs, key=lambda x: x[0])
            used.add(index)
            relative = center - expected
            bar_delta = center - np.array(prev["center"])
            px, py = prev["center"]
            body = old / scale
            inside = (
                (np.abs(body[:, 0] - px) < 50 * width / 1280)
                & (body[:, 1] > prev["y"] + prev["height"] + 5 * width / 1280)
                & (body[:, 1] < prev["y"] + prev["height"] + 90 * width / 1280)
            )
            body_delta = (new[inside] - old[inside]) / scale
            count = len(body_delta)
            consistent = False
            body_error = None
            if count >= 3:
                median = np.median(body_delta, axis=0)
                dispersion = float(
                    np.median(np.linalg.norm(body_delta - median, axis=1))
                )
                body_error = float(np.linalg.norm(median - bar_delta))
                tolerance = max(
                    1.5 * width / 1280,
                    min(4 * width / 1280, 0.35 * float(np.linalg.norm(bar_delta))),
                )
                consistent = body_error <= tolerance and dispersion <= max(
                    3, 6 * width / 1280
                )
            hits = prev.get("motion_hits", 1) + 1
            duration = self.last_time + dt - prev.get("motion_since", self.last_time)
            speed = float(np.linalg.norm(relative) / dt)
            classification = "insufficient_evidence"
            if hits >= 4 and duration >= 0.3 and consistent:
                classification = (
                    "moving_with_body"
                    if speed > 25 * width / 1280
                    else (
                        "stationary_relative_camera"
                        if speed < 12 * width / 1280
                        else "ambiguous_motion"
                    )
                )
            item["motion_evidence"] = {
                "camera_valid": True,
                "body_bar_consistent": bool(consistent),
                "body_points": count,
                "body_bar_error_px": (
                    round(body_error, 3) if body_error is not None else None
                ),
                "relative_camera_displacement_px": [
                    round(float(v), 3) for v in relative
                ],
                "relative_camera_speed_px_s": round(speed, 3),
                "observations": hits,
                "duration_s": round(duration, 3),
                "classification": classification,
            }
            if (
                item["kind"] != "jogador_candidato"
                and classification == "moving_with_body"
            ):
                item["visual_kind_before_motion"] = item["kind"]
                item["kind"] = "personagem_possivel"
                item["confidence"] = max(item["confidence"], 0.65)
            elif (
                item["kind"] == "caixa_provavel"
                and classification == "stationary_relative_camera"
            ):
                item["confidence"] = min(0.88, item["confidence"] + 0.04)
            current.append(
                {
                    **item,
                    "motion_hits": hits,
                    "motion_since": prev.get("motion_since", self.last_time),
                }
            )
