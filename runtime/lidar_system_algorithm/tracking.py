from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment
except Exception:  # pragma: no cover - exercised only when scipy is unavailable
    scipy_linear_sum_assignment = None


@dataclass
class Detection:
    frame_id: str
    center_xyz: np.ndarray
    size_xyz: np.ndarray
    yaw: float = 0.0
    score: float | None = None
    class_name: str = "object"
    source: str = "detector"


@dataclass
class TrackState:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    size_xyz: np.ndarray
    yaw: float
    class_name: str
    source: str
    first_frame_id: str
    last_frame_id: str
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    history: list[np.ndarray] = field(default_factory=list)

    @property
    def center_xyz(self) -> np.ndarray:
        return np.array([self.state[0], self.state[1], self.size_xyz[2] / 2.0], dtype=np.float64)


def _hungarian(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    if cost_matrix.size == 0:
        return []
    rows, cols = cost_matrix.shape
    size = max(rows, cols)
    pad_value = float(cost_matrix.max() + 1.0) if cost_matrix.size > 0 else 1.0
    square = np.full((size, size), pad_value, dtype=np.float64)
    square[:rows, :cols] = cost_matrix
    u = np.zeros(size + 1, dtype=np.float64)
    v = np.zeros(size + 1, dtype=np.float64)
    p = np.zeros(size + 1, dtype=np.int32)
    way = np.zeros(size + 1, dtype=np.int32)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = np.full(size + 1, np.inf, dtype=np.float64)
        used = np.zeros(size + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                cur = square[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignments: list[tuple[int, int]] = []
    for j in range(1, size + 1):
        if p[j] != 0 and p[j] - 1 < rows and j - 1 < cols:
            assignments.append((p[j] - 1, j - 1))
    return assignments


class MultiObjectTracker:
    def __init__(
        self,
        distance_threshold: float = 4.0,
        max_age: int = 2,
        min_hits: int = 2,
        process_noise: float = 1.0,
        measurement_noise: float = 1.0,
    ) -> None:
        self.distance_threshold = distance_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self._next_track_id = 1
        self.tracks: list[TrackState] = []

    def _create_track(self, detection: Detection) -> TrackState:
        state = np.array([detection.center_xyz[0], detection.center_xyz[1], 0.0, 0.0], dtype=np.float64)
        covariance = np.eye(4, dtype=np.float64) * 10.0
        track = TrackState(
            track_id=self._next_track_id,
            state=state,
            covariance=covariance,
            size_xyz=detection.size_xyz.astype(np.float64),
            yaw=float(detection.yaw),
            class_name=detection.class_name,
            source=detection.source,
            first_frame_id=detection.frame_id,
            last_frame_id=detection.frame_id,
        )
        track.history.append(detection.center_xyz.astype(np.float64))
        self._next_track_id += 1
        return track

    def _predict(self, track: TrackState, dt: float = 1.0) -> None:
        f = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        q = np.eye(4, dtype=np.float64) * self.process_noise
        track.state = f @ track.state
        track.covariance = f @ track.covariance @ f.T + q
        track.age += 1
        track.time_since_update += 1

    def _update(self, track: TrackState, detection: Detection) -> None:
        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        r = np.eye(2, dtype=np.float64) * self.measurement_noise
        z = detection.center_xyz[:2].astype(np.float64)
        y = z - h @ track.state
        s = h @ track.covariance @ h.T + r
        k = track.covariance @ h.T @ np.linalg.inv(s)
        track.state = track.state + k @ y
        track.covariance = (np.eye(4, dtype=np.float64) - k @ h) @ track.covariance
        track.size_xyz = detection.size_xyz.astype(np.float64)
        track.yaw = float(detection.yaw)
        track.class_name = detection.class_name
        track.last_frame_id = detection.frame_id
        track.hits += 1
        track.time_since_update = 0
        track.history.append(detection.center_xyz.astype(np.float64))

    def update(self, detections: list[Detection], frame_id: str) -> list[TrackState]:
        for track in self.tracks:
            self._predict(track)

        if self.tracks and detections:
            cost = np.zeros((len(self.tracks), len(detections)), dtype=np.float64)
            for i, track in enumerate(self.tracks):
                for j, detection in enumerate(detections):
                    predicted_center = track.state[:2]
                    cost[i, j] = float(np.linalg.norm(predicted_center - detection.center_xyz[:2]))
            raw_matches = _hungarian(cost)
        else:
            cost = np.zeros((len(self.tracks), len(detections)), dtype=np.float64)
            raw_matches = []

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for track_index, det_index in raw_matches:
            if cost[track_index, det_index] > self.distance_threshold:
                continue
            self._update(self.tracks[track_index], detections[det_index])
            self.tracks[track_index].last_frame_id = frame_id
            matched_tracks.add(track_index)
            matched_detections.add(det_index)

        for det_index, detection in enumerate(detections):
            if det_index not in matched_detections:
                self.tracks.append(self._create_track(detection))

        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_age]
        visible_tracks = [
            track
            for track in self.tracks
            if track.hits >= self.min_hits or track.time_since_update == 0
        ]
        return visible_tracks


class OptimizedMultiObjectTracker(MultiObjectTracker):
    """Vectorized center-distance tracker with gated Hungarian association."""

    def __init__(
        self,
        distance_threshold: float = 4.0,
        max_age: int = 2,
        min_hits: int = 2,
        process_noise: float = 1.0,
        measurement_noise: float = 1.0,
    ) -> None:
        super().__init__(
            distance_threshold=distance_threshold,
            max_age=max_age,
            min_hits=min_hits,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
        )
        self.last_stats: dict = {}

    def _match(self, detections: list[Detection]) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
        if not self.tracks or not detections:
            empty_cost = np.zeros((len(self.tracks), len(detections)), dtype=np.float64)
            empty_gate = np.zeros_like(empty_cost, dtype=bool)
            return [], empty_cost, empty_gate

        track_centers = np.asarray([track.state[:2] for track in self.tracks], dtype=np.float64)
        det_centers = np.asarray([detection.center_xyz[:2] for detection in detections], dtype=np.float64)
        deltas = track_centers[:, None, :] - det_centers[None, :, :]
        cost = np.linalg.norm(deltas, axis=2)
        gate = cost <= float(self.distance_threshold)
        if not np.any(gate):
            return [], cost, gate

        components: list[tuple[list[int], list[int]]] = []
        remaining_rows = set(np.where(gate.any(axis=1))[0].tolist())
        while remaining_rows:
            row_stack = [remaining_rows.pop()]
            rows_seen: set[int] = set()
            cols_seen: set[int] = set()
            while row_stack:
                row = row_stack.pop()
                if row in rows_seen:
                    continue
                rows_seen.add(row)
                cols = np.where(gate[row])[0].tolist()
                for col in cols:
                    if col in cols_seen:
                        continue
                    cols_seen.add(col)
                    linked_rows = np.where(gate[:, col])[0].tolist()
                    for linked_row in linked_rows:
                        if linked_row not in rows_seen:
                            row_stack.append(linked_row)
                            remaining_rows.discard(linked_row)
            components.append((sorted(rows_seen), sorted(cols_seen)))

        matches: list[tuple[int, int]] = []
        for row_ids, col_ids in components:
            sub_cost = cost[np.ix_(row_ids, col_ids)]
            sub_gate = gate[np.ix_(row_ids, col_ids)]
            gated_cost = sub_cost.copy()
            gated_cost[~sub_gate] = 1e6
            if scipy_linear_sum_assignment is not None:
                rows, cols = scipy_linear_sum_assignment(gated_cost)
                matches.extend(
                    (row_ids[int(row)], col_ids[int(col)])
                    for row, col in zip(rows, cols)
                    if sub_gate[int(row), int(col)]
                )
            else:
                matches.extend(
                    (row_ids[row], col_ids[col])
                    for row, col in _hungarian(gated_cost)
                    if sub_gate[row, col]
                )
        return matches, cost, gate

    def update_with_stats(self, detections: list[Detection], frame_id: str) -> tuple[list[TrackState], dict]:
        previous_track_ids = {track.track_id for track in self.tracks}
        for track in self.tracks:
            self._predict(track)

        matches, cost, gate = self._match(detections)
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for track_index, det_index in matches:
            self._update(self.tracks[track_index], detections[det_index])
            self.tracks[track_index].last_frame_id = frame_id
            matched_tracks.add(track_index)
            matched_detections.add(det_index)

        spawned_tracks = 0
        for det_index, detection in enumerate(detections):
            if det_index not in matched_detections:
                self.tracks.append(self._create_track(detection))
                spawned_tracks += 1

        before_prune_ids = {track.track_id for track in self.tracks}
        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_age]
        after_prune_ids = {track.track_id for track in self.tracks}
        expired_tracks = len(before_prune_ids - after_prune_ids)

        visible_tracks = [
            track
            for track in self.tracks
            if track.hits >= self.min_hits or track.time_since_update == 0
        ]
        matched_track_ids = {self.tracks[index].track_id for index in matched_tracks if index < len(self.tracks)}
        self.last_stats = {
            "num_detections": len(detections),
            "num_tracks_before": len(previous_track_ids),
            "num_tracks_after": len(self.tracks),
            "visible_track_count": len(visible_tracks),
            "association_matrix_size": int(cost.size),
            "gated_pair_count": int(gate.sum()),
            "match_count": len(matches),
            "spawned_tracks": spawned_tracks,
            "expired_tracks": expired_tracks,
            "track_id_switch_proxy": max(0, len(matches) - len(matched_track_ids)),
            "scipy_available": scipy_linear_sum_assignment is not None,
        }
        return visible_tracks, self.last_stats

    def update(self, detections: list[Detection], frame_id: str) -> list[TrackState]:
        tracks, _ = self.update_with_stats(detections, frame_id)
        return tracks
