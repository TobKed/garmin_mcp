"""
Activity Management functions for Garmin Connect MCP Server
"""
import datetime
import functools
import json
from typing import Any, Dict, List, Optional, Union

import requests

# The garmin_client will be set by the main file
garmin_client = None


def configure(client):
    """Configure the module with the Garmin client instance"""
    global garmin_client
    garmin_client = client


def _put_activity_update(activity_id: int, payload: Dict[str, Any]) -> Any:
    """Send a partial activity update via PUT to the activity-service endpoint.

    Garmin's activity endpoint accepts a partial ActivityDTO keyed by
    activityId, so callers only include the top-level fields they want to
    change. This mirrors how the library's set_activity_name works.
    """
    url = f"{garmin_client.garmin_connect_activity}/{activity_id}"
    body = {"activityId": activity_id, **payload}
    return garmin_client.client.put("connectapi", url, json=body, api=True)


def _update_activity_summary(activity_id: int, fields: Dict[str, Any]) -> Any:
    """Update fields nested in an activity's summaryDTO.

    Garmin merges a partial summaryDTO into the stored summary, so we send only
    the fields being changed and the other recorded metrics are preserved.

    We deliberately avoid a read-modify-write. The full summaryDTO returned by
    get_activity contains a complete GPS coordinate pair (startLatitude and
    startLongitude); PUTting both halves of a coordinate together is rejected by
    the endpoint with a 400. Each field is individually writable, so it is the
    coordinate pair specifically that trips validation, and a minimal update
    sidesteps it.
    """
    return _put_activity_update(activity_id, {"summaryDTO": fields})


EXERCISE_CATALOG_URL = "https://connect.garmin.com/web-data/exercises/Exercises.json"
# Catalog names that Garmin's exerciseSets endpoint rejects with
# "Invalid Sub-Category". Add pairs here as they turn up.
KNOWN_REJECTED_EXERCISES = {("BENCH_PRESS", "BENCH_PRESS")}
# Watch start times are whole seconds, so recorded sets overlap by up to ~1 s.
_SET_TIME_TOLERANCE = datetime.timedelta(seconds=1)


def _exercise(category: str, name: Optional[str]) -> Dict[str, Any]:
    """An exerciseSets exercise entry with category and name upper-cased."""
    return {
        "category": category.strip().upper(),
        "name": (name or "").strip().upper() or None,
        "probability": 100.0,
    }


def _parse_set_time(value: str) -> datetime.datetime:
    """Parse '2026-11-05T15:51:28Z' / '...28.0' (UTC) to a naive datetime."""
    return datetime.datetime.strptime(
        value.strip().rstrip("Z").split(".")[0], "%Y-%m-%dT%H:%M:%S"
    )


def _recorded_to_entry(recorded_set: Dict[str, Any]) -> Dict[str, Any]:
    """A stored set in replace_activity_exercise_sets input form (for undo)."""
    exercise = (recorded_set.get("exercises") or [{}])[0]
    weight = recorded_set.get("weight")
    return {
        "type": recorded_set.get("setType"),
        "category": exercise.get("category"),
        "name": exercise.get("name"),
        "reps": recorded_set.get("repetitionCount"),
        "weight_kg": weight / 1000 if weight is not None else None,
        "start_time": recorded_set["startTime"].split(".")[0] + "Z",
        "duration_s": recorded_set.get("duration"),
    }


def _plan_set_times(entries, activity_start, activity_end):
    """Resolve (start, duration) for every entry, then validate the timeline.

    Entries with a start_time open a block; following entries without one
    belong to it. Within a block, entries with duration_s keep it and the rest
    share the remaining time up to the next block (or the activity end) evenly,
    so splitting a set into two untimed entries halves it, and an entry with
    only a duration goes right after the one before it.
    """
    starts = [_parse_set_time(e["start_time"]) if e.get("start_time") else None for e in entries]
    if starts and starts[0] is None:
        starts[0] = activity_start
    anchors = [i for i, s in enumerate(starts) if s is not None] + [len(entries)]
    planned = []
    for a, b in zip(anchors, anchors[1:]):
        block = entries[a:b]
        window_end = starts[b] if b < len(entries) else activity_end
        durations = [e.get("duration_s") for e in block]
        fixed = sum(float(d) for d in durations if d is not None)
        free = durations.count(None)
        share = ((window_end - starts[a]).total_seconds() - fixed) / free if free else 0
        if free and share <= 0:
            raise ValueError(f"set {a + 1}: no time left to place the untimed sets after it")
        t = starts[a]
        for d in durations:
            d = float(d) if d is not None else share
            planned.append((t, d))
            t += datetime.timedelta(seconds=d)

    prev_end = activity_start
    for number, (start, duration) in enumerate(planned, start=1):
        if duration <= 0:
            raise ValueError(f"set {number}: duration_s must be > 0")
        end = start + datetime.timedelta(seconds=duration)
        if start < prev_end - _SET_TIME_TOLERANCE:
            raise ValueError(
                f"set {number}: starts at {start.isoformat()}Z, before the previous "
                f"set ends ({prev_end.isoformat()}Z); sets must be in time order "
                "without overlaps"
            )
        if end > activity_end + _SET_TIME_TOLERANCE:
            raise ValueError(
                f"set {number}: ends at {end.isoformat()}Z, after the activity ends "
                f"({activity_end.isoformat()}Z)"
            )
        prev_end = end
    return planned


@functools.lru_cache(maxsize=1)
def _fetch_exercise_catalog() -> Dict[str, List[str]]:
    """Garmin's public strength exercise catalog as {category: [names]}."""
    response = requests.get(EXERCISE_CATALOG_URL, timeout=30)
    response.raise_for_status()
    return {
        category: sorted(details.get("exercises") or {})
        for category, details in response.json()["categories"].items()
    }


def register_tools(app):
    """Register all activity management tools with the MCP server app"""

    @app.tool()
    async def get_activities_by_date(
        start_date: str,
        end_date: str,
        activity_type: str = "",
        page: int = 0,
        page_size: int = 100,
    ) -> str:
        """Get activities between specified dates with pagination support.

        For accounts with large activity histories, broad date ranges can return
        thousands of activities in a single response. Use page and page_size to
        retrieve activities in manageable chunks and avoid "result too large" errors.
        Activities are ordered newest-first.

        Pagination: when has_more is true the response includes next_page — pass
        that value as page on the next call to retrieve the following page. Repeat
        until has_more is false.

        Note: total_count for a date range is not available from the Garmin API
        without fetching all results. Use has_more / next_page to walk pages.

        Each activity includes an event_type field with values such as:
          - "race"          — explicitly tagged as a race by the user
          - "training"      — explicitly tagged as a training activity
          - "uncategorized" — no event type set; common for Peloton imports and
                              untagged outdoor runs. Distinct from "training":
                              filter for races with event_type == "race" rather
                              than excluding "training", since many non-race
                              activities appear as "uncategorized" not "training"
          - field absent    — API returned no eventType for this activity; not
                              observed in practice in any activity back to 2012
                              (oldest activities sampled on this account)

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            activity_type: Optional activity type filter (e.g., cycling, running, swimming)
            page: Zero-based page number (default 0)
            page_size: Number of activities per page, max 200 (default 100)
        """
        try:
            # Clamp page_size to [1, 200]
            page_size = min(max(1, page_size), 200)
            start = page * page_size

            # Call the Garmin API directly with explicit pagination params.
            # The library's get_activities_by_date auto-fetches ALL matching
            # activities in a loop (hardcoded limit=20 per request), which causes
            # "Tool result is too large" errors on accounts with large histories.
            # Calling connectapi directly lets us fetch exactly one page at a time.
            params: Dict[str, Any] = {
                "startDate": start_date,
                "endDate": end_date,
                "start": str(start),
                "limit": str(page_size),
            }
            if activity_type:
                params["activityType"] = activity_type

            activities = garmin_client.connectapi(
                garmin_client.garmin_connect_activities,
                params=params,
            )

            if not activities:
                return json.dumps({
                    "count": 0,
                    "page": page,
                    "page_size": page_size,
                    "has_more": False,
                    "date_range": {"start": start_date, "end": end_date},
                    "activities": [],
                }, indent=2)

            has_more = len(activities) == page_size
            curated: Dict[str, Any] = {
                "count": len(activities),
                "page": page,
                "page_size": page_size,
                "has_more": has_more,
                "date_range": {"start": start_date, "end": end_date},
                "activities": [],
            }
            if has_more:
                curated["next_page"] = page + 1

            for a in activities:
                activity = {
                    "id": a.get('activityId'),
                    "name": a.get('activityName'),
                    "type": a.get('activityType', {}).get('typeKey'),
                    "event_type": (a.get('eventType') or {}).get('typeKey'),
                    "start_time": a.get('startTimeLocal'),
                    "distance_meters": a.get('distance'),
                    "duration_seconds": a.get('duration'),
                    "calories": a.get('calories'),
                    "avg_hr_bpm": a.get('averageHR'),
                    "max_hr_bpm": a.get('maxHR'),
                    "steps": a.get('steps'),
                    "elevation_gain_meters": a.get('elevationGain'),
                    "elevation_loss_meters": a.get('elevationLoss'),
                }
                # Remove None values
                activity = {k: v for k, v in activity.items() if v is not None}
                curated["activities"].append(activity)

            return json.dumps(curated, indent=2)
        except Exception as e:
            return f"Error retrieving activities by date: {str(e)}"

    @app.tool()
    async def get_activities_fordate(date: str) -> str:
        """Get activities for a specific date

        Args:
            date: Date in YYYY-MM-DD format
        """
        try:
            data = garmin_client.get_activities_fordate(date)
            if not data:
                return f"No activities found for {date}"

            # Extract just the activities, not the embedded HR data
            activities_data = data.get('ActivitiesForDay', {})
            payload = activities_data.get('payload', [])

            if not payload:
                return f"No activities found for {date}"

            curated = {
                "date": date,
                "count": len(payload),
                "activities": []
            }

            for a in payload:
                activity = {
                    "id": a.get('activityId'),
                    "name": a.get('activityName'),
                    "type": a.get('activityType', {}).get('typeKey'),
                    "event_type": (a.get('eventType') or {}).get('typeKey'),
                    "start_time": a.get('startTimeLocal'),
                    "distance_meters": a.get('distance'),
                    "duration_seconds": a.get('duration'),
                    "calories": a.get('calories'),
                    "avg_hr_bpm": a.get('averageHR'),
                    "steps": a.get('steps'),
                    "lap_count": a.get('lapCount'),
                    "moderate_intensity_minutes": a.get('moderateIntensityMinutes'),
                    "vigorous_intensity_minutes": a.get('vigorousIntensityMinutes'),
                }
                # Remove None values
                activity = {k: v for k, v in activity.items() if v is not None}
                curated["activities"].append(activity)

            return json.dumps(curated, indent=2)
        except Exception as e:
            return f"Error retrieving activities for date: {str(e)}"

    @app.tool()
    async def get_activity(activity_id: Union[int, str]) -> str:
        """Get detailed information for a single activity.

        Returns a comprehensive summary including timing, distance, heart rate,
        elevation, training effect, and an event_type field. Common event_type
        values: "race", "training", "uncategorized" (no event type set by the
        user). The field is omitted for very old activities that pre-date event
        type support in the Garmin API.

        Args:
            activity_id: ID of the activity to retrieve
        """
        try:
            activity_id = int(activity_id)
            activity = garmin_client.get_activity(activity_id)
            if not activity:
                return f"No activity found with ID {activity_id}"

            # Extract summary data
            summary = activity.get('summaryDTO', {})
            activity_type = activity.get('activityTypeDTO', {})
            metadata = activity.get('metadataDTO', {})

            curated = {
                "id": activity.get('activityId'),
                "name": activity.get('activityName'),
                "description": activity.get('description'),
                "type": activity_type.get('typeKey'),
                "event_type": (activity.get('eventTypeDTO') or {}).get('typeKey'),
                "parent_type": activity_type.get('parentTypeId'),

                # Timing
                "start_time_local": summary.get('startTimeLocal'),
                "start_time_gmt": summary.get('startTimeGMT'),
                "duration_seconds": summary.get('duration'),
                "moving_duration_seconds": summary.get('movingDuration'),
                "elapsed_duration_seconds": summary.get('elapsedDuration'),

                # Distance and speed
                "distance_meters": summary.get('distance'),
                "avg_speed_mps": summary.get('averageSpeed'),
                "max_speed_mps": summary.get('maxSpeed'),

                # Heart rate
                "avg_hr_bpm": summary.get('averageHR'),
                "max_hr_bpm": summary.get('maxHR'),
                "min_hr_bpm": summary.get('minHR'),

                # Calories
                "calories": summary.get('calories'),
                "bmr_calories": summary.get('bmrCalories'),

                # Running metrics
                "avg_cadence": summary.get('averageRunCadence'),
                "max_cadence": summary.get('maxRunCadence'),
                "avg_stride_length_cm": summary.get('strideLength'),
                "avg_ground_contact_time_ms": summary.get('groundContactTime'),
                "avg_vertical_oscillation_cm": summary.get('verticalOscillation'),
                "steps": summary.get('steps'),

                # Power
                "avg_power_watts": summary.get('averagePower'),
                "max_power_watts": summary.get('maxPower'),
                "normalized_power_watts": summary.get('normalizedPower'),

                # Training effect
                "training_effect": summary.get('trainingEffect'),
                "anaerobic_training_effect": summary.get('anaerobicTrainingEffect'),
                "training_effect_label": summary.get('trainingEffectLabel'),
                "training_load": summary.get('activityTrainingLoad'),

                # Intensity minutes
                "moderate_intensity_minutes": summary.get('moderateIntensityMinutes'),
                "vigorous_intensity_minutes": summary.get('vigorousIntensityMinutes'),

                # Elevation
                "elevation_gain_meters": summary.get('elevationGain'),
                "elevation_loss_meters": summary.get('elevationLoss'),
                "max_elevation_meters": summary.get('maxElevation'),
                "min_elevation_meters": summary.get('minElevation'),

                # Recovery
                "recovery_hr_bpm": summary.get('recoveryHeartRate'),
                "body_battery_impact": summary.get('differenceBodyBattery'),

                # Workout feedback
                "workout_feel": summary.get('directWorkoutFeel'),
                "workout_rpe": summary.get('directWorkoutRpe'),

                # Metadata
                "lap_count": metadata.get('lapCount'),
                "has_splits": metadata.get('hasSplits'),
                "device_manufacturer": metadata.get('manufacturer'),
            }

            # Remove None values
            curated = {k: v for k, v in curated.items() if v is not None}

            return json.dumps(curated, indent=2)
        except Exception as e:
            return f"Error retrieving activity: {str(e)}"

    @app.tool()
    async def set_activity_name(activity_id: Union[int, str], activity_name: str) -> str:
        """Set or update the name of an activity.

        Args:
            activity_id: ID of the activity to update
            activity_name: New activity name
        """
        try:
            activity_id = int(activity_id)
            activity_name = activity_name.strip()
            if not activity_name:
                return "Activity name cannot be empty"

            garmin_client.set_activity_name(activity_id, activity_name)

            return json.dumps(
                {
                    "success": True,
                    "activity_id": activity_id,
                    "activity_name": activity_name,
                    "message": "Activity name successfully updated",
                },
                indent=2,
            )
        except Exception as e:
            return f"Error updating activity name: {str(e)}"

    @app.tool()
    async def set_activity_type(activity_id: Union[int, str], type_key: str) -> str:
        """Change the activity type (sport) of an activity.

        Useful for reclassifying a mislabelled activity, e.g. flipping a run
        logged as 'trail_running' to 'running', or a 'treadmill_running' walk to
        'treadmill_walking'. Call get_activity_types to see all valid type keys.

        Args:
            activity_id: ID of the activity to update
            type_key: Target activity type key (e.g. 'running', 'trail_running',
                'treadmill_running', 'cycling', 'lap_swimming')
        """
        try:
            activity_id = int(activity_id)
            type_key = type_key.strip()

            types = garmin_client.get_activity_types() or []
            match = next((t for t in types if t.get("typeKey") == type_key), None)
            if not match:
                valid = ", ".join(
                    sorted(t.get("typeKey") for t in types if t.get("typeKey"))
                )
                return f"Unknown activity type '{type_key}'. Valid type keys: {valid}"

            garmin_client.set_activity_type(
                activity_id,
                match["typeId"],
                match["typeKey"],
                match.get("parentTypeId"),
            )

            return json.dumps(
                {
                    "success": True,
                    "activity_id": activity_id,
                    "type_key": match["typeKey"],
                    "type_id": match["typeId"],
                    "message": "Activity type successfully updated",
                },
                indent=2,
            )
        except Exception as e:
            return f"Error updating activity type: {str(e)}"

    @app.tool()
    async def set_activity_description(
        activity_id: Union[int, str], description: str
    ) -> str:
        """Set or update the free-text description (notes) of an activity.

        This is the notes field shown on the activity page — useful for
        recording how a session felt, kit used, conditions, niggles, etc.
        Pass an empty string to clear an existing description.

        Args:
            activity_id: ID of the activity to update
            description: New description text (empty string clears it)
        """
        try:
            activity_id = int(activity_id)
            _put_activity_update(activity_id, {"description": description})

            return json.dumps(
                {
                    "success": True,
                    "activity_id": activity_id,
                    "description": description,
                    "message": "Activity description successfully updated",
                },
                indent=2,
            )
        except Exception as e:
            return f"Error updating activity description: {str(e)}"

    @app.tool()
    async def set_activity_event_type(
        activity_id: Union[int, str], event_type: str
    ) -> str:
        """Set the event type of an activity.

        Event type categorises the activity's purpose. Valid keys:
        race, recreation, specialEvent, training, transportation, touring,
        geocaching, fitness, uncategorized.

        Args:
            activity_id: ID of the activity to update
            event_type: Target event type key (e.g. 'race', 'training')
        """
        try:
            activity_id = int(activity_id)
            event_type = event_type.strip()

            event_types = (
                garmin_client.connectapi("/activity-service/activity/eventTypes") or []
            )
            match = next(
                (e for e in event_types if e.get("typeKey") == event_type), None
            )
            if not match:
                valid = ", ".join(e.get("typeKey") for e in event_types if e.get("typeKey"))
                return f"Unknown event type '{event_type}'. Valid event types: {valid}"

            _put_activity_update(
                activity_id,
                {
                    "eventTypeDTO": {
                        "typeId": match["typeId"],
                        "typeKey": match["typeKey"],
                        "sortOrder": match.get("sortOrder"),
                    }
                },
            )

            return json.dumps(
                {
                    "success": True,
                    "activity_id": activity_id,
                    "event_type": match["typeKey"],
                    "message": "Activity event type successfully updated",
                },
                indent=2,
            )
        except Exception as e:
            return f"Error updating activity event type: {str(e)}"

    @app.tool()
    async def set_perceived_effort(
        activity_id: Union[int, str], rpe: float
    ) -> str:
        """Set the perceived effort (RPE) for an activity.

        Mirrors Garmin Connect's 'Perceived Effort' rating on a 0-10 scale,
        where 0 clears the rating. Internally Garmin stores this multiplied by
        10 (so RPE 7 is stored as 70); this tool handles the conversion.

        Args:
            activity_id: ID of the activity to update
            rpe: Perceived effort from 0 to 10 (0 clears the rating)
        """
        try:
            activity_id = int(activity_id)
            rpe = float(rpe)
            if not 0 <= rpe <= 10:
                return "rpe must be between 0 and 10"

            _update_activity_summary(
                activity_id, {"directWorkoutRpe": int(round(rpe * 10))}
            )

            return json.dumps(
                {
                    "success": True,
                    "activity_id": activity_id,
                    "rpe": rpe,
                    "message": "Perceived effort successfully updated",
                },
                indent=2,
            )
        except Exception as e:
            return f"Error updating perceived effort: {str(e)}"

    @app.tool()
    async def set_activity_feel(activity_id: Union[int, str], feel: int) -> str:
        """Set how an activity felt ('How did you feel?').

        Mirrors Garmin Connect's 5-point feel rating, stored as one of:
          0   = very tired / poor
          25  = tired
          50  = normal
          75  = good
          100 = strong
        Higher is better.

        Args:
            activity_id: ID of the activity to update
            feel: One of 0, 25, 50, 75, 100
        """
        try:
            activity_id = int(activity_id)
            feel = int(feel)
            if feel not in (0, 25, 50, 75, 100):
                return "feel must be one of 0, 25, 50, 75, 100"

            _update_activity_summary(activity_id, {"directWorkoutFeel": feel})

            return json.dumps(
                {
                    "success": True,
                    "activity_id": activity_id,
                    "feel": feel,
                    "message": "Activity feel successfully updated",
                },
                indent=2,
            )
        except Exception as e:
            return f"Error updating activity feel: {str(e)}"

    @app.tool()
    async def get_activity_splits(activity_id: Union[int, str]) -> str:
        """Get splits for an activity

        Args:
            activity_id: ID of the activity to retrieve splits for
        """
        try:
            activity_id = int(activity_id)
            splits = garmin_client.get_activity_splits(activity_id)
            if not splits:
                return f"No splits found for activity with ID {activity_id}"

            # Curate the splits data
            laps = splits.get('lapDTOs', [])

            curated = {
                "activity_id": splits.get('activityId'),
                "lap_count": len(laps),
                "laps": []
            }

            for lap in laps:
                lap_data = {
                    "lap_number": lap.get('lapIndex'),
                    "start_time": lap.get('startTimeGMT'),
                    "distance_meters": lap.get('distance'),
                    "duration_seconds": lap.get('duration'),
                    "moving_duration_seconds": lap.get('movingDuration'),
                    "elapsed_duration_seconds": lap.get('elapsedDuration'),
                    "avg_speed_mps": lap.get('averageSpeed'),
                    "avg_moving_speed_mps": lap.get('averageMovingSpeed'),
                    "max_speed_mps": lap.get('maxSpeed'),
                    "avg_hr_bpm": lap.get('averageHR'),
                    "max_hr_bpm": lap.get('maxHR'),
                    "calories": lap.get('calories'),
                    "bmr_calories": lap.get('bmrCalories'),
                    "avg_cadence": lap.get('averageRunCadence'),
                    "avg_power_watts": lap.get('averagePower'),
                    "avg_swim_cadence": lap.get('averageSwimCadence'),
                    "active_length_count": lap.get('numberOfActiveLengths'),
                    "total_strokes": lap.get('totalNumberOfStrokes'),
                    "avg_strokes": lap.get('averageStrokes'),
                    "avg_swolf": lap.get('averageSWOLF'),
                    "avg_stroke_distance": lap.get('averageStrokeDistance'),
                    "intensity_type": lap.get('intensityType'),
                    "elevation_gain_meters": lap.get('elevationGain'),
                    "elevation_loss_meters": lap.get('elevationLoss'),
                    "workout_step_index": lap.get('wktStepIndex'),
                }

                length_dtos = lap.get('lengthDTOs', [])
                if length_dtos:
                    lap_data["lengths"] = []
                    for length in length_dtos:
                        length_data = {
                            "length_number": length.get('lengthIndex'),
                            "start_time": length.get('startTimeGMT'),
                            "distance_meters": length.get('distance'),
                            "duration_seconds": length.get('duration'),
                            "avg_speed_mps": length.get('averageSpeed'),
                            "max_speed_mps": length.get('maxSpeed'),
                            "calories": length.get('calories'),
                            "avg_hr_bpm": length.get('averageHR'),
                            "max_hr_bpm": length.get('maxHR'),
                            "total_strokes": length.get('totalNumberOfStrokes'),
                            "avg_swolf": length.get('averageSWOLF'),
                            "stroke": length.get('swimStroke'),
                        }
                        length_data = {
                            k: v for k, v in length_data.items() if v is not None
                        }
                        lap_data["lengths"].append(length_data)

                # Remove None values
                lap_data = {k: v for k, v in lap_data.items() if v is not None}
                curated["laps"].append(lap_data)

            return json.dumps(curated, indent=2)
        except Exception as e:
            return f"Error retrieving activity splits: {str(e)}"

    @app.tool()
    async def get_activity_typed_splits(activity_id: Union[int, str]) -> str:
        """Get typed splits for an activity

        Args:
            activity_id: ID of the activity to retrieve typed splits for
        """
        try:
            activity_id = int(activity_id)
            typed_splits = garmin_client.get_activity_typed_splits(activity_id)
            if not typed_splits:
                return f"No typed splits found for activity with ID {activity_id}"

            return json.dumps(typed_splits, indent=2)
        except Exception as e:
            return f"Error retrieving activity typed splits: {str(e)}"

    @app.tool()
    async def get_activity_split_summaries(activity_id: Union[int, str]) -> str:
        """Get split summaries for an activity

        Args:
            activity_id: ID of the activity to retrieve split summaries for
        """
        try:
            activity_id = int(activity_id)
            split_summaries = garmin_client.get_activity_split_summaries(activity_id)
            if not split_summaries:
                return f"No split summaries found for activity with ID {activity_id}"

            return json.dumps(split_summaries, indent=2)
        except Exception as e:
            return f"Error retrieving activity split summaries: {str(e)}"

    @app.tool()
    async def get_activity_weather(activity_id: Union[int, str]) -> str:
        """Get weather data for an activity.

        Garmin's weather endpoint returns temperatures in Fahrenheit (from the
        weather-station source) with no unit indicator, regardless of account
        settings. This tool converts them to the account's display unit: metric
        accounts get Celsius, statute_us accounts keep Fahrenheit. The
        temperature_unit field ("F" or "C") states which unit was returned.

        Wind speed, unlike temperature, is already returned in the account's
        display unit (km/h for metric, mph for statute_us), so it is passed
        through unconverted and labeled via the wind_speed_unit field.

        Args:
            activity_id: ID of the activity to retrieve weather data for
        """
        try:
            activity_id = int(activity_id)
            weather = garmin_client.get_activity_weather(activity_id)
            if not weather:
                return f"No weather data found for activity with ID {activity_id}"

            # Garmin's activity-weather endpoint returns temperatures in
            # FAHRENHEIT (from the METAR/station source), regardless of the
            # account's measurement system — the payload carries no unit field.
            # (Confirmed: a metric account still gets °F here, e.g. temp 57.)
            # So convert the raw Fahrenheit values to the account's display unit.
            def _f_to_c(value):
                return round((value - 32) * 5 / 9, 1) if value is not None else None

            try:
                unit_system = garmin_client.get_unit_system()
            except Exception:
                unit_system = None

            raw_temp = weather.get('temp')
            raw_apparent = weather.get('apparentTemp')
            raw_dew = weather.get('dewPoint')

            if unit_system is not None and unit_system != "statute_us":
                # Metric (or non-US-statute) account → present in Celsius.
                temp_unit = "C"
                out_temp, out_apparent, out_dew = (
                    _f_to_c(raw_temp), _f_to_c(raw_apparent), _f_to_c(raw_dew)
                )
            else:
                # statute_us account (or unknown) → leave the raw Fahrenheit.
                temp_unit = "F" if unit_system == "statute_us" else None
                out_temp, out_apparent, out_dew = raw_temp, raw_apparent, raw_dew

            # Wind, unlike temperature, IS already returned in the account's
            # display unit (verified against a local weather station: metric
            # accounts get km/h, not mph). So don't convert it — just label it.
            wind_unit = "mph" if unit_system == "statute_us" else (
                "km/h" if unit_system is not None else None
            )

            weather_type_dto = weather.get('weatherTypeDTO') or {}
            station_dto = weather.get('weatherStationDTO') or {}

            curated = {
                "activity_id": activity_id,
                "temperature": out_temp,
                "temperature_unit": temp_unit,
                "apparent_temperature": out_apparent,
                "dew_point": out_dew,
                "humidity_percent": weather.get('relativeHumidity'),
                "wind_speed": weather.get('windSpeed'),
                "wind_speed_unit": wind_unit,
                "wind_direction_degrees": weather.get('windDirection'),
                "wind_direction_compass": weather.get('windDirectionCompassPoint'),
                "wind_gust": weather.get('windGust'),
                "weather_description": weather_type_dto.get('desc'),
                "station_id": station_dto.get('id'),
                "station_name": station_dto.get('name'),
                "issue_time": weather.get('issueDate'),
            }

            # Remove None values
            curated = {k: v for k, v in curated.items() if v is not None}

            return json.dumps(curated, indent=2)
        except Exception as e:
            return f"Error retrieving activity weather data: {str(e)}"

    @app.tool()
    async def get_activity_hr_in_timezones(activity_id: Union[int, str]) -> str:
        """Get heart rate data in different time zones for an activity

        Args:
            activity_id: ID of the activity to retrieve heart rate time zone data for
        """
        try:
            activity_id = int(activity_id)
            hr_zones = garmin_client.get_activity_hr_in_timezones(activity_id)
            if not hr_zones:
                return f"No heart rate time zone data found for activity with ID {activity_id}"

            return json.dumps(hr_zones, indent=2)
        except Exception as e:
            return f"Error retrieving activity heart rate time zone data: {str(e)}"

    @app.tool()
    async def get_activity_power_in_timezones(activity_id: Union[int, str]) -> str:
        """Get power distribution across training zones for an activity.

        Returns time spent in each power zone with watt thresholds and duration.
        Requires a power meter. Zones are based on the athlete's FTP configured in Garmin Connect.

        Args:
            activity_id: ID of the activity to retrieve power zone data for
        """
        try:
            activity_id = int(activity_id)
            power_zones = garmin_client.get_activity_power_in_timezones(activity_id)
            if not power_zones:
                return f"No power zone data found for activity {activity_id}. Ensure the activity was recorded with a power meter."

            return json.dumps(power_zones, indent=2)
        except Exception as e:
            return f"Error retrieving activity power zone data: {str(e)}"

    @app.tool()
    async def get_activity_gear(activity_id: Union[int, str]) -> str:
        """Get gear data used for an activity

        Args:
            activity_id: ID of the activity to retrieve gear data for
        """
        try:
            activity_id = int(activity_id)
            gear = garmin_client.get_activity_gear(activity_id)
            if not gear:
                return f"No gear data found for activity with ID {activity_id}"

            return json.dumps(gear, indent=2)
        except Exception as e:
            return f"Error retrieving activity gear data: {str(e)}"

    @app.tool()
    async def get_activity_exercise_sets(activity_id: Union[int, str]) -> str:
        """Get exercise sets for strength training activities

        Args:
            activity_id: ID of the activity to retrieve exercise sets for
        """
        try:
            activity_id = int(activity_id)
            exercise_sets = garmin_client.get_activity_exercise_sets(activity_id)
            if not exercise_sets:
                return f"No exercise sets found for activity with ID {activity_id}"

            return json.dumps(exercise_sets, indent=2)
        except Exception as e:
            return f"Error retrieving activity exercise sets: {str(e)}"

    @app.tool()
    async def set_activity_exercise_sets(
        activity_id: Union[int, str], sets: List[Dict[str, Any]]
    ) -> str:
        """Correct the exercises, reps, and weights of a strength activity's sets.

        Works on top of what the watch recorded: pass exactly one entry per
        recorded ACTIVE set, in order (see get_activity_exercise_sets). Rest
        sets, durations, and timestamps are kept as recorded.

        Each entry is either:
          {"category": "SQUAT", "name": null, "reps": 10, "weight_kg": 50}
            category is required (e.g. BENCH_PRESS, SQUAT, ROW, DEADLIFT).
            name is an optional specific exercise within the category
            (e.g. "T_BAR_ROW"); leave it null when unsure, since Garmin rejects
            some names that its own exercise catalog lists (BENCH_PRESS/BENCH_PRESS).
            reps and weight_kg are optional; omitted values keep the recorded ones.
          {"rest": true}
            turns the recorded set into rest, e.g. a phantom set from a button press.

        Args:
            activity_id: ID of the strength activity to update
            sets: One entry per recorded active set, in order
        """
        try:
            activity_id = int(activity_id)
            recorded = garmin_client.get_activity_exercise_sets(activity_id) or {}
            exercise_sets = recorded.get("exerciseSets") or []
            active = [s for s in exercise_sets if s.get("setType") == "ACTIVE"]
            if len(sets) != len(active):
                return (
                    f"Activity {activity_id} has {len(active)} active sets recorded, "
                    f"got {len(sets)} entries. Pass one entry per active set, in order "
                    '(use {"rest": true} for sets that should become rest).'
                )

            for number, (entry, recorded_set) in enumerate(zip(sets, active), start=1):
                if entry.get("rest"):
                    recorded_set.update(
                        setType="REST", exercises=[], repetitionCount=None, weight=None
                    )
                    continue
                category = (entry.get("category") or "").strip().upper()
                if not category:
                    return f"set {number}: category is required unless rest is true"
                recorded_set["exercises"] = [_exercise(category, entry.get("name"))]
                if entry.get("reps") is not None:
                    recorded_set["repetitionCount"] = int(entry["reps"])
                if entry.get("weight_kg") is not None:
                    recorded_set["weight"] = float(entry["weight_kg"]) * 1000

            url = f"{garmin_client.garmin_connect_activity}/{activity_id}/exerciseSets"
            garmin_client.client.put(
                "connectapi",
                url,
                json={"activityId": activity_id, "exerciseSets": exercise_sets},
            )

            return json.dumps(
                {
                    "success": True,
                    "activity_id": activity_id,
                    "active_sets": [
                        {
                            "category": s["exercises"][0]["category"],
                            "name": s["exercises"][0]["name"],
                            "reps": s.get("repetitionCount"),
                            "weight_kg": (s.get("weight") or 0) / 1000,
                        }
                        for s in exercise_sets
                        if s.get("setType") == "ACTIVE"
                    ],
                },
                indent=2,
            )
        except Exception as e:
            message = f"Error updating activity exercise sets: {str(e)}"
            if "Sub-Category" in str(e):
                message += (
                    ". Garmin rejected an exercise name; retry with name null for "
                    "the affected sets."
                )
            return message

    @app.tool()
    async def replace_activity_exercise_sets(
        activity_id: Union[int, str], sets: List[Dict[str, Any]]
    ) -> str:
        """Rewrite a strength activity's sets in full: add, remove, split, merge, re-time.

        Unlike set_activity_exercise_sets, the list replaces everything the watch
        recorded and can have any length. Pass ACTIVE and REST sets in time order:
          {"type": "ACTIVE", "category": "PUSH_UP", "name": "PIKE_PUSH_UP",
           "reps": 3, "weight_kg": null,
           "start_time": "2026-11-05T15:51:28Z", "duration_s": 20}
          {"type": "REST", "start_time": "2026-11-05T15:51:48Z", "duration_s": 5}
        type defaults to ACTIVE; category is required for ACTIVE sets, name is
        optional (see get_exercise_catalog). start_time (UTC) and duration_s are
        optional: an entry without start_time follows the entry before it, and
        untimed entries share the time up to the next timed entry (or the activity
        end) evenly. So to split a recorded set, keep its start_time on the first
        part and leave both parts without duration_s.

        Sets must stay inside the recorded activity, in order, without overlaps.
        If Garmin rejects an exercise name, the write is retried with names
        dropped (name null) and the affected sets are listed in names_dropped.

        Returns the stored sets plus "backup": the previous sets in this tool's
        input form. Pass the backup back to this tool to undo.

        Args:
            activity_id: ID of the strength activity to update
            sets: The complete, ordered list of sets to store
        """
        try:
            activity_id = int(activity_id)
            recorded = (garmin_client.get_activity_exercise_sets(activity_id) or {}).get(
                "exerciseSets"
            ) or []
            if not recorded:
                return f"Activity {activity_id} has no recorded exercise sets to replace."
            if not sets:
                return "sets must contain at least one set."
            backup = [_recorded_to_entry(s) for s in recorded]
            # ponytail: the activity window is taken as the span of the recorded
            # sets, which covers the whole activity on the watches seen so far.
            # Read summaryDTO.startTimeGMT/elapsedDuration if that stops holding.
            activity_start = _parse_set_time(recorded[0]["startTime"])
            activity_end = max(
                _parse_set_time(s["startTime"])
                + datetime.timedelta(seconds=s.get("duration") or 0)
                for s in recorded
            )

            exercise_sets = []
            names_dropped = []
            for number, entry in enumerate(sets, start=1):
                set_type = (entry.get("type") or "ACTIVE").strip().upper()
                if set_type not in ("ACTIVE", "REST"):
                    return f"set {number}: type must be ACTIVE or REST, got {set_type}"
                reps = int(entry["reps"]) if entry.get("reps") is not None else None
                weight_kg = entry.get("weight_kg")
                weight = float(weight_kg) * 1000 if weight_kg is not None else None
                if reps is not None and reps < 0:
                    return f"set {number}: reps must be >= 0"
                if weight is not None and weight < 0:
                    return f"set {number}: weight_kg must be >= 0"
                exercises = []
                if set_type == "ACTIVE":
                    if not (entry.get("category") or "").strip():
                        return f"set {number}: category is required for ACTIVE sets"
                    exercise = _exercise(entry["category"], entry.get("name"))
                    if (exercise["category"], exercise["name"]) in KNOWN_REJECTED_EXERCISES:
                        names_dropped.append({"set": number, "name": exercise["name"]})
                        exercise["name"] = None
                    exercises = [exercise]
                else:
                    reps = weight = None
                exercise_sets.append(
                    {
                        "exercises": exercises,
                        "repetitionCount": reps,
                        "weight": weight,
                        "setType": set_type,
                        "wktStepIndex": None,
                        "messageIndex": number - 1,
                    }
                )

            try:
                planned = _plan_set_times(sets, activity_start, activity_end)
            except ValueError as e:
                return str(e)
            for exercise_set, (start, duration) in zip(exercise_sets, planned):
                start = start + datetime.timedelta(microseconds=500_000)  # round
                exercise_set["startTime"] = start.strftime("%Y-%m-%dT%H:%M:%S") + ".0"
                exercise_set["duration"] = round(duration, 3)

            url = f"{garmin_client.garmin_connect_activity}/{activity_id}/exerciseSets"
            body = {"activityId": activity_id, "exerciseSets": exercise_sets}
            try:
                garmin_client.client.put("connectapi", url, json=body)
            except Exception as e:
                if "Sub-Category" not in str(e):
                    raise
                # ponytail: Garmin doesn't say which name it rejected, so every
                # name is dropped; retry per set if keeping valid names matters.
                for number, exercise_set in enumerate(exercise_sets, start=1):
                    exercise = (exercise_set["exercises"] or [{}])[0]
                    if exercise.get("name"):
                        names_dropped.append({"set": number, "name": exercise["name"]})
                        exercise["name"] = None
                garmin_client.client.put("connectapi", url, json=body)

            stored = garmin_client.get_activity_exercise_sets(activity_id) or {}
            return json.dumps(
                {
                    "success": True,
                    "activity_id": activity_id,
                    "names_dropped": names_dropped,
                    "exerciseSets": [
                        {k: v for k, v in s.items() if v is not None}
                        for s in stored.get("exerciseSets") or []
                    ],
                    "backup": backup,
                },
                indent=2,
            )
        except Exception as e:
            return f"Error replacing activity exercise sets: {str(e)}"

    @app.tool()
    async def get_exercise_catalog(category: Optional[str] = None) -> str:
        """List the strength exercise names Garmin knows, by category.

        Use it to pick category/name pairs for set_activity_exercise_sets,
        replace_activity_exercise_sets and create_strength_workout. A category
        with name null is always accepted. known_rejected lists catalog names
        that Garmin's set editor still rejects.

        Args:
            category: Optional category (e.g. PUSH_UP) to list only its names
        """
        try:
            catalog = _fetch_exercise_catalog()
            rejected = KNOWN_REJECTED_EXERCISES
            if category:
                category = category.strip().upper()
                if category not in catalog:
                    return json.dumps(
                        {
                            "error": f"Unknown category {category}",
                            "categories": sorted(catalog),
                        },
                        indent=2,
                    )
                catalog = {category: catalog[category]}
                rejected = {(c, n) for c, n in rejected if c == category}
            return json.dumps(
                {
                    "categories": catalog,
                    "known_rejected": sorted(f"{c}/{n}" for c, n in rejected),
                },
                indent=2
            )
        except Exception as e:
            return f"Error retrieving exercise catalog: {str(e)}"

    @app.tool()
    async def count_activities() -> str:
        """Get total count of activities in the user's Garmin account

        Returns the total number of activities recorded.
        """
        try:
            count = garmin_client.count_activities()
            if count is None:
                return "Unable to retrieve activity count"

            return json.dumps({
                "total_activities": count,
                "note": "Use get_activities() with pagination to retrieve activity details"
            }, indent=2)
        except Exception as e:
            return f"Error counting activities: {str(e)}"

    @app.tool()
    async def get_activities(start: int = 0, limit: int = 20) -> str:
        """Get activities with pagination support.

        Retrieves a paginated list of activities ordered newest-first. Use this
        for browsing through large activity lists when you do not need to filter
        by date range, or as a complement to get_activities_by_date.

        Each activity includes an event_type field. Common values: "race",
        "training", "uncategorized" (no event type set by the user — common for
        Peloton imports and untagged runs). Filter for races with
        event_type == "race" rather than excluding "training", as many non-race
        activities appear as "uncategorized" rather than "training".

        Args:
            start: Starting index (default 0)
            limit: Maximum number of activities to return (default 20, max 100)
        """
        try:
            # Cap limit at 100 for safety and performance
            limit = min(max(1, limit), 100)

            activities = garmin_client.get_activities(start, limit)
            if not activities:
                return f"No activities found at index {start}"

            # Curate the activity list
            curated = {
                "start": start,
                "limit": limit,
                "count": len(activities),
                "has_more": len(activities) == limit,
                "next_start": start + limit if len(activities) == limit else None,
                "activities": []
            }

            for a in activities:
                activity = {
                    "id": a.get('activityId'),
                    "name": a.get('activityName'),
                    "type": a.get('activityType', {}).get('typeKey'),
                    "event_type": (a.get('eventType') or {}).get('typeKey'),
                    "start_time": a.get('startTimeLocal'),
                    "distance_meters": a.get('distance'),
                    "duration_seconds": a.get('duration'),
                    "moving_duration_seconds": a.get('movingDuration'),
                    "calories": a.get('calories'),
                    "avg_hr_bpm": a.get('averageHR'),
                    "max_hr_bpm": a.get('maxHR'),
                    "steps": a.get('steps'),
                    "elevation_gain_meters": a.get('elevationGain'),
                    "elevation_loss_meters": a.get('elevationLoss'),
                    "owner_display_name": a.get('ownerDisplayName'),
                }
                # Remove None values
                activity = {k: v for k, v in activity.items() if v is not None}
                curated["activities"].append(activity)

            return json.dumps(curated, indent=2)
        except Exception as e:
            return f"Error retrieving activities: {str(e)}"

    @app.tool()
    async def create_manual_activity(
        type_key: str,
        date: str,
        duration_minutes: int,
        start_time: str = "09:00",
        activity_name: str = "",
        distance_km: float = 0.0,
        time_zone: str = "UTC",
    ) -> str:
        """Log a manual activity in Garmin Connect — useful for activities done without a watch.

        The type_key must match a Garmin activity type. Use get_activity_types to see
        the full list. Common values: yoga, strength_training, meditation, indoor_cycling,
        pilates, bouldering, fitness_equipment.

        Args:
            type_key: Activity type key (e.g. "yoga", "strength_training")
            date: Date of the activity in YYYY-MM-DD format
            duration_minutes: Duration of the activity in minutes
            start_time: Start time as HH:MM (24-hour, default 09:00)
            activity_name: Optional title; defaults to the type_key if not provided
            distance_km: Distance in kilometres (default 0.0 for non-distance activities)
            time_zone: IANA time zone for the activity (default UTC)
        """
        try:
            if not type_key.strip():
                return "Error: type_key is required"
            if duration_minutes <= 0:
                return "Error: duration_minutes must be greater than 0"

            name = activity_name.strip() or type_key.replace("_", " ").title()
            start_datetime = f"{date}T{start_time}:00.000"

            result = garmin_client.create_manual_activity(
                start_datetime=start_datetime,
                time_zone=time_zone,
                type_key=type_key,
                distance_km=distance_km,
                duration_min=duration_minutes,
                activity_name=name,
            )

            return json.dumps({
                "success": True,
                "activity": result,
            }, indent=2)
        except Exception as e:
            return f"Error creating manual activity: {str(e)}"

    @app.tool()
    async def get_activity_types() -> str:
        """Get all available activity types

        Returns a list of all activity types supported by Garmin Connect,
        useful for filtering activities by type.
        """
        try:
            activity_types = garmin_client.get_activity_types()
            if not activity_types:
                return "No activity types found"

            # Curate the activity types list
            curated = {
                "count": len(activity_types),
                "activity_types": []
            }

            for at in activity_types:
                activity_type = {
                    "type_id": at.get('typeId'),
                    "type_key": at.get('typeKey'),
                    "display_name": at.get('displayName'),
                    "parent_type_id": at.get('parentTypeId'),
                    "is_hidden": at.get('isHidden'),
                }
                # Remove None values
                activity_type = {k: v for k, v in activity_type.items() if v is not None}
                curated["activity_types"].append(activity_type)

            return json.dumps(curated, indent=2)
        except Exception as e:
            return f"Error retrieving activity types: {str(e)}"

    return app
