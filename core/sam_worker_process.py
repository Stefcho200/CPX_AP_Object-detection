"""Runs in a completely separate OS process from the main GUI. Receives
(frame, points, labels) requests over a multiprocessing Queue, runs SAM,
sends back the resulting polygon (or None) over another Queue.

Why a subprocess instead of a thread: SAM/torch on CPU has a real memory
leak across repeated predict() calls, and eventually crashes the whole
process it's running in (this is what was killing the entire app before).
By running it in its own process, a crash there just ends that process —
it cannot bring down the main Tkinter app or lose any already-saved
labels, since labels are written to disk by the main process immediately
after each successful result, not by this worker."""

import os
import sys


def sam_worker_main(request_queue, response_queue, model_path):
    # Imported inside the function so this only happens in the child
    # process, keeping the parent process's memory footprint untouched.
    import numpy as np
    from ultralytics import SAM

    try:
        model = SAM(model_path)
    except Exception as e:
        response_queue.put(("init_error", str(e)))
        return

    response_queue.put(("ready", None))

    call_count = 0
    RELOAD_EVERY_N_CALLS = 30

    while True:
        try:
            msg = request_queue.get()
        except (EOFError, OSError):
            break

        if msg is None:  # shutdown signal
            break

        request_id, frame, points_xy, labels = msg

        try:
            results = model.predict(
                frame, points=[[list(p) for p in points_xy]],
                labels=[labels], verbose=False)

            polygon = None
            if results and results[0].masks is not None and len(results[0].masks.xy) > 0:
                polygons = results[0].masks.xy
                largest = max(polygons, key=lambda p: len(p))
                polygon = np.array(largest, dtype=np.float32).copy()

            response_queue.put(("result", (request_id, polygon)))

            del results
            call_count += 1

            if call_count >= RELOAD_EVERY_N_CALLS:
                call_count = 0
                del model
                import gc
                gc.collect()
                model = SAM(model_path)

        except Exception as e:
            response_queue.put(("call_error", (request_id, str(e))))
            # Don't try to keep going after an error inside predict —
            # safer to let the parent detect this and restart us fresh.
            break
