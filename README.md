# yt-dlp Queue

An interactive command-line wrapper around [yt-dlp](https://github.com/yt-dlp/yt-dlp) that focuses on reliability, ergonomics, and session persistence.

## Highlights

- **Persistent Queue & Sessions** – Save and restore pending URLs, worker directories, and failed download metadata.
- **Smart Worker Orchestration** – Configure per-worker download directories, pause/resume globally, and auto-start workers if pending URLs exist.
- **Live Monitoring** – Real-time worker log streaming with progress indicators.
- **Robust Failure Handling** – Automatic retries with configurable delays, failed-download tracking, and `retry` command to requeue everything that failed (even across restarts).
- **Flexible Directory Control** – Session-wide default directories, per-worker overrides, flattening options, and support for quoted paths.
- **CLI UX** – Command aliases, async status updates, auto-update checks for yt-dlp, and colorized outputs that don’t interfere with workers.

## Key Commands

- `start` / `pause` / `resume` – Control worker threads.
- `worker <id>` / `worker all` – Follow live logs for one worker or all workers simultaneously.
- `workerdir [id] [path]` – Inspect or set download directories per worker (or session defaults).
- `retry` – Requeue every failed download tracked during this or prior sessions.
- `save <file>` / `load <file>` – Persist or restore the entire queue state (pending URLs, configuration, worker dirs, failed entries).

## Workflow Overview

1. Add URLs directly at the prompt (the shell validates, deduplicates, and records sesion history).
2. Use `workerdir` to steer downloads into specific folders before starting.
3. Run `start` (or let the auto-start timer kick in) to spawn worker threads; each worker launches yt-dlp with enforced `-P` download directory and progress capture.
4. Use `worker <id>` or `worker all` to observe stdout/stderr output and live progress without disrupting the download processes.
5. If a download fails, it’s tracked and can be retried explicitly or automatically across sessions.

## Requirements

- Python 3.9+
- `yt-dlp` binary accessible via PATH or configured in `config.yt_dlp_path`
- Optional: `colorama` (auto-detected) for cross-platform colored output

## Getting Started

```bash
python yt-dlp.queue.py
```

From the prompt:

```bash
> config workers 4                  # tune worker count
> workerdir "F:\\Downloads\\Art"      # optional session default dir
> https://example.com/some/gallery  # queue URLs directly
> start                             # workers will also auto-start after 5s
> worker all                        # follow every worker’s log + progress
```

Press `Ctrl+C` while inside `worker`/`worker all` views to return to the main prompt without interrupting downloads. Use `quit` when you’re finished (remember to `save` if you want to resume later). For the full list of commands (and all aliases), run `?` inside the shell to open the built-in help sheet.
