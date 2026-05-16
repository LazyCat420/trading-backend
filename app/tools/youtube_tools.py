"""
YouTube maintenance tools for the trading bot agents.

These tools allow the agent to debug and fix broken YouTube scrapers by:
1. Finding the correct handle for a channel name.
2. Testing a handle to see if it has a valid videos tab.
"""

import json
import logging
import sys

from app.tools.registry import registry

logger = logging.getLogger(__name__)


@registry.register(
    name="youtube_search_handle",
    description="Search for the correct YouTube channel handle (e.g. @markets) given a channel name or old handle.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The name or old handle of the YouTube channel (e.g., 'Bloomberg Television' or 'FundstratTomLee').",
            },
        },
        "required": ["query"],
    },
    tier=0,
    source="playwright",
)
async def youtube_search_handle(query: str) -> str:
    """Search the web to find the YouTube channel URL for a given query."""
    from app.tools.web_tools import search_web

    logger.info(f"[YouTubeTools] Searching for handle for query: {query}")

    # We use the existing web search tool but modify the query slightly to target YouTube channels
    search_query = f"{query} youtube channel url"
    result_str = await search_web(search_query, num_results=5)

    try:
        result_json = json.loads(result_str)
        if result_json.get("status") == "success":
            results = result_json.get("results", [])
            urls = [
                r.get("url", "") for r in results if "youtube.com/@" in r.get("url", "")
            ]

            if urls:
                # Extract handles from URLs
                handles = []
                for url in urls:
                    parts = url.split("youtube.com/")
                    if len(parts) > 1:
                        handle_part = parts[1].split("/")[0].split("?")[0]
                        if handle_part.startswith("@") and handle_part not in handles:
                            handles.append(handle_part)

                if handles:
                    return json.dumps(
                        {
                            "status": "success",
                            "query": query,
                            "suggested_handles": handles,
                            "message": f"Found {len(handles)} potential handles. You should test them using youtube_test_channel.",
                        }
                    )

            return json.dumps(
                {
                    "status": "not_found",
                    "message": "Could not extract any @handles from the search results.",
                    "raw_search_results": results,
                }
            )

    except Exception as e:
        logger.error(f"[YouTubeTools] Error in search handle: {e}")
        return json.dumps({"status": "error", "error": str(e)})

    return result_str


@registry.register(
    name="youtube_test_channel",
    description="Test if a YouTube channel handle is valid and has a videos tab. Returns success or the specific yt-dlp error.",
    parameters={
        "type": "object",
        "properties": {
            "handle": {
                "type": "string",
                "description": "The YouTube handle to test, must start with @ (e.g., '@markets').",
            },
        },
        "required": ["handle"],
    },
    tier=0,
    source="local",
)
async def youtube_test_channel(handle: str) -> str:
    """Run yt-dlp to verify a channel handle."""
    import asyncio

    if not handle.startswith("@"):
        handle = f"@{handle}"

    logger.info(f"[YouTubeTools] Testing channel handle: {handle}")

    try:
        cmd = [
            sys.executable,
            "-m",
            "yt_dlp",
            f"https://www.youtube.com/{handle}/videos",
            "--flat-playlist",
            "--dump-json",
            "--playlist-end=1",
            "--no-download",
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)

        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()

        if process.returncode == 0 and stdout_text:
            try:
                # Verify it returned valid JSON representing a video
                video_data = json.loads(stdout_text.split("\n")[0])
                title = video_data.get("title", "Unknown")
                channel = video_data.get("channel", "Unknown")

                return json.dumps(
                    {
                        "status": "success",
                        "handle": handle,
                        "is_valid": True,
                        "channel_name": channel,
                        "latest_video_title": title,
                    }
                )
            except json.JSONDecodeError:
                pass

        # If it failed or wasn't valid JSON
        is_404 = "HTTP Error 404" in stderr_text
        no_videos_tab = "This channel does not have a videos tab" in stderr_text

        return json.dumps(
            {
                "status": "error",
                "handle": handle,
                "is_valid": False,
                "error_summary": "404 Not Found"
                if is_404
                else ("No videos tab" if no_videos_tab else "Unknown error"),
                "full_stderr": stderr_text[:500]
                + ("..." if len(stderr_text) > 500 else ""),
            }
        )

    except asyncio.TimeoutError:
        return json.dumps({"status": "error", "error": "Timeout while testing channel"})
    except Exception as e:
        logger.error(f"[YouTubeTools] Error testing channel: {e}")
        return json.dumps({"status": "error", "error": str(e)})
