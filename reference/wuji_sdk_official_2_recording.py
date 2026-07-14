#!/usr/bin/env python3
# =============================================================================
# VENDORED REFERENCE — 官方录制示例(未改动逻辑,仅加本头注)
#
# 来源 : wuji-technology/wuji-sdk
#        examples/python/wuji_glove/2.recording.py
#        https://github.com/wuji-technology/wuji-sdk/blob/main/examples/python/wuji_glove/2.recording.py
# 许可 : MIT License (© Wuji Technology)
#
# 放这里是给客户做"官方 MCAP 录制"的对照参考;本仓库自带的
# record_glove_data.py 是 CSV/JSONL 版,两者定位不同,见
# docs/customer-guide-calibration-and-recording.md 的对比表。
#
# 运行前:pip install "wuji-sdk"，python reference/wuji_sdk_official_2_recording.py
# 产物 :./data/<时间戳>.mcap，用 Foxglove Studio 或 `mcap info <file>` 打开。
# =============================================================================
"""
Recording example.

Connect to a Wuji Glove and record sensor data to an MCAP file.
Uses TopicRecorder with LZ4 compression. The resulting file can be
viewed with Foxglove Studio or `mcap info <file>`.

Usage: python 2.recording.py
"""

import asyncio
import os
from datetime import datetime

from wuji_sdk import SdkManager, TopicRecorder


async def main():
    manager = SdkManager.instance()
    devices = manager.scan()

    if not devices:
        print("No devices found")
        return

    glove = manager.connect(sn=devices[0].sn, device_name="glove_0")
    print(f"Connected: {glove.serial_number}")

    # Create a recorder with LZ4 compression
    recorder = TopicRecorder(compression="lz4")

    # Register channels — each .subscribe() feeds data into the recorder
    recorder.record(glove.tactile().subscribe())
    recorder.record(glove.emf_poses().subscribe())
    recorder.record(glove.hand_skeleton().subscribe())

    # Start recording to an MCAP file
    os.makedirs("./data", exist_ok=True)
    path = f"./data/{datetime.now().strftime('%Y%m%d_%H%M%S')}.mcap"
    print(f"Recording to {path} ...")
    handle = await recorder.start(path)

    try:
        # Record for 10 seconds, then stop
        await asyncio.sleep(10)
    finally:
        stop_task = asyncio.ensure_future(handle.stop())
        try:
            summary = await asyncio.shield(stop_task)
        except asyncio.CancelledError:
            summary = await stop_task
        print(f"Done — {summary.total_frames} frames, "
              f"{summary.file_size / 1024 / 1024:.2f} MB, "
              f"{summary.duration_s:.1f}s")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user")
