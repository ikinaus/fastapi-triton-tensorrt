import asyncio
import os
import time

import httpx

stats = {
    "classifier_onnx": {"count": 0, "total_time": 0.0},
    "classifier_trt": {"count": 0, "total_time": 0.0},
}


async def load_worker(worker_id, client, image_bytes, model_name):
    url = "http://localhost:5000/predict"
    params = {"model_name": model_name}
    files = {"file": image_bytes}

    for _ in range(100):
        try:
            start_time = time.time()
            response = await client.post(url, params=params, files=files)
            latency = time.time() - start_time

            stats[model_name]["count"] += 1
            stats[model_name]["total_time"] += latency

            total_model_requests = stats[model_name]["count"]

            if total_model_requests % 100 == 0:
                avg_time = stats[model_name]["total_time"] / total_model_requests
                print(
                    f"[{model_name.upper()}] Requests: {total_model_requests} | "
                    f"Current worker #{worker_id} | "
                    f"Last: {latency:.3f}s | "
                    f"Avg time: {avg_time:.3f}s | "
                    f"Prediction: {response.json()}"
                )

        except Exception as e:
            print(f"[Error] Worker #{worker_id} ({model_name}) failed: {e}")
            await asyncio.sleep(1)


async def main():
    image_path = "images/7.jpg"
    concurrency_per_model = 32

    if not os.path.exists(image_path):
        print(f"Error: place the file {image_path} next to the script!")
        return

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    print("=== STARTING COMPARATIVE LOAD TEST ===")
    print(f"Workers for ONNX: {concurrency_per_model}")
    print(f"Workers for TRT: {concurrency_per_model}")
    print("Press Ctrl+C to stop...\n")

    timeout = httpx.Timeout(30.0, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        workers = []

        # Start workers for ONNX
        for i in range(1, concurrency_per_model + 1):
            workers.append(
                load_worker(f"ONNX-{i}", client, image_bytes, "classifier_onnx")
            )

        # Start workers for TensorRT
        for i in range(1, concurrency_per_model + 1):
            workers.append(
                load_worker(f"TRT-{i}", client, image_bytes, "classifier_trt")
            )

        # Start all workers concurrently
        await asyncio.gather(*workers)

    print("\n=== TEST COMPLETED ===")
    for model, data in stats.items():
        if data["count"] > 0:
            print(
                f"Model: {model} | Total successful requests: {data['count']} | Final average latency: {data['total_time'] / data['count']:.4f} sec"
            )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest stopped by user.")
