import os
import time
import queue
import threading
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

file_queue = queue.Queue(maxsize=10000)
stats = {"downloaded": 0, "skipped": 0, "failed": 0}
stats_lock = threading.Lock()
session_local = threading.local()

def get_session():
    if not hasattr(session_local, "session"):
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=3)
        s.mount("https://", adapter)
        session_local.session = s
    return session_local.session

def worker_thread(output_dir, retries=5):
    while True:
        item = file_queue.get()
        if item is None:
            file_queue.task_done()
            break

        outfile = os.path.join(output_dir, item.file_name)
        os.makedirs(os.path.dirname(outfile), exist_ok=True)

        if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
            with stats_lock:
                stats["skipped"] += 1
                curr_skipped = stats["skipped"]
                if curr_skipped % 1000 == 0:
                    print(f"Skipped {curr_skipped} existing files...", flush=True)
            file_queue.task_done()
            continue

        sess = get_session()
        success = False
        for attempt in range(1, retries + 1):
            try:
                r = sess.get(item.url, stream=True, timeout=30)
                r.raise_for_status()
                with open(outfile, "wb") as f:
                    for chunk in r.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
                success = True
                break
            except Exception:
                if attempt < retries:
                    time.sleep(1.0 * attempt)

        with stats_lock:
            if success:
                stats["downloaded"] += 1
                curr = stats["downloaded"]
                if curr % 100 == 0:
                    print(f"Progress: [{curr} newly downloaded, {stats['skipped']} skipped / 29,129 total]", flush=True)
            else:
                stats["failed"] += 1
                print(f"Failed file after {retries} retries: {item.file_name}", flush=True)

        file_queue.task_done()

def producer_thread(kernel_slug: str):
    api = KaggleApi()
    api.authenticate()
    owner_slug, k_slug, _ = api.parse_kernel_string(kernel_slug)

    token = None
    page_size = 500

    while True:
        try:
            with api.build_kaggle_client() as kaggle:
                req = ApiListKernelSessionOutputRequest()
                req.user_name = owner_slug
                req.kernel_slug = k_slug
                api._set_paging(req, page_size, token)
                response = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
                token = response.next_page_token

            files = response.files or []
            for item in files:
                file_queue.put(item)

            if not token:
                break
        except Exception as e:
            print(f"Producer temporary error: {e}, backing off 3s...", flush=True)
            time.sleep(3)

def start_download(kernel_slug: str, output_dir: str, num_workers: int = 16):
    print(f"Starting 16-worker downloader...", flush=True)
    
    prod = threading.Thread(target=producer_thread, args=(kernel_slug,), daemon=True)
    prod.start()

    workers = []
    for _ in range(num_workers):
        w = threading.Thread(target=worker_thread, args=(output_dir,), daemon=True)
        w.start()
        workers.append(w)

    prod.join()
    print("Producer finished listing all files!", flush=True)

    file_queue.join()

    for _ in range(num_workers):
        file_queue.put(None)

    for w in workers:
        w.join()

    print(f"\nCOMPLETED! Total Downloaded: {stats['downloaded']}, Total Skipped: {stats['skipped']}, Total Failed: {stats['failed']}", flush=True)

if __name__ == "__main__":
    start_download("anamariabarrosnonato/sensecv-six-model-dataset-audit", "output_audit", num_workers=16)
