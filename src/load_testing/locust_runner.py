import json
import random
import subprocess
import sys
from pathlib import Path


LOCUST_SCRIPT = '''
import random, json
from locust import HttpUser, task, between
from faker import Faker

fake = Faker("ar")

class CompletionLoadTest(HttpUser):
    wait_time = between(1, 3)

    @task
    def post_completion(self):
        model_id = "{model_id}"
        prompt = fake.text(max_nb_chars=random.randint(150, 200))

        response = self.client.post("/v1/completions", json={{
            "model": model_id,
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.3,
        }})

        if response.status_code == 200:
            with open("{output_file}", "a", encoding="utf-8") as dest:
                dest.write(json.dumps({{
                    "prompt": prompt,
                    "response": response.json()["choices"][0]["text"],
                }}, ensure_ascii=False) + "\\n")
'''


def run_load_test(
    host: str = "http://localhost:8000",
    model_id: str = "news-lora",
    users: int = 20,
    spawn_rate: int = 1,
    duration: str = "60s",
    output_dir: str = ".",
    html_report: str = "locust_results.html",
) -> None:
    output_file = str(Path(output_dir) / "vllm_tokens.txt")
    script_path = str(Path(output_dir) / "locust_generated.py")

    script = LOCUST_SCRIPT.format(model_id=model_id, output_file=output_file)

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)

    cmd = [
        sys.executable, "-m", "locust",
        "--headless",
        "-f", script_path,
        "--host", host,
        "-u", str(users),
        "-r", str(spawn_rate),
        "-t", duration,
        f"--html={html_report}",
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nTokens log saved to: {output_file}")
    print(f"HTML report: {html_report}")