import subprocess


def run(step):
    print(f"\nRunning: {step}")
    subprocess.run(["python", "-m", step], check=True)


if __name__ == "__main__":
    run("src.preprocess")
    run("src.train")
    run("src.evaluate")
