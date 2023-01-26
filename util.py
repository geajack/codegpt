from datetime import datetime
from pathlib import Path


def output_directory(name):
    output_home = Path("../results")
    now = datetime.now().strftime("%d-%m-%y@%H:%M:%S")
    output_directory_name = f"{name}-{now}"
    output_directory = output_home / output_directory_name
    return output_directory


def save_output(output_file, output_source):
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    with open(output_file, "wb") as output_file:
        for prediction in output_source:
            buffer = prediction.encode("utf-8")
            output_file.write(buffer)
            output_file.write(b"\0")
            output_file.flush()
            print(prediction)