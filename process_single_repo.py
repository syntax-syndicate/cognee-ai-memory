import os
import cognee


def process_repo():
    instance_data = os.getenv("INSTANCE_DATA")
    disable_cognee = os.getenv("DISABLE_COGNEE", "False") == "True"

    print(instance_data)
    print(disable_cognee)

    return instance_data


if __name__ == "__main__":
    process_repo()
