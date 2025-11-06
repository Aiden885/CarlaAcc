import time

import carla


def main():
    """
    Minimal helper script to load a specific CARLA map.
    Edit `MAP_NAME` below to switch towns before running the script.
    """
    # === User-configurable section ===
    MAP_NAME = "Town04"  # e.g. "Town01", "Town03_Opt", "Town10HD"
    HOST = "localhost"
    PORT = 2000
    TIMEOUT = 10.0  # seconds

    print(f"[INFO] Connecting to CARLA at {HOST}:{PORT} ...")
    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT)

    try:
        # Attempt to load the requested map
        print(f"[INFO] Loading map '{MAP_NAME}' ...")
        world = client.load_world(MAP_NAME)
        time.sleep(2.0)  # give CARLA a moment to finish loading assets
        loaded_name = world.get_map().name
        print(f"[SUCCESS] Map loaded: {loaded_name}")
    except RuntimeError as exc:
        print(f"[ERROR] Failed to load map '{MAP_NAME}': {exc}")
    except Exception as exc:
        print(f"[ERROR] Unexpected failure: {exc}")


if __name__ == "__main__":
    main()
