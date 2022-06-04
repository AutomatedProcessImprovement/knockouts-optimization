import platform


def is_windows():
    system = platform.system()

    return "windows" in system.lower()
