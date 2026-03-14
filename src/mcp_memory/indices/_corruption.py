def is_corruption_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        pattern in message
        for pattern in (
            "database disk image is malformed",
            "disk i/o error",
            "file is not a database",
            "database is locked",
            "unable to open database file",
        )
    )