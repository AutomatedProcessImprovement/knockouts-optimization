def seconds_to_hms(seconds):
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    days, h = divmod(h, 24)
    if (days > 0) or (h > 24):
        return f'{days:d} days, {h:d}:{m:02d}:{s:02d}'

    return f'{h:d}:{m:02d}:{s:02d}'
