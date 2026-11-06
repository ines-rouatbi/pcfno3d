"""Fetch a dated NOAA GFS forecast snapshot for the national terrain view."""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['wind_speed_10m', 'wind_direction_10m', 'wind_speed_80m', 'wind_direction_80m']


def main() -> None:
    points = [(30 + y * .5, 7.5 + x * .5) for y in range(17) for x in range(10)]
    records = []
    times = None
    for offset in range(0, len(points), 32):
        batch = points[offset:offset + 32]
        params = urllib.parse.urlencode({
            'latitude': ','.join(str(p[0]) for p in batch),
            'longitude': ','.join(str(p[1]) for p in batch),
            'hourly': ','.join(FIELDS), 'wind_speed_unit': 'ms',
            'timeformat': 'unixtime', 'timezone': 'Africa/Tunis',
            'forecast_days': 2, 'cell_selection': 'nearest',
        })
        with urllib.request.urlopen('https://api.open-meteo.com/v1/gfs?' + params, timeout=45) as response:
            results = json.load(response)
        if not isinstance(results, list) or len(results) != len(batch):
            raise ValueError('Unexpected national forecast response')
        for (lat, lon), result in zip(batch, results):
            hourly = result['hourly']
            if times is None:
                times = hourly['time']
            if hourly['time'] != times or len(times) < 24:
                raise ValueError('Forecast times are inconsistent')
            if result['hourly_units']['wind_speed_80m'] != 'm/s':
                raise ValueError('Forecast speed units are not m/s')
            records.append({
                'latitude': lat, 'longitude': lon,
                'modelLatitude': result['latitude'], 'modelLongitude': result['longitude'],
                **{key: hourly[key] for key in FIELDS},
            })
        print(f'Fetched {len(records)} / {len(points)} forecast locations', flush=True)
    snapshot = {
        'version': 1, 'source': 'NOAA GFS via Open-Meteo',
        'sourceUrl': 'https://open-meteo.com/en/docs/gfs-api',
        'attribution': 'Weather data by Open-Meteo.com, NOAA GFS; CC BY 4.0',
        'retrievedAt': int(time.time() * 1000), 'timezone': 'Africa/Tunis',
        'grid': {'west': 7.5, 'south': 30, 'step': .5, 'nx': 10, 'ny': 17},
        'heights': [10, 80], 'times': times, 'points': records,
    }
    for relative in ['frontend/public/weather/tunisia-gfs.json', 'webapp/weather/tunisia-gfs.json']:
        target = ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(snapshot, separators=(',', ':')), encoding='utf-8')
        temporary.replace(target)
    print(f'National forecast saved: {len(records)} locations, {len(times)} hours, real retrieval timestamp.')


if __name__ == '__main__':
    main()
