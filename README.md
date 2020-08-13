# PersonCount

## Dependency

```bash
$ pip install -r requirements.txt
```

## Count method

detected people number in sliding window of size=(fps * interval) as stored data to calculate average person count.

choose the max number in each second, and calculate average of 5 max values.

The calculation is written in utils/flow_data.py

## Testing

put the People_sample_2.mp4 in data/People_sample_2.mp4

```bash
$ python person_count.py
```

Fps: 50 on GeForce GTX 1080 Ti, 0.02s per frame
