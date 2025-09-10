import json
import random

num_records = 100

data = []

for _ in range(num_records):
    record = {
        "SleepHours": round(random.uniform(4, 10), 1),
        "HeartRate": random.randint(50, 100),
        "Steps": random.randint(1000, 20000)
    }
    data.append(record)

with open('fitness_data.json', 'w') as file:
    json.dump(data, file, indent=4)