import csv
import random


num_records = 100


with open('fitness_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['SleepHours', 'HeartRate', 'Steps'])
    
   
    for _ in range(num_records):
        sleep_hours = round(random.uniform(4, 10), 1)      
        heart_rate = random.randint(50, 100)               
        steps = random.randint(1000, 20000)               
        writer.writerow([sleep_hours, heart_rate, steps])