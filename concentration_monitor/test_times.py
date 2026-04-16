import csv
counts = {0:0, 1:0, 2:0}
with open('data/training_data.csv') as f:
    for row in csv.DictReader(f):
        counts[int(row['label'])] += 1
print(f'专注: {counts[0]}条')
print(f'分心: {counts[1]}条')
print(f'疲劳: {counts[2]}条')
print(f'总计: {sum(counts.values())}条')