# utils/logger.py

import csv
import os

def log_run_results(filepath, run_name, learning_rate, batch_size, num_epochs, rouge1, rouge2, rougeL):
    file_exists = os.path.isfile(filepath)

    with open(filepath, mode='a', newline='') as csv_file:
        fieldnames = ['run_name', 'learning_rate', 'batch_size', 'num_epochs', 'rouge1', 'rouge2', 'rougeL']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'run_name': run_name,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'rouge1': rouge1,
            'rouge2': rouge2,
            'rougeL': rougeL,
        })
