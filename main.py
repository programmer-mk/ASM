import os
import zipfile
import os.path as path
import csv
import networkx as nx

DATA_DIR = 'data'
RESULTS_DIR = 'results'
ZIP_FILE_NAME = 'ASM_PZ2_podaci_2021.zip'
ATP_MATCHES_2018_DATASET = 'atp_matches_2018.csv'
ATP_MATCHES_2019_DATASET = 'atp_matches_2019.csv'

atp_mathces_2018_dataset = []
atp_mathces_2019_dataset = []
players_2018_dictionary = {}
players_2019_dictionary = {}


def data_path(file_name):
    print("Returns relative path to data file passed as the argument.")
    return os.path.join(DATA_DIR, file_name)


def extract_csv_from_zip(clean: bool = False):
    print("Extracts the data from the provided zip file if no extracted data is found.")
    if (not clean) and path.isfile(data_path(ATP_MATCHES_2018_DATASET)):
        print(ATP_MATCHES_2018_DATASET + ' already extracted.')
    else:
        print('Extracting data from ' + ZIP_FILE_NAME)
        exists = os.path.isfile(data_path(ZIP_FILE_NAME))

        if not exists:
            raise OSError("Error -file '" + ZIP_FILE_NAME + "' not found. Aborting.")

        with zipfile.ZipFile(data_path(ZIP_FILE_NAME), 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)


def extract_secondary_dataset(clean: bool = False):
    extract_csv_from_zip(clean)


def read_atp_matches_2018_dataset():
    atp_mathces_2018_dataset_header = None
    with open(data_path(ATP_MATCHES_2018_DATASET), 'r') as csvFile:
        reader = csv.reader(csvFile)

        # Read primary dataset.
        for row in reader:
            if atp_mathces_2018_dataset_header is None:
                atp_mathces_2018_dataset_header = row
            else:
                atp_mathces_2018_dataset.append(row)
    csvFile.close()

    # Get indexes of columns of interest.
    winner_id_index = atp_mathces_2018_dataset_header.index("winner_id")
    winner_name_index = atp_mathces_2018_dataset_header.index("winner_name")
    loser_id_index = atp_mathces_2018_dataset_header.index("loser_id")
    loser_name_index = atp_mathces_2018_dataset_header.index("loser_name")

    for row in atp_mathces_2018_dataset:
        winner_id = row[winner_id_index]
        winner_name = row[winner_name_index]
        loser_id = row[loser_id_index]
        loser_name = row[loser_name_index]

        if winner_id in players_2018_dictionary:
            players_2018_dictionary[winner_id].append(loser_id)
        else:
            players_2018_dictionary[winner_id] = [loser_id]


def read_atp_matches_2019_dataset():
    atp_mathces_2019_dataset_header = None
    with open(data_path(ATP_MATCHES_2019_DATASET), 'r') as csvFile:
        reader = csv.reader(csvFile)

        # Read primary dataset.
        for row in reader:
            if atp_mathces_2019_dataset_header is None:
                atp_mathces_2019_dataset_header = row
            else:
                atp_mathces_2019_dataset.append(row)
    csvFile.close()

    # Get indexes of columns of interest.
    winner_id_index = atp_mathces_2019_dataset_header.index("winner_id")
    winner_name_index = atp_mathces_2019_dataset_header.index("winner_name")
    loser_id_index = atp_mathces_2019_dataset_header.index("loser_id")
    loser_name_index = atp_mathces_2019_dataset_header.index("loser_name")

    for row in atp_mathces_2019_dataset:
        winner_id = row[winner_id_index]
        winner_name = row[winner_name_index]
        loser_id = row[loser_id_index]
        loser_name = row[loser_name_index]

        if winner_id in players_2019_dictionary:
            players_2019_dictionary[winner_id].append(loser_id)
        else:
            players_2019_dictionary[winner_id] = [loser_id]


def main():
    print("Starting script...")
    extract_secondary_dataset()
    read_atp_matches_2018_dataset()
    read_atp_matches_2019_dataset()
    print(players_2018_dictionary)
    print(players_2019_dictionary)

if __name__ == "__main__":
    main()
