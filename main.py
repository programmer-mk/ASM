import math
import os
import zipfile
import os.path as path
import csv
import networkx as nx
import collections
import matplotlib.pyplot as pl
import time
import math
import datetime

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


def results_path(file_name):
    return os.path.join(RESULTS_DIR, file_name)


def suffix():
    ts = time.time()
    return "___"+datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')


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

        if loser_id in players_2018_dictionary:
            players_2018_dictionary[loser_id].append(winner_id)
        else:
            players_2018_dictionary[loser_id] = [winner_id]


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

        if loser_id in players_2019_dictionary:
            players_2019_dictionary[loser_id].append(winner_id)
        else:
            players_2019_dictionary[loser_id] = [winner_id]


def save_actor_graph_as_pdf(actor_graph: nx.Graph, color='r', file_name=""):
    #pos = nx.spring_layout(actor_graph, iterations=5000, )
    #pos = nx.random_layout(actor_graph)
    number_of_nodes: int = len(actor_graph.nodes())
    n: int = 4
    pos = nx.spring_layout(actor_graph, k=(1/math.sqrt(number_of_nodes))*n)
    pl.figure(figsize=(20, 20))  # Don't create a humongous figure
    nx.draw_networkx(actor_graph, pos, node_size=30, font_size='xx-small', with_labels=False, node_color=color)
    pl.axis('off')
    pl.show()
    pl.savefig(file_name, format='pdf', dpi=900)


def create_atp_matches_2018_network():
    player_graph = nx.Graph()

    player_graph.add_nodes_from(players_2018_dictionary.keys())

    all_players_played_in_2018 = players_2018_dictionary.keys()

    for player1 in all_players_played_in_2018:
        distinct_opponents = list(set(players_2018_dictionary[player1]))
        opponents_match_num = collections.Counter(players_2018_dictionary[player1])
        print(opponents_match_num)
        print(
            '################################################################################################################################')
        for player2 in distinct_opponents:
            print('Player {player1} win over player {player2} {matches} times'.format(player1=player1, player2=player2,
                                                                                      matches=opponents_match_num[
                                                                                          player2]))

            if player_graph.has_edge(player1, player2):
                player_graph[player1][player2]['weight'] += opponents_match_num[player2]
            else:
                player_graph.add_edge(player1, player2, weight=opponents_match_num[player2])

    player_graph.remove_edges_from(nx.selfloop_edges(player_graph))
    return player_graph


def create_atp_matches_2019_network():
    player_graph = nx.Graph()

    player_graph.add_nodes_from(players_2019_dictionary.keys())

    all_players_played_in_2019 = players_2019_dictionary.keys()

    for player1 in all_players_played_in_2019:
        distinct_opponents = list(set(players_2019_dictionary[player1]))
        opponents_match_num = collections.Counter(players_2019_dictionary[player1])
        print(opponents_match_num)
        print(
            '################################################################################################################################')
        for player2 in distinct_opponents:
            print('Player {player1} win over player {player2} {matches} times'.format(player1=player1, player2=player2,
                                                                                      matches=opponents_match_num[
                                                                                          player2]))

            if player_graph.has_edge(player1, player2):
                player_graph[player1][player2]['weight'] += opponents_match_num[player2]
            else:
                player_graph.add_edge(player1, player2, weight=opponents_match_num[player2])

    player_graph.remove_edges_from(nx.selfloop_edges(player_graph))
    return player_graph


def main():
    print("Starting script...")
    extract_secondary_dataset()
    read_atp_matches_2018_dataset()
    read_atp_matches_2019_dataset()

    matches_2018_graph = create_atp_matches_2018_network()
    matches_2019_graph = create_atp_matches_2019_network()
    save_actor_graph_as_pdf(matches_2018_graph, 'r', 'player_matches_2018_graph.pdf')
    save_actor_graph_as_pdf(matches_2019_graph, 'r', 'player_matches_2019_graph.pdf')

if __name__ == "__main__":
    main()
