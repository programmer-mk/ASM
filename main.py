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
import inspect
import pandas as pd


DATA_DIR = 'data'
RESULTS_DIR = 'results'
ZIP_FILE_NAME = 'ASM_PZ2_podaci_2021.zip'
ATP_MATCHES_2018_DATASET = 'atp_matches_2018.csv'
ATP_MATCHES_2019_DATASET = 'atp_matches_2019.csv'
ATP_CURRENT_RANKING_DATASET = 'data/atp_rankings_current.csv'
ATP_PLAYERS_DATASET = 'data/atp_players.csv'

atp_mathces_2018_dataset = []
atp_mathces_2019_dataset = []
players_2018_dictionary = {}
players_2019_dictionary = {}

current_player_ranking = pd.DataFrame().empty
atp_players = pd.DataFrame().empty
atp_matches_2018 = pd.DataFrame().empty

players_2018_data = {}
players_2019_data = {}

def data_path(file_name):
    print("Returns relative path to data file passed as the argument.")
    return os.path.join(DATA_DIR, file_name)


def results_path(file_name):
    return os.path.join(RESULTS_DIR, file_name)


def check():
    print("Executing "+inspect.stack()[1].function)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)


def suffix():
    ts = time.time()
    return "___"+datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')


def add_missing_header_to_dataset(headers, dataset):
    with open(dataset,newline='') as f:
        r = csv.reader(f)
        data = [line for line in r]
    with open(dataset,'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(data)


def load_current_player_ranking():
    global current_player_ranking
    current_player_ranking = pd.read_csv(ATP_CURRENT_RANKING_DATASET)
    print(current_player_ranking.head())


def load_atp_players():
    global atp_players
    atp_players = pd.read_csv(ATP_PLAYERS_DATASET)
    print(atp_players.head())


def load_atp_matches_2018():
    global atp_matches_2018
    atp_matches_2018 = pd.read_csv(f'data/{ATP_MATCHES_2018_DATASET}')
    print(atp_matches_2018.head())


def extract_csv_from_zip(clean: bool = False):
    print("Extracts the data from the provided zip file if no extracted data is found.")
    if (not clean) and path.isfile(data_path(ATP_MATCHES_2018_DATASET)):
        print(ATP_MATCHES_2018_DATASET + ' already extracted.')
        return True
    else:
        print('Extracting data from ' + ZIP_FILE_NAME)
        exists = os.path.isfile(data_path(ZIP_FILE_NAME))

        if not exists:
            raise OSError("Error -file '" + ZIP_FILE_NAME + "' not found. Aborting.")

        with zipfile.ZipFile(data_path(ZIP_FILE_NAME), 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        return False


def extract_secondary_dataset(clean: bool = False):
    already_extracted = extract_csv_from_zip(clean)
    if not already_extracted:
        add_missing_header_to_dataset(['ranking_date', 'rank', 'player_id', 'points'], ATP_CURRENT_RANKING_DATASET)
        add_missing_header_to_dataset(['player_id', 'first_name', 'last_name', 'hand', 'birth_date', 'country_code'], ATP_PLAYERS_DATASET)
    load_current_player_ranking()
    load_atp_players()
    load_atp_matches_2018()


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


def sort_nodes_by_degree(graph):
    ret = list(graph.degree(graph.nodes()))
    ret.sort(key=lambda x: x[1], reverse=True)
    return ret


def sort_nodes_by_weighed_degree(graph):
    ret = list(graph.degree(graph.nodes(), 'weight'))
    ret.sort(key=lambda x: x[1], reverse=True)
    return ret


def question1(player_network: nx.Graph):
    check()

    n = player_network.number_of_nodes()
    with open(results_path("q1.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Avg degree", "Avg weighed degree"])

        avg_degree = sum(degrees[1] for degrees in sort_nodes_by_degree(player_network))/n
        avg_wdegree = sum(degrees[1] for degrees in sort_nodes_by_weighed_degree(player_network))/n

        writer.writerow([avg_degree, avg_wdegree])
    csvFile.close()


def question2(players: nx.Graph, top: int = 10):
    check()

    lst1 = sort_nodes_by_degree(players)[0:top]
    lst2 = sort_nodes_by_weighed_degree(players)[0:top]

    with open(results_path("q2.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Rank", "Top players", "Degree", "Top players", "Weighed degree"])

        for rank in range(0, top):
            writer.writerow([rank+1, lst1[rank][0], lst1[rank][1], lst2[rank][0], lst2[rank][1]])

    csvFile.close()


def question3():
    print('skipped...')


def question4():
    print('Look into results of question 2')


def get_atp_rank(player_id):
    global current_player_ranking
    atp_rang = current_player_ranking[current_player_ranking['player_id'] == int(player_id)].sort_values('ranking_date', ascending=False)['rank'].head(1).values[0]
    print(atp_rang)
    return atp_rang


def question5(players: nx.Graph, top: int = 10):
    lst1 = sort_nodes_by_degree(players)[0:top]
    lst2 = sort_nodes_by_weighed_degree(players)[0:top]

    with open(results_path("q5.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Rank", "Top players", "Degree", "Atp Rank", "Top players", "Weighed degree", "Atp Rank"])

        for rank in range(0, top):
            writer.writerow([rank+1, lst1[rank][0], lst1[rank][1], get_atp_rank(lst1[rank][0]), lst2[rank][0], lst2[rank][1], get_atp_rank(lst2[rank][0])])
    csvFile.close()


def count_players_by_country():
    full_winning_player_data = pd.merge(atp_players, atp_matches_2018,left_on='player_id',right_on='winner_id').drop_duplicates(['player_id'])
    full_winning_player_ids = full_winning_player_data['player_id']
    full_loser_player_data = pd.merge(atp_players,atp_matches_2018,left_on='player_id',right_on='loser_id').drop_duplicates(['player_id'])
    full_loser_player_ids = full_loser_player_data['player_id']
    all_player_ids = pd.concat([full_winning_player_ids, full_loser_player_ids], axis=0)
    distinct_players = all_player_ids.drop_duplicates(keep='first').to_frame()
    count_players_by_country = pd.merge(atp_players,distinct_players,left_on='player_id',right_on='player_id')
    return count_players_by_country.groupby('country_code')['player_id'].count().sort_values(ascending=False)


def question6():
    country_count = count_players_by_country()
    print(country_count.head(10))

def main():
    print("Starting script...")
    extract_secondary_dataset()
    read_atp_matches_2018_dataset()
    read_atp_matches_2019_dataset()

    matches_2018_graph = create_atp_matches_2018_network()
    matches_2019_graph = create_atp_matches_2019_network()
    # save_actor_graph_as_pdf(matches_2018_graph, 'r', 'player_matches_2018_graph.pdf')
    # save_actor_graph_as_pdf(matches_2019_graph, 'r', 'player_matches_2019_graph.pdf')

    # make this generic, do compute for all graphs
    question1(matches_2018_graph)
    question2(matches_2018_graph)

    question5(matches_2018_graph)
    question6()


if __name__ == "__main__":
    main()
