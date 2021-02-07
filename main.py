import math
import os
import zipfile
import os.path as path
import csv
import matplotlib
import networkx as nx
from networkx.algorithms import community
import collections
import matplotlib.pyplot as pl
from matplotlib import colors as mcolors
import time
import math
import datetime
import inspect
import pandas as pd
import numpy as np
from collections import Counter
from matplotlib.backends.backend_pdf import PdfPages
import networkx.algorithms.community.quality as qual
import scipy.stats as stats
import operator
from operator import itemgetter
from pandas import DataFrame

DATA_DIR = 'data'
RESULTS_DIR = 'results'
ZIP_FILE_NAME = 'ASM_PZ2_podaci_2021.zip'
ATP_MATCHES_2018_DATASET = 'atp_matches_2018.csv'
ATP_MATCHES_2019_DATASET = 'atp_matches_2019.csv'
ATP_MATCHES_2020_DATASET = 'atp_matches_2020.csv'
ATP_CURRENT_RANKING_DATASET = 'data/atp_rankings_current.csv'
ATP_PLAYERS_DATASET = 'data/atp_players.csv'

atp_mathces_2018_dataset = []
atp_mathces_2019_dataset = []
atp_mathces_2020_dataset = []
players_2018_dictionary = {}
players_2019_dictionary = {}
players_2020_dictionary = {}

current_player_ranking = pd.DataFrame().empty
atp_players = pd.DataFrame().empty
atp_matches_2018 = pd.DataFrame().empty
atp_matches_2019 = pd.DataFrame().empty
atp_matches_2020 = pd.DataFrame().empty

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


def load_atp_matches_2019():
    global atp_matches_2019
    atp_matches_2019 = pd.read_csv(f'data/{ATP_MATCHES_2019_DATASET}')
    print(atp_matches_2019.head())


def load_atp_matches_2020():
    global atp_matches_2020
    atp_matches_2020 = pd.read_csv(f'data/{ATP_MATCHES_2020_DATASET}')
    print(atp_matches_2020.head())


def sored_nodes_on_betweenness_centrality(graph: nx.Graph):
    ret = []
    bc = nx.betweenness_centrality(graph)
    for node in graph.nodes():
        ret.append([node,  bc[node]])

    ret.sort(key=lambda x: x[1], reverse=True)
    return ret


def get_player_name(player_id):
    first_name = atp_players[atp_players['player_id'] == int(player_id)]['first_name'].head(1).values[0]
    last_name = atp_players[atp_players['player_id'] == int(player_id)]['last_name'].head(1).values[0]
    return first_name + ' '+ last_name


def sored_nodes_on_closeness_centrality(graph: nx.Graph):
    ret = []
    cc = nx.closeness_centrality(graph)
    for node in graph.nodes():
        ret.append([node,  cc[node]])

    ret.sort(key=lambda x: x[1], reverse=True)
    return ret


def sorted_nodes_on_degree_centrality(graph: nx.Graph):
    ret = list()
    dc = nx.degree_centrality(graph)
    for node in graph.nodes():
        ret.append([node, dc[node]])

    ret.sort(key=lambda x: x[1], reverse=True)
    return ret


def sorted_nodes_on_two_centralities(bc: list, dc: list):
    ret = list()
    for item1 in bc:
        for item2 in dc:
            if item1[0] == item2[0]:
                ret.append([item1[0], item1[1]*item2[1]])
                break

    ret.sort(key=lambda x: x[1], reverse=True)
    return ret


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
    load_atp_matches_2019()
    load_atp_matches_2020()


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


def read_atp_matches_2020_dataset():
    atp_mathces_2020_dataset_header = None
    with open(data_path(ATP_MATCHES_2020_DATASET), 'r') as csvFile:
        reader = csv.reader(csvFile)

        # Read primary dataset.
        for row in reader:
            if atp_mathces_2020_dataset_header is None:
                atp_mathces_2020_dataset_header = row
            else:
                atp_mathces_2020_dataset.append(row)
    csvFile.close()

    # Get indexes of columns of interest.
    winner_id_index = atp_mathces_2020_dataset_header.index("winner_id")
    winner_name_index = atp_mathces_2020_dataset_header.index("winner_name")
    loser_id_index = atp_mathces_2020_dataset_header.index("loser_id")
    loser_name_index = atp_mathces_2020_dataset_header.index("loser_name")

    for row in atp_mathces_2020_dataset:
        winner_id = row[winner_id_index]
        winner_name = row[winner_name_index]
        loser_id = row[loser_id_index]
        loser_name = row[loser_name_index]

        if winner_id in players_2020_dictionary:
            players_2020_dictionary[winner_id].append(loser_id)
        else:
            players_2020_dictionary[winner_id] = [loser_id]

        if loser_id in players_2020_dictionary:
            players_2020_dictionary[loser_id].append(winner_id)
        else:
            players_2020_dictionary[loser_id] = [winner_id]


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


def add_players_to_graph(player_graph, player_dictionary):
    players = list(player_dictionary.keys())
    player_countries = list(atp_players[atp_players['player_id'] == int(player)]['country_code'].unique()[0] for player in players)

    player_rankings = []
    for player in players:
        # latest rank will be used
        rank = int()
        if current_player_ranking[current_player_ranking['player_id'] == int(player)].sort_values('ranking_date', ascending=False)['rank'].empty:
            # not active players on last noticed date(don't have rank)
            rank = -1
        else:
            rank = current_player_ranking[current_player_ranking['player_id'] == int(player)].sort_values('ranking_date', ascending=False)['rank'].head(1).values[0]
        player_rankings.append(rank)
    # TODO: group by country and give list of nodes to add_nodes_from
    for index in range(len(players)):
        s = players[index]
        player_graph.add_nodes_from([players[index]], country=player_countries[index], rank=player_rankings[index])
    return player_graph


def create_atp_matches_2018_network():
    player_graph = nx.Graph()
    player_graph = add_players_to_graph(player_graph,players_2018_dictionary)
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
    player_graph = add_players_to_graph(player_graph, players_2019_dictionary)
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


def create_aggregated_network(first_graph, second_graph, third_graph):
    first_half = nx.compose(first_graph,second_graph)
    full_graph = nx.compose(first_half, third_graph)
    return full_graph


def create_atp_matches_2020_network():
    player_graph = nx.Graph()
    player_graph = add_players_to_graph(player_graph, players_2020_dictionary)
    all_players_played_in_2020 = players_2020_dictionary.keys()

    for player1 in all_players_played_in_2020:
        distinct_opponents = list(set(players_2020_dictionary[player1]))
        opponents_match_num = collections.Counter(players_2020_dictionary[player1])
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


def question1(player_network_2018: nx.Graph, player_network_2019: nx.Graph,
              player_network_2020: nx.Graph, player_network_aggregated: nx.Graph):
    check()
    players_num_2018 = player_network_2018.number_of_nodes()
    players_num_2019 = player_network_2019.number_of_nodes()
    players_num_2020 = player_network_2020.number_of_nodes()
    players_num_aggregated = player_network_aggregated.number_of_nodes()
    with open(results_path("q1.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Avg degree-2018", "Avg weighed degree-2018", "Avg degree-2019", "Avg weighed degree-2019",
                         "Avg degree-2020", "Avg weighed degree-2020", "Avg degree-aggregated", "Avg weighed degree-aggregated"])
        avg_degree_2018 = sum(degrees[1] for degrees in sort_nodes_by_degree(player_network_2018))/players_num_2018
        avg_wdegree_2018 = sum(degrees[1] for degrees in sort_nodes_by_weighed_degree(player_network_2018))/players_num_2018
        avg_degree_2019 = sum(degrees[1] for degrees in sort_nodes_by_degree(player_network_2019))/players_num_2019
        avg_wdegree_2019 = sum(degrees[1] for degrees in sort_nodes_by_weighed_degree(player_network_2019))/players_num_2019
        avg_degree_2020 = sum(degrees[1] for degrees in sort_nodes_by_degree(player_network_2020))/players_num_2020
        avg_wdegree_2020 = sum(degrees[1] for degrees in sort_nodes_by_weighed_degree(player_network_2020))/players_num_2020
        avg_degree_aggregated = sum(degrees[1] for degrees in sort_nodes_by_degree(player_network_aggregated))/players_num_aggregated
        avg_wdegree_aggregated = sum(degrees[1] for degrees in sort_nodes_by_weighed_degree(player_network_aggregated))/players_num_aggregated
        writer.writerow([avg_degree_2018, avg_wdegree_2018, avg_degree_2019, avg_wdegree_2019,
                         avg_degree_2020, avg_wdegree_2020, avg_degree_aggregated, avg_wdegree_aggregated ])
    csvFile.close()


def question2(players_2018: nx.Graph, players_2019: nx.Graph, players_2020: nx.Graph, players_aggregated: nx.Graph,top: int = 10):
    check()

    lst1_2018 = sort_nodes_by_degree(players_2018)[0:top]
    lst2_2018 = sort_nodes_by_weighed_degree(players_2018)[0:top]

    lst1_2019 = sort_nodes_by_degree(players_2019)[0:top]
    lst2_2019 = sort_nodes_by_weighed_degree(players_2019)[0:top]

    lst1_2020 = sort_nodes_by_degree(players_2020)[0:top]
    lst2_2020 = sort_nodes_by_weighed_degree(players_2020)[0:top]

    lst1_aggregated = sort_nodes_by_degree(players_aggregated)[0:top]
    lst2_aggregated = sort_nodes_by_weighed_degree(players_aggregated)[0:top]

    with open(results_path("q2.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Rank", "Top players", "Degree", "Top players", "Weighed degree", "year"])

        for rank in range(0, top):
            first_name = atp_players[atp_players['player_id'] == int(lst1_2018[rank][0])]['first_name'].head(1).values[0]
            last_name = atp_players[atp_players['player_id'] == int(lst1_2018[rank][0])]['last_name'].head(1).values[0]

            first_name_weighted = atp_players[atp_players['player_id'] == int(lst2_2018[rank][0])]['first_name'].head(1).values[0]
            last_name_weighted = atp_players[atp_players['player_id'] == int(lst2_2018[rank][0])]['last_name'].head(1).values[0]

            writer.writerow([rank+1, first_name + ' ' + last_name, lst1_2018[rank][1], first_name_weighted + ' ' + last_name_weighted, lst2_2018[rank][1], '2018'])

        writer.writerow([])
        for rank in range(0, top):
            first_name = atp_players[atp_players['player_id'] == int(lst1_2019[rank][0])]['first_name'].head(1).values[0]
            last_name = atp_players[atp_players['player_id'] == int(lst1_2019[rank][0])]['last_name'].head(1).values[0]

            first_name_weighted = atp_players[atp_players['player_id'] == int(lst2_2019[rank][0])]['first_name'].head(1).values[0]
            last_name_weighted = atp_players[atp_players['player_id'] == int(lst2_2019[rank][0])]['last_name'].head(1).values[0]

            writer.writerow([rank+1, first_name + ' ' + last_name, lst1_2019[rank][1], first_name_weighted + ' ' + last_name_weighted, lst2_2019[rank][1], '2019'])

        writer.writerow([])
        for rank in range(0, top):
            first_name = atp_players[atp_players['player_id'] == int(lst1_2020[rank][0])]['first_name'].head(1).values[0]
            last_name = atp_players[atp_players['player_id'] == int(lst1_2020[rank][0])]['last_name'].head(1).values[0]

            first_name_weighted = atp_players[atp_players['player_id'] == int(lst2_2020[rank][0])]['first_name'].head(1).values[0]
            last_name_weighted = atp_players[atp_players['player_id'] == int(lst2_2020[rank][0])]['last_name'].head(1).values[0]

            writer.writerow([rank+1, first_name + ' ' + last_name, lst1_2020[rank][1], first_name_weighted + ' ' + last_name_weighted, lst2_2020[rank][1], '2020'])

        writer.writerow([])
        for rank in range(0, top):
            first_name = atp_players[atp_players['player_id'] == int(lst1_aggregated[rank][0])]['first_name'].head(1).values[0]
            last_name = atp_players[atp_players['player_id'] == int(lst1_aggregated[rank][0])]['last_name'].head(1).values[0]

            first_name_weighted = atp_players[atp_players['player_id'] == int(lst2_aggregated[rank][0])]['first_name'].head(1).values[0]
            last_name_weighted = atp_players[atp_players['player_id'] == int(lst2_aggregated[rank][0])]['last_name'].head(1).values[0]

            writer.writerow([rank+1, first_name + ' ' + last_name, lst1_aggregated[rank][1], first_name_weighted + ' ' + last_name_weighted, lst2_aggregated[rank][1], 'aggregated'])

    csvFile.close()


def most_tournaments_played(year, top=10):
    matches = {
        '2018': atp_matches_2018,
        '2019': atp_matches_2018,
        '2020': atp_matches_2018
    }

    players = {
        '2018': players_2018_dictionary,
        '2019': players_2019_dictionary,
        '2020': players_2020_dictionary
    }

    result_dictionary = {}
    for player in players.get(year):
        tournaments_with_win = set(matches.get(year)[matches.get(year)['winner_id'] == int(player)]['tourney_id'])
        tournaments_with_lose = set(matches.get(year)[matches.get(year)['loser_id'] == int(player)]['tourney_id'])
        distinct_tournaments_num = tournaments_with_win.union(tournaments_with_lose)
        result_dictionary[player] = distinct_tournaments_num

    print('sorting dictionary ...')
    result_dictionary = sorted(result_dictionary.items(), key=lambda x: len(x[1]), reverse=True)
    return result_dictionary[0:top]


def create_output_question3(results_2018, results_2019, results_2020, top=10):

    with open(results_path("q3.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Rank", "Top players", "Tournaments played", "Year"])

        for rank in range(0, top):
            first_name = atp_players[atp_players['player_id'] == int(results_2018[rank][0])]['first_name'].head(1).values[0]
            last_name = atp_players[atp_players['player_id'] == int(results_2018[rank][0])]['last_name'].head(1).values[0]
            writer.writerow([rank+1, first_name + ' ' + last_name, len(results_2018[rank][1]), '2018'])

        writer.writerow([])
        for rank in range(0, top):
            first_name = atp_players[atp_players['player_id'] == int(results_2019[rank][0])]['first_name'].head(1).values[0]
            last_name = atp_players[atp_players['player_id'] == int(results_2019[rank][0])]['last_name'].head(1).values[0]
            writer.writerow([rank+1, first_name + ' ' + last_name, len(results_2019[rank][1]), '2019'])

        writer.writerow([])
        for rank in range(0, top):
            first_name = atp_players[atp_players['player_id'] == int(results_2020[rank][0])]['first_name'].head(1).values[0]
            last_name = atp_players[atp_players['player_id'] == int(results_2020[rank][0])]['last_name'].head(1).values[0]
            writer.writerow([rank+1, first_name + ' ' + last_name, len(results_2020[rank][1]), '2020'])
        writer.writerow([])

    csvFile.close()


def question3():
    results_2018 = most_tournaments_played('2018', 10)
    results_2019 = most_tournaments_played('2019', 10)
    results_2020 = most_tournaments_played('2020', 10)
    create_output_question3(results_2018, results_2019, results_2020, 10)



def question4_year_output(network, year, top):

    with open(results_path(f'q4-{year}.csv'), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        row = ["", "Player", "Top DC", "Player", "Top CC", "Player", "Top DC*CC"]
        writer.writerow(row)

        cc = sored_nodes_on_closeness_centrality(network)
        dc = sorted_nodes_on_degree_centrality(network)
        dc_cc = sorted_nodes_on_two_centralities(dc, cc)

        for i in range(0,top):
            row = [i,get_player_name(dc[i][0]), dc[i][1],get_player_name(cc[i][0]), cc[i][1],get_player_name(dc_cc[i][0]), dc_cc[i][1]]
            writer.writerow(row)
    csvFile.close()


def question4(matches_2018_graph,matches_2019_graph, matches_2020_graph, matches_year_aggregated_graph):
    question4_year_output(matches_2018_graph, '2018', 10)
    question4_year_output(matches_2019_graph, '2019', 10)
    question4_year_output(matches_2020_graph, '2020', 10)
    question4_year_output(matches_year_aggregated_graph, 'aggregated', 10)


def get_atp_rank(player_id):
    global current_player_ranking
    atp_rang = current_player_ranking[current_player_ranking['player_id'] == int(player_id)].sort_values('ranking_date', ascending=False)['rank'].head(1).values[0]
    print(atp_rang)
    return atp_rang


def question5(players_2018: nx.Graph, players_2019: nx.Graph,players_2020: nx.Graph, players_aggregated: nx.Graph, top = 10):
    lst1_2018 = sort_nodes_by_degree(players_2018)[0:top]
    lst2_2018 = sort_nodes_by_weighed_degree(players_2018)[0:top]

    lst1_2019 = sort_nodes_by_degree(players_2019)[0:top]
    lst2_2019 = sort_nodes_by_weighed_degree(players_2019)[0:top]

    lst1_2020 = sort_nodes_by_degree(players_2020)[0:top]
    lst2_2020 = sort_nodes_by_weighed_degree(players_2020)[0:top]

    lst1_aggregated = sort_nodes_by_degree(players_aggregated)[0:top]
    lst2_aggregated = sort_nodes_by_weighed_degree(players_aggregated)[0:top]

    with open(results_path("q5.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Rank", "Top players", "Degree", "Atp Rank", "Top players", "Weighed degree", "Atp Rank"])

        for rank in range(0, top):
            writer.writerow([rank+1, lst1_2018[rank][0], lst1_2018[rank][1], get_atp_rank(lst1_2018[rank][0]), lst2_2018[rank][0],
                             lst2_2018[rank][1], get_atp_rank(lst2_2018[rank][0])])

        writer.writerow([])
        for rank in range(0, top):
            writer.writerow([rank+1, lst1_2019[rank][0], lst1_2019[rank][1], get_atp_rank(lst1_2019[rank][0]), lst2_2019[rank][0],
                             lst2_2019[rank][1], get_atp_rank(lst2_2019[rank][0])])

        writer.writerow([])
        for rank in range(0, top):
            writer.writerow([rank+1, lst1_2020[rank][0], lst1_2020[rank][1], get_atp_rank(lst1_2020[rank][0]), lst2_2020[rank][0],
                             lst2_2020[rank][1], get_atp_rank(lst2_2020[rank][0])])

        writer.writerow([])
        for rank in range(0, top):
            writer.writerow([rank+1, lst1_aggregated[rank][0], lst1_aggregated[rank][1], get_atp_rank(lst1_aggregated[rank][0]),
                             lst2_aggregated[rank][0], lst2_aggregated[rank][1], get_atp_rank(lst2_aggregated[rank][0])])
    csvFile.close()


def compute_count_players_by_country(year):
    matches = {
        '2018': atp_matches_2018,
        '2019': atp_matches_2019,
        '2020': atp_matches_2020,
        'aggregated': pd.concat([atp_matches_2018, atp_matches_2019, atp_matches_2020])
    }

    full_winning_player_data = pd.merge(atp_players, matches.get(year),left_on='player_id',right_on='winner_id').drop_duplicates(['player_id'])
    full_winning_player_ids = full_winning_player_data['player_id']
    full_loser_player_data = pd.merge(atp_players,matches.get(year),left_on='player_id',right_on='loser_id').drop_duplicates(['player_id'])
    full_loser_player_ids = full_loser_player_data['player_id']
    all_player_ids = pd.concat([full_winning_player_ids, full_loser_player_ids], axis=0)
    distinct_players = all_player_ids.drop_duplicates(keep='first').to_frame()
    count_players_by_country = pd.merge(atp_players,distinct_players,left_on='player_id',right_on='player_id')
    return count_players_by_country.groupby('country_code')['player_id'].count().sort_values(ascending=False)


def question6():
    country_count_2018 = compute_count_players_by_country('2018')
    country_count_2019 = compute_count_players_by_country('2019')
    country_count_2020 = compute_count_players_by_country('2020')
    country_count_aggregated = compute_count_players_by_country('aggregated')
    print(country_count_2018.head(5))
    print(country_count_2019.head(5))
    print(country_count_2020.head(5))
    print(country_count_aggregated.head(5))


# this method retrieves countries of best atp players on current date(last date in dataset)
def question7():
    current_date = current_player_ranking['ranking_date'].max()
    full_player_data = pd.merge(current_player_ranking, atp_players,left_on='player_id',right_on='player_id')
    countries = full_player_data[full_player_data['ranking_date'] == current_date].sort_values('rank')['country_code'].head(10)
    print(countries)


def generate_communities(actor_network: nx.Graph):
    communities_generator = community.girvan_newman(actor_network)
    top_level_communities = next(communities_generator)
    next_level_communities = next(communities_generator)
    answer = sorted(map(sorted, next_level_communities))
    return answer


def random_color():
    #dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    cls = [name for hsv, name in by_hsv]
    return cls[np.random.choice(range(len(cls)))]


def question_9_output(answer, player_network, year):
    with open(results_path(f"q9-{year}.csv"), 'w', newline='') as csvFile:
        #writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        for index in range(-1, len(answer[0])+1):
            if index == -1:
                csvFile.write("")
                for i in range(0, len(answer)):
                    csvFile.write(", Commune "+str(i))
            else:
                for commune in answer:
                    csvFile.write(", ")
                    if index < len(commune):
                        csvFile.write(commune[index])
            csvFile.write("\r\n")

    csvFile.close()
    colors = list()
    community_colors = list()
    for comm in answer:
        community_colors.append(random_color())

    for node in player_network.nodes():
        for index in range(0, len(answer)):
            if answer[index].__contains__(node):
                break
        colors.append(community_colors[index])

    #save_actor_graph_as_pdf(actor_network, color=colors, fileName="q4.pdf")

    pdf = matplotlib.backends.backend_pdf.PdfPages(results_path(f"q9-{year}.pdf"))
    number_of_nodes: int = len(player_network.nodes())
    n: int = 4
    pos = nx.spring_layout(player_network, k=(1 / math.sqrt(number_of_nodes)) * n)
    pl.figure(figsize=(20, 20))  # Don't create a humongous figure
    nx.draw_networkx(player_network, pos, node_size=30, font_size='xx-small', with_labels=False, node_color=colors)
    pl.axis('off')
    pdf.savefig(pl.gcf(), dpi=900)

    fig = pl.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    pl.title('Distribution of players across clusters')

    cluter_counter = Counter(colors)

    frequencies = cluter_counter.values()
    names = list(cluter_counter.keys())

    x_coordinates = np.arange(len(cluter_counter))
    ax.bar(x_coordinates, frequencies, align='center', color=names)

    ax.xaxis.set_major_locator(pl.FixedLocator(x_coordinates))
    ax.xaxis.set_major_formatter(pl.FixedFormatter(names))
    pdf.savefig(fig, dpi=900)

    pdf.close()


def question9(player_network_2018: nx.Graph, player_network_2019: nx.Graph, player_network_2020: nx.Graph, player_network_aggregated: nx.Graph):
    check()
    answer_2018 = generate_communities(player_network_2018)
    answer_2019 = generate_communities(player_network_2019)
    answer_2020 = generate_communities(player_network_2020)
    answer_aggregated = generate_communities(player_network_aggregated)
    question_9_output(answer_2018, player_network_2018, '2018')
    question_9_output(answer_2019, player_network_2019, '2019')
    question_9_output(answer_2020, player_network_2020, '2020')
    question_9_output(answer_aggregated, player_network_aggregated, 'aggregated')


def clustering_analyse(player_network: nx.Graph):
    player_id, clustering_coef = zip(*nx.clustering(player_network, weight = "weight").items())
    non_zero = [(id_ig, cc)  for id_ig, cc in zip(player_id, clustering_coef) if cc > 0]

    df = pd.DataFrame(non_zero, columns = ["id", "cc"])
    df.sort_values('cc', inplace = True)

    max_local_clustering_degree = max(clustering_coef)
    average_clustering_degree = nx.average_clustering(player_network)

    global_clustering_coef = nx.transitivity(player_network)
    communities = generate_communities(player_network)
    modularity_undirected = qual.modularity(player_network, communities)

    print(f"Max local cc: {max_local_clustering_degree}")
    print(f"Average cc: {average_clustering_degree}")
    print(f"Global cc: {global_clustering_coef}")
    print(f"clusters modularity: {modularity_undirected}")
    print("Local non-zero cc:")
    print(df)


def question10(player_network: nx.Graph):
    clustering_analyse(player_network)
    print(nx.attribute_assortativity_coefficient(player_network, "rank"))
    print(nx.attribute_assortativity_coefficient(player_network, "country"))
    print(nx.degree_assortativity_coefficient(player_network))


def compare_weighted_and_regular_graph(player_network: nx.Graph):
    regular = list(player_network.degree(player_network.nodes()))
    weighted = list(player_network.degree(player_network.nodes(), 'weight'))
    sum = 0
    results = []
    player_ids = []
    for i in range(0, len(regular)):
        results.append(weighted[i][1] / regular[i][1])
        sum += weighted[i][1] / regular[i][1]
        player_ids.append(regular[i][0])


    return sum / len(player_network.nodes()),results,player_ids


def tendency_to_play_with_same_players(player_network, year):
    average, results, players = compare_weighted_and_regular_graph(player_network)
    print(average)
    players_rank = []
    test = int()
    for player in players:
        if current_player_ranking[current_player_ranking['player_id'] == int(player)].sort_values('ranking_date', ascending=False)['rank'].empty:
            # not active players on last noticed date(don't have rank)
            rank = -10
            test += 1
            players_rank.append(rank)
        else:
            rank = current_player_ranking[current_player_ranking['player_id'] == int(player)].sort_values('ranking_date', ascending=False)['rank'].head(1).values[0]
            players_rank.append(rank)

    print(test)
    pl.plot(results, players_rank, 'ro')
    #pl.axis([0, 20, 0, 5000])
    pl.show()
    pl.savefig(f'results/q11-{year}.pdf')


def question11(player_network_2018: nx.Graph, player_network_2019: nx.Graph, player_network_2020: nx.Graph,player_network_aggregated: nx.Graph):
    tendency_to_play_with_same_players(player_network_2018, '2018')
    tendency_to_play_with_same_players(player_network_2019, '2019')
    tendency_to_play_with_same_players(player_network_2020, '2020')
    tendency_to_play_with_same_players(player_network_aggregated, 'aggregated')


def found_center_of_network(player_network: nx.Graph, year, top: int=10):
    check()
    with open(results_path(f"q12-{year}.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        row = ["Players in the center of the network"]
        writer.writerow(row)

        subgraph = player_network.subgraph(generate_communities(player_network)[0])
        #periphery = nx.periphery(subgraph)
        answer = nx.closeness_centrality(subgraph)
        answer = sorted(answer.items(), key= lambda x: x[1], reverse=True)[:top]
        writer.writerows(answer)

        '''
        for item in generate_communities(actor_network)[0]:
            if actor_network.degree(item) > degree_treshold:
                row = [item]
                writer.writerow(row)
        '''
    csvFile.close()


def question12(player_network_2018: nx.Graph, player_network_2019: nx.Graph, player_network_2020: nx.Graph,player_network_aggregated: nx.Graph):
    found_center_of_network(player_network_2018, '2018')
    found_center_of_network(player_network_2019, '2019')
    found_center_of_network(player_network_2020, '2020')
    found_center_of_network(player_network_aggregated, 'aggregated')


def question13(player_network: nx.Graph, top: int = 10):
    check()

    with open(results_path("q13.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        row = ["", "Player", "Top BC", "Player", "Top DC", "Player", "Top DC*BC"]
        writer.writerow(row)

        bc = sored_nodes_on_betweenness_centrality(player_network)
        dc = sorted_nodes_on_degree_centrality(player_network)
        bc_dc = sorted_nodes_on_two_centralities(bc, dc)

        for i in range(0,top):
            row = [i,bc[i][0], bc[i][1],dc[i][0], dc[i][1],bc_dc[i][0], bc_dc[i][1]]
            writer.writerow(row)

    csvFile.close()


def question14(player_network_2018: nx.Graph, player_network_2019: nx.Graph, player_network_2020: nx.Graph, player_network_aggregated: nx.Graph):
    check()

    with open(results_path("q14.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        row = ["Player18 density", "Player19 Density", "Player20 density", "PlayerAggregated density"]
        writer.writerow(row)
        row = [nx.density(player_network_2018),nx.density(player_network_2019),nx.density(player_network_2020), nx.density(player_network_aggregated)]
        writer.writerow(row)

    csvFile.close()


def my_sum(lst: dict):
    return sum(x for x in lst.values())


def my_avg(lst: dict):
    return my_sum(lst)/len(lst)


def question15(player_network_2018: nx.Graph, player_network_2019: nx.Graph, player_network_2020: nx.Graph, player_network_aggregated: nx.Graph):
    check()

    with open(results_path("q15.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        row = ["", "Player18", "Player19", "Player20", "Player Aggregated"]
        writer.writerow(row)

        n1 = my_avg(nx.closeness_centrality(player_network_2018))
        n2 = my_avg(nx.closeness_centrality(player_network_2019))
        n3 = my_avg(nx.closeness_centrality(player_network_2020))
        n4 = my_avg(nx.closeness_centrality(player_network_aggregated))

        row = ["Closeness centrality", n1, n2, n3, n4]
        writer.writerow(row)

        n1 = my_avg(nx.betweenness_centrality(player_network_2018))
        n2 = my_avg(nx.betweenness_centrality(player_network_2019))
        n3 = my_avg(nx.betweenness_centrality(player_network_2020))
        n4 = my_avg(nx.closeness_centrality(player_network_aggregated))

        row = ["Betweenness centrality", n1, n2, n3, n4]
        writer.writerow(row)

        n1 = my_avg(nx.degree_centrality(player_network_2018))
        n2 = my_avg(nx.degree_centrality(player_network_2019))
        n3 = my_avg(nx.degree_centrality(player_network_2020))
        n4 = my_avg(nx.closeness_centrality(player_network_aggregated))

        row = ["Degree centrality", n1, n2, n3, n4]
        writer.writerow(row)
    csvFile.close()


def my_average_shortest_path(graph: nx.Graph):
    cnt: int = 0
    length: int = 0

    for node1 in graph.nodes():
        for node2 in graph.nodes():
            if node1 != node2 and nx.has_path(graph, node1, node2):
                length += len(nx.shortest_path(graph, node1, node2))-1
                cnt += 1

    return length/cnt


def question16(player_network_2018: nx.Graph, player_network_2019: nx.Graph, player_network_2020: nx.Graph, player_network_aggregated: nx.Graph):
    check()

    with open(results_path("q16.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        row = ["Player18 average distance", "Player19 average distance", "Player20 average distance", "Player Aggregated average distance"]
        writer.writerow(row)

        try:
            n1 = my_average_shortest_path(player_network_2018)
        except nx.exception.NetworkXError:
            n1 = 'graph is not connected'

        try:
            n2 = my_average_shortest_path(player_network_2019)
        except nx.exception.NetworkXError:
            n2 = 'graph is not connected'

        try:
            n3 = my_average_shortest_path(player_network_2020)
        except nx.exception.NetworkXError:
            n3 = 'graph is not connected'

        try:
            n4 = my_average_shortest_path(player_network_2020)
        except nx.exception.NetworkXError:
            n4 = 'graph is not connected'

        row = [n1,n2,n3, n4]
        writer.writerow(row)

        row = ["","",""]
        writer.writerow(row)

        row = ["Player18 diameter", "Player19 diameter", "Player20 diameter", "Player Aggregated diameter"]
        writer.writerow(row)
        try:
            n1 = nx.diameter(player_network_2018)
        except nx.exception.NetworkXError:
            n1 = 'graph is not connected'

        try:
            n2 = nx.diameter(player_network_2019)
        except nx.exception.NetworkXError:
            n2 = 'graph is not connected'

        try:
            n3 = nx.diameter(player_network_2020)
        except nx.exception.NetworkXError:
            n3 = 'graph is not connected'

        try:
            n4 = nx.diameter(player_network_aggregated)
        except nx.exception.NetworkXError:
            n4 = 'graph is not connected'

        row = [n1,n2,n3,n4]
        writer.writerow(row)

    csvFile.close()


def compute_correlation_rank_and_degree(player_network):
    degrees = pd.DataFrame([val for (node, val) in player_network.degree()])
    players = [node for (node, val) in player_network.degree()]
    ranks = []
    for player in players:
        if current_player_ranking[current_player_ranking['player_id'] == int(player)].sort_values('ranking_date', ascending=False)['rank'].empty:
            # not active players on last noticed date(don't have rank)
            rank = -10
            ranks.append(rank)
        else:
            rank = current_player_ranking[current_player_ranking['player_id'] == int(player)].sort_values('ranking_date', ascending=False)['rank'].head(1).values[0]
            ranks.append(rank)

    frame_ranks = pd.DataFrame(ranks)
    x, y = stats.kendalltau(degrees,frame_ranks)
    print(x) #  correlation
    return x


def node_degrees_distribution(player_network, year):
    check()

    pdf = matplotlib.backends.backend_pdf.PdfPages(results_path(f"q17-{year}.pdf"))
    fig = pl.figure()
    pl.title(f'Player network - distribution of node degrees, year: {year}')
    pl.hist([val for (node, val) in player_network.degree()], bins=50)
    pdf.savefig(fig, dpi=900)
    pdf.close()

    print(compute_correlation_rank_and_degree(player_network))


def question17(player_network_2018: nx.Graph, player_network_2019: nx.Graph, player_network_2020: nx.Graph, player_network_aggregated: nx.Graph):
    node_degrees_distribution(player_network_2018, '2018')
    node_degrees_distribution(player_network_2019, '2019')
    node_degrees_distribution(player_network_2020, '2020')
    node_degrees_distribution(player_network_aggregated, 'aggregated')


def question18(player_network_2018: nx.Graph, player_network_2019: nx.Graph, player_network_2020: nx.Graph, player_network_aggregated: nx.Graph):
    # hubs and authority scores for each node
    # not sure how much iterations i need to set here
    scores_2018 = nx.hits(player_network_2018, 100)[0]
    scores_2019 = nx.hits(player_network_2019, 100)[0]
    scores_2020 = nx.hits(player_network_2020, 100)[0]
    scores_aggregated = nx.hits(player_network_aggregated, 100)[0]

    scores_2018 = {k: v for k, v in sorted(scores_2018.items(), key=lambda item: item[1], reverse=True)}
    scores_2018 = {k: v for k, v in sorted(scores_2018.items(), key=lambda item: item[1], reverse=True)}
    scores_2018 = {k: v for k, v in sorted(scores_2018.items(), key=lambda item: item[1], reverse=True)}
    scores_aggregated = {k: v for k, v in sorted(scores_aggregated.items(), key=lambda item: item[1], reverse=True)}

    N = 3
    print(list(scores_2018.items())[:N])
    print(list(scores_2019.items())[:N])
    print(list(scores_2020.items())[:N])
    print(list(scores_aggregated.items())[:N])
    print('hubs founded')



def number_of_triangles(graph: nx.Graph):
    return sum(x for x in nx.triangles(graph).values())/3


def number_of_paths_len_2(graph: nx.Graph):
    cnt: int = 0
    for node1 in graph.nodes():
        for node2 in graph.nodes():
            if node1 != node2:
                for path in nx.all_simple_paths(graph, source=node1, target=node2, cutoff=2):
                    if len(path) == 2:
                        cnt+=1
    return cnt


def c_del(graph: nx.Graph):
    return (3*number_of_triangles(graph))/number_of_paths_len_2(graph)


def gam(graph: nx.Graph, rand: nx.Graph):
    return c_del(graph)/c_del(rand)


def lam(graph: nx.Graph, rand: nx.Graph):
    return my_average_shortest_path(graph)/ my_average_shortest_path(rand)


def s(graph: nx.Graph):
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    p = 2*m/(n*(n-1))
    rand = nx.erdos_renyi_graph(n,p)

    return gam(graph, rand)/lam(graph,rand)


def is_small_world(graph: nx.Graph):
    return s(graph) > 1


def question19(player_network_2018: nx.Graph, player_network_2019: nx.Graph, player_network_2020: nx.Graph, player_network_aggregated: nx.Graph):
    check()

    with open(results_path("q19.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        row = ["Network", "Is small world?"]
        writer.writerow(row)

        row = ["Player18", is_small_world(player_network_2018)]
        writer.writerow(row)

        row = ["Player19", is_small_world(player_network_2019)]
        writer.writerow(row)

        row = ["Player20", is_small_world(player_network_2020)]
        writer.writerow(row)

        row = ["Player aggregated", is_small_world(player_network_aggregated)]
        writer.writerow(row)

    csvFile.close()


# iscrtati ego mreze za nadala, djokovica i federera -> q20
# proveri gde se ti cvorovi nalaze u mrezi, kako su rasporedjeni u odnosu na ostale, koliko veza imaju i slicno -> q21
# videti da iscrtas mrezu celu(3 puta) posebno za novaka, nadala i federera i obojis te cvorove posebnom bojom na dijagramu -> q22
# spojiti sve tri mreze i izvrsiti obradu iz pitanja


def draw_ego_network(player_network: nx.Graph, player_name):
    player_id = {
        'Novak Djokovic': '104925',
        'Rafael Nadal': '104745',
        'Roger Federer': '103819',
    }
    ego_network = nx.ego_graph(player_network, player_id.get(player_name))
    print(f"Nodes ego network player {player_name} are:  {ego_network.nodes}")

    # find node with largest degree
    node_and_degree = ego_network.degree()
    (largest_hub, degree) = sorted(node_and_degree, key=itemgetter(1))[-1]

    # Create ego graph of main hub
    hub_ego = nx.ego_graph(ego_network, largest_hub)

    # Draw graph
    pl.figure(figsize=(20, 20))  # Don't create a humongous figure
    pos = nx.spring_layout(hub_ego)
    nx.draw(hub_ego, pos, node_color="b", node_size=50, with_labels=False)

    # Draw ego as large and red
    options = {"node_size": 300, "node_color": "r"}
    nx.draw_networkx_nodes(hub_ego, pos, nodelist=[largest_hub], **options)
    #pl.show()
    pl.savefig(f"results/q20_{player_name}.pdf", format='pdf', dpi=900)


def question20(player_network: nx.Graph):
    draw_ego_network(player_network, 'Novak Djokovic')
    draw_ego_network(player_network, 'Rafael Nadal')
    draw_ego_network(player_network, 'Roger Federer')


def draw_ego_network_position_in_full_network(player_network, player_name):
    # player_name on which ego network is built
    player_id = {
        'Novak Djokovic': '104925',
        'Rafael Nadal': '104745',
        'Roger Federer': '103819',
    }

    ego_network = nx.ego_graph(player_network, player_id.get(player_name))
    print(f"Nodes ego network player {player_name} are:  {ego_network.nodes}")

    all_players = list(player_network.nodes)
    ego_network_players = list(ego_network.nodes)
    node_colors = []
    for player in all_players:
        if player in ego_network_players:
            node_colors.append(0.7)
        else:
            node_colors.append(0.1)

    pl.figure(figsize=(10, 10))  # Don't create a humongous figure
    nx.draw(player_network, node_color=node_colors, with_labels=False, font_color='white', node_size=5)
    pl.savefig(f"results/q22_{player_name}.pdf", format='pdf', dpi=900)


def question22(player_network: nx.Graph):
    draw_ego_network_position_in_full_network(player_network, 'Novak Djokovic')
    draw_ego_network_position_in_full_network(player_network, 'Rafael Nadal')
    draw_ego_network_position_in_full_network(player_network, 'Roger Federer')


def question23(player_network: nx.Graph):
    djokovic_ego_network = nx.ego_graph(player_network, '104925')
    nadal_ego_network = nx.ego_graph(player_network, '104745')
    federer_ego_network = nx.ego_graph(player_network, '103819')

    combined1_network = nx.compose(djokovic_ego_network,nadal_ego_network)
    full_combined = nx.compose(combined1_network, federer_ego_network)
    print(full_combined)
    # do clustering of this network to 3 clusters


def matches_and_players_distribution(player_network: nx.Graph, year):
    check()
    ret = list(player_network.degree(player_network.nodes(), 'weight'))
    # here column shouldn't be called 'number_of_players', it should be 'player_id',
    # but bacause rename is not working we don't want 'player_id' on plot
    df = DataFrame(ret, columns=['number_of_players', 'game_played'])
    df.groupby('game_played')['number_of_players'].count().to_frame().plot(kind='bar', color='r',
      figsize=(20,10), title=f'Distribution of games played by player numbers - {year}')

    #pdf = matplotlib.backends.backend_pdf.PdfPages(results_path("q24.pdf"))

    #pl.title('Player network - distribution of node degrees')
    #pl.plot(games_played(0), games_played(1), 'ro')
    #pl.axis([0, 20, 0, 5000])
    #pdf.savefig(games_played, dpi=900)
    #pdf.close()


def tournaments_and_surface_distribution(year):
    check()
    # put here 2019 and 2020
    matches = {
        '2018': atp_matches_2018,
        '2019': atp_matches_2018,
        '2020': atp_matches_2018
    }
    distinct_tournaments = matches.get(year).drop_duplicates( subset='tourney_id', keep='first', inplace=False)
    df = pd.DataFrame(distinct_tournaments.groupby('surface')['tourney_id'].count())
    df.columns = ['number_of_tournaments']
    df.plot(kind='bar', color='r', figsize=(20,10), title=f'Distribution of tournaments by surface - {year}')
    print(df)


def tournaments_per_year_distribution(year):
    check()
    # put here 2019 and 2020
    matches = {
        '2018': atp_matches_2018,
        '2019': atp_matches_2018,
        '2020': atp_matches_2018
    }
    distinct_tournaments = matches.get(year).drop_duplicates( subset='tourney_id', keep='first', inplace=False)
    return distinct_tournaments.shape


def matches_per_year_distribution(year):
    check()
    # put here 2019 and 2020
    matches = {
        '2018': atp_matches_2018,
        '2019': atp_matches_2018,
        '2020': atp_matches_2018
    }
    return matches.get(year).shape


def question24(player_network: nx.Graph):
    matches_and_players_distribution(player_network, '2018')
    matches_and_players_distribution(player_network, '2019')
    matches_and_players_distribution(player_network, '2020')


def question25():
    tournaments_and_surface_distribution('2018')
    tournaments_and_surface_distribution('2019')
    tournaments_and_surface_distribution('2020')

    tournaments_2018, _ = tournaments_per_year_distribution('2018')
    tournaments_2019, _ = tournaments_per_year_distribution('2019')
    tournaments_2020, _ = tournaments_per_year_distribution('2020')
    #pl.figure()
    pl.plot(['2018', '2019', '2020'], [tournaments_2018, tournaments_2019, tournaments_2020], 'ro')


def matches_number_and_surface_distribution(year):
    check()
    # put here 2019 and 2020
    matches = {
        '2018': atp_matches_2018,
        '2019': atp_matches_2018,
        '2020': atp_matches_2018
    }
    distinct_matches = matches.get(year)
    df = pd.DataFrame(distinct_matches['surface'].value_counts())
    df.columns = ['number_of_games']
    df.plot(kind='bar', color='r', figsize=(20,10), title=f'Distribution of matches number by surface - {year}', )
    print(df)


def question26():
    matches_number_and_surface_distribution('2018')
    matches_number_and_surface_distribution('2019')
    matches_number_and_surface_distribution('2020')

    matches_count_2018, _ = matches_per_year_distribution('2018')
    matches_count_2019, _ = matches_per_year_distribution('2019')
    matches_count_2020, _ = matches_per_year_distribution('2020')
    pl.figure(figsize=(20,10))
    pl.plot(['2018', '2019', '2020'], [matches_count_2018, matches_count_2019, matches_count_2020], 'ro')


def main():
    print("Starting script...")
    extract_secondary_dataset()
    read_atp_matches_2018_dataset()
    read_atp_matches_2019_dataset()
    read_atp_matches_2020_dataset()

    matches_2018_graph = create_atp_matches_2018_network()
    matches_2019_graph = create_atp_matches_2019_network()
    matches_2020_graph = create_atp_matches_2020_network()
    matches_year_aggregated_graph = create_aggregated_network(matches_2018_graph, matches_2019_graph, matches_2020_graph)

    # save_actor_graph_as_pdf(matches_2018_graph, 'r', 'player_matches_2018_graph.pdf')
    # save_actor_graph_as_pdf(matches_2019_graph, 'r', 'player_matches_2019_graph.pdf')

    #question1(matches_2018_graph,matches_2019_graph, matches_2020_graph, matches_year_aggregated_graph)
    #question2(matches_2018_graph,matches_2019_graph, matches_2020_graph, matches_year_aggregated_graph)
    #question3()
    #question4(matches_2018_graph,matches_2019_graph, matches_2020_graph, matches_year_aggregated_graph)
    #question5(matches_2018_graph,matches_2019_graph, matches_2020_graph, matches_year_aggregated_graph)
    #question6()
    #question7()
    #question9(matches_2018_graph,matches_2019_graph, matches_2020_graph, matches_year_aggregated_graph)

    #question10(matches_2018_graph)
    #question11(matches_2018_graph,matches_2019_graph, matches_2020_graph, matches_year_aggregated_graph)
    #question12(matches_2018_graph,matches_2019_graph, matches_2020_graph, matches_year_aggregated_graph)
    #question13(matches_2018_graph)
    #question14(matches_2018_graph,matches_2019_graph, matches_2020_graph, matches_year_aggregated_graph)
    #question15(matches_2018_graph,matches_2019_graph, matches_2020_graph, matches_year_aggregated_graph)
    #question16(matches_2018_graph,matches_2019_graph, matches_2020_graph, matches_year_aggregated_graph)
    #question17(matches_2018_graph,matches_2019_graph, matches_2020_graph, matches_year_aggregated_graph)
    #question18(matches_2018_graph,matches_2019_graph, matches_2020_graph, matches_year_aggregated_graph)
    #question19(matches_2018_graph,matches_2019_graph, matches_2020_graph, matches_year_aggregated_graph)
    #question20(matches_2018_graph)
    #question22(matches_2018_graph)
    #question23(matches_2018_graph)
    #question24(matches_2018_graph)
    #question25()
    #question26()


if __name__ == "__main__":
    main()
