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


def sored_nodes_on_betweenness_centrality(graph: nx.Graph):
    ret = []
    bc = nx.betweenness_centrality(graph)
    for node in graph.nodes():
        ret.append([node,  bc[node]])

    ret.sort(key=lambda x: x[1], reverse=True)
    return ret


def sorted_nodes_on_degree_centrality(graph: nx.Graph):
    ret = list()
    dc = nx.degree_centrality(graph)
    for node in graph.nodes():
        ret.append([node, dc[node]])

    ret.sort(key=lambda x: x[1], reverse=True)
    return ret


def sorted_nodes_on_bc_dc(bc: list, dc: list):
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


def compute_count_players_by_country():
    full_winning_player_data = pd.merge(atp_players, atp_matches_2018,left_on='player_id',right_on='winner_id').drop_duplicates(['player_id'])
    full_winning_player_ids = full_winning_player_data['player_id']
    full_loser_player_data = pd.merge(atp_players,atp_matches_2018,left_on='player_id',right_on='loser_id').drop_duplicates(['player_id'])
    full_loser_player_ids = full_loser_player_data['player_id']
    all_player_ids = pd.concat([full_winning_player_ids, full_loser_player_ids], axis=0)
    distinct_players = all_player_ids.drop_duplicates(keep='first').to_frame()
    count_players_by_country = pd.merge(atp_players,distinct_players,left_on='player_id',right_on='player_id')
    return count_players_by_country.groupby('country_code')['player_id'].count().sort_values(ascending=False)


def question6():
    country_count = compute_count_players_by_country()
    print(country_count.head(10))


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


def question9(player_network: nx.Graph):
    check()
    answer = generate_communities(player_network)

    with open(results_path("q9.csv"), 'w', newline='') as csvFile:
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

    pdf = matplotlib.backends.backend_pdf.PdfPages(results_path("q9.pdf"))
    number_of_nodes: int = len(player_network.nodes())
    n: int = 4
    pos = nx.spring_layout(player_network, k=(1 / math.sqrt(number_of_nodes)) * n)
    pl.figure(figsize=(20, 20))  # Don't create a humongous figure
    nx.draw_networkx(player_network, pos, node_size=30, font_size='xx-small', with_labels=False, node_color=colors)
    pl.axis('off')
    pdf.savefig(pl.gcf(), dpi=900)

    fig = pl.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    pl.title('Distribution of actors across clusters')

    cluter_counter = Counter(colors)

    frequencies = cluter_counter.values()
    names = list(cluter_counter.keys())

    x_coordinates = np.arange(len(cluter_counter))
    ax.bar(x_coordinates, frequencies, align='center', color=names)

    ax.xaxis.set_major_locator(pl.FixedLocator(x_coordinates))
    ax.xaxis.set_major_formatter(pl.FixedFormatter(names))
    pdf.savefig(fig, dpi=900)

    pdf.close()


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


def question11(player_network: nx.Graph):
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

    # compare weighted and non-weighted graph
    #write with matplotlib 2D diagram based on that number and atp rank
    #compare results there


def question12(player_network: nx.Graph, top: int=10):
    check()
    with open(results_path("q12.csv"), 'w', newline='') as csvFile:
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


def question13(player_network: nx.Graph, top: int = 10):
    check()

    with open(results_path("q13.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        row = ["", "Player", "Top BC", "Player", "Top DC", "Player", "Top DC*BC"]
        writer.writerow(row)

        bc = sored_nodes_on_betweenness_centrality(player_network)
        dc = sorted_nodes_on_degree_centrality(player_network)
        bc_dc = sorted_nodes_on_bc_dc(bc, dc)

        for i in range(0,top):
            row = [i,bc[i][0], bc[i][1],dc[i][0], dc[i][1],bc_dc[i][0], bc_dc[i][1]]
            writer.writerow(row)

    csvFile.close()


def question14(player_network: nx.Graph):
    check()

    with open(results_path("q14.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        row = ["Player18 density", "Player19 Density", "Player20 density"]
        writer.writerow(row)
        row = [nx.density(player_network),nx.density(player_network),nx.density(player_network)]
        writer.writerow(row)

    csvFile.close()


def my_sum(lst: dict):
    return sum(x for x in lst.values())


def my_avg(lst: dict):
    return my_sum(lst)/len(lst)


def question15(player_network: nx.Graph):
    check()

    with open(results_path("q15.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        row = ["", "Player18", "Player19", "Player20"]
        writer.writerow(row)

        n1 = my_avg(nx.closeness_centrality(player_network))
        n2 = my_avg(nx.closeness_centrality(player_network))
        n3 = my_avg(nx.closeness_centrality(player_network))

        row = ["Closeness centrality", n1, n2, n3]
        writer.writerow(row)

        n1 = my_avg(nx.betweenness_centrality(player_network))
        n2 = my_avg(nx.betweenness_centrality(player_network))
        n3 = my_avg(nx.betweenness_centrality(player_network))

        row = ["Betweenness centrality", n1, n2, n3]
        writer.writerow(row)

        n1 = my_avg(nx.degree_centrality(player_network))
        n2 = my_avg(nx.degree_centrality(player_network))
        n3 = my_avg(nx.degree_centrality(player_network))

        row = ["Normalized degree centrality", n1, n2, n3]
        writer.writerow(row)

        # n1 = my_avg(nx.eigenvector_centrality_numpy(player_network))
        # n2 = my_avg(nx.eigenvector_centrality_numpy(player_network))
        # n3 = my_avg(nx.eigenvector_centrality_numpy(player_network))
        #
        # row = ["Eigenvector centrality", n1, n2, n3]
        # writer.writerow(row)
        #
        # n1 = my_avg(nx.katz_centrality_numpy(player_network))
        # n2 = my_avg(nx.katz_centrality_numpy(player_network))
        # n3 = my_avg(nx.katz_centrality_numpy(player_network))
        #
        # row = ["Katz centrality", n1, n2, n3]
        # writer.writerow(row)

        n1 = my_avg(nx.edge_betweenness_centrality(player_network))
        n2 = my_avg(nx.edge_betweenness_centrality(player_network))
        n3 = my_avg(nx.edge_betweenness_centrality(player_network))

        row = ["Edge betweenness centrality", n1, n2, n3]
        writer.writerow(row)

        # Error
        '''
        print(nx.percolation_centrality(actor_network))
        n1 = my_avg(nx.percolation_centrality(actor_network))
        n2 = my_avg(nx.percolation_centrality(genre_network))
        n3 = my_avg(nx.percolation_centrality(movie_network))
        
        row = ["Percolation centrality", n1, n2, n3]
        writer.writerow(row)
        '''

        n1 = my_avg(nx.pagerank(player_network))
        n2 = my_avg(nx.pagerank(player_network))
        n3 = my_avg(nx.pagerank(player_network))

        row = ["PageRank centrality", n1, n2, n3]
        writer.writerow(row)

        n1 = nx.global_reaching_centrality(player_network)
        n2 = nx.global_reaching_centrality(player_network)
        n3 = nx.global_reaching_centrality(player_network)

        row = ["Global reaching centrality", n1, n2, n3]
        writer.writerow(row)

        row = []
        writer.writerow(row)

        n1 = nx.node_connectivity(player_network)
        n2 = nx.node_connectivity(player_network)
        n3 = nx.node_connectivity(player_network)

        row = ["Node connectivity", n1, n2, n3]
        writer.writerow(row)

        n1 = nx.edge_connectivity(player_network)
        n2 = nx.edge_connectivity(player_network)
        n3 = nx.edge_connectivity(player_network)

        row = ["Edge connectivity", n1, n2, n3]
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


def question16(player_network: nx.Graph):
    check()

    with open(results_path("q16.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        row = ["Player18 average distance", "Player19 average distance", "Player20 average distance"]
        writer.writerow(row)

        try:
            n1 = my_average_shortest_path(player_network)
        except nx.exception.NetworkXError:
            n1 = 'graph is not connected'

        try:
            n2 = my_average_shortest_path(player_network)
        except nx.exception.NetworkXError:
            n2 = 'graph is not connected'

        try:
            n3 = my_average_shortest_path(player_network)
        except nx.exception.NetworkXError:
            n3 = 'graph is not connected'

        row = [n1,n2,n3]
        writer.writerow(row)

        row = ["","",""]
        writer.writerow(row)

        row = ["Player18 diameter", "Player19 diameter", "Player20 diameter"]
        writer.writerow(row)
        try:
            n1 = nx.diameter(player_network)
        except nx.exception.NetworkXError:
            n1 = 'graph is not connected'

        try:
            n2 = nx.diameter(player_network)
        except nx.exception.NetworkXError:
            n2 = 'graph is not connected'

        try:
            n3 = nx.diameter(player_network)
        except nx.exception.NetworkXError:
            n3 = 'graph is not connected'

        row = [n1,n2,n3]
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


def question17(player_network: nx.Graph):
    check()

    pdf = matplotlib.backends.backend_pdf.PdfPages(results_path("q17.pdf"))
    fig = pl.figure()
    pl.title('Player network - distribution of node degrees')
    pl.hist([val for (node, val) in player_network.degree()], bins=50)
    pdf.savefig(fig, dpi=900)
    pdf.close()

    print(compute_correlation_rank_and_degree(player_network))


def question18(player_network: nx.Graph):
    # hubs and authority scores for each node
    scores = nx.hits(player_network, 5)
    return scores


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


def question19(player_network: nx.Graph):
    check()

    with open(results_path("q12.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        row = ["Network", "Is small world?"]
        writer.writerow(row)

        row = ["Player18", is_small_world(player_network)]
        writer.writerow(row)

        row = ["Player19", is_small_world(player_network)]
        writer.writerow(row)

        row = ["Player20", is_small_world(player_network)]
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
    question2(matches_2019_graph)

    question5(matches_2018_graph)
    question6()
    question7()
    #question9(matches_2018_graph)

    #question10(matches_2018_graph)
    #question11(matches_2018_graph)
    #question12(matches_2018_graph)
    #question13(matches_2018_graph)
    #question14(matches_2018_graph)
    #question15(matches_2018_graph)
    #question16(matches_2018_graph)
    #question17(matches_2018_graph)
    #question20(matches_2018_graph)
    question22(matches_2018_graph)

if __name__ == "__main__":
    main()
