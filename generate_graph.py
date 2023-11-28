import csv
import pandas as pd
import rdflib
from rdflib import Graph, Namespace, Literal, URIRef
import networkx as nx
# from lm import Embedder

def removeSpaces(string:str)->str:
    return string.strip().replace(" ", "_")

def graph_to_ttl(csv_file:str='IMDB-Movie-Data.csv',filename:str='imdb_kg.ttl')->rdflib.graph:
    # Create an RDF graph
    g = Graph()

    # Define namespaces
    movie_ns = Namespace("movie:")
    actor_ns = Namespace("actor:")
    director_ns = Namespace("director:")
    year_ns = Namespace("year:")
    genre_ns = Namespace("genre:")
    relation_ns = Namespace("relation:")
    
    g.bind("movie", movie_ns)
    g.bind("actor", actor_ns)
    g.bind("director", director_ns)
    g.bind("year", year_ns)
    g.bind("genre", genre_ns)
    g.bind("relation", relation_ns)


    # Load IMDb data from CSV file (modify the filename accordingly)
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        # Iterate through the CSV data and add triples to the graph
        for row in reader:
            movie_uri = URIRef(movie_ns[removeSpaces(row["Title"])])
            for actor in row['Actors'].split(','):
                actor_uri = URIRef(actor_ns[f'{removeSpaces(actor)}'])
                g.add((movie_uri, relation_ns.hasActor, actor_uri))
            
            for genre in row['Genre'].split(','):
                genre_uri = URIRef(genre_ns[f'{genre}'])
                g.add((movie_uri, relation_ns.hasGenre, genre_uri))
            director_uri = URIRef(director_ns[f'{removeSpaces(row["Director"])}'])
            g.add((movie_uri, relation_ns.directedBy, director_uri))

            # Add triples representing movies, actors, and their relationships
            g.add((movie_uri, relation_ns.hasReleaseYear, URIRef(year_ns[row['Year']])))

    g.serialize(filename, format='turtle')
    return g

attr_to_num = {'movie2actor':0,'movie2genre':1,'movie2director':2,'movie2year':3,'movie2desc':4}

def graph_to_nx(csv_file:str='IMDB-Movie-Data.csv',save_desc:bool=False,doc_mapping=None)->nx.graph:
    G = nx.Graph()
    # Load IMDb data from CSV file (modify the filename accordingly)

    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        # Iterate through the CSV data and add triples to the graph
        for row in reader:
            movie_uri = f"movie:{removeSpaces(row['Title'])}"
            # actor_uri = URIRef(ns[row['actor_id']])
            for actor in row['Actors'].split(','):
                actor_uri = f'actor:{removeSpaces(actor)}'
                G.add_edges_from([
                    (movie_uri, actor_uri, {'type': attr_to_num['movie2actor']}),
                    # (actor_uri, movie_uri, {'type': 'actor2movie'}),
                ])
        
            for genre in row['Genre'].split(','):
                genre_uri = f'genre:{genre}'
                G.add_edges_from([
                    (movie_uri, genre_uri, {'type': attr_to_num['movie2genre']}),
                ])
            director_uri = f'director:{removeSpaces(row["Director"])}'
            G.add_edges_from([
                    (movie_uri, director_uri, {'type': attr_to_num['movie2director']}),
                    # (director_uri, movie_uri, {'type': 'director2movie'}),
                    (movie_uri, f"year:{row['Year']}", {'type': attr_to_num['movie2year']})
                ])
            if save_desc:
                G.add_edges_from([
                    (movie_uri, f'desc:{doc_mapping(row["Description"])}', {'type': attr_to_num['movie2desc']})
                ])
                ### doc_mapping fn will embed the description to a vector and return an id
    return G