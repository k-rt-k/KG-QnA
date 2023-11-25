import csv
import pandas as pd
import rdflib
from rdflib import Graph, Namespace, Literal, URIRef
import rdflib.term as rdflib_term
from rdflib.namespace import RDF as rdf
import networkx as nx

def removeSpaces(string:str)->str:
    return string.replace(" ", "_")

def graph_to_ttl(csv_file:str='IMDB-Movie-Data.csv',filename:str='imdb_kg.ttl')->rdflib.graph:
    # Create an RDF graph
    g = Graph()

    # Define namespaces
    ns = Namespace("ns:")

    # Load IMDb data from CSV file (modify the filename accordingly)
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        # Iterate through the CSV data and add triples to the graph
        for row in reader:
            movie_uri = URIRef(ns[removeSpaces(row["Title"])])
            # actor_uri = URIRef(ns[row['actor_id']])
            for actor in row['Actors'].split(','):
                actor_uri = URIRef(ns[f'actor:{removeSpaces(actor)}'])
                g.add((actor_uri, ns.hasName, Literal(actor)))
                g.add((movie_uri, ns.hasActor, actor_uri))
            
            for genre in row['Genre'].split(','):
                genre_uri = URIRef(ns[f'genre:{genre}'])
                g.add((movie_uri, ns.hasGenre, genre_uri))
            director_uri = URIRef(ns[f'director:{removeSpaces(row["Director"])}'])
            # director_resource = g.resource(director_uri)
            g.add((movie_uri, ns.directed_by, director_uri))

            # Add triples representing movies, actors, and their relationships
            g.add((movie_uri, ns.hasTitle, Literal(row['Title'])))
            g.add((movie_uri, ns.hasReleaseYear, Literal(row['Year'])))
            # g.add((movie_uri, ns.hasReleaseYear, Literal(row['Year'], datatype=ns.xsd.int)))
    # Serialize the RDF graph to a file (modify the filename accordingly)
    g.serialize(filename, format='turtle')
    return g

attr_to_num = {'movie2actor':1,'movie2genre':2,'movie2director':3,'movie2year':4}

def graph_to_nx(csv_file:str='IMDB-Movie-Data.csv')->nx.graph:
    G = nx.Graph()
    # Load IMDb data from CSV file (modify the filename accordingly)

    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        # Iterate through the CSV data and add triples to the graph
        for row in reader:
            movie_uri = removeSpaces(row["Title"])
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
                    (movie_uri, row['Year'], {'type': attr_to_num['movie2year']})
                ])
    return G