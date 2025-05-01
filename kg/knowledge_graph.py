from config.config import Config

from rdflib import Graph, URIRef, Namespace, Literal
from rdflib.namespace import RDF, RDFS, SKOS
from SPARQLWrapper import SPARQLWrapper, JSON

DBPEDIA_SPARQL = "https://dbpedia.org/sparql"
DBPEDIA = Namespace("http://dbpedia.org/resource/")
DBPEDIA_ONT = Namespace("http://dbpedia.org/ontology/")

manual_translations = {
    "EU cooperation": ("http://dbpedia.org/resource/European_Union", "співпраця ЄС"),
    "financial assistance": (None, "фінансова допомога"),
    "European Union": ("http://dbpedia.org/resource/European_Union", "Європейський Союз"),
    "donors": (None, "донори")
}


def query_dbpedia(term):
    if term in manual_translations:
        return manual_translations[term] 
    sparql = SPARQLWrapper(DBPEDIA_SPARQL)
    query = f"""
    SELECT ?ukLabel WHERE {{
      ?resource rdfs:label "{term}"@en .
      ?resource rdfs:label ?ukLabel .
      FILTER (lang(?ukLabel) = "uk")
    }} LIMIT 1
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        if results["results"]["bindings"]:
            return (
                results["results"]["bindings"][0]["resource"]["value"],
                results["results"]["bindings"][0]["ukLabel"]["value"]
            )
    except Exception as e:
        print(f"Request error for '{term}': {e}")
    return None, None

def create_knowledge_graph_dbpedia(terms):
    print("Creating DBpedia-based KG...")
    g = Graph()
    g.bind("skos", SKOS)
    g.bind("rdfs", RDFS)
    g.bind("dbpedia", DBPEDIA)

    for term in terms:
        uri, uk_label = query_dbpedia(term)
        if uri and uk_label:
            term_uri = URIRef(uri)
            g.add((term_uri, RDFS.label, Literal(term, lang="en")))
            g.add((term_uri, RDFS.label, Literal(uk_label, lang="uk")))
            g.add((term_uri, SKOS.prefLabel, Literal(term, lang="en")))
            g.add((term_uri, SKOS.altLabel, Literal(uk_label, lang="uk")))
        else:
            print(f"No match found for '{term}'.")

    g.serialize("dbpedia_kg.ttl", format="turtle")
    print("Knowledge graph saved as 'dbpedia_kg.ttl'")
