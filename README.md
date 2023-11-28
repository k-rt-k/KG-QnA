# CS635: Assignment 2
# Answering Queries with Knowledge Graphs
## Kartik Nair,  Arhaan Ahmad, and Prerak Contractor

[Problem Statement](https://docs.google.com/document/d/1W2QxciT8i-uT6fomxfmHih0SE2Z1BLNEoLXzplHgvcM/edit)

We will be making a Knowledge Graph for the IMDB top 1000 movies for 2006-2016 and using it to answer queries over title, genres, director, actors and years.


---

## Final Results:

- The pipeline to construct and query the knowledge graph is present in ``sparql.ipynb``.
- ``main.ipynb``contains the now ditched effort of trying to use a GNN and a LM to query.
- We were unable to fine-tune a LM to convert a natural language query to the format required by us, due to time constraints. We however are able to convert it using GPT 3. Here is an example:

![image](https://github.com/k-rt-k/KG-QnA/assets/93263179/700d1725-c7d2-435c-9629-6151d000dda4)

 - We attempted to create a data set using GPT 3 API which can be found in ``GenerateData.ipynb`` but were unable to do so due to rate limit imposed by OpenAI.

**Note**: The API Key present has been disabled and will be removed in next push.
