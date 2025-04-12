import openai
import csv
import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen

import certifi
import json
urlCheck = " "

def get_jsonparsed_data(url):
    """
    Receive the content of `url`, parse it as JSON and return the object.

    Parameters
    ----------
    url : str

    Returns
    -------
    dict
    """
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    print("Appel de l'API")
    return json.loads(data)
while True :
    print("pd")
    
    urlData = "https://financialmodelingprep.com/api/v4/forex_news?page=0&apikey=6337f5a5a3024939d88b837e41aac0b3"
    dataNews = get_jsonparsed_data(urlData)
    urls = [item['url'] for item in dataNews if 'url' in item]
    titles = [item['title'] for item in dataNews if 'title' in item]
    dates = [item['publishedDate'] for item in dataNews if 'publishedDate' in item]
    sites = [item['site'] for item in dataNews if 'site' in item]
    urlForex = urls[0]
    prompt = f"Read this article {urlForex} and tell me in this order : Which pair of currencies this article is talking about ?  In your opinion give a grade to this pair of currencies from 0 to 100 where 0 is -I should definitly Sell- and 100 - I should definetly Buy- you answer should only be two words no more: one for the pair and one for the grade. If the article doesn't mention a pair of currencies return 0"
    if urlForex == urlCheck :   
        time.sleep(25)
        continue
    for i in range(1,11) :
        if urls[i] == urlCheck : 
            print(f"Il y a eu donc {i-1} article poster entre temps")
            urlForex = urls[i-1]
            title = titles[i-1]
            date=dates[i-1]
            site = sites[i-1]
            break
        else :
            title = titles[0]
            date=dates[0]
            site = sites[0]


                  

    
    




    # optional; defaults to os.environ['OPENAI_API_KEY']
    openai.api_key = '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'


    print("One appele Chat")
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    print(completion.choices[0].message.content)
    if completion.choices[0].message.content != "0":
        tabString=completion.choices[0].message.content.split(' ', 1)
    else : 
        print("L'article ne mentionne pas de paires de devise" )
        urlCheck = urlForex
        continue
        
        

    urlCheck = urlForex

    import csv

    # Chemin vers votre fichier CSV
    fichier_csv = "C:/Users/Administrator/Documents/Asso finance/ECE_Finance/Tables_Ronde/FOREX/analyse_FOREX_chat_Desktop.csv"
    # Lire le contenu existant du fichier CSV
    donnees = []
    with open(fichier_csv, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            donnees.append(row)

    # Ajouter la nouvelle ligne à la fin
    nouvelle_ligne = [site,date,title, tabString[0], tabString[1],urlForex]
    donnees.append(nouvelle_ligne)

    # Réécrire le fichier CSV avec les données mises à jour
    with open(fichier_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(donnees)
        print("Tout est bon pour les fichiers")

    api_key = "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"  # Replace with your API key
