import json
import requests
import pandas as pd


class CustomSearch:

    def __init__(self):
        self.key = "AIzaSyCMmo6IBp5w5g8oL-1LUANLa6OmxOr-i8U"
        self.cx = "012678121589857172556:3ck87h1vxfj"
        self.url = "https://www.googleapis.com/customsearch/v1"
        self.results = ''


    def search(self ,query):
        parameters = {"q":query,"cx": self.cx,"key": self.key,}
        page = requests.request("GET", self.url, params=parameters)
        return json.loads(page.text)

    def process_search(self ,results):
        link_list = [item["link"] for item in results["items"]]
        df = pd.DataFrame(link_list, columns=["link"])
        df["title"] = [item["title"] for item in results["items"]]
        df["snippet"] = [item["snippet"] for item in results["items"]]
        return df
    
    def custom_search(self,query):
        results = self.search(query)
        df= self.process_search(results)[:6]
        data =self.pre_process(query,df)
        input_data =self.initialize_data(data);
        return input_data

    def pre_process(self,query,df):
        data=[]
        for index, row in df.iterrows():
            url =row['link']
            name=row['link'].split('.',2)[1]
            Headline=row['title']
            articleBody = query
            data.append([url,name,Headline,articleBody,''])
        return data

    def initialize_data(self,data):
         df = pd.DataFrame(data, columns = ['url','name','Headline', 'articleBody','stance']) 
         return df

    




