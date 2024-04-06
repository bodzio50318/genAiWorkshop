import requests
import wget


def fetchListOfLinks(url):
    try:
        response = requests.get(url)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Assuming the response contains a JSON array
            data = response.json()
            return data
        else:
            print("Failed to fetch data. Status code:", response.status_code)
            return None
    except requests.exceptions.RequestException as e:
        print("Error fetching data:", e)
        return None
  
url = 'https://wolnelektury.pl/api/books/' 
dataList = fetchListOfLinks(url)
numberOfBooks=len(dataList)
with open('dataSet.txt', "w") as file:
    for i in range(numberOfBooks):
        if i%100==0 :print(i)
        name=str(dataList[i]['url']).split('/')[-2]
        url='https://wolnelektury.pl/media/book/txt/'+name+'.txt'
        # print(url)
        file.write(requests.get(url).text)
        # print(file_content)





