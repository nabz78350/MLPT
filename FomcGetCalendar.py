import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import os


def getFOMCChanges():
    url = (
        "https://en.wikipedia.org/wiki/History_of_Federal_Open_Market_Committee_actions"
    )
    html = requests.get(url).content
    df_list = pd.read_html(html)
    massivechanges = df_list[1]
    return massivechanges


def getFOMCDates(decade):
    url = f"https://fraser.stlouisfed.org/title/federal-open-market-committee-meeting-minutes-transcripts-documents-677?browse={decade}s"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")

    dateList = soup.find_all("span", {"class": "list-item-title"})
    dList = []

    def parseString():
        for i in range(len(dateList)):
            s = dateList[i].text
            ind = s.find("Meeting, ")
            if ind != None and ind != -1:
                dateStr = s[ind + 9 :]
                dateStr = dateStr[: dateStr.find(",") + 6]
                if dateStr.find("-") != -1:
                    dateStr = (
                        dateStr[: dateStr.find("-")] + dateStr[dateStr.find(",") :]
                    )
                dList.append(dateStr)

    parseString()
    return dList


if __name__ == "__main__":
    dateList = (
        getFOMCDates(1960)
        + getFOMCDates(1970)
        + getFOMCDates(1980)
        + getFOMCDates(1990)
        + getFOMCDates(2000)
        + getFOMCDates(2010)
        + getFOMCDates(2020)
    )

    def strToCal(s):
        return datetime.strptime(s, "%B %d, %Y")

    dateTimeList = list(map(strToCal, dateList))

    dfList = []
    d = datetime(1960, 12, 10)
    for elem in dateTimeList:
        if elem > d:
            dfList.append([False, elem])
    df = pd.DataFrame(dfList, columns=["unscheduled", "datetime"])
    os.makedirs("data/FOMC", exist_ok=True)
    df.to_parquet("data/FOMC/fomc_calendar.parquet")
