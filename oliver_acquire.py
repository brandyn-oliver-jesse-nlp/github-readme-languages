"""
A module for obtaining repo readme and language data from the github API.
Before using this module, read through it, and follow the instructions marked
TODO.
After doing so, run it like this:
    python acquire.py
To create the `data.json` file that contains the data.
"""
import os
import json
from typing import Dict, List, Optional, Union, cast
import requests

from env import github_token, github_username

# required imports
from requests import get
import pandas as pd
from bs4 import BeautifulSoup
import time
import random

# goes to github and pages through the most forked repos, grabbing the 'num_pages' number of pages
def fetch_github_repos(num_pages):
    items_list = []
    # to make this simple, we will grab repos with the most forks and with stars > 1
    # the top pages have the format https://github.com/search?o=desc&p=1&q=stars%3A%3E1&s=forks&type=Repositories
    # so we need to increment the p= parameter to go to each subsequent page
    for i in range(1,num_pages+1):
        # add a sleep amount of random time so that we don't get HTTP 429s
        time.sleep(random.random())
        headers = {'User-Agent': 'Codeup Data Science'} # Some websites don't accept the pyhon-requests default user-agent
        url = f'https://github.com/search?o=desc&p={i}&q=stars%3A%3E1&s=forks&type=Repositories'
        response = get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')

        # each page has 10 results. Let's loop through and find each instance of page element element = 'a' class = 'v-align-middle'
        links = soup.find_all('a', class_='v-align-middle')
        for repo_link in links:
            time.sleep(random.random())
            repo_name = repo_link.text
            # add it to our output array
            items_list.append(repo_name)
    return items_list

# returns a list of github repo names, either from a file on disk or from github using the fetch_github_repos function
def get_github_repos(cached=False, num_pages = 40):
    # If the cached parameter is false, or the csv file is absent, use github
    if cached == False or os.path.isfile('git_urls.csv') == False:
        # read from github
        repo_list = fetch_github_repos(num_pages)
        # and write to the cache file for next time
        df = pd.DataFrame(repo_list, columns=['repos'])
        df.to_csv('git_urls.csv', index=False)
    else:
        # read from the cache file on disk
        df = pd.read_csv("git_urls.csv")
        repo_list = df['repos'].to_list()

    # either way, return the list of repos
    return repo_list


# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token: https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.

REPOS = [
 'jtleek/datasharing',
 'rdpeng/ProgrammingAssignment2',
 'octocat/Spoon-Knife',
 'SmartThingsCommunity/SmartThingsPublic',
 'tensorflow/tensorflow',
 'twbs/bootstrap',
 'LSPosed/MagiskOnWSA',
 'github/gitignore',
 'Pierian-Data/Complete-Python-3-Bootcamp',
 'jwasham/coding-interview-university',
 'rdpeng/ExData_Plotting1',
 'github/docs',
 'opencv/opencv',
 'EbookFoundation/free-programming-books',
 'eugenp/tutorials',
 'tensorflow/models',
 'jackfrued/Python-100-Days',
 'firstcontributions/first-contributions',
 'torvalds/linux',
 'Snailclimb/JavaGuide',
 'donnemartin/system-design-primer',
 'kubernetes/kubernetes',
 'vuejs/vue',
 'mrdoob/three.js',
 'DataScienceSpecialization/courses',
 'getify/You-Dont-Know-JS',
 'freeCodeCamp/freeCodeCamp',
 'kamranahmedse/developer-roadmap',
 'PanJiaChen/vue-element-admin',
 'DefinitelyTyped/DefinitelyTyped',
 'PanJiaChen/vue-element-admin',
 'mui/material-ui',
 'justjavac/free-programming-books-zh_CN',
 'apache/spark',
 'google/it-cert-automation-practice',
 'macrozheng/mall',
 'facebook/create-react-app',
 'dotnet/AspNetCore.Docs',
 'ohmyzsh/ohmyzsh',
 'airbnb/javascript',
 'qmk/qmk_firmware',
 'apache/spark',
 'apache/dubbo',
 'google/it-cert-automation-practice',
 'macrozheng/mall',
 'facebook/create-react-app',
 'dotnet/AspNetCore.Docs',
 'ohmyzsh/ohmyzsh',
 'airbnb/javascript',
 'git/git',
 'rails/rails',
 'pjreddie/darknet',
 'MicrosoftDocs/mslearn-tailspin-spacegame-web',
 'shadowsocks/shadowsocks',
 'academicpages/academicpages.github.io',
 'mmistakes/minimal-mistakes',
 'keras-team/keras',
 'BVLC/caffe',
 'apache/echarts',
 'CSSEGISandData/COVID-19',
 'atralice/Curso.Prep.Henry',
 'udacity/frontend-nanodegree-resume',
 'laravel/laravel',
 'microsoft/vscode',
 'public-apis/public-apis',
 'facebook/react-native',
 'flutter/flutter',
 'ansible/ansible',
 'elastic/elasticsearch',
 '996icu/996.ICU',
 'flutter/flutter',
 'ansible/ansible',
 'elastic/elasticsearch',
 '996icu/996.ICU',
 'angular/angular',
 'slatedocs/slate',
 'redis/redis',
 'vinta/awesome-python',
 'labuladong/fucking-algorithm',
 'LarryMad/recipes',
 'animate-css/animate.css',
 'gabrielecirulli/2048',
 'udacity/course-collaboration-travel-plans',
 'lewagon/dotfiles',
 'coding-boot-camp/prework-about-me',
 'pytorch/pytorch',
 'jakevdp/PythonDataScienceHandbook',
 'aymericdamien/TensorFlow-Examples',
 'reduxjs/redux',
 'pallets/flask',
 'lewagon/dotfiles',
 'coding-boot-camp/prework-about-me',
 'spring-projects/spring-petclinic',
 'aymericdamien/TensorFlow-Examples',
 'reduxjs/redux',
 'pallets/flask',
 'TheAlgorithms/Java',
 'scutan90/DeepLearning-500-questions',
 'huggingface/transformers',
 'github/opensource.guide',
 'huggingface/transformers',
 'golang/go',
 'github/opensource.guide',
 'TheOdinProject/css-exercises',
 'netty/netty',
 'Azure/azure-quickstart-templates',
 'pandas-dev/pandas',
 'protocolbuffers/protobuf',
 'othneildrew/Best-README-Template',
 'tastejs/todomvc',
 'jakevdp/PythonDataScienceHandbook',
 'aymericdamien/TensorFlow-Examples',
 'reduxjs/redux',
 'scutan90/DeepLearning-500-questions',
 'huggingface/transformers',
 'golang/go',
 'github/opensource.guide',
 'TheOdinProject/css-exercises',
 'selfteaching/the-craft-of-selfteaching',
 'netty/netty',
 'ethereum/go-ethereum',
 'trustwallet/assets',
 'codecrafters-io/build-your-own-x',
 '233boy/v2ray',
 'electron/electron',
 'MisterBooo/LeetCodeAnimation',
 'archway-network/testnets',
 'josephmisiti/awesome-machine-learning',
 'springframeworkguru/spring5webapp',
 'linuxacademy/devops-essentials-sample-app',
 'bailicangdu/vue2-elm',
 'angular/angular-cli',
 'kdn251/interviews',
 'google/styleguide',
 'Trinea/android-open-project',
 'FortAwesome/Font-Awesome',
 'h5bp/html5-boilerplate',
 'udacity/fullstack-nanodegree-vm',
 'ityouknow/spring-boot-examples',
 'Homebrew/legacy-homebrew',
 'h5bp/html5-boilerplate',
 'udacity/fullstack-nanodegree-vm',
 'Homebrew/legacy-homebrew',
 'PX4/PX4-Autopilot',
 'deepfakes/faceswap',
 'jeecgboot/jeecg-boot',
 'shadowsocks/shadowsocks-android',
 'mybatis/mybatis-3',
 'rafaballerini/rafaballerini',
 'getlantern/lantern',
 'udacity/fullstack-nanodegree-vm',
 'ityouknow/spring-boot-examples',
 'Homebrew/legacy-homebrew',
 'PX4/PX4-Autopilot',
 'deepfakes/faceswap',
 'jeecgboot/jeecg-boot',
 'shadowsocks/shadowsocks-android',
 'mybatis/mybatis-3',
 'getlantern/lantern',
 'apache/kafka',
 'Homebrew/legacy-homebrew',
 'PX4/PX4-Autopilot',
 'deepfakes/faceswap',
 'jeecgboot/jeecg-boot',
 'mybatis/mybatis-3',
 'rafaballerini/rafaballerini',
 'getlantern/lantern',
 'apache/kafka',
 'chartjs/Chart.js',
 'jonasschmedtmann/complete-javascript-course',
 'socketio/socket.io',
 'learn-co-students/python-practice-with-datatypes-data-science-intro-000',
 'google/guava',
 'Homebrew/homebrew-cask',
 'avelino/awesome-go',
 'alibaba/nacos',
 'lenve/vhr',
 'bettiolo/node-echo',
 'apache/rocketmq',
 'udacity/create-your-own-adventure',
 'scrapy/scrapy',
 'grafana/grafana',
 'FFmpeg/FFmpeg',
 'RedHatTraining/DO288-apps',
 'fengdu78/Coursera-ML-AndrewNg-Notes',
 'vuejs/vuex',
 'testerSunshine/12306',
 'youzan/vant',
 'ultralytics/yolov5',
 'google/material-design-icons',
 'woocommerce/woocommerce',
 'fatedier/frp',
 'ageron/handson-ml2',
 'jhu-ep-coursera/fullstack-course4',
 'practical-tutorials/project-based-learning',
 'kelseyhightower/kubernetes-the-hard-way',
 '30-seconds/30-seconds-of-code',
 'socketio/socket.io',
 'learn-co-students/python-practice-with-datatypes-data-science-intro-000',
 'google/guava']
 
headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )


def github_api_request(url: str) -> Union[List, Dict]:
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        if "language" not in repo_info:
            raise Exception(
                "'language' key not round in response\n{}".format(json.dumps(repo_info))
            )
        return repo_info["language"]
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_download_url = get_readme_download_url(contents)
    if readme_download_url == "":
        readme_contents = ""
    else:
        readme_contents = requests.get(readme_download_url).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


def scrape_github_data() -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    return [process_repo(repo) for repo in REPOS]


if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data.json", "w"), indent=1)
