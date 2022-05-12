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
from github import Github

from env import github_token, github_username

# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token: https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.

REPOS = ['chenyuntc/pytorch-best-practice',
 'knavesec/Max',
 'adilmoujahid/deeplearning-cats-dogs-tutorial',
 'dogsheep/dogsheep-photos',
 'dogsheep/dogsheep-beta',
 'ahangchen/keras-dogs',
 'mrgloom/kaggle-dogs-vs-cats-caffe',
 'holyhao/Baidu-Dogs',
 'maples1993/Cats_vs_Dogs',
 'ponicode/dogstring-action',
 'lvyilin/pytorch-fgvc-dataset',
 'kastnerkyle/kaggle-dogs-vs-cats',
 'desimone/pytorch-cat-vs-dogs',
 'ehsanik/dogTorch',
 'zhangpengpengpeng/kaggle-dogs-vs-cats',
 'gzshan/dogsVScats',
 'xuetsing/cats-vs-dogs',
 'espectre/Kaggle_Dogs_vs_Cats_PyTorch',
 'DukeNgn/Dog-facts-API-DEPRECATED-',
 'zygmuntz/kaggle-cats-and-dogs'
]

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

def get_repo_list(query='data science', sort_method='stars', how_many = 100):
    """Return a list of repositories based on search query and sorted by sort method"""
    
    g = Github(github_token)
    
    repositories = g.search_repositories(query=query, sort=sort_method)
    
    repo_list = []
    for repo in repositories[:how_many]:
        repo_list.append(repo.full_name)
        
    return repo_list

if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data.json", "w"), indent=1)