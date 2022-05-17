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
import pandas as pd

from env import github_token, github_username

# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token: https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.

REPOS = ['janishar/mit-deep-learning-book-pdf',
 'Angel-ML/angel',
 'Alluxio/alluxio',
 'haifengl/smile',
 'alibaba/Alink',
 'ICT-BDA/EasyML',
 'OryxProject/oryx',
 'kermitt2/grobid',
 'SeldonIO/seldon-server',
 'MindorksOpenSource/AndroidTensorFlowMachineLearningExample',
 'o19s/elasticsearch-learning-to-rank',
 'datumbox/datumbox-framework',
 'oracle/tribuo',
 'mimno/Mallet',
 'imageprocessor/cv4j',
 'kaiwaehner/kafka-streams-machine-learning-examples',
 'EdwardRaff/JSAT',
 'amitshekhariitbhu/Android-TensorFlow-Lite-Example',
 'apache/submarine',
 'Waikato/moa',
 'antlr/codebuff',
 'pmerienne/trident-ml',
 'etsy/Conjecture',
 'hanuor/onyx',
 'yuantiku/ytk-learn',
 'linkedin/dagli',
 'opendistro-for-elasticsearch/k-NN',
 'pushkar/ABAGAIL',
 'ShifuML/shifu',
 'Waikato/meka',
 'HongZhaoHua/jstarcraft-ai',
 'cheng-li/pyramid',
 'sea-boat/TextAnalyzer',
 'afsalashyana/FakeImageDetection',
 'RumbleDB/rumble',
 'ogrisel/pignlproc',
 'projectmatris/antimalwareapp',
 'kermitt2/entity-fishing',
 'SmartDataAnalytics/DL-Learner',
 'yinlou/mltk',
 'apache/flink-ml',
 'Daniel-Liu-c0deb0t/Java-Machine-Learning',
 'algorithmfoundry/Foundry',
 'ClearTK/cleartk',
 'uea-machine-learning/tsml',
 'wen-fei/choice',
 'goessl/MachineLearning',
 'Nova41/SnowLeopard',
 'amidst/toolbox',
 'anbud/MarkovComposer',
         'tensorflow/tensorflow',
 'TheAlgorithms/C-Plus-Plus',
 'PaddlePaddle/Paddle',
 'microsoft/LightGBM',
 'onnx/onnx',
 'davisking/dlib',
 'apple/turicreate',
 'bulletphysics/bullet3',
 'VowpalWabbit/vowpal_wabbit',
 'tensorflow/serving',
 'interpretml/interpret',
 'amazon-archives/amazon-dsstne',
 'flashlight/flashlight',
 'mlpack/mlpack',
 'aksnzhy/xlearn',
 'ARM-software/ComputeLibrary',
 'facebookresearch/TensorComprehensions',
 'ermig1979/Simd',
 '4paradigm/OpenMLDB',
 'IBM/fhe-toolkit-linux',
 'microsoft/EdgeML',
 'ml4a/ml4a-ofx',
 'dmlc/mshadow',
 'novak-99/MLPP',
 'Samsung/veles',
 'turi-code/SFrame',
 'node-tensorflow/node-tensorflow',
 'microsoft/Windows-Machine-Learning',
 'dmlc/dmlc-core',
 'microsoft/Multiverso',
 'jubatus/jubatus',
 'nladuo/captcha-break',
 'niessner/Matterport',
 'neoml-lib/neoml',
 'NVIDIA/nvvl',
 'fastmachinelearning/hls4ml',
 'StanfordSNR/puffer',
 'bytefish/opencv',
 'perone/euclidesdb',
 'mldbai/mldb',
 'PaddlePaddle/Serving',
 'deepmind/reverb',
 'dmlc/rabit',
 'pennyliang/MachineLearning-C---code',
 'memo/ofxMSATensorFlow',
 'neo-ai/neo-ai-dlr',
 'FidoProject/Fido',
 'chanyn/3Dpose_ssl',
 'NVIDIA/tensorflow',
 'alibaba/BladeDISC',
         'huggingface/transformers',
 'josephmisiti/awesome-machine-learning',
 'scikit-learn/scikit-learn',
 'fighting41love/funNLP',
 'eriklindernoren/ML-From-Scratch',
 'RasaHQ/rasa',
 'aleju/imgaug',
 'gunthercox/ChatterBot',
 'mlflow/mlflow',
 'microsoft/nni',
 'ddbourgin/numpy-ml',
 'PySimpleGUI/PySimpleGUI',
 'ml-tooling/best-of-ml-python',
 'rushter/MLAlgorithms',
 'EpistasisLab/tpot',
 'ludwig-ai/ludwig',
 'clips/pattern',
 'gradio-app/gradio',
 'instillai/machine-learning-course',
 'mindsdb/mindsdb',
 'lazyprogrammer/machine_learning_examples',
 'pymc-devs/pymc',
 'automl/auto-sklearn',
 'doccano/doccano',
 'Jack-Cherish/Machine-Learning',
 'scikit-learn-contrib/imbalanced-learn',
 'alan-turing-institute/sktime',
 'humphd/have-fun-with-machine-learning',
 'kootenpv/whereami',
 'mdbloice/Augmentor',
 'lawlite19/MachineLearning_Python',
 'ujjwalkarn/DataSciencePython',
 'wepe/MachineLearning',
 'rasbt/mlxtend',
 'wandb/client',
 'DistrictDataLabs/yellowbrick',
 'aladdinpersson/Machine-Learning-Collection',
 'online-ml/river',
 'cleanlab/cleanlab',
 'arielf/weight-loss',
 'feast-dev/feast',
 'Avik-Jain/100-Days-of-ML-Code-Chinese-Version',
 'TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials',
 'polyaxon/polyaxon',
 'uber/causalml',
 'Trusted-AI/adversarial-robustness-toolbox',
 'nidhaloff/igel',
 'hudson-and-thames/mlfinlab',
 'kubeflow/pipelines',
 'ml-tooling/opyrator'
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

def get_repo_list(query='machine learning', sort_method='stars', how_many = 50):
    """Return a list of repositories based on search query and sorted by sort method"""
    
    g = Github(github_token)
    
    repositories = g.search_repositories(query=query, sort=sort_method)
    
    repo_list = []
    for repo in repositories[:how_many]:
        repo_list.append(repo.full_name)
        
    return repo_list

def get_repo_data():
    """Acquires scraped data either from cached data.json file or using the Github API"""
    
    # Checks if cache exists
    if os.path.exists('data.json'):
        print("Reading data from cache")
        # Read data from json into Pandas Dataframe
        return pd.read_json('data.json')
    
    else:
        print("Acquiring new data from Github")
        
        # Data was scraped using the acquire script from Zach
        scraped_data = scrape_github_data()

        # Scrapped data saved to dataframe and json
        data = pd.DataFrame(scraped_data)

        data.to_json('data.json')
        
        return data

if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data.json", "w"), indent=1)