"""
Data Science Case Study - data preprocessing module
Author: BO
Data: 2026-06-03
Description: This versatile data module serves as an all-in-one ingestion and engineering engine:
***Smart Data Ingestion:** Automatically reads and parses files from specified paths, dynamically routing data into **Pandas** (for non-spatial data) or **GeoPandas** (for GeoJSON/spatial data) while strictly enforcing file-safety checks.
***Resilient Error Management:** Handles unsupported formats and missing files gracefully without crashing your notebook.
***Random Forest Feature Selection:** Integrates an advanced feature pruning mechanism. It fits a Random Forest model on the combined dataset to ranking feature contributions and automatically strips away low-importance noise to optimize downstream model training.
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
import requests
import subprocess   
from pandas.api.types import is_string_dtype, is_bool_dtype, is_numeric_dtype
from dython.nominal import associations
from sklearn.ensemble import RandomForestRegressor




def get_release_df(filename: str, tag: str = "data") -> pd.DataFrame:
    """from the current repository's Release, automatically construct the URL to the data file and read it into a DataFrame.

    Parameters:
    - filename: full name of the file (e.g., 'calendar.csv.gz')
    - tag: Release tag, default is 'data'

    Returns:
    - On success, returns a pd.DataFrame; on failure, returns None
    """
    try:
        # 1. get the repo URL from git config
        repo_url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"]).decode("utf-8").strip()
        # get the github repo path (e.g., "username/repo")
        repo_path = repo_url.split("github.com/")[-1].replace(".git", "")
        # 3. automatically construct the URL to the data in the Release
        data_url = f"https://github.com/{repo_path}/releases/download/{tag}/{filename}"
        
        # get url response status code
        response = requests.head(data_url, allow_redirects=True, timeout=10)
        # check if the file already exists locally before trying to download and read the data into a DataFrame
        if os.path.exists(filename):
            print(
                f"Codespaces already has '{filename}'"
            )
            df = pd.read_csv(filename)
            return df

        # if not, try to fetch from the remote URL
        # print(f"Attempting to validate URL: {data_url}")
        print(f"Attempting to validate URL link")

        if response.status_code == 200:
            # print(f"URL successfully validated! Preparing to read data from: {data_url}")
            print(f"URL successfully validated! Preparing to read data from link")

            # 5. read the data into a DataFrame
            df = pd.read_csv(data_url)

            return df
        
        elif response.status_code == 404:
            print(
                f"failed to validate URL (404 Not Found): Please check if your Release tag is '{tag}' and the filename '{filename}' is correct."
            )
            return None
        else:
            print(
                f"failed to validate URL (Status Code: {response.status_code}): Please check the URL."
            )
            return None # type: ignore

    except subprocess.CalledProcessError:
        print("Git command failed: Unable to retrieve repository URL. Please ensure you are running this script within a Git repository and that Git is properly installed.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
        return None
    



def get_release_geodf(filename: str, tag: str = "data") -> gpd.GeoDataFrame:
    """from the current repository's Release, automatically construct the URL to the data file and read it into a GeoDataFrame.

    Parameters:
    - filename: full name of the file (e.g., "neighbourhoods.geojson")
    - tag: Release tag, default is 'data'

    Returns:
    - On success, returns a gpd.GeoDataFrame; on failure, returns None
    """
    try:
        # 1. get the repo URL from git config
        repo_url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"]).decode("utf-8").strip()
        # get the github repo path (e.g., "username/repo")
        repo_path = repo_url.split("github.com/")[-1].replace(".git", "")
        # 3. automatically construct the URL to the data in the Release
        data_url = f"https://github.com/{repo_path}/releases/download/{tag}/{filename}"
        
        # get url response status code
        response = requests.head(data_url, allow_redirects=True, timeout=10)
        # check if the file already exists locally before trying to download and read the data into a GeoDataFrame
        if os.path.exists(filename):
            print(
                f"Codespaces already has '{filename}'"
            )
            #df = pd.read_csv(filename)
            geodf = gpd.read_file(filename)
            return geodf

        # if not, try to fetch from the remote URL
        # print(f"Attempting to validate URL: {data_url}")
        print(f"Attempting to validate URL link")

        if response.status_code == 200:
            # print(f"URL successfully validated! Preparing to read data from: {data_url}")
            print(f"URL successfully validated! Preparing to read data from link")

            # 5. read the data into a GeoDataFrame
            #df = pd.read_csv(data_url)
            geodf = gpd.read_file(data_url)

            return geodf
        elif response.status_code == 404:
            print(
                f"failed to validate URL (404 Not Found): Please check if your Release tag is '{tag}' and the filename '{filename}' is correct."
            )
            return None
        else:
            print(
                f"failed to validate URL (Status Code: {response.status_code}): Please check the URL."
            )
            return None

    except subprocess.CalledProcessError:
        print("Git command failed: Unable to retrieve repository URL. Please ensure you are running this script within a Git repository and that Git is properly installed.")
        return None # type: ignore
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
        return None # type: ignore


def get_validate_url(filename:str, tag:str = "data") -> str:
    """from the current repository's Release, automatically construct the URL to the data file.

    Parameters:
    - filename: full name of the file (e.g., 'calendar.csv.gz')
    - tag: Release tag, default is 'data'

    Returns:
    - On success, returns a string URL; on failure, returns None
    """
    try:
        # 1. get the repo URL from git config
        repo_url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"]).decode("utf-8").strip()
        # get the github repo path (e.g., "username/repo")
        repo_path = repo_url.split("github.com/")[-1].replace(".git", "")
        # 3. automatically construct the URL to the data in the Release
        data_url = f"https://github.com/{repo_path}/releases/download/{tag}/{filename}"
        # get url response status code
        response = requests.head(data_url, allow_redirects=True, timeout=10)
        
       
        # print(f"Attempting to validate URL: {data_url}")
        print(f"Attempting to validate URL link")
        print(f"response.status_code:{response.status_code  }")
        if response.status_code == 200:
            # print(f"URL successfully validated! Preparing to read data from: {data_url}")
            print(f"URL successfully validated! Preparing to read data from link")

            

            return data_url
        elif response.status_code == 404:
            print(
                f"failed to validate URL (404 Not Found): Please check if your Release tag is '{tag}' and the filename '{filename}' is correct."
            )
            return None
        else:
            print(
                f"failed to validate URL (Status Code: {response.status_code}): Please check the URL."
            )
            return None

    except subprocess.CalledProcessError:
        print("Git command failed: Unable to retrieve repository URL. Please ensure you are running this script within a Git repository and that Git is properly installed.")
        return None # type: ignore
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
        return None # type: ignore
    

def infer_json_schema(data):
    """
    recursively infer the schema of a JSON-like data structure, which can be a mix of dictionaries, lists, and basic types.
    """
    # 1. if it's a dictionary, we need to look at each key-value pair and infer the schema of the value, then combine them into a new dictionary
    if isinstance(data, dict):
        return {key: infer_json_schema(value) for key, value in data.items()}
    
    # 2. if it's a list, we need to check what's inside the list
    elif isinstance(data, list):
        if len(data) == 0:
            return "List[Empty]"
        
        # we take the first element as a representative and continue probing downward, adding a marker
        return f"List -> {infer_json_schema(data[0])}"
    
    # 3. if we encounter a standard basic type (leaf node), return its type name directly
    else:
        if data is None:
            return "NoneType"
        return f"{type(data).__name__}"



def select_features_by_rf(X, y, threshold, max_depth, n_estimators, random_state):
    """
    Random Forest to select the feautes 
    return:
    - selected_features: list,
    - importance_df: pd.DataFrame,
    """
    print(f"Beginning: original features {X.shape[1]} )")
    # 1. Build Random Model
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=random_state, 
        n_jobs=-1
    )
    rf_model.fit(X, y)
    
    # 2. select the importance 
    importance_df = pd.DataFrame({
        'Feature': X.columns, 
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    importance_df['Cumulative_Importance'] = importance_df['Importance'].cumsum()
    passed_mask = importance_df['Cumulative_Importance'] <= threshold
    
    # error dealing
    if not passed_mask.any():
        selected_features = [importance_df['Feature'].iloc[0]]
    else:
        
        selected_features = importance_df[passed_mask]['Feature'].tolist()
        
        

    print(f"finished:original features {X.shape[1]} 个 -> selected features: {len(selected_features)} ,cumulative importance {threshold})")
    
    return selected_features, importance_df


