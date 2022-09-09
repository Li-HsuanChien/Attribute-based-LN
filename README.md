# Attribute-based Injection Transformer for Personalized Sentiment Analysis

This repo contains PyTorch deep learning models of AI-Transformer in *Attribute-based Injection Transformer for Personalized Sentiment Analysis*.

## Usage

- Downloading datasets

  All three datasets, imdb, yelp_2013, and yelp_2014, are followed by Tang. 2015 and available at [here](http://ir.hit.edu.cn/~dytang/paper/acl2015/dataset.7z).

  Unzip for getting datasets listed as the following folders:

  ```
  |-- corpus # 数据集
    |-- PSC
    	|-- imdb
    		|-- ...
    	|-- yelp_13
    		|-- ...
    	|-- yelp_14
    		|-- ...
  ```

- Running the AI-Transformer

  ```
  # run: train, val, test
  # dataset: imdb, yelp_13, yelp_14
  # gpu: 0,1 (a list of gpu ids)
  
  # running code of an instance for IMDB datasets
  python run_PSC.py --run train --dataset imdb --gpu 0,1 --version test  ```

## Noting

All Hyper-Parameters are set in the directory of cfgs.